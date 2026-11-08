"""Public campaign persistence and opening gates, using small actual files."""

import json

import pytest

from music_genre.evaluation.final_evaluation import PredictionStore


def campaign():
    return {"schema_version": 1, "split": "validation", "lock_sha256": "fixed-lock",
            "runner_sources": {"runner.py": "fixed-source"}, "labels": ["a", "b"],
            "models": {"model": "model-id"}, "track_ids": ["000001", "000002"],
            "source_sha256": {"000001": "audio1", "000002": "audio2"}}


def prediction():
    return {"schema_version": 1, "model_id": "model-id", "mode": "fma",
            "labels": ["a", "b"], "scores": [0.75, 0.25],
            "windows": [{"start_seconds": float(t), "valid_seconds": 10.,
                         "padded_seconds": 0., "scores": [0.75, 0.25]} for t in (0, 10, 20)]}


def test_resume_preserves_completed_predictions_and_requires_whole_cohort(tmp_path):
    store = PredictionStore(tmp_path, campaign())
    store.save("model", "000001", prediction())
    reopened = PredictionStore(tmp_path, campaign())
    assert reopened.read("model", "000001")["scores"] == [0.75, 0.25]
    assert reopened.read("model", "000002") is None
    with pytest.raises(ValueError, match="Incomplete"):
        reopened.receipts()
    reopened.save("model", "000002", prediction())
    assert len(reopened.receipts()) == 2


@pytest.mark.parametrize("field,value", [("lock_sha256", "changed"), ("split", "test"),
                                         ("runner_sources", {"runner.py": "new"}),
                                         ("track_ids", ["000001"])])
def test_resume_refuses_changed_provenance(tmp_path, field, value):
    PredictionStore(tmp_path, campaign())
    changed = campaign()
    changed[field] = value
    with pytest.raises(ValueError, match="Campaign differs"):
        PredictionStore(tmp_path, changed)


def test_resume_refuses_corrupted_prediction_and_unknown_files(tmp_path):
    store = PredictionStore(tmp_path, campaign())
    for track in campaign()["track_ids"]:
        store.save("model", track, prediction())
    path = store.path("model", "000001")
    record = json.loads(path.read_text("utf-8"))
    record["payload"]["prediction"]["scores"] = [0.1, 0.9]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        store.read("model", "000001")


@pytest.mark.parametrize("fault", ["nan", "label", "model", "mean", "windows", "padding"])
def test_invalid_inference_cannot_be_checkpointed(tmp_path, fault):
    store = PredictionStore(tmp_path, campaign())
    item = prediction()
    if fault == "nan":
        item["scores"][0] = float("nan")
    elif fault == "label":
        item["labels"] = ["b", "a"]
    elif fault == "model":
        item["model_id"] = "another-model"
    elif fault == "mean":
        item["scores"] = [0.5, 0.5]
    elif fault == "windows":
        item["windows"] = item["windows"][:2]
    else:
        item["windows"][0]["padded_seconds"] = 1
    with pytest.raises(ValueError):
        store.save("model", "000001", item)
    assert store.read("model", "000001") is None


def test_test_gate_requires_complete_matching_validation_evidence(tmp_path):
    from music_genre.evaluation.final_evaluation import (
        atomic_json,
        check_validation_evidence,
        json_digest,
    )

    expected = campaign()
    with pytest.raises(ValueError, match="Missing validation"):
        check_validation_evidence(tmp_path, expected)
    store = PredictionStore(tmp_path, expected)
    for track in expected["track_ids"]:
        store.save("model", track, prediction())
    completion = {"status": "validation_passed", "campaign_sha256": json_digest(expected),
                  "predictions": store.receipts(), "systems": {
                      "model": {"compared_tracks": 2, "max_probability_error": 0.,
                                "class_disagreements": 0}}}
    atomic_json(tmp_path / "completion.json", completion)
    assert check_validation_evidence(tmp_path, expected)["status"] == "validation_passed"
    completion["systems"]["model"]["max_probability_error"] = 0.002
    atomic_json(tmp_path / "completion.json", completion)
    with pytest.raises(ValueError, match="Validation comparison"):
        check_validation_evidence(tmp_path, expected)


def test_campaign_has_one_writer_and_can_resume_after_release(tmp_path):
    from music_genre.evaluation.final_evaluation import exclusive_campaign

    with (exclusive_campaign(tmp_path),
          pytest.raises(RuntimeError, match="already running"), exclusive_campaign(tmp_path)):
        pass
    with exclusive_campaign(tmp_path):
        PredictionStore(tmp_path, campaign())


def test_validation_comparison_requires_matching_ids_and_saved_scores(tmp_path):
    from music_genre.evaluation.final_evaluation import compare_validation

    store = PredictionStore(tmp_path, campaign())
    for track in campaign()["track_ids"]:
        store.save("model", track, prediction())
    reference = [{"track_id": t, "p_a": "0.75", "p_b": "0.25"}
                 for t in reversed(campaign()["track_ids"])]
    assert compare_validation(store, "model", reference) == {
        "compared_tracks": 2, "max_probability_error": 0., "class_disagreements": 0}
    with pytest.raises(ValueError, match="cohort"):
        compare_validation(store, "model", reference[:1])
    reference[0]["p_a"], reference[0]["p_b"] = "0.8", "0.2"
    with pytest.raises(ValueError, match="Validation comparison"):
        compare_validation(store, "model", reference)


def test_interrupted_atomic_write_does_not_prevent_fresh_campaign(tmp_path):
    (tmp_path / ".pending-example.tmp").write_text('{"interrupted":')
    store = PredictionStore(tmp_path, campaign())
    store.save("model", "000001", prediction())
    assert store.read("model", "000001")["scores"] == [0.75, 0.25]
