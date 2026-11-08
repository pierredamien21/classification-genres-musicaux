import pytest

from music_genre.inference.demo import overview_starts


@pytest.mark.parametrize("duration,expected", [
    (0.5, (0.0,)), (10.0, (0.0,)), (30.0, (0.0, 10.0, 20.0)),
    (60.0, (0.0, 25.0, 50.0)),
])
def test_demo_covers_beginning_middle_end_without_duplicate_short_votes(duration, expected):
    assert overview_starts(duration) == expected


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_demo_refuses_invalid_duration(duration):
    with pytest.raises(ValueError):
        overview_starts(duration)


def test_demo_keeps_fma_predictions_and_pads_short_overview(tmp_path):
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    from music_genre.inference.demo import DemoPredictor
    from music_genre.inference.predict import CandidatePredictor

    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/PREPARATION_EVALUATION_FINALE.json").exists():
        pytest.skip("Pinned local model required")
    predictor = DemoPredictor(root)
    path = tmp_path / "short.wav"
    tone = .1 * np.sin(2 * np.pi * 440 * np.arange(4000) / 8000)
    sf.write(path, np.column_stack((tone, tone)), 8000, subtype="FLOAT")
    fixed = CandidatePredictor.predict_file(predictor, path, mode="fma")
    demo = predictor.predict_file(path, mode="overview")
    assert len(demo["windows"]) == 1
    assert demo["windows"][0]["valid_seconds"] == .5
    assert demo["windows"][0]["padded_seconds"] == 9.5
    np.testing.assert_array_equal(demo["scores"], fixed["scores"])
    sf.write(path, np.zeros(4000), 8000)
    with pytest.raises(ValueError, match="silent"):
        predictor.predict_file(path)
