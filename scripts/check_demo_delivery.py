"""Exercise the real demo callback without a browser or a new experiment."""

import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("music_demo", ROOT / "app/main.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
from music_genre.data.audit import sha256_file, write_json


def main():
    manifest = ROOT / "data/manifests/baseline-20261020T142750008115Z/manifest.csv"
    with manifest.open(encoding="utf-8") as stream:
        row = next(r for r in csv.DictReader(stream) if r["split"] == "validation")
    source = ROOT / "data/raw" / row["audio_path"]
    assert sha256_file(source) == row["source_sha256"]
    result = app.analyze(str(source), "fma")
    assert result[4] is not None and result[3] is not None, result[0]
    saved_path = ROOT / "data/interim/final-validation-v1/predictions/vit_block6_hpss" / (row["track_id"] + ".json")
    saved = json.loads(saved_path.read_text())["payload"]["prediction"]
    assert result[4]["model_id"] == saved["model_id"]
    error = float(np.max(np.abs(np.asarray(saved["scores"]) - result[4]["scores"])))
    assert error <= 1e-6
    assert app.analyze(None, "fma")[4] is None
    with tempfile.TemporaryDirectory(dir=ROOT / "data/interim") as temp:
        silence = Path(temp) / "silence.wav"
        sf.write(silence, np.zeros(1600), 16000)
        corrupt = Path(temp) / "corrupt.mp3"
        corrupt.write_bytes(b"invalid")
        for path in (silence, corrupt):
            failed = app.analyze(str(path), "overview")
            assert failed[0].startswith("Analyse impossible")
            assert failed[1:] == (None, [], None, None)
    write_json(ROOT / "reports/DEMO_CALLBACK_CHECK.json", {
        "status": "passed", "validation_track": row["track_id"],
        "max_probability_error": error, "model_id": saved["model_id"],
        "source_sha256": {p: sha256_file(ROOT / p) for p in (
            "app/main.py", "src/music_genre/inference/demo.py", "scripts/check_demo_delivery.py")},
        "checks": ["pinned_validation_prediction", "plot", "empty_input", "digital_silence", "corrupt_mp3"],
        "browser_check": "separate", "test_audio_loaded": False,
    })
    print(f"Demo callback passed; validation probability difference {error:.3g}")


if __name__ == "__main__":
    main()
