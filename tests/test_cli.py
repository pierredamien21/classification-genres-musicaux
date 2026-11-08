import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_genre.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_does_not_import_the_ml_stack():
    code = ("import sys; from music_genre.cli import main; main(['doctor', '--root', '.']);"
            "assert 'torch' not in sys.modules and 'librosa' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_predict_prints_the_shared_prediction_json(tmp_path, capsys):
    if not (ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").exists():
        pytest.skip("Pinned local model required")
    path = tmp_path / "tone.wav"
    sf.write(path, .1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000), 16000, subtype="FLOAT")
    assert main(["predict", str(path), "--root", str(ROOT)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "fma" and len(result["scores"]) == 8 and len(result["windows"]) == 1


@pytest.mark.parametrize("arguments", [["--system", "unknown"], ["--system", "vit", "--mode", "overview"]])
def test_predict_rejects_unknown_system_or_unsupported_mode(tmp_path, arguments):
    with pytest.raises(SystemExit):
        main(["predict", str(tmp_path / "missing.wav"), "--root", str(ROOT), *arguments])
