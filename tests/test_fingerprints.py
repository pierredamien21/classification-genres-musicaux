import json
from pathlib import Path

import numpy as np

from music_genre.data.fingerprints import compare_fingerprints, find_candidates

SETTINGS = json.loads((Path(__file__).resolve().parents[1] /
                       "configs/review_policy.json").read_text())["fingerprint"]


def test_alignment_with_bit_errors_and_shift():
    rng = np.random.default_rng(123)
    a = rng.integers(0, 2**32, size=220).tolist()
    b = [0] * 12 + a[:-12]
    for i in range(20, 170, 2):
        b[i] ^= 3
    score = compare_fingerprints(a, b, SETTINGS)
    assert score["offset_words"] == 12
    assert score["overlap_words"] == 208
    assert 0 < score["bit_error_rate"] < 0.1
    matches = find_candidates([{"track_id": "a", "returncode": 0, "fingerprint": a},
                               {"track_id": "b", "returncode": 0, "fingerprint": b}], SETTINGS)
    assert len(matches) == 1


def test_low_information_and_unrelated_fingerprints_not_matches():
    rng = np.random.default_rng(456)
    records = [{"track_id": str(i), "returncode": 0,
                "fingerprint": rng.integers(0, 2**32, size=220).tolist()} for i in range(4)]
    records += [{"track_id": "silent1", "returncode": 0, "fingerprint": [123] * 220},
                {"track_id": "silent2", "returncode": 0, "fingerprint": [123] * 220}]
    assert find_candidates(records, SETTINGS) == []
