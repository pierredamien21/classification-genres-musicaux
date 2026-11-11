import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# These modules check the frozen V1 exports; their runs, caches and manifests are local, not versioned.
NEEDS_V1_EXPORTS = {"test_cli.py", "test_cnn_inference.py", "test_demo_inference.py", "test_inference.py",
                    "test_inference_adapters.py", "test_linear_comparators.py"}


def missing_v1_exports():
    prepared = json.loads((ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").read_text(encoding="utf-8"))
    needed = {prepared["sample"]["path"], "data/manifests/robustness-validation-v1.csv"}
    for pin in prepared["runs"].values():
        needed |= {pin["model_path"], f"{pin['run']['folder']}/report.json", f"{pin['feature_cache']}/report.json"}
    return sorted(p for p in needed if not (ROOT / p).exists())


def pytest_collection_modifyitems(config, items):
    if not any(Path(item.fspath).name in NEEDS_V1_EXPORTS for item in items):
        return
    missing = missing_v1_exports()
    if missing:
        skip = pytest.mark.skip(reason=f"local V1 exports absent (not versioned), e.g. {missing[0]}")
        for item in items:
            if Path(item.fspath).name in NEEDS_V1_EXPORTS:
                item.add_marker(skip)
