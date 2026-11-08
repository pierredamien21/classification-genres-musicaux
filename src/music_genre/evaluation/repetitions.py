"""Validate controlled CNN repetitions and summarize across seeds, not tracks."""

import numpy as np


def summarize_repetitions(reports, seeds):
    actual = [r["config"]["seed"] for r in reports]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or actual != seeds:
        raise ValueError("Expected exactly the distinct prespecified seeds in order")
    reference = reports[0]
    config = {k: v for k, v in reference["config"].items() if k != "seed"}
    for report in reports:
        if report["status"] != "complete" or report["test_loaded"] or report["test_evaluated"]:
            raise ValueError("Expected complete validation-only runs")
        if {k: v for k, v in report["config"].items() if k != "seed"} != config:
            raise ValueError("Repetitions must change only the seed")
        for field in ("labels", "baseline_manifest_sha256", "feature_cache", "feature_report_sha256",
                      "index_sha256", "track_counts", "window_counts", "parameters"):
            if report[field] != reference[field]:
                raise ValueError(f"Incomparable repetitions: {field}")
        for field in ("torch_version", "numpy_version", "benchmark_sha256"):
            if report["identity"][field] != reference["identity"][field]:
                raise ValueError(f"Changed training environment: {field}")
        for name, expected in reference["identity"]["source_sha256"].items():
            if name != "scripts/train_cnn.py" and report["identity"]["source_sha256"][name] != expected:
                raise ValueError("Training or model implementation changed")
    summary = {}
    for metric in ("macro_f1", "accuracy"):
        values = np.array([r["validation"][metric] for r in reports], dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid scores")
        summary[metric] = {"values": values.tolist(), "mean": float(values.mean()),
                           "sample_std": float(values.std(ddof=1)),
                           "min": float(values.min()), "max": float(values.max())}
    return summary
