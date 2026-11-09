"""Collect the latest runs/v2-cv/<system>-<ts>/report.json per system into reports/v2/cv_results.csv.

Each report's out-of-fold file is re-hashed; the table keeps the report SHA-256 as evidence.
Systems are sorted by mean outer-fold macro-F1 (the V2 selection statistic).
"""

import csv
import json
from pathlib import Path

from music_genre.data.audit import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/v2/cv_results.csv"


def main():
    latest = {}
    for path in sorted((ROOT / "runs/v2-cv").glob("*/report.json")):  # timestamped names: last wins
        report = json.loads(path.read_text(encoding="utf-8"))
        if sha256_file(path.parent / "oof_tracks.csv") != report["oof_tracks_sha256"]:
            raise ValueError(f"Out-of-fold file changed: {path.parent.name}")
        latest[report.get("system", report["features"])] = (path, report)
    rows = []
    for system, (path, report) in latest.items():
        rows.append({
            "system": system, "dimensions": report["feature_dimensions"],
            "mean_fold_macro_f1": f"{report['mean_fold_macro_f1']:.4f}",
            "std_fold_macro_f1": f"{report['std_fold_macro_f1']:.4f}",
            "pooled_oof_macro_f1": f"{report['pooled_oof_macro_f1']:.4f}",
            "selected_c_per_fold": " ".join(str(r["selected_c"]) for r in report["outer_folds"]),
            "final_c": report["final_c"], "minutes": round(report["elapsed_seconds"] / 60, 1),
            "run": path.parent.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(path),
        })
    rows.sort(key=lambda r: -float(r["mean_fold_macro_f1"]))
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['system']:24} {row['mean_fold_macro_f1']} ± {row['std_fold_macro_f1']}  C={row['final_c']}")


if __name__ == "__main__":
    main()
