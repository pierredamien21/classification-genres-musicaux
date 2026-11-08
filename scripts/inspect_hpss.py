"""Training-only HPSS diagnostics and a shared-scale figure; no model tuning."""

import csv
import json
import os
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/interim/matplotlib"))
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.features.hpss import hpss_log_mel


def main():
    benchmark_path = ROOT / "reports/VIT_HPSS_BENCHMARK.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    config = benchmark["provenance"]["config"]
    for source, expected in benchmark["provenance"]["source_sha256"].items():
        if sha256_file(ROOT / source) != expected:
            raise ValueError("Extraction sources changed since benchmark")
    for name, expected in benchmark["provenance"]["versions"].items():
        if version(name) != expected:
            raise ValueError("Extraction dependencies changed since benchmark")
    pointer = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text(encoding="utf-8"))
    manifest = (ROOT / pointer["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != benchmark["provenance"]["manifest_sha256"]):
        raise ValueError("Manifest differs from benchmark")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = {r["track_id"]: r for r in csv.DictReader(stream)}
    samples = benchmark["samples"]
    if benchmark["split"] != "train" or len(samples) != 64:
        raise ValueError("Expected the 64 training benchmark windows")
    figure_id = next(r["track_id"] for r in samples if r["genre"] == "Folk")
    names = ["Original", "Harmonique", "Percussif"]
    statistics, figure_values = [], None
    with threadpool_limits(limits=1):
        for sample in samples:
            row = rows[sample["track_id"]]
            if row["split"] != "train" or row["status"] != "valid" or row["decision"] != "keep":
                raise ValueError("Diagnostics are restricted to selected training tracks")
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if (not path.is_relative_to((ROOT / "data/raw").resolve())
                    or sha256_file(path) != row["source_sha256"]):
                raise ValueError("Audio source changed")
            window = segment_audio(load_audio(path, sample_rate=config["sample_rate"]),
                                   (10.0,), config["window_seconds"])[0]
            values = hpss_log_mel(window, config["logmel"], config["hpss"])
            if row["track_id"] == figure_id:
                figure_values = values.copy()
            for name, channel in zip(names, values, strict=True):
                statistics.append({
                    "track_id": row["track_id"], "genre": row["genre"], "channel": name,
                    "below_db_min_fraction": float(np.mean(channel < config["db_min"])),
                    "above_db_max_fraction": float(np.mean(channel > config["db_max"])),
                })
    folder = ROOT / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    time_centers = (np.arange(figure_values.shape[-1]) * config["logmel"]["hop_length"]
                    + config["logmel"]["n_fft"] / 2) / config["sample_rate"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for ax, name, channel in zip(axes, names, figure_values, strict=True):
        mesh = ax.pcolormesh(time_centers, np.arange(channel.shape[0]), channel,
                             shading="nearest", cmap="magma", vmin=config["db_min"],
                             vmax=config["db_max"], rasterized=True)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel("Bande Mel")
        ax.set_xlim(0, config["window_seconds"])
    axes[-1].set_xlabel("Temps dans la fenêtre (secondes)")
    fig.colorbar(mesh, ax=axes, label="Puissance Mel (dB, référence 1)", extend="both", shrink=0.85)
    fig.suptitle(f"HPSS avant redimensionnement — FMA {figure_id}, Folk, train\n"
                 "Extrait 10–20 s · même échelle dB sur les trois canaux", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"hpss_channels.{extension}", dpi=160)
    plt.close(fig)
    means = {
        name: {field: float(np.mean([r[field] for r in statistics if r["channel"] == name]))
               for field in ("below_db_min_fraction", "above_db_max_fraction")}
        for name in names
    }
    report = {
        "benchmark_sha256": sha256_file(benchmark_path), "config": config,
        "sample_count": len(samples), "split": "train", "test_audio_loaded": False,
        "figure_track_id": figure_id, "figure_selection": "first_Folk_track_in_fixed_benchmark",
        "statistics_stage": "native_logmel_before_clipping_and_resize",
        "mean_clipped_fractions": means, "per_window": statistics,
        "source_sha256": sha256_file(Path(__file__)), "matplotlib_version": version("matplotlib"),
        "figure_sha256": {ext: sha256_file(folder / f"hpss_channels.{ext}") for ext in ("png", "pdf")},
    }
    write_json(ROOT / "reports/HPSS_INPUT_DIAGNOSTICS.json", report)
    print(json.dumps({"figure_track_id": figure_id, "mean_clipped_fractions": means}, indent=2))


if __name__ == "__main__":
    main()
