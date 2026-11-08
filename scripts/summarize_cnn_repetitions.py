"""Verify three planned CNN runs and report variability without seed selection."""

import csv
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import classification_metrics
from music_genre.evaluation.repetitions import summarize_repetitions

ROOT = Path(__file__).resolve().parents[1]


def plot_histories(reports):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/interim/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for report in reports:
        epochs = [r["epoch"] for r in report["history"]]
        scores = [r["validation"]["macro_f1"] for r in report["history"]]
        label = f"Graine {report['config']['seed']}"
        axes[0].plot(epochs, [r["training_window_loss"] for r in report["history"]],
                     marker="o", markersize=3, label=label)
        line, = axes[1].plot(epochs, scores, marker="o", markersize=3, label=label)
        chosen = report["selected_epoch"]
        axes[1].scatter([chosen], [scores[chosen - 1]], s=120, marker="*", color=line.get_color(), zorder=5)
    axes[0].set_ylabel("Entropie croisée moyenne par fenêtre")
    axes[0].set_title("Apprentissage sur train")
    axes[1].set_ylabel("Macro-F1 par morceau")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation — étoile : époque retenue")
    for ax in axes:
        ax.set_xlabel("Époque")
        ax.set_xticks(range(1, max(r["epochs_completed"] for r in reports) + 1))
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    fig.suptitle("CNN : trois graines, même protocole — test non évalué")
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"cnn_repetitions.{ext}", dpi=160)
    plt.close(fig)
    return {ext: sha256_file(folder / f"cnn_repetitions.{ext}") for ext in ("png", "pdf")}


def read_repetitions():
    plan = json.loads((ROOT / "configs/cnn_repetitions.json").read_text(encoding="utf-8"))
    if (sha256_file(ROOT / "configs/cnn.json") != plan["base_config_sha256"]
            or sha256_file(ROOT / "scripts/train_cnn.py") != plan["new_trainer_sha256"]
            or sha256_file(ROOT / "reports/CNN_BENCHMARK.json") != plan["benchmark_sha256"]):
        raise ValueError("Prespecified protocol changed")
    pointers, reports, keys_reference = [], [], None
    for seed in plan["seeds"]:
        pointer = (plan["reference_run"] if seed == plan["base_config"]["seed"] else
                   json.loads((ROOT / f"reports/latest_cnn_seed{seed}_run.json").read_text(encoding="utf-8")))
        folder = (ROOT / pointer["folder"]).resolve()
        if (not folder.is_relative_to((ROOT / "runs").resolve())
                or sha256_file(folder / "report.json") != pointer["report_sha256"]):
            raise ValueError("Run report changed")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        expected_config = {**plan["base_config"], "seed": seed}
        expected_sources = dict(plan["reference_source_sha256"])
        if seed != plan["base_config"]["seed"]:
            expected_sources["scripts/train_cnn.py"] = plan["new_trainer_sha256"]
        if (report["config"] != expected_config or report["identity"]["config"] != expected_config
                or report["identity"]["source_sha256"] != expected_sources
                or report["identity"]["benchmark_sha256"] != plan["benchmark_sha256"]):
            raise ValueError("Run does not follow the prespecified training protocol")
        for filename, field in (("model.safetensors", "model_sha256"),
                                ("validation_predictions.csv", "validation_predictions_sha256")):
            if sha256_file(folder / filename) != report[field]:
                raise ValueError("Model or predictions changed")
        cache = (ROOT / report["feature_cache"]).resolve()
        if (not cache.is_relative_to((ROOT / "data/features").resolve())
                or sha256_file(cache / "report.json") != report["feature_report_sha256"]
                or sha256_file(cache / "index.csv") != report["index_sha256"]):
            raise ValueError("Cache provenance changed")
        with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
            expected = sorted({(r["track_id"], report["labels"][int(r["label_index"])])
                               for r in csv.DictReader(stream) if r["split"] == "validation"})
        with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = [(r["track_id"], r["true_label"]) for r in rows]
        if keys_reference is None:
            keys_reference = keys
        if keys != expected or keys != keys_reference:
            raise ValueError("Incomplete or misaligned validation predictions")
        labels = report["labels"]
        truth = np.array([labels.index(r["true_label"]) for r in rows])
        probabilities = np.array([[float(r[f"p_{label}"]) for label in labels] for r in rows])
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1)):
            raise ValueError("Invalid probabilities")
        metrics = classification_metrics(truth, probabilities, labels)
        history = report["history"]
        if [r["epoch"] for r in history] != list(range(1, report["epochs_completed"] + 1)):
            raise ValueError("Incomplete epoch history")
        best = max(history, key=lambda r: r["validation"]["macro_f1"])
        if (metrics != report["validation"] or best["validation"] != metrics
                or best["epoch"] != report["selected_epoch"]
                or report["export_max_probability_error"] > 1e-6):
            raise ValueError("Selection, metrics or export mismatch")
        pointers.append(pointer)
        reports.append(report)
    return plan, pointers, reports, summarize_repetitions(reports, plan["seeds"])


def main():
    plan, pointers, reports, summary = read_repetitions()
    near_cap = [r["config"]["seed"] for r in reports
                if r["selected_epoch"] >= r["config"]["max_epochs"] - 1]
    per_class = {}
    for label in reports[0]["labels"]:
        values = [r["validation"]["per_class"][label]["f1-score"] for r in reports]
        per_class[label] = {"values": values, "mean": float(np.mean(values)),
                            "sample_std": float(np.std(values, ddof=1))}
    result = {
        "schema_version": 1, "plan_sha256": sha256_file(ROOT / "configs/cnn_repetitions.json"),
        "seeds": plan["seeds"], "runs": pointers, "summary": summary, "per_class_f1": per_class,
        "selected_epochs": [r["selected_epoch"] for r in reports],
        "seeds_with_selected_epoch_in_last_two_budgeted_epochs": near_cap,
        "status": "complete", "test_evaluated": False, "ensemble_evaluated": False,
        "baseline_manifest_sha256": reports[0]["baseline_manifest_sha256"],
        "figure_sha256": plot_histories(reports), "matplotlib_version": version("matplotlib"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/summarize_cnn_repetitions.py", "src/music_genre/evaluation/repetitions.py")},
    }
    write_json(ROOT / "reports/CNN_REPETITIONS.json", result)
    write_json(ROOT / "reports/latest_cnn_repetitions.json", {
        "report": "reports/CNN_REPETITIONS.json",
        "report_sha256": sha256_file(ROOT / "reports/CNN_REPETITIONS.json"),
    })
    lines = ["# CNN : stabilité sur trois graines", "",
             "Même protocole, mêmes morceaux ; seules initialisation et permutation des lots changent.", "",
             "| Graine | Époques réalisées | Époque choisie | Macro-F1 validation | Accuracy | Macro-F1 train |",
             "|---:|---:|---:|---:|---:|---:|"]
    for seed, r in zip(plan["seeds"], reports, strict=True):
        lines.append(f"| {seed} | {r['epochs_completed']} | {r['selected_epoch']} | "
                     f"{r['validation']['macro_f1']:.4f} | {r['validation']['accuracy']:.4f} | "
                     f"{r['training']['macro_f1']:.4f} |")
    lines += ["", "## Moyenne et dispersion entre graines", ""]
    for metric, values in summary.items():
        lines.append(f"- {metric} : **{values['mean']:.4f} ± {values['sample_std']:.4f}** ; "
                     f"minimum {values['min']:.4f}, maximum {values['max']:.4f}.")
    lines += ["", ("Écart-type d’échantillon, n − 1 = 2. Ces valeurs résument trois entraînements, "
                    "pas un ensemble de réseaux ni un intervalle de confiance sur les artistes. "
                    "L’époque de chaque run est choisie sur validation ; aucune graine n’est éliminée."), "",
              "## F1 par genre", "", "| Genre | Graine 42 | Graine 43 | Graine 44 | Moyenne | Écart-type |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, values in per_class.items():
        lines.append(f"| {label} | " + " | ".join(f"{x:.4f}" for x in values["values"])
                     + f" | {values['mean']:.4f} | {values['sample_std']:.4f} |")
    lines += ["", "## Conséquence pour la suite", "",
              (f"Pour {len(near_cap)} graine(s) sur trois, l’époque retenue appartient aux deux "
               f"dernières époques du budget : {near_cap}. Cela signale une limite de durée à examiner, "
               "sans démontrer à lui seul un défaut de convergence."), ""]
    if len(near_cap) >= 2:
        lines += [("Piste prioritaire issue de ces résultats : préparer une comparaison avec un plafond "
                   "de vingt époques, sur les mêmes graines et sans changer architecture, optimiseur "
                   "ou patience. Le coût doit être vérifié et le nouveau protocole consigné avant "
                   "l’exécution. Cette extension n’est pas réalisée dans le présent rapport. "
                   "Elle peut modifier le classement ; le test reste réservé."), ""]
    lines += ["## Courbes d’apprentissage", "",
              "![Courbes des trois graines](figures/cnn_repetitions.png)", "",
              "[Figure PDF](figures/cnn_repetitions.pdf). Les étoiles indiquent les époques sélectionnées.", "",
              "## Contrôles et limites", "",
              ("Rapports, poids et prédictions vérifiés par empreintes ; métriques recalculées, "
               "sélection d’époque revérifiée, validation alignée avec le cache. Versions numériques, "
               "modèle, code d’apprentissage, données et configuration identiques sauf graine. "
               "Le script a seulement reçu une option de graine et des pointeurs séparés ; "
               "les empreintes des deux versions sont conservées dans le protocole."), "",
              ("La graine 42 provient du développement antérieur et est réutilisée, sans prétendre "
               "qu’il s’agit de trois nouveaux runs indépendants du choix de la recette. Le plafond "
               "de dix époques ne démontre pas la convergence maximale du CNN. Les durées dépendent "
               "de la charge du poste ; aucune comparaison contrôlée de vitesse n’est revendiquée."), "",
              ("Le test reste non évalué. Un point de décision sur validation précède tout verrouillage : "
               "analyse des erreurs, nouvelles hypothèses et budget peuvent faire évoluer le plan."), "",
              "## Reproduction", "", "Voir [le protocole](../docs/REPETITIONS_CNN.md).", ""]
    for seed, pointer in zip(plan["seeds"], pointers, strict=True):
        lines.append(f"- [Rapport graine {seed}](../{pointer['folder']}/report.json)")
    lines += ["- [Résumé numérique](CNN_REPETITIONS.json)", ""]
    (ROOT / "reports/REPETITIONS_CNN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
