"""Verify paired CNN continuations and report the adaptive budget comparison."""

import csv
import json
import os
from pathlib import Path

import numpy as np
from summarize_cnn_repetitions import read_repetitions

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import classification_metrics
from music_genre.evaluation.repetitions import summarize_repetitions
from music_genre.training.cnn import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def read_extension():
    plan_path = ROOT / "configs/cnn_extension.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _, parent_pointers, parents, previous = read_repetitions()
    if sha256_file(ROOT / "configs/cnn_repetitions.json") != plan["parent_plan_sha256"]:
        raise ValueError("Parent protocol changed")
    for name, expected in plan["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Extension source changed")
    reports, pointers = [], []
    for seed, parent, parent_pointer, frozen in zip(
            plan["seeds"], parents, parent_pointers, plan["parents"], strict=True):
        parent_folder = ROOT / parent_pointer["folder"]
        if (frozen["seed"] != seed
                or {k: frozen[k] for k in ("folder", "report_sha256")} != parent_pointer
                or sha256_file(parent_folder / "checkpoint.pt") != frozen["checkpoint_sha256"]):
            raise ValueError("Parent checkpoint or pointer changed")
        pointer = json.loads((ROOT / f"reports/latest_cnn20_seed{seed}_run.json").read_text(encoding="utf-8"))
        folder = (ROOT / pointer["folder"]).resolve()
        if (not folder.is_relative_to((ROOT / "runs").resolve())
                or sha256_file(folder / "report.json") != pointer["report_sha256"]):
            raise ValueError("Invalid extension run")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        expected_config = {**parent["config"], "max_epochs": plan["max_epochs"]}
        expected_identity = {**parent["identity"], "config": expected_config,
                             "source_sha256": plan["source_sha256"], "parent": frozen,
                             "extension_plan_sha256": sha256_file(plan_path)}
        if (report["config"] != expected_config or report["identity"] != expected_identity
                or report["parent"] != frozen or report["extension_plan_sha256"] != sha256_file(plan_path)):
            raise ValueError("Extension does not match frozen protocol")
        for field in ("labels", "baseline_manifest_sha256", "feature_cache", "feature_report_sha256",
                      "index_sha256", "track_counts", "window_counts", "parameters"):
            if report[field] != parent[field]:
                raise ValueError(f"Extension changed {field}")
        for filename, key in (("model.safetensors", "model_sha256"),
                              ("validation_predictions.csv", "validation_predictions_sha256")):
            if sha256_file(folder / filename) != report[key]:
                raise ValueError("Extension artifact changed")
        with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (parent_folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
            parent_rows = list(csv.DictReader(stream))
        if [(r["track_id"], r["true_label"]) for r in rows] != [
                (r["track_id"], r["true_label"]) for r in parent_rows]:
            raise ValueError("Validation tracks changed")
        labels = report["labels"]
        truth = np.array([labels.index(r["true_label"]) for r in rows])
        probabilities = np.array([[float(r[f"p_{label}"]) for label in labels] for r in rows])
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1)):
            raise ValueError("Invalid extension probabilities")
        metrics = classification_metrics(truth, probabilities, labels)
        history = report["history"]
        if (history[:parent["epochs_completed"]] != parent["history"]
                or [r["epoch"] for r in history] != list(range(1, report["epochs_completed"] + 1))
                or not parent["epochs_completed"] < len(history) <= plan["max_epochs"]):
            raise ValueError("Parent history or epoch continuity changed")
        best = max(history, key=lambda r: r["validation"]["macro_f1"])
        if (metrics != report["validation"] or metrics != best["validation"]
                or best["epoch"] != report["selected_epoch"]
                or not 0 <= report["export_max_probability_error"] <= 1e-6):
            raise ValueError("Extension selection or export mismatch")
        best_epoch, score = 0, -1.0
        for row in history:
            if row["epoch"] - 1 - best_epoch >= report["config"]["patience"]:
                raise ValueError("Training continued after patience was exhausted")
            if row["validation"]["macro_f1"] > score:
                score, best_epoch = row["validation"]["macro_f1"], row["epoch"]
        stop = report["stop_reason"]
        if not ((stop == "max_epochs" and len(history) == plan["max_epochs"])
                or (stop == "early_stopping" and len(history) - best_epoch >= report["config"]["patience"])):
            raise ValueError("Unjustified stopping condition")
        reports.append(report)
        pointers.append(pointer)
    summary = summarize_repetitions(reports, plan["seeds"])
    differences = {}
    for metric in ("macro_f1", "accuracy"):
        values = np.array(summary[metric]["values"]) - np.array(previous[metric]["values"])
        differences[metric] = {"values": values.tolist(), "mean": float(values.mean()),
                               "sample_std": float(values.std(ddof=1))}
    return plan, pointers, parents, reports, summary, differences


def plot_histories(reports):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/interim/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for report in reports:
        history = report["history"]
        epochs = [r["epoch"] for r in history]
        label = f"Graine {report['config']['seed']}"
        axes[0].plot(epochs, [r["training_window_loss"] for r in history], marker="o", markersize=3, label=label)
        line, = axes[1].plot(epochs, [r["validation"]["macro_f1"] for r in history],
                            marker="o", markersize=3, label=label)
        axes[1].scatter([report["selected_epoch"]], [report["validation"]["macro_f1"]],
                        s=100, marker="*", color=line.get_color(), zorder=5)
    axes[0].set(title="Perte sur train", ylabel="Entropie croisée par fenêtre")
    axes[1].set(title="Validation — étoile : époque retenue", ylabel="Macro-F1 par morceau", ylim=(0, 1))
    for ax in axes:
        ax.axvline(10, color="gray", linestyle="--", label="Plafond initial")
        ax.set_xlabel("Époque")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("CNN : prolongation contrôlée — plafond 20, patience 3")
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"cnn_extension.{ext}", dpi=160)
    plt.close(fig)
    return {ext: sha256_file(folder / f"cnn_extension.{ext}") for ext in ("png", "pdf")}


def main():
    plan, pointers, parents, reports, summary, differences = read_extension()
    costs = []
    for frozen, parent, report in zip(plan["parents"], parents, reports, strict=True):
        inherited = load_checkpoint(ROOT / frozen["folder"] / "checkpoint.pt", parent["identity"])["elapsed_seconds"]
        costs.append({
            "seed": report["config"]["seed"],
            "additional_training_validation_seconds": report["training_validation_seconds"] - inherited,
            "additional_seconds_including_final_export":
                report["elapsed_including_export_and_train_evaluation_seconds"] - inherited,
        })
    per_class = {}
    for label in reports[0]["labels"]:
        old = np.array([r["validation"]["per_class"][label]["f1-score"] for r in parents])
        new = np.array([r["validation"]["per_class"][label]["f1-score"] for r in reports])
        per_class[label] = {"before_mean": float(old.mean()), "after_mean": float(new.mean()),
                            "after_sample_std": float(new.std(ddof=1)), "delta_mean": float((new - old).mean())}
    result = {"schema_version": 1, "status": "complete", "seeds": plan["seeds"], "runs": pointers,
              "parents": plan["parents"], "plan_sha256": sha256_file(ROOT / "configs/cnn_extension.json"),
              "summary": summary, "paired_differences": differences, "per_class_f1": per_class,
              "selected_epochs": [r["selected_epoch"] for r in reports],
              "epochs_completed": [r["epochs_completed"] for r in reports],
              "measured_continuation_costs": costs,
              "figure_sha256": plot_histories(reports), "test_evaluated": False, "ensemble_evaluated": False,
              "source_sha256": sha256_file(Path(__file__))}
    write_json(ROOT / "reports/CNN_EXTENSION.json", result)
    write_json(ROOT / "reports/latest_cnn_extension.json", {
        "report": "reports/CNN_EXTENSION.json", "report_sha256": sha256_file(ROOT / "reports/CNN_EXTENSION.json")})
    lines = ["# CNN : comparaison des plafonds de dix et vingt époques", "",
             "Prolongation adaptative des trois runs, sans réinitialiser optimiseur, patience ou générateur aléatoire.", "",
             "| Graine | Époques réalisées | Retenue avant → après | F1 avant | F1 après | Différence | Arrêt |",
             "|---:|---:|---:|---:|---:|---:|---|"]
    for seed, old, new in zip(plan["seeds"], parents, reports, strict=True):
        delta = new["validation"]["macro_f1"] - old["validation"]["macro_f1"]
        lines.append(f"| {seed} | {new['epochs_completed']} | {old['selected_epoch']} → {new['selected_epoch']} | "
                     f"{old['validation']['macro_f1']:.4f} | {new['validation']['macro_f1']:.4f} | "
                     f"{delta:+.4f} | {new['stop_reason']} |")
    lines += ["", "## Moyenne et dispersion", ""]
    for metric, stats in summary.items():
        lines.append(f"- {metric} : **{stats['mean']:.4f} ± {stats['sample_std']:.4f}** ; "
                     f"différence moyenne appariée : {differences[metric]['mean']:+.4f}.")
    lines += ["", "Écart-type d’échantillon entre trois graines ; aucun ensemble ni choix de la meilleure graine.", "",
              "## Genres", "", "| Genre | F1 moyenne avant | Après | Écart-type après | Différence |",
              "|---|---:|---:|---:|---:|"]
    for label, stats in per_class.items():
        lines.append(f"| {label} | {stats['before_mean']:.4f} | {stats['after_mean']:.4f} | "
                     f"{stats['after_sample_std']:.4f} | {stats['delta_mean']:+.4f} |")
    lines += ["", "## Coût observé", "",
              "| Graine | Apprentissage + validation supplémentaires (min) | Avec export final et évaluation train (min) |",
              "|---:|---:|---:|"]
    for cost in costs:
        lines.append(f"| {cost['seed']} | {cost['additional_training_validation_seconds'] / 60:.2f} | "
                     f"{cost['additional_seconds_including_final_export'] / 60:.2f} |")
    lines += ["", ("Temps cumulés des invocations d’apprentissage après le checkpoint parent, hors initialisation "
                    "et vérification initiale du cache. Les pauses entre invocations ne sont pas comptées. "
                    "L’estimation initiale avec marge était de 69 minutes ; les arrêts anticipés peuvent la réduire."), "",
              "## Courbes", "", "![Courbes prolongées](figures/cnn_extension.png)", "",
              "[Figure PDF](figures/cnn_extension.pdf). Les dix premières époques sont héritées et vérifiées à l’identique.", "",
              "## Interprétation et limites", "",
              ("L’extension a été décidée après examen des courbes à dix époques. Elle est une expérience de développement "
              "adaptative, pas une confirmation indépendante. Le meilleur score est recherché parmi davantage d’époques : "
              "une hausse sur cette validation ne démontre pas à elle seule une meilleure généralisation."), "",
              ("Le plafond est vingt époques ; la patience reste trois. Un arrêt anticipé ne prouve pas la convergence "
              "maximale de l’architecture. Trois graines ne permettent pas de revendiquer une significativité statistique "
              "sur les artistes. Le test reste non chargé et non évalué."), "",
              "## Vérifications et reproduction", "",
              ("Parents, checkpoints, configurations, sources, poids et prédictions contrôlés par empreintes. "
              "Métriques recalculées ; historique initial, règle d’arrêt, époque choisie et rechargement des exports vérifiés."), "",
              ("Voir [le protocole](../docs/EXTENSION_CNN.md), [la référence à dix époques](REPETITIONS_CNN.md) "
              "et [la synthèse globale](SYNTHESE_VALIDATION.md)."), "",
              "Commande : `.\\.venv\\Scripts\\python.exe scripts/summarize_cnn_extension.py`.", "",
              "- [Résumé numérique](CNN_EXTENSION.json)"]
    for seed, pointer in zip(plan["seeds"], pointers, strict=True):
        lines.append(f"- [Rapport graine {seed}](../{pointer['folder']}/report.json)")
    (ROOT / "reports/EXTENSION_CNN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "differences": differences}, indent=2))


if __name__ == "__main__":
    main()
