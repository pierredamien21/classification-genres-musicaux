"""Local Gradio demonstration; no training, download or public sharing."""

import os
import sys
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
UPLOADS = ROOT / "data/interim/demo-uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(UPLOADS)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.features.hpss import hpss_log_mel
from music_genre.inference.demo import DemoPredictor

_predictor = None
_lock = Lock()


def analyze(path, mode):
    """One shared callback for UI and end-to-end verification."""
    global _predictor
    if not path:
        return "Ajoute un fichier WAV ou MP3 avant de lancer l’analyse.", None, [], None, None
    try:
        with _lock:
            if _predictor is None:
                _predictor = DemoPredictor(ROOT)
            result = _predictor.predict_file(path, mode=mode)
            audio = load_audio(Path(path), _predictor.config["sample_rate"])
            window = segment_audio(audio, (result["windows"][0]["start_seconds"],), 10)[0]
            mel = hpss_log_mel(window, _predictor.config["logmel"], _predictor.config["hpss"])[0]
            fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
            im = ax.imshow(mel, origin="lower", aspect="auto", extent=[0, 10, 0, 128],
                           cmap="magma", vmin=-80, vmax=20)
            ax.set(xlabel="Temps dans la première fenêtre (s)", ylabel="Bande Mel",
                   title="Log-Mel original · première fenêtre")
            fig.colorbar(im, ax=ax, label="dB, référence 1")
            plt.close(fig)
        top = result["top3"][0]
        message = (f"### Genre proposé : {top['label']}\n"
                   f"{result['duration_seconds']:.1f} s de fichier · "
                   f"{len(result['windows'])} passages · "
                   f"{result['elapsed_seconds']:.2f} s d’inférence (hors chargement).\n\n"
                   "Les scores comparent huit genres ; ils ne mesurent pas une certitude.")
        positions = [[w["start_seconds"], w["valid_seconds"], w["padded_seconds"]]
                     for w in result["windows"]]
        return message, {p["label"]: p["score"] for p in result["top3"]}, positions, fig, result
    except (ValueError, OSError) as error:
        return f"Analyse impossible : {error}", None, [], None, None


def build_app():
    with gr.Blocks(title="Genres musicaux — exploration locale", delete_cache=(600, 3600)) as demo:
        gr.Markdown("# Explorer le genre d’un morceau\n"
                    "Un Vision Transformer gelé et des représentations audio. "
                    "Démo locale du projet · huit genres FMA.")
        with gr.Row():
            with gr.Column():
                source = gr.Audio(label="Ton extrait · WAV ou MP3", sources=["upload"],
                                  type="filepath", buttons=[])
                mode = gr.Radio([("Début, milieu et fin", "overview"),
                                 ("Protocole FMA : 0, 10 et 20 secondes", "fma")],
                                value="overview", label="Passages à analyser")
                gr.Markdown("Maximum 50 Mo et 5 minutes. Aucun fichier envoyé à un service externe.\n\n"
                            "Le mode début/milieu/fin est destiné à l’exploration ; "
                            "les résultats scientifiques utilisent le mode FMA.")
                run = gr.Button("Analyser le morceau", variant="primary")
                clear = gr.ClearButton(value="Effacer")
            with gr.Column():
                status = gr.Markdown("Ajoute un morceau, puis lance l’analyse.")
                scores = gr.Label(label="Trois genres proposés · scores non calibrés", num_top_classes=3)
                windows = gr.Dataframe(headers=["Début (s)", "Audio valide (s)", "Zéros ajoutés (s)"],
                                       label="Passages analysés", interactive=False)
        plot = gr.Plot(label="Représentation audio")
        with gr.Accordion("Détails de la prédiction et version du modèle", open=False):
            details = gr.JSON(label="Résultat traçable")
        gr.Markdown("**Limites connues.** Gospel et genres inconnus non couverts ; "
                    "musique multigenre ramenée à un seul label. Sensibilité au bruit. "
                    "V1 : macro-F1 test 0,5035, accuracy 51,16 %. "
                    "Le résultat sur un fichier personnel peut être incorrect. "
                    "Les copies temporaires sont purgées toutes les dix minutes "
                    "lorsqu’elles ont plus d’une heure.")
        run.click(analyze, [source, mode], [status, scores, windows, plot, details],
                  concurrency_limit=1, api_name="analyze")
        clear.add([source, status, scores, windows, plot, details])
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=4).launch(server_name="127.0.0.1", server_port=7860,
                                       share=False, inbrowser=False, max_file_size="50mb")
