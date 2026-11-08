"""Ligne de commande : `doctor` sans dépendance ML, `predict` avec le modèle épinglé."""

import argparse
import json
from pathlib import Path

from music_genre.diagnostics import environment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="music-genre",
        description="Projet de classification musicale : diagnostic et prédiction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="Afficher un diagnostic en JSON.")
    predict = subparsers.add_parser("predict", help="Prédire le genre d'un fichier WAV ou MP3.")
    predict.add_argument("audio", type=Path, help="Fichier WAV ou MP3 (50 Mo et 5 minutes au plus).")
    predict.add_argument("--system", default="vit_block6_hpss",
                         help="Système V1 épinglé (défaut : le candidat vit_block6_hpss).")
    predict.add_argument("--mode", choices=("fma", "overview"), default="fma",
                         help="fma : fenêtres 0/10/20 s du protocole ; overview : début/milieu/fin "
                              "(candidat seulement).")
    for command in (doctor, predict):
        command.add_argument(
            "--root",
            type=Path,
            default=Path.cwd(),
            help="Racine existante du projet (défaut : répertoire courant).",
        )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Le répertoire du projet n'existe pas : {root}")
    if args.command == "doctor":
        print(json.dumps(environment_report(root), ensure_ascii=True, indent=2))
        return 0
    # Imported here so that `doctor` never loads the ML stack.
    from music_genre.inference.demo import DemoPredictor
    from music_genre.inference.predict import CANDIDATE, SYSTEMS, Predictor

    if args.system not in SYSTEMS:
        parser.error(f"Système inconnu : {args.system} (choix : {', '.join(SYSTEMS)})")
    if args.mode == "overview" and args.system != CANDIDATE:
        parser.error("Le mode overview n'existe que pour le candidat vit_block6_hpss")
    predictor = DemoPredictor(root) if args.system == CANDIDATE else Predictor(root, args.system)
    try:
        result = predictor.predict_file(args.audio, mode=args.mode)
    except ValueError as error:
        parser.exit(1, f"Analyse impossible : {error}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0
