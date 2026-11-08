"""Commandes disponibles du socle, sans import de dépendances ML."""

import argparse
import json
from pathlib import Path

from music_genre.diagnostics import environment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="music-genre",
        description="Projet de classification musicale : diagnostic de l'environnement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="Afficher un diagnostic en JSON.")
    doctor.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Racine existante du projet (défaut : répertoire courant).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Le répertoire du projet n'existe pas : {root}")
    report = environment_report(root)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0

