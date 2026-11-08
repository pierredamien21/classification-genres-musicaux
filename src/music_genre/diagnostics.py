"""Inspection sans installation, import de torch ou modification des fichiers."""

import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

from music_genre import __version__

DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "librosa",
    "soundfile",
    "scikit-learn",
    "torch",
    "timm",
    "gradio",
    "pytest",
    "ruff",
)


def environment_report(root: Path) -> dict[str, object]:
    """Retourner des observations ; ne pas conclure à la disponibilité d'un GPU."""
    versions: dict[str, str | None] = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    usage = shutil.disk_usage(root)
    return {
        "project_version": __version__,
        "root": str(root),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "process_machine": platform.machine(),
            "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
        },
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "processor_architecture_env": os.environ.get("PROCESSOR_ARCHITECTURE"),
            "processor_architew6432_env": os.environ.get("PROCESSOR_ARCHITEW6432"),
        },
        "disk": {
            "total_gib": round(usage.total / 2**30, 2),
            "free_gib": round(usage.free / 2**30, 2),
        },
        "dependencies": versions,
        "accelerator": {
            "status": "not_probed",
            "note": "La présence de torch ne démontre pas une accélération GPU utilisable.",
        },
        "note": "Une version installée ne garantit ni l'import ni la compatibilité du pipeline.",
    }

