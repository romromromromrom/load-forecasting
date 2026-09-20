"""Chargement de la configuration et des chemins du projet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge le YAML de configuration (défaut : configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Chemin absolu d'une entrée `paths.<key>` (relative à la racine du projet)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
