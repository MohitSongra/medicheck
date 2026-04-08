"""
MediCheck - API utility helpers.
"""

import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISEASE_INFO_PATH = PROJECT_ROOT / "data" / "disease_info.json"

_disease_info_cache: Dict[str, dict] | None = None


def load_disease_info() -> Dict[str, dict]:
    """Load disease descriptions and suggested actions from JSON."""
    global _disease_info_cache
    if _disease_info_cache is None:
        with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
            _disease_info_cache = json.load(f)
    return _disease_info_cache


def get_disease_details(disease_name: str) -> tuple[str, str]:
    """
    Return (description, suggested_action) for a disease.
    Falls back to generic text if the disease is not in the info file.
    """
    info = load_disease_info()
    entry = info.get(disease_name, {})
    description = entry.get(
        "description",
        "A medical condition that requires further evaluation by a healthcare professional.",
    )
    action = entry.get(
        "suggested_action",
        "Consult a healthcare professional for proper diagnosis and treatment.",
    )
    return description, action


def format_symptom_name(raw: str) -> str:
    """Convert 'burning_urination' -> 'Burning Urination'."""
    return raw.replace("_", " ").title()


def format_disease_name(raw: str) -> str:
    """Convert 'Urinary_Tract_Infection' -> 'Urinary Tract Infection'."""
    return raw.replace("_", " ")
