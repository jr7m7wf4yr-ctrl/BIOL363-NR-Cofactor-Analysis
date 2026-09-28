import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("NR_DATA_DIR", ROOT / "data" / "generated"))
STRUCTURES = Path(os.environ.get("NR_STRUCTURE_DIR", DATA_ROOT / "structures"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)
