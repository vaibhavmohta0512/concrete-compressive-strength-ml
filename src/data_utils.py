
"""Dataset acquisition and validation for the UCI concrete strength data."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "concrete_compressive_strength.csv"

FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"
]
TARGET = "Concrete compressive strength"

def load_dataset(save_local=True):
    """Load local CSV if present; otherwise fetch UCI dataset 165."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        dataset = fetch_ucirepo(id=165)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        # Normalize labels to stable names used throughout this project.
        X.columns = FEATURES
        y.columns = [TARGET]
        df = pd.concat([X, y], axis=1)
        if save_local:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
    # Some mirrors use a different target spelling; normalize known variants.
    aliases = {
        "Concrete compressive strength(MPa)": TARGET,
        "Concrete compressive strength (MPa)": TARGET,
        "strength": TARGET,
    }
    df = df.rename(columns=aliases)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}. Found: {list(df.columns)}")
    df = df[FEATURES + [TARGET]].copy()
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
