from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
IMG_DIR = ROOT / "img"
IMG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CLASS_COL = "class"

N_SAMPLES = 10
SAMPLE_FRAC = 0.8
SAMPLES_FILE = CSV_DIR / "7_imtys.csv"

def load_B():
    # Grazinam:
    # X - pozymiu matrica klasterizavimui,
    # Y - klases interpretacijai

    df = pd.read_csv(CSV_DIR / "B_no_outlier.csv")
    y = df[CLASS_COL]
    X = df.drop(columns=CLASS_COL)
    return X, y

def load_B_sample(i):
    X, y = load_B()
    samples = pd.read_csv(SAMPLES_FILE)
    idx = samples.loc[samples["imtis"] == i, "indeksas"].to_numpy()
    return X.loc[idx], y.loc[idx]