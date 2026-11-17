import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import CSV_DIR, CLASS_COL, load_B

COMPONENTS = [4, 5, 6, 7]
THRESHOLD = 0.90


def cumulative_variance(X):
    pca = PCA().fit(X)
    return np.cumsum(pca.explained_variance_ratio_)


def main():
    df_full = pd.read_csv(CSV_DIR / "B.csv")
    X_full = df_full.drop(columns=CLASS_COL)
    X_clean, _ = load_B()

    rows = {
        f"B aibe ({len(X_full)} obj.)": cumulative_variance(X_full),
        f"B aibe be isskirties ({len(X_clean)} obj.)": cumulative_variance(X_clean),
    }

    table = pd.DataFrame(
        {name: [round(cum[c - 1] * 100, 1) for c in COMPONENTS] for name, cum in rows.items()},
        index=COMPONENTS,
    ).T
    table.columns.name = "Komponenciu sk."

    print("PCA kumuliatyvi paaiskintos dispersijos dalis, %")
    print(table.to_string())
    print()

    for name, cum in rows.items():
        n = int(np.argmax(cum >= THRESHOLD)) + 1
        print(f"{name}: {int(THRESHOLD * 100)} % riba pasiekiama ties {n} komponentemis ({cum[n - 1] * 100:.1f} %)")


if __name__ == "__main__":
    main()