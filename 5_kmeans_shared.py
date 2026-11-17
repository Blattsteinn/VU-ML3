import numpy as np
import pandas as pd

from common import CSV_DIR

K = 4
SOURCES = {
    "5_kmeans_labels_tsne.csv": ["tsne_5d", "tsne_2d"],
    "5_kmeans_labels_pca.csv": ["pca_5d", "pca_2d"],
}
NAMES = {
    "tsne_5d": "t-SNE 5D",
    "tsne_2d": "t-SNE 2D",
    "pca_5d": "PCA 5D",
    "pca_2d": "PCA 2D",
}


def overlap_table(original, reduced):
    table = pd.crosstab(original, reduced).reindex(index=range(K), columns=range(K), fill_value=0)
    table.index.name = "originali"
    table.columns.name = "sumazinta"
    return table


def main():
    rows = []
    for file, columns in SOURCES.items():
        labels = pd.read_csv(CSV_DIR / file)
        for column in columns:
            table = overlap_table(labels["original"], labels[column])
            same = int(np.trace(table.to_numpy()))
            total = int(table.to_numpy().sum())
            pct = same / total * 100
            rows.append({"erdve": NAMES[column], "sutampa": same, "is_viso": total, "sutampa_proc": round(pct, 1)})
            print(f"{NAMES[column]}: su originalia erdve sutampa {same} is {total} objektu ({pct:.1f} %)")
            print("eilutes - originalios erdves klasteriai, stulpeliai - sumazintos erdves klasteriai")
            print(table.to_string())
            print()
    summary = pd.DataFrame(rows)
    summary.to_csv(CSV_DIR / "5_kmeans_agreement.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

# SVARBU PRIES PALEIDZIANT:
# NAUDOJAM .csv failus, kuriuos sugeneruoja:
# 5th_kmeans_tsne.py
# 5th_kmeans_pca.py