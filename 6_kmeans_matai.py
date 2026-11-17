import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix

from common import CSV_DIR, load_B

K = 4
N_COMP = 5


def jaccard_pairs(reference, labels):
    c = pair_confusion_matrix(reference, labels)
    return c[1, 1] / (c[1, 1] + c[0, 1] + c[1, 0])


def load_spaces(X):
    pca = PCA(n_components=N_COMP).fit_transform(X.to_numpy())
    return {
        "Originali (17D)": X.to_numpy(),
        "t-SNE 5D": pd.read_csv(CSV_DIR / "5_tsne_5d.csv").to_numpy(),
        "t-SNE 2D": pd.read_csv(CSV_DIR / "4_tsne_2d.csv").to_numpy(),
        "PCA 5D": pca,
        "PCA 2D": pca[:, :2],
    }


def load_labels():
    tsne = pd.read_csv(CSV_DIR / "5_kmeans_labels_tsne.csv")
    pca = pd.read_csv(CSV_DIR / "5_kmeans_labels_pca.csv")
    if not (tsne["original"] == pca["original"]).all():
        raise ValueError("original labels differ between 5_kmeans_labels_tsne.csv and 5_kmeans_labels_pca.csv")
    return {
        "Originali (17D)": tsne["original"].to_numpy(),
        "t-SNE 5D": tsne["tsne_5d"].to_numpy(),
        "t-SNE 2D": tsne["tsne_2d"].to_numpy(),
        "PCA 5D": pca["pca_5d"].to_numpy(),
        "PCA 2D": pca["pca_2d"].to_numpy(),
    }


def build_table(spaces, labels):
    reference = labels["Originali (17D)"]
    rows = []
    for space, coords in spaces.items():
        lab = labels[space]
        if len(lab) != len(coords):
            raise ValueError(f"{space}: {len(lab)} labels but {len(coords)} objects")
        row = {
            "Erdve": space,
            "Algoritmas": "k-means",
            "k": len(set(lab)),
            "Silhouette": round(silhouette_score(coords, lab), 3),
            "CH": round(calinski_harabasz_score(coords, lab), 1),
            "DB": round(davies_bouldin_score(coords, lab), 3),
        }
        if space == "Originali (17D)":
            row.update({"ARI": "atskaita", "NMI": "atskaita", "Jaccard": "atskaita"})
        else:
            row.update({
                "ARI": round(adjusted_rand_score(reference, lab), 3),
                "NMI": round(normalized_mutual_info_score(reference, lab), 3),
                "Jaccard": round(jaccard_pairs(reference, lab), 3),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    X, _ = load_B()
    table = build_table(load_spaces(X), load_labels())
    table.to_csv(CSV_DIR / "6_kmeans_metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()