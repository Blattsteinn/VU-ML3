import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

from common import CSV_DIR, IMG_DIR, RANDOM_STATE, load_B

K = 4
INIT = "random_coords"
SEED = 4
LOW_E = 16000
GRID_SEEDS = [1, 2, 4, 8]
COLORS = np.array(plt.get_cmap("tab10").colors)


def tsne_coords(Xv):
    path = CSV_DIR / "4_tsne_2d.csv"
    if path.exists():
        coords = pd.read_csv(path).to_numpy()
        if coords.shape[0] == Xv.shape[0]:
            return coords
    coords = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE
    ).fit_transform(Xv)
    pd.DataFrame(coords, columns=["tsne1", "tsne2"]).to_csv(path, index=False)
    return coords


def rank_labels(labels):
    order = np.argsort(-np.bincount(labels, minlength=K))
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))
    return mapping[labels]


def stains_split(labels, y):
    ct = pd.crosstab(labels, y)
    main = ct["Stains"].idxmax()
    sizes = ct.sum(axis=1)
    return pd.Series(
        {
            "stains_in_main_cluster_pct": 100 * ct.loc[main, "Stains"] / ct["Stains"].sum(),
            "stains_main_cluster_size": sizes[main],
            "stains_separated": sizes[main] != sizes.max(),
        }
    )


def low_error_runs(results, labels, y):
    low = results[(results["init"] == INIT) & (results["E"] < LOW_E)].copy()
    split = low["random_state"].apply(lambda s: stains_split(labels[f"{INIT}_{s}"].to_numpy(), y))
    low = pd.concat([low, split], axis=1)
    columns = [
        "random_state",
        "E",
        "silhouette",
        "sizes",
        "stains_in_main_cluster_pct",
        "stains_main_cluster_size",
        "stains_separated",
    ]
    return low[columns].sort_values("E")


def cluster_profile(Xv, labels, y):
    share = pd.crosstab(labels, y, normalize="index").mul(100).round(1)
    sizes = np.bincount(labels, minlength=K)
    share.insert(0, "size", sizes)
    share.insert(1, "size_pct", (100 * sizes / sizes.sum()).round(1))
    values = silhouette_samples(Xv, labels)
    share["silhouette"] = [round(values[labels == c].mean(), 3) for c in range(K)]
    share.index.name = "cluster"
    return share


def plot_grid(coords, results, labels):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    for ax, seed in zip(axes.ravel(), GRID_SEEDS):
        row = results[(results["init"] == INIT) & (results["random_state"] == seed)].iloc[0]
        run_labels = rank_labels(labels[f"{INIT}_{seed}"].to_numpy())
        ax.scatter(coords[:, 0], coords[:, 1], c=COLORS[run_labels], s=5, linewidths=0)
        ax.set_title(
            f"seed={seed}\nE={row['E']:.0f}, sil={row['silhouette']:.3f}\nsizes {row['sizes']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"k-means, k={K}, init={INIT}:\nmažiausią paklaidą turintys paleidimai (t-SNE 2D projekcija)", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "4_kmeans_tsne_grid.png", dpi=130)
    plt.close(fig)


def plot_chosen(coords, labels, error):
    counts = np.bincount(labels, minlength=K)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
    for c in range(K):
        mask = labels == c
        ax1.scatter(coords[mask, 0], coords[mask, 1], s=7, color=COLORS[c], label=f"klasteris {c}")
    ax1.legend(markerscale=3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Pasirinktas paleidimas: init={INIT}, seed={SEED}, E={error:.0f}")
    ax2.bar(range(K), counts, color=COLORS[:K])
    for c in range(K):
        ax2.text(c, counts[c], f"{counts[c]}\n({100 * counts[c] / counts.sum():.1f}%)", ha="center", va="bottom")
    ax2.set_xticks(range(K))
    ax2.set_xlabel("klasteris")
    ax2.set_ylabel("objektų skaičius")
    ax2.set_ylim(0, counts.max() * 1.2)
    ax2.set_title("Klasterių dydžiai")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "4_kmeans_chosen.png", dpi=150)
    plt.close(fig)


def main():
    X, y = load_B()
    Xv = X.to_numpy()
    results = pd.read_csv(CSV_DIR / "4_kmeans_experiments.csv")
    labels = pd.read_csv(CSV_DIR / "4_kmeans_all_labels.csv")
    coords = tsne_coords(Xv)

    low = low_error_runs(results, labels, y)
    chosen = rank_labels(labels[f"{INIT}_{SEED}"].to_numpy())
    error = results[(results["init"] == INIT) & (results["random_state"] == SEED)]["E"].iloc[0]
    profile = cluster_profile(Xv, chosen, y)
    class_in_cluster = pd.crosstab(chosen, y, normalize="columns").mul(100).round(1)
    class_in_cluster.index.name = "cluster"
    medians = (X.groupby(chosen).median() - X.median()).round(2).T

    low.to_csv(CSV_DIR / "4_kmeans_low_error_runs.csv", index=False)
    profile.to_csv(CSV_DIR / "4_kmeans_class_share.csv")
    class_in_cluster.to_csv(CSV_DIR / "4_kmeans_class_in_cluster.csv")
    medians.to_csv(CSV_DIR / "4_kmeans_median_diff.csv")
    pd.DataFrame({"cluster": chosen}).to_csv(CSV_DIR / "4_kmeans_labels.csv", index=False)

    plot_grid(coords, results, labels)
    plot_chosen(coords, chosen, error)

    print(f"{INIT} runs with E < {LOW_E}:")
    print(low.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"chosen run: k = {K}, init = {INIT}, random_state = {SEED}, E = {error:.4f}")
    print()
    print("class share inside each cluster, % (rows sum to 100), and cluster silhouette:")
    print(profile.to_string())
    print()
    print("share of each class falling into each cluster, % (columns sum to 100):")
    print(class_in_cluster.to_string())
    print()
    print("feature median minus overall median (scaled units):")
    print(medians.to_string())


if __name__ == "__main__":
    main()