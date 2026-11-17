import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE

from common import CSV_DIR, IMG_DIR, RANDOM_STATE, load_B

K = 4
MAX_ITER = 300
INIT = "random_coords"
SEED = 4
N_COMP = 5
COLORS = np.array(plt.get_cmap("tab10").colors)


def assign(Xv, centers):
    d = ((Xv[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def update_centers(Xv, labels, centers):
    new_centers = centers.copy()
    for l in range(K):
        members = Xv[labels == l]
        if len(members) > 0:
            new_centers[l] = members.mean(axis=0)
    return new_centers


def squared_error(Xv, labels, centers):
    return float(((Xv - centers[labels]) ** 2).sum())


def initial_centers(Xv, init, rng):
    redraws = 0
    while True:
        if init == "random_coords":
            centers = rng.uniform(Xv.min(axis=0), Xv.max(axis=0), size=(K, Xv.shape[1]))
            labels = assign(Xv, centers)
        else:
            labels = rng.integers(0, K, size=Xv.shape[0])
            centers = None
        if len(np.unique(labels)) == K:
            break
        redraws += 1
    if centers is None:
        centers = update_centers(Xv, labels, np.zeros((K, Xv.shape[1])))
    return centers, redraws


def kmeans(Xv, init, seed):
    rng = np.random.default_rng(seed)
    centers, redraws = initial_centers(Xv, init, rng)
    labels = None
    errors = []
    for iteration in range(1, MAX_ITER + 1):
        new_labels = assign(Xv, centers)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = update_centers(Xv, labels, centers)
        errors.append(squared_error(Xv, labels, centers))
    return labels, centers, errors[-1], iteration, redraws


def rank_labels(labels):
    order = np.argsort(-np.bincount(labels, minlength=K))
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))
    return mapping[labels]


def tsne_2d(Xv):
    path = CSV_DIR / "4_tsne_2d.csv"
    if path.exists():
        coords = pd.read_csv(path).to_numpy()
        if coords.shape[0] == Xv.shape[0]:
            return coords
    coords = TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE).fit_transform(Xv)
    pd.DataFrame(coords, columns=["tsne1", "tsne2"]).to_csv(path, index=False)
    return coords


def tsne_space(Xv):
    path = CSV_DIR / "5_tsne_5d.csv"
    if path.exists():
        coords = pd.read_csv(path).to_numpy()
        if coords.shape == (Xv.shape[0], N_COMP):
            return coords
    print("running t-SNE 5D with method=exact, this can take several minutes")
    coords = TSNE(
        n_components=N_COMP,
        perplexity=30,
        init="pca",
        method="exact",
        random_state=RANDOM_STATE,
    ).fit_transform(Xv)
    names = [f"tsne{i + 1}" for i in range(N_COMP)]
    pd.DataFrame(coords, columns=names).to_csv(path, index=False)
    return coords


def align_to(ref, labels):
    overlap = np.zeros((K, K), dtype=int)
    np.add.at(overlap, (labels, ref), 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(K, dtype=int)
    mapping[rows] = cols
    return mapping[labels]


def plot_grid(coords, labels_by_space):
    fig, axes = plt.subplots(1, len(labels_by_space), figsize=(5.2 * len(labels_by_space), 5.0))
    for ax, (space, labels) in zip(axes, labels_by_space.items()):
        ax.scatter(coords[:, 0], coords[:, 1], c=COLORS[labels], s=6, linewidths=0)
        sizes = "/".join(str(n) for n in np.bincount(labels, minlength=K))
        ax.set_title(f"{space}\nklasterių dydžiai {sizes}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS[c], label=f"{c} klasteris")
        for c in range(K)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=K, frameon=False)
    fig.suptitle(
        f"k-means (k = {K}, init={INIT}, random_state = {SEED}). t-SNE 2D projekcija",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(IMG_DIR / "5_kmeans_tsne_5d_2d.png", dpi=150)
    plt.close(fig)


def main():
    X, y = load_B()
    Xv = X.to_numpy()

    tsne_coords = tsne_space(Xv)

    spaces = {
        "Original (17D)": Xv,
        "t-SNE 5D": tsne_coords,
        "t-SNE 2D": tsne_2d(Xv),
    }

    labels_by_space = {}
    reference = None
    for space, data in spaces.items():
        labels, _, error, n_iter, redraws = kmeans(data, INIT, SEED)
        if reference is None:
            labels = rank_labels(labels)
            reference = labels
        else:
            labels = align_to(reference, labels)
        labels_by_space[space] = labels
        same = int((labels == reference).sum())
        print(f"{space}: E = {error:.2f}  n_iter = {n_iter}  init_redraws = {redraws}")
        print(f"  sizes: {np.bincount(labels, minlength=K).tolist()}")
        print(f"  same cluster as original: {same} of {len(labels)} ({same / len(labels) * 100:.1f} %)")
        share = pd.crosstab(labels, y, normalize="index") * 100
        share.index.name = "cluster"
        print("  class share inside each cluster, % (rows sum to 100):")
        print(share.round(1).to_string())
        print()

    out = pd.DataFrame(
        {
            "original": labels_by_space["Original (17D)"],
            "tsne_5d": labels_by_space["t-SNE 5D"],
            "tsne_2d": labels_by_space["t-SNE 2D"],
            "class": y,
        }
    )
    out.to_csv(CSV_DIR / "5_kmeans_labels_tsne.csv", index=False)

    plot_grid(tsne_2d(Xv), {s: labels_by_space[s] for s in ["t-SNE 5D", "t-SNE 2D"]})


if __name__ == "__main__":
    main()