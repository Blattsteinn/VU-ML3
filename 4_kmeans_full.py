import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from common import CSV_DIR, IMG_DIR, RANDOM_STATE, load_B

K = 4
INITS = ["random_coords", "random_partition"]
SEEDS = list(range(10)) + [RANDOM_STATE]
MAX_ITER = 300


def assign(Xv, centers):
    distances = ((Xv[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


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
    return labels, errors[-1], iteration, redraws


def partition_ids(labels_list):
    reps = []
    ids = []
    for labels in labels_list:
        found = None
        for i, rep in enumerate(reps):
            if adjusted_rand_score(rep, labels) > 0.9999:
                found = i
                break
        if found is None:
            reps.append(labels)
            found = len(reps) - 1
        ids.append(found)
    return ids


def run_all(Xv):
    rows = []
    labels_by_run = {}
    for init in INITS:
        for seed in SEEDS:
            labels, error, n_iter, redraws = kmeans(Xv, init, seed)
            sizes = np.sort(np.bincount(labels, minlength=K))[::-1]
            rows.append(
                {
                    "init": init,
                    "random_state": seed,
                    "E": error,
                    "n_iter": n_iter,
                    "init_redraws": redraws,
                    "silhouette": silhouette_score(Xv, labels),
                    "calinski_harabasz": calinski_harabasz_score(Xv, labels),
                    "davies_bouldin": davies_bouldin_score(Xv, labels),
                    "sizes": "/".join(str(s) for s in sizes),
                }
            )
            labels_by_run[f"{init}_{seed}"] = labels
    results = pd.DataFrame(rows)
    results["partition_id"] = partition_ids(list(labels_by_run.values()))
    return results, pd.DataFrame(labels_by_run)


def summarize(results):
    summary = results.groupby("init").agg(
        E_min=("E", "min"),
        E_median=("E", "median"),
        E_max=("E", "max"),
        silhouette_median=("silhouette", "median"),
        silhouette_min=("silhouette", "min"),
        silhouette_max=("silhouette", "max"),
        calinski_harabasz_median=("calinski_harabasz", "median"),
        davies_bouldin_median=("davies_bouldin", "median"),
        n_iter_median=("n_iter", "median"),
        distinct_partitions=("partition_id", "nunique"),
        init_redraws=("init_redraws", "sum"),
    )
    return summary.loc[INITS].T


def plot_params(results):
    metrics = [
        ("E", "E (kvadratinė paklaida)"),
        ("silhouette", "Silhouette"),
        ("calinski_harabasz", "Calinski-Harabasz"),
        ("davies_bouldin", "Davies-Bouldin"),
    ]
    rng = np.random.default_rng(RANDOM_STATE)
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        data = [results.loc[results["init"] == i, col].to_numpy() for i in INITS]
        ax.boxplot(data, showfliers=False)
        for pos, d in enumerate(data, start=1):
            ax.scatter(pos + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color="black", zorder=3)
        ax.set_xticks(range(1, len(INITS) + 1))
        ax.set_xticklabels(INITS)
        ax.set_title(title)
    fig.suptitle(f"k-means, k={K}: metrikos per {len(SEEDS)} sėklų kiekvienam inicializacijos būdui")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "4_kmeans_params_boxplot.png", dpi=150)
    plt.close(fig)


def main():
    X, _ = load_B()
    Xv = X.to_numpy()

    results, labels = run_all(Xv)
    summary = summarize(results)

    results.to_csv(CSV_DIR / "4_kmeans_experiments.csv", index=False)
    labels.to_csv(CSV_DIR / "4_kmeans_all_labels.csv", index=False)
    summary.to_csv(CSV_DIR / "4_kmeans_init_summary.csv")
    plot_params(results)

    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("summary by init:")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("runs:", len(results), "distinct partitions:", results["partition_id"].nunique())


if __name__ == "__main__":
    main()