import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from common import IMG_DIR, RANDOM_STATE, load_B

K_RANGE = range(2, 11)
K_CHOSEN = 3


def fit_kmeans(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    return model.fit_predict(X)


def average_silhouettes(X):
    rows = []
    for k in K_RANGE:
        labels = fit_kmeans(X, k)
        rows.append({"k": k, "avg_silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(rows)


def plot_average(results):
    best_k = int(results.loc[results["avg_silhouette"].idxmax(), "k"])

    ks = [1] + results["k"].tolist()
    scores = [0.0] + results["avg_silhouette"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, marker="o", color="steelblue")
    ax.axvline(best_k, linestyle="--", color="steelblue")
    fig.suptitle("Optimalus klasterių skaičius", x=0.125, ha="left", fontsize=14)
    ax.set_title("Vidutinio silueto metodas (k-means, B aibė)", loc="left", fontsize=11)
    ax.set_xlabel("Klasterių skaičius k")
    ax.set_ylabel("Vidutinis siluetas")
    ax.set_xticks(ks)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "3_section_silhouette_average.png", dpi=150)
    plt.close(fig)
    return best_k


def plot_samples(X, k):
    labels = fit_kmeans(X, k)
    values = silhouette_samples(X, labels)
    avg = values.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    for c in range(k):
        cluster_values = np.sort(values[labels == c])
        y_upper = y_lower + len(cluster_values)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                         alpha=0.7, label=f"Klasteris {c} (n = {len(cluster_values)})")
        y_lower = y_upper + 10
    ax.axvline(avg, linestyle="--", color="red", label=f"Vidurkis = {avg:.3f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Objektų siluetai (k-means, k = {k}, B aibė)")
    ax.set_xlabel("Siluetas S_i")
    ax.set_ylabel("Objektai, surikiuoti klasteriuose")
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"3_section_silhouette_k{k}.png", dpi=150)
    plt.close(fig)

    print(f"\nSilhouette by cluster, k = {k}")
    for c in range(k):
        cluster_values = values[labels == c]
        negative = (cluster_values < 0).mean() * 100
        print(f"Cluster {c}: n = {len(cluster_values)}, "
              f"avg = {cluster_values.mean():.3f}, negative = {negative:.1f}%")
    print(f"All objects: avg = {avg:.3f}, negative = {(values < 0).mean() * 100:.1f}%")


def main():
    X, _ = load_B()

    results = average_silhouettes(X)
    print("Average silhouette by k")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_k = plot_average(results)
    print(f"\nBest k by average silhouette: {best_k}")

    plot_samples(X, K_CHOSEN)


if __name__ == "__main__":
    main()