from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix

from common import CSV_DIR, IMG_DIR, N_SAMPLES, load_B, load_B_sample

K = 4
INIT = "random_coords"
SEED = 4
MAX_ITER = 300
PARAMS = f"k = {K}, init = {INIT}, random_state = {SEED}"
EXTERNAL = ["ARI", "NMI", "Jaccard"]
INTERNAL = ["Silhouette", "CH", "DB"]
COLOR = "#2a78d6"


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


def initial_centers(Xv, rng):
    while True:
        centers = rng.uniform(Xv.min(axis=0), Xv.max(axis=0), size=(K, Xv.shape[1]))
        if len(np.unique(assign(Xv, centers))) == K:
            return centers


def kmeans(Xv, seed):
    rng = np.random.default_rng(seed)
    centers = initial_centers(Xv, rng)
    labels = None
    for _ in range(MAX_ITER):
        new_labels = assign(Xv, centers)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = update_centers(Xv, labels, centers)
    return labels


def pair_jaccard(a, b):
    C = pair_confusion_matrix(a, b)
    return C[1, 1] / (C[1, 1] + C[0, 1] + C[1, 0])


def compare_pair(la, lb):
    common = la.index.intersection(lb.index)
    a = la.loc[common].to_numpy()
    b = lb.loc[common].to_numpy()
    return {
        "bendri_objektai": len(common),
        "ARI": adjusted_rand_score(a, b),
        "NMI": normalized_mutual_info_score(a, b),
        "Jaccard": pair_jaccard(a, b),
    }


def main():
    X_full, _ = load_B()
    reference = pd.Series(kmeans(X_full.to_numpy(), SEED), index=X_full.index)

    labels = []
    internal_rows = []
    label_rows = []
    for i in range(N_SAMPLES):
        X, _ = load_B_sample(i)
        Xv = X.to_numpy()
        l = kmeans(Xv, SEED)
        labels.append(pd.Series(l, index=X.index))
        sizes = np.sort(np.bincount(l, minlength=K))[::-1]
        vs_reference = compare_pair(reference, labels[-1])
        internal_rows.append({
            "imtis": i,
            "Silhouette": silhouette_score(Xv, l),
            "CH": calinski_harabasz_score(Xv, l),
            "DB": davies_bouldin_score(Xv, l),
            "klasteriu_dydziai": " / ".join(str(s) for s in sizes),
            **{f"{col}_su_atskaita": vs_reference[col] for col in EXTERNAL},
        })
        label_rows.append(pd.DataFrame({"imtis": i, "indeksas": X.index, "klasteris": l}))

    pd.concat(label_rows, ignore_index=True).to_csv(CSV_DIR / "7_kmeans_imciu_zymes.csv", index=False)

    internal = pd.DataFrame(internal_rows)
    internal.to_csv(CSV_DIR / "7_kmeans_vidiniai_matai_imtys.csv", index=False)

    pairs = pd.DataFrame([
        {"imtis_a": i, "imtis_b": j, **compare_pair(labels[i], labels[j])}
        for i, j in combinations(range(N_SAMPLES), 2)
    ])
    pairs.to_csv(CSV_DIR / "7_kmeans_poru_matai.csv", index=False)

    summary = {"algoritmas": "k-means", "parametrai": PARAMS, "poru_sk": len(pairs)}
    for col in EXTERNAL:
        summary[f"{col}_vid"] = round(pairs[col].mean(), 3)
        summary[f"{col}_sn"] = round(pairs[col].std(), 3)
        summary[f"{col}_min"] = round(pairs[col].min(), 3)
    for col in INTERNAL:
        summary[f"{col}_vid"] = round(internal[col].mean(), 3)
        summary[f"{col}_sn"] = round(internal[col].std(), 3)
    pd.DataFrame([summary]).to_csv(CSV_DIR / "7_kmeans_stabilumo_lentele.csv", index=False)

    print(f"k-means parametrai: {PARAMS}")
    print(f"Imciu sk.: {N_SAMPLES}, poru sk.: {len(pairs)}, bendru objektu poroje vidurkis: {pairs['bendri_objektai'].mean():.1f}")
    print()
    print("Stabilumas, vidurkis +/- SN [min]:")
    for col in EXTERNAL:
        print(f"  {col}: {summary[f'{col}_vid']:.3f} +/- {summary[f'{col}_sn']:.3f} [{summary[f'{col}_min']:.3f}]")
    print()
    print("Vidiniai matai per imtis, vidurkis +/- SN:")
    for col in INTERNAL:
        print(f"  {col}: {summary[f'{col}_vid']:.3f} +/- {summary[f'{col}_sn']:.3f}")
    print()
    ref_sizes = " / ".join(str(s) for s in np.sort(np.bincount(reference, minlength=K))[::-1])
    print(f"Atskaitos suskirstymas (visa B aibe, {len(reference)} obj.): {ref_sizes}")
    print("Imciu palyginimas su atskaitos suskirstymu ir klasteriu dydziai:")
    for _, r in internal.iterrows():
        print(f"  imtis {r['imtis']}: ARI {r['ARI_su_atskaita']:.3f}, NMI {r['NMI_su_atskaita']:.3f}, "
              f"Jaccard {r['Jaccard_su_atskaita']:.3f}, dydziai {r['klasteriu_dydziai']}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot([pairs[col].to_numpy() for col in EXTERNAL], widths=0.5, patch_artist=True,
                    medianprops={"color": "#1f1f1e", "linewidth": 2},
                    whiskerprops={"color": "#6b6a64", "linewidth": 1},
                    capprops={"color": "#6b6a64", "linewidth": 1},
                    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "none",
                                "markeredgecolor": "#6b6a64"},
                    boxprops={"linewidth": 0})
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR)
        patch.set_alpha(0.85)
    for x, col in enumerate(EXTERNAL, start=1):
        ax.scatter(np.full(len(pairs), x) + np.random.default_rng(x).uniform(-0.12, 0.12, len(pairs)),
                   pairs[col], s=10, color="#1f1f1e", alpha=0.45, zorder=3, linewidths=0)
    ax.set_xticks(range(1, len(EXTERNAL) + 1))
    ax.set_xticklabels(EXTERNAL)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mato reikšmė")
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_title(f"k-means stabilumas originalioje erdvėje\n({N_SAMPLES} imčių po 80 %, {len(pairs)} poros)")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "7_kmeans_stabilumas_boxplot.png", dpi=200)
    print()
    print(f"Failas issaugotas i {IMG_DIR / '7_kmeans_stabilumas_boxplot.png'}")


if __name__ == "__main__":
    main()