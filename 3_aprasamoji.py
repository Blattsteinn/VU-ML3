import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import CSV_DIR, IMG_DIR, RANDOM_STATE, load_B

K_VALUES = [2, 3, 4, 5, 6]
K_PLOT = [4, 5]
CLIP = 3.0

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def fit_labels(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    return pd.Series(model.fit_predict(X), index=X.index, name="cluster")


def median_profile(X, labels):
    overall = X.median()
    profile = X.groupby(labels).median().sub(overall, axis=1).T
    profile.columns.name = "cluster"
    return profile


def plot_profiles(profiles, sizes, path):
    fig, axes = plt.subplots(1, len(K_PLOT), figsize=(5.5 * len(K_PLOT), 7.5),
                             sharey=True, layout="constrained")
    for ax, k in zip(axes, K_PLOT):
        values = profiles[k].to_numpy()
        im = ax.imshow(np.clip(values, -CLIP, CLIP), cmap="RdBu_r",
                       vmin=-CLIP, vmax=CLIP, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center",
                        fontsize=8, color="white" if abs(v) > 2 else "black")
        ax.set_xticks(range(values.shape[1]),
                      [f"{c}\nn = {sizes[k][c]}" for c in profiles[k].columns])
        ax.set_yticks(range(values.shape[0]), profiles[k].index)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Klasteris")
    fig.colorbar(im, ax=axes, shrink=0.6,
                 label="Klasterio mediana - bendra mediana (IQR vienetais)")
    fig.suptitle("Klasterių požymių medianų nuokrypiai nuo bendros medianos")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    X, _ = load_B()

    profiles, sizes, rows = {}, {}, []
    for k in K_VALUES:
        labels = fit_labels(X, k)
        profile = median_profile(X, labels)
        profiles[k] = profile
        sizes[k] = labels.value_counts().sort_index().to_dict()

        print("\n" + "=" * 70)
        print(f"k = {k}, klasteriu dydziai: {sizes[k]}")
        print("Medianu nuokrypiai nuo bendros medianos (IQR vienetais)")
        print(profile.round(2))

        long = profile.rename_axis(index="feature").stack().rename("median_diff").reset_index()
        long.insert(0, "k", k)
        rows.append(long)

    csv_path = CSV_DIR / "3_aprasomosios_medianos.csv"
    pd.concat(rows).to_csv(csv_path, index=False)
    img_path = IMG_DIR / f"3_aprasomosios_k{K_PLOT[0]}_k{K_PLOT[-1]}.png"
    plot_profiles(profiles, sizes, img_path)
    print(f"\nIssaugota: {csv_path.name}, {img_path.name}")


if __name__ == "__main__":
    main()