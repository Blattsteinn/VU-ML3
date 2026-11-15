import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from common import CSV_DIR, IMG_DIR, RANDOM_STATE, load_B

K_GROUPS = [[2, 3], [4, 5]]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def fit_labels(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    return pd.Series(model.fit_predict(X), index=X.index, name="cluster")


def sizes_table(labels):
    counts = labels.value_counts().sort_index()
    percent = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "percent": percent})


def class_split(labels, y):
    return (pd.crosstab(y, labels, normalize="index") * 100).round(1)


def plot_group(ks, sizes_by_k, split_by_k):
    fig, axes = plt.subplots(1, len(ks), figsize=(5.5 * len(ks), 4.8), sharey=True, layout="constrained")
    for ax, k in zip(axes, ks):
        split = split_by_k[k]
        sizes = sizes_by_k[k]
        values = split.to_numpy()
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 60 else "black")
        ax.set_xticks(range(len(split.columns)),
                      [f"{c}\nn = {sizes.loc[c, 'count']}" for c in split.columns])
        ax.set_yticks(range(len(split.index)), split.index)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Klasteris")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Klasės dalis, %")
    fig.suptitle("Klasių pasiskirstymas k-means klasteriuose, %")
    path = IMG_DIR / f"3_section_class_split_k{ks[0]}_k{ks[-1]}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def main():
    X, y = load_B()
    ks = [k for group in K_GROUPS for k in group]

    sizes_by_k, split_by_k = {}, {}
    sizes_rows, split_rows = [], []

    for k in ks:
        labels = fit_labels(X, k)
        sizes = sizes_table(labels)
        split = class_split(labels, y)
        sizes_by_k[k] = sizes
        split_by_k[k] = split

        print("\n" + "=" * 70)
        print(f"k = {k}")
        print("=" * 70)
        print("\nCluster sizes:")
        print(sizes)
        print("\nKlasiu pasiskirstymas k-means klasteriuose, %")
        print(split)

        sizes_rows.append(sizes.reset_index().assign(k=k)[["k", "cluster", "count", "percent"]])
        long = split.rename_axis(index="class", columns="cluster").stack().rename("percent").reset_index()
        long.insert(0, "k", k)
        split_rows.append(long)

    pd.concat(sizes_rows).to_csv(CSV_DIR / "3_section_class_split_sizes.csv", index=False)
    pd.concat(split_rows).to_csv(CSV_DIR / "3_section_class_split.csv", index=False)
    print("\nSaved 3_section_class_split_sizes.csv and 3_section_class_split.csv")

    for group in K_GROUPS:
        plot_group(group, sizes_by_k, split_by_k)


if __name__ == "__main__":
    main()