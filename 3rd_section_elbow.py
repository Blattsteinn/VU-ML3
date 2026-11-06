import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from common import CSV_DIR, IMG_DIR, RANDOM_STATE, load_B

K_RANGE = range(1, 11)


def elbow(X, k_range=K_RANGE):
    rows = []
    for k in k_range:
        # var(k) = E_k-means (sum of squared distances to own cluster center)
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "var_k": km.inertia_})
    df = pd.DataFrame(rows)
    df["decrease"] = -df["var_k"].diff()
    return df


def plot_elbow(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["k"], df["var_k"], marker="o")
    for k, v in zip(df["k"], df["var_k"]):
        ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(df["k"])
    ax.set_xlabel("Klasterių skaičius k")
    ax.set_ylabel("var(k)")
    ax.set_title("Elbow metodas (k-means, B aibė)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    X, _ = load_B()
    df = elbow(X.to_numpy())
    print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    csv_path = CSV_DIR / "elbow_B.csv"
    df.to_csv(csv_path, index=False)
    img_path = IMG_DIR / "elbow_B.png"
    plot_elbow(df, img_path)
    print(f"Table: {csv_path}")
    print(f"Plot: {img_path}")


if __name__ == "__main__":
    main()