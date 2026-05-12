from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import umap


INPUT_PATH = Path(
    "data/outputs/experiments/buffer_sensitivity_embeddings.csv"
)

OUTPUT_PATH = Path(
    "data/outputs/experiments/buffer_umap.png"
)


MARKERS = {
    25: "o",
    50: "s",
    100: "^",
    250: "*",
}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    embedding_cols = [
        c for c in df.columns
        if c.startswith("embedding_")
    ]

    if not embedding_cols:
        raise ValueError("No embedding columns found.")

    X = df[embedding_cols].astype(float).to_numpy()

    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
    )

    embedding_2d = reducer.fit_transform(X)

    df["umap_x"] = embedding_2d[:, 0]
    df["umap_y"] = embedding_2d[:, 1]

    plt.figure(figsize=(12, 8))

    unique_fields = sorted(df["field_id"].unique())

    cmap = plt.get_cmap("tab20")

    color_map = {
        field: cmap(i % 20)
        for i, field in enumerate(unique_fields)
    }

    for _, row in df.iterrows():
        plt.scatter(
            row["umap_x"],
            row["umap_y"],
            color=color_map[row["field_id"]],
            marker=MARKERS.get(row["buffer_meters"], "o"),
            s=180,
            alpha=0.8,
            edgecolors="black",
        )

        plt.text(
            row["umap_x"] + 0.1,
            row["umap_y"] + 0.1,
            f"{row['field_id']}_{row['buffer_meters']}m",
            fontsize=8,
        )

    plt.title(
        "UMAP of AlphaEarth Embeddings Across Buffer Sizes",
        fontsize=16,
    )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    legend_elements = []

    for buffer_size, marker in MARKERS.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=f"{buffer_size}m buffer",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=10,
            )
        )

    plt.legend(
        handles=legend_elements,
        title="Buffer Size",
        loc="best",
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_PATH, dpi=300)

    print(f"Saved UMAP plot to {OUTPUT_PATH}")

    print("\nInterpretation guide:")
    print("- Same field clustering tightly = stable embeddings")
    print("- Separation by marker shape = buffer-size sensitivity")
    print("- Large spread = spatial instability")


if __name__ == "__main__":
    main()
