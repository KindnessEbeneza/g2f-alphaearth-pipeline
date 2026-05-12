from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/outputs/experiments/buffer_sensitivity_embeddings.csv")
OUTPUT_PATH = Path("data/outputs/experiments/buffer_similarity.csv")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return float("nan")

    return float(np.dot(a, b) / (a_norm * b_norm))


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
    if not embedding_cols:
        raise ValueError("No embedding columns found.")

    rows = []

    for (field_id, year), group in df.groupby(["field_id", "year"]):
        group = group.sort_values("buffer_meters")

        for _, row_a in group.iterrows():
            for _, row_b in group.iterrows():
                if row_a["buffer_meters"] >= row_b["buffer_meters"]:
                    continue

                vec_a = row_a[embedding_cols].astype(float).to_numpy()
                vec_b = row_b[embedding_cols].astype(float).to_numpy()

                rows.append(
                    {
                        "field_id": field_id,
                        "year": year,
                        "buffer_a": int(row_a["buffer_meters"]),
                        "buffer_b": int(row_b["buffer_meters"]),
                        "cosine_similarity": cosine_similarity(vec_a, vec_b),
                        "cropland_fraction_a": row_a.get("cropland_fraction"),
                        "cropland_fraction_b": row_b.get("cropland_fraction"),
                        "quality_flag_a": row_a.get("quality_flag"),
                        "quality_flag_b": row_b.get("quality_flag"),
                    }
                )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved similarity results to {OUTPUT_PATH}")
    print(result.head(20).to_string(index=False))
    print("\nSimilarity summary:")
    print(result["cosine_similarity"].describe())


if __name__ == "__main__":
    main()
