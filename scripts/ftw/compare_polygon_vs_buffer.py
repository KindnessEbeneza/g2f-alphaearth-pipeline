"""
Analytical script to compare embeddings generated via circular buffering 
vs. precise field delineation polygons using cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


POLYGON_PATH = Path("data/outputs/ftw_test/site_2024/polygon_alphaearth_embedding.csv")
BUFFER_PATH = Path("data/outputs/experiments/buffer_sensitivity_embeddings.csv")
OUTPUT_PATH = Path("data/outputs/ftw_test/site_2024/polygon_vs_buffer_similarity.csv")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two numeric vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return float("nan")

    return float(np.dot(a, b) / (a_norm * b_norm))


def main() -> None:
    """
    Reads the polygon-derived embedding and buffer-derived embeddings, 
    computes the cosine similarities across different configurations, 
    and saves a comparison report.
    """
    polygon = pd.read_csv(POLYGON_PATH)
    buffers = pd.read_csv(BUFFER_PATH)

    embedding_cols = [c for c in polygon.columns if c.startswith("embedding_")]

    if not embedding_cols:
        raise ValueError("No embedding columns found in polygon file.")

    polygon_vec = polygon.iloc[0][embedding_cols].astype(float).to_numpy()

    rows = []

    for _, row in buffers.iterrows():
        buffer_vec = row[embedding_cols].astype(float).to_numpy()

        rows.append(
            {
                "polygon_field_id": polygon.iloc[0].get("field_id"),
                "buffer_field_id": row.get("field_id"),
                "buffer_year": row.get("year"),
                "buffer_meters": row.get("buffer_meters"),
                "cosine_similarity": cosine_similarity(polygon_vec, buffer_vec),
                "buffer_cropland_fraction": row.get("cropland_fraction"),
                "buffer_quality_flag": row.get("quality_flag"),
                "polygon_area_m2": polygon.iloc[0].get("polygon_area_m2"),
                "polygon_distance_to_point_m": polygon.iloc[0].get("distance_to_point_m"),
                "polygon_valid_pixel_count": polygon.iloc[0].get("valid_pixel_count"),
            }
        )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved comparison to {OUTPUT_PATH}")
    print(result.sort_values("cosine_similarity", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
