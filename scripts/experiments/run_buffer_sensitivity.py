from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


BUFFER_SIZES = [25, 50, 100, 250]

FIELDS_PATH = Path("data/real/fields_buffer_test.csv")
OUTPUT_PATH = Path("data/outputs/experiments/buffer_sensitivity_embeddings.csv")
TEMP_CONFIG_PATH = Path("configs/_tmp_buffer_experiment.yaml")


def write_temp_config(buffer_meters: int, output_path: Path) -> None:
    TEMP_CONFIG_PATH.write_text(
        f"""input:
  fields_path: {FIELDS_PATH}
  environment_path: data/real/environment.csv

output:
  embeddings_path: {output_path}

pipeline:
  mode: earth-engine-cropland-buffer
  field_id_column: field_id
  year_column: year
  longitude_column: longitude
  latitude_column: latitude
  scale_meters: 30
  embedding_band_count: 64
  alphaearth_collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
  buffer_meters: {buffer_meters}
  cropland_mask: USDA_CDL
  cropland_fraction_threshold: 0.5
"""
    )


def main() -> None:
    if not os.getenv("EE_PROJECT"):
        raise RuntimeError("EE_PROJECT is not set. Run: export EE_PROJECT=your-project-id")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    for buffer_meters in BUFFER_SIZES:
        print(f"\nRunning buffer size: {buffer_meters}m")

        buffer_output = OUTPUT_PATH.parent / f"buffer_{buffer_meters}m_embeddings.csv"
        write_temp_config(buffer_meters, buffer_output)

        command = (
            "python scripts/run_pipeline.py "
            f"--config {TEMP_CONFIG_PATH} "
            "--mode earth-engine-cropland-buffer"
        )

        exit_code = os.system(command)
        if exit_code != 0:
            raise RuntimeError(f"Pipeline failed for buffer {buffer_meters}m")

        df = pd.read_csv(buffer_output)
        df["buffer_meters"] = buffer_meters
        df["environment_buffer_id"] = (
            df["field_id"].astype(str)
            + "_"
            + df["year"].astype(str)
            + "_bb_"
            + df["buffer_meters"].astype(str)
        )
        all_outputs.append(df)

    combined = pd.concat(all_outputs, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved combined experiment output to {OUTPUT_PATH}")
    print("Rows:", len(combined))
    print("Columns:", len(combined.columns))


if __name__ == "__main__":
    main()
