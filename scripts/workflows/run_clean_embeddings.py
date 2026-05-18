"""
End-to-end workflow runner for generating clean AlphaEarth embeddings.
This script orchestrates the full process:
1. Validates the raw input CSV to ensure required columns exist.
2. Runs the preprocessing script to generate field and environment datasets.
3. Executes the main embedding pipeline (mock or earth-engine).
4. Generates a final embedding quality summary report.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def run(cmd: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def validate_input(raw_csv: Path, columns: dict, report_dir: Path) -> None:
    """
    Checks the raw input CSV for required columns mapped in the config.
    Saves a validation report and raises an error if required data is missing.
    """
    df = pd.read_csv(raw_csv)

    required = {
        "field_id": columns["field_id"],
        "year": columns["year"],
        "latitude": columns["latitude"],
        "longitude": columns["longitude"],
    }

    rows = []

    for logical_name, column_name in required.items():
        rows.append({
            "required_field": logical_name,
            "mapped_column": column_name,
            "exists": column_name in df.columns,
            "missing_values": df[column_name].isna().sum() if column_name in df.columns else None,
        })

    report = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_dir / "input_validation_report.csv", index=False)

    missing = [r["mapped_column"] for r in rows if not r["exists"]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\nInput validation report:")
    print(report.to_string(index=False))


def write_quality_report(output_csv: Path, report_dir: Path) -> None:
    """
    Reads the final generated embeddings and produces a summary report 
    describing the confidence scores and quality flags.
    """
    df = pd.read_csv(output_csv)

    report = {
        "rows": len(df),
        "mean_confidence_score": df["confidence_score"].mean() if "confidence_score" in df.columns else None,
        "min_confidence_score": df["confidence_score"].min() if "confidence_score" in df.columns else None,
        "max_confidence_score": df["confidence_score"].max() if "confidence_score" in df.columns else None,
        "good_count": (df["quality_flag"] == "good").sum() if "quality_flag" in df.columns else None,
        "review_count": (df["quality_flag"] == "review").sum() if "quality_flag" in df.columns else None,
        "bad_count": (df["quality_flag"] == "bad").sum() if "quality_flag" in df.columns else None,
    }

    report_df = pd.DataFrame([report])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_dir / "embedding_quality_summary.csv", index=False)

    print("\nEmbedding quality summary:")
    print(report_df.to_string(index=False))


def main() -> None:
    """
    Parses the workflow configuration and executes the end-to-end process:
    validation -> preprocessing -> pipeline extraction -> reporting.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clean_embedding_workflow.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    raw_csv = Path(config["input"]["raw_csv"])
    columns = config["columns"]
    preprocess_output_dir = Path(config["preprocess"]["output_dir"])
    min_year = int(config["preprocess"].get("min_year", 2017))
    pipeline_config = Path(config["pipeline"]["config_path"])
    mode = config["pipeline"]["mode"]
    report_dir = Path(config["reports"]["output_dir"])

    validate_input(raw_csv, columns, report_dir)

    run([
        "python",
        "scripts/preprocess/build_inputs_from_pheno.py",
        "--input",
        str(raw_csv),
        "--output-dir",
        str(preprocess_output_dir),
        "--min-year",
        str(min_year),
    ])

    max_rows = config["preprocess"].get("max_rows")

    if max_rows:
        max_rows = int(max_rows)

        fields_path = preprocess_output_dir / "fields.csv"
        environment_path = preprocess_output_dir / "environment.csv"

        fields = pd.read_csv(fields_path).head(max_rows)
        environment = pd.read_csv(environment_path).head(max_rows)

        fields.to_csv(fields_path, index=False)
        environment.to_csv(environment_path, index=False)

        print(f"\nLimited workflow test to {max_rows} site-year rows.")

    run([
        "python",
        "scripts/run_pipeline.py",
        "--config",
        str(pipeline_config),
        "--mode",
        mode,
    ])

    output_csv = Path("data/outputs/g2f_real_embeddings.csv")
    if output_csv.exists():
        write_quality_report(output_csv, report_dir)
    else:
        print(f"Warning: expected output not found: {output_csv}")

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()