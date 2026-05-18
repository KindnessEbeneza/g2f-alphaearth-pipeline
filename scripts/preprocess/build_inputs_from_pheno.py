"""
Data preprocessing script for Genomes to Fields (G2F).
Takes a raw phenotype CSV, standardizes column names, filters by year,
and splits it into pipeline-ready `fields.csv` and `environment.csv`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


COMMON_COLUMN_ALIASES = {
    "field_id": [
        "Field Location",
        "field_id",
        "FieldLocation",
        "location",
        "site",
        "site_id",
        "field",
    ],
    "year": [
        "Year",
        "year",
        "season_year",
        "trial_year",
    ],
    "latitude": [
        "Latitude",
        "latitude",
        "lat",
        "Lat",
    ],
    "longitude": [
        "Longitude",
        "longitude",
        "lon",
        "Long",
        "lng",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Build AlphaEarth-ready input tables from a merged phenotype CSV."
    )
    parser.add_argument(
        "--input",
        default="user_files/01-merged_pheno-1-.csv",
        help="Path to the merged phenotype CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/real",
        help="Directory where the cleaned output files will be written.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2017,
        help="Minimum year to keep. AlphaEarth annual embeddings start in 2017.",
    )
    parser.add_argument(
        "--field-id-column",
        help="Column to use as the field or site ID. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--year-column",
        help="Column to use as the year. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--latitude-column",
        help="Column to use as latitude. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--longitude-column",
        help="Column to use as longitude. If omitted, the script will try to infer it.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def infer_column_name(
    df: pd.DataFrame,
    requested_name: str | None,
    semantic_name: str,
) -> str:
    if requested_name:
        if requested_name not in df.columns:
            raise ValueError(
                f"Requested {semantic_name} column '{requested_name}' was not found in the input file."
            )
        return requested_name

    for candidate in COMMON_COLUMN_ALIASES[semantic_name]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not infer the {semantic_name} column. Pass --{semantic_name.replace('_', '-').lower()}-column explicitly."
    )


def standardize_core_columns(
    df: pd.DataFrame,
    field_id_column: str,
    year_column: str,
    latitude_column: str,
    longitude_column: str,
) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            field_id_column: "field_id",
            year_column: "year",
            latitude_column: "latitude",
            longitude_column: "longitude",
        }
    ).copy()
    return renamed


def build_fields_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a unique table of field locations and coordinates
    required by the AlphaEarth extraction engine.
    """
    fields = (
        df[["field_id", "year", "longitude", "latitude"]]
        .drop_duplicates()
        .sort_values(["field_id", "year"])
        .reset_index(drop=True)
    )
    return fields


def build_plot_level_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def build_site_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in df.select_dtypes(include=["number"]).columns
        if column not in {"year", "latitude", "longitude"}
    ]

    aggregated = (
        df.groupby(["field_id", "year"], dropna=False)[numeric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )

    pedigree_counts = (
        df.groupby(["field_id", "year"], dropna=False)["Pedigree"]
        .nunique(dropna=True)
        .reset_index(name="pedigree_count")
        if "Pedigree" in df.columns
        else pd.DataFrame(columns=["field_id", "year", "pedigree_count"])
    )
    tester_counts = (
        df.groupby(["field_id", "year"], dropna=False)["Tester"]
        .nunique(dropna=True)
        .reset_index(name="tester_count")
        if "Tester" in df.columns
        else pd.DataFrame(columns=["field_id", "year", "tester_count"])
    )
    plot_counts = (
        df.groupby(["field_id", "year"], dropna=False)
        .size()
        .reset_index(name="plot_count")
    )

    summary = aggregated.merge(pedigree_counts, on=["field_id", "year"], how="left")
    summary = summary.merge(tester_counts, on=["field_id", "year"], how="left")
    summary = summary.merge(plot_counts, on=["field_id", "year"], how="left")
    summary = summary.sort_values(["field_id", "year"]).reset_index(drop=True)
    return summary


def main() -> None:
    """
    Executes the preprocessing logic: parses arguments, infers columns, 
    cleans the data, and exports `fields.csv` and `environment.csv`.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(input_path)
    field_id_column = infer_column_name(df, args.field_id_column, "field_id")
    year_column = infer_column_name(df, args.year_column, "year")
    latitude_column = infer_column_name(df, args.latitude_column, "latitude")
    longitude_column = infer_column_name(df, args.longitude_column, "longitude")

    require_columns(
        df,
        [field_id_column, year_column, latitude_column, longitude_column],
    )

    standardized = standardize_core_columns(
        df,
        field_id_column,
        year_column,
        latitude_column,
        longitude_column,
    )

    filtered = standardized.loc[standardized["year"] >= args.min_year].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows remain after filtering to Year >= {args.min_year}. "
            "Choose a smaller --min-year if needed."
        )

    fields = build_fields_table(filtered)
    plot_level = build_plot_level_table(filtered)
    site_year_summary = build_site_year_summary(filtered)

    output_dir.mkdir(parents=True, exist_ok=True)
    fields_path = output_dir / "fields.csv"
    environment_path = output_dir / "environment.csv"
    plot_level_path = output_dir / "phenotypes_plot_level.csv"
    site_year_path = output_dir / "phenotypes_site_year.csv"

    fields.to_csv(fields_path, index=False)
    site_year_summary.to_csv(environment_path, index=False)
    plot_level.to_csv(plot_level_path, index=False)
    site_year_summary.to_csv(site_year_path, index=False)

    print(f"Read {len(df)} rows from {input_path}")
    print(f"Kept {len(filtered)} rows with Year >= {args.min_year}")
    print(
        "Mapped columns: "
        f"field_id='{field_id_column}', year='{year_column}', "
        f"latitude='{latitude_column}', longitude='{longitude_column}'"
    )
    print(f"Saved {len(fields)} site-year rows to {fields_path}")
    print(f"Saved {len(site_year_summary)} pipeline-ready rows to {environment_path}")
    print(f"Saved {len(plot_level)} plot-level rows to {plot_level_path}")
    print(f"Saved {len(site_year_summary)} site-year summary rows to {site_year_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
