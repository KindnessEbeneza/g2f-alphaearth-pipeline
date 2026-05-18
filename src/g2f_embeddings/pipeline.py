"""
Core pipeline orchestrator for extracting and merging AlphaEarth embeddings.
Reads field coordinates and metadata, dispatches to the correct extraction engine
(mock or earth-engine), and merges the results into a single output dataframe.
"""

from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .earth_engine import (
    build_alphaearth_embeddings,
    build_cropland_buffered_alphaearth_embeddings,
    build_mock_embeddings,
)
from .io_utils import ensure_parent_dir, load_environment, load_fields


def _validate_required_columns(
    fields_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    field_id_column: str,
    year_column: str,
    longitude_column: str,
    latitude_column: str,
) -> None:
    missing_field_columns = [
        column
        for column in [field_id_column, year_column, longitude_column, latitude_column]
        if column not in fields_df.columns
    ]
    if missing_field_columns:
        raise ValueError(f"Fields file is missing required columns: {missing_field_columns}")

    missing_environment_columns = [
        column
        for column in [field_id_column, year_column]
        if column not in environment_df.columns
    ]
    if missing_environment_columns:
        raise ValueError(
            f"Environment file is missing required columns: {missing_environment_columns}"
        )


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """
    Executes the embedding extraction pipeline based on the provided configuration.
    
    Loads input field and environment data, validates schemas, dispatches to the
    appropriate embedding generation function (e.g., mock or earth-engine-cropland-buffer),
    and joins the resulting embeddings with the environment metadata.
    
    Args:
        config (PipelineConfig): The parsed configuration object.
        
    Returns:
        pd.DataFrame: A merged dataframe containing the original environment data
                      alongside the extracted embedding vectors and quality flags.
    """
    fields = load_fields(config.fields_path)
    environment = load_environment(config.environment_path)

    _validate_required_columns(
        fields,
        environment,
        config.field_id_column,
        config.year_column,
        config.longitude_column,
        config.latitude_column,
    )

    if config.mode == "mock":
        embeddings = build_mock_embeddings(
            fields,
            config.field_id_column,
            config.year_column,
            config.longitude_column,
            config.latitude_column,
            config.embedding_band_count,
        )
    elif config.mode == "earth-engine-cropland-buffer":
        embeddings = build_cropland_buffered_alphaearth_embeddings(fields, config)
        
    else:
        raise ValueError(
            f"Unsupported pipeline mode '{config.mode}'. Use 'mock' or 'earth-engine' or 'earth-engine-cropland-buffer'."
        )

    merged = embeddings.merge(
        environment,
        on=[config.field_id_column, config.year_column],
        how="left",
        validate="one_to_one",
    )

    ensure_parent_dir(config.output_path)
    merged.to_csv(config.output_path, index=False)
    return merged
