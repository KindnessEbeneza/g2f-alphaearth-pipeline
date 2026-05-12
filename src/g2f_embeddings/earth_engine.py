from __future__ import annotations

import hashlib
import os
from typing import Iterable

import numpy as np
import pandas as pd
import ee

from .config import PipelineConfig


def initialize_earth_engine() -> None:
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "earthengine-api is not installed. Run `pip install -e '.[ee]'` before using earth-engine mode."
        ) from exc

    project = os.getenv("EE_PROJECT", "").strip()
    if not project:
        raise RuntimeError(
            "EE_PROJECT is empty. Export your Google Cloud project ID before using earth-engine mode."
        )
    ee.Initialize(project=project)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values)
    if norm == 0:
        return values
    return values / norm


def build_mock_embeddings(
    fields: pd.DataFrame,
    field_id_column: str,
    year_column: str,
    longitude_column: str,
    latitude_column: str,
    embedding_band_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in fields.iterrows():
        seed_text = (
            f"{row[field_id_column]}|{row[year_column]}|"
            f"{row[longitude_column]:.6f}|{row[latitude_column]:.6f}"
        )
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        vector = _normalize_vector(rng.normal(size=embedding_band_count).astype(float))

        output_row: dict[str, object] = {
            field_id_column: row[field_id_column],
            year_column: int(row[year_column]),
            "source_mode": "mock",
        }
        for index, value in enumerate(vector):
            output_row[f"embedding_{index:02d}"] = float(value)
        rows.append(output_row)

    return pd.DataFrame(rows)


def _points_to_ee_feature_collection(
    fields: pd.DataFrame,
    field_id_column: str,
    year_column: str,
    longitude_column: str,
    latitude_column: str,
):
    import ee

    features: list = []
    for _, row in fields.iterrows():
        feature = ee.Feature(
            ee.Geometry.Point([float(row[longitude_column]), float(row[latitude_column])]),
            {
                field_id_column: str(row[field_id_column]),
                year_column: int(row[year_column]),
            },
        )
        features.append(feature)
    return ee.FeatureCollection(features)


def _band_names(embedding_band_count: int) -> list[str]:
    return [f"A{index:02d}" for index in range(embedding_band_count)]


def _extract_feature_rows(
    properties_list: Iterable[dict[str, object]],
    field_id_column: str,
    year_column: str,
    embedding_band_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for properties in properties_list:
        row: dict[str, object] = {
            field_id_column: properties[field_id_column],
            year_column: int(properties[year_column]),
            "source_mode": "earth-engine",
        }
        for band_index, band_name in enumerate(_band_names(embedding_band_count)):
            row[f"embedding_{band_index:02d}"] = properties.get(band_name)
        rows.append(row)

    return pd.DataFrame(rows)


def build_alphaearth_embeddings(
    fields: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    initialize_earth_engine()
    import ee

    all_rows: list[pd.DataFrame] = []

    for year, year_fields in fields.groupby(config.year_column):
        start_date = ee.Date.fromYMD(int(year), 1, 1)
        end_date = start_date.advance(1, "year")

        image = (
            ee.ImageCollection(config.alphaearth_collection)
            .filterDate(start_date, end_date)
            .filterBounds(
                ee.Geometry.MultiPoint(
                    year_fields[[config.longitude_column, config.latitude_column]].values.tolist()
                )
            )
            .mosaic()
            .select(_band_names(config.embedding_band_count))
        )

        feature_collection = _points_to_ee_feature_collection(
            year_fields,
            config.field_id_column,
            config.year_column,
            config.longitude_column,
            config.latitude_column,
        )

        sampled = image.sampleRegions(
            collection=feature_collection,
            scale=config.scale_meters,
            geometries=False,
        )

        properties_list = [feature["properties"] for feature in sampled.getInfo()["features"]]
        all_rows.append(
            _extract_feature_rows(
                properties_list,
                config.field_id_column,
                config.year_column,
                config.embedding_band_count,
            )
        )
def build_cropland_buffered_alphaearth_embeddings(
    fields: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    initialize_earth_engine()

    rows: list[dict[str, object]] = []

    for _, row in fields.iterrows():
        field_id = row[config.field_id_column]
        year = int(row[config.year_column])
        lon = float(row[config.longitude_column])
        lat = float(row[config.latitude_column])

        point = ee.Geometry.Point([lon, lat])

        region = point.buffer(config.buffer_meters)

        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = start_date.advance(1, "year")

        alphaearth = (
            ee.ImageCollection(config.alphaearth_collection)
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .mosaic()
            .select(_band_names(config.embedding_band_count))
        )

        if config.cropland_mask == "USDA_CDL":
            cdl = (
                ee.ImageCollection("USDA/NASS/CDL")
                .filterDate(start_date, end_date)
                .first()
            )

            crop_classes = [
                1, 2, 3, 4, 5, 6,
                21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 41, 42, 43, 44, 45,
                46, 47, 48, 49, 50, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 66, 67,
                68, 69, 70, 71, 72, 74, 75, 76, 77
            ]

            crop_mask = (
                cdl.select("cropland")
                .remap(crop_classes, [1] * len(crop_classes), 0)
                .eq(1)
            )

            mask_source = "USDA_CDL"

        else:
            raise ValueError(
                f"Unsupported cropland mask: {config.cropland_mask}"
            )

        masked_alphaearth = alphaearth.updateMask(crop_mask)

        embedding_stats = masked_alphaearth.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=config.scale_meters,
            maxPixels=1_000_000,
        ).getInfo()

        crop_count = crop_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=config.scale_meters,
            maxPixels=1_000_000,
        ).getInfo().get("remapped", 0)

        total_count = ee.Image.constant(1).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=config.scale_meters,
            maxPixels=1_000_000,
        ).getInfo().get("constant", 0)

        cropland_fraction = (
            float(crop_count) / float(total_count)
            if total_count
            else 0.0
        )

        if cropland_fraction >= 0.6:
            quality_flag = "good"
        elif cropland_fraction >= 0.3:
            quality_flag = "review"
        else:
            quality_flag = "bad"

        output_row: dict[str, object] = {
            config.field_id_column: field_id,
            config.year_column: year,
            "source_mode": "earth-engine-cropland-buffer",
            "cropland_fraction": cropland_fraction,
            "valid_pixel_count": crop_count,
            "buffer_meters": config.buffer_meters,
            "mask_source": mask_source,
            "quality_flag": quality_flag,
        }

        for band_index, band_name in enumerate(
            _band_names(config.embedding_band_count)
        ):
            output_row[f"embedding_{band_index:02d}"] = (
                embedding_stats.get(band_name)
            )

        rows.append(output_row)

    return pd.DataFrame(rows)        

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
