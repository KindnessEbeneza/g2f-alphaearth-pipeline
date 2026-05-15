from __future__ import annotations

import hashlib
import os
from typing import Iterable

import ee
import numpy as np
import pandas as pd

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


def _band_names(embedding_band_count: int) -> list[str]:
    return [f"A{index:02d}" for index in range(embedding_band_count)]


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
    features: list = []

    for _, row in fields.iterrows():
        feature = ee.Feature(
            ee.Geometry.Point(
                [
                    float(row[longitude_column]),
                    float(row[latitude_column]),
                ]
            ),
            {
                field_id_column: str(row[field_id_column]),
                year_column: int(row[year_column]),
            },
        )
        features.append(feature)

    return ee.FeatureCollection(features)


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

    all_rows: list[pd.DataFrame] = []

    for year, year_fields in fields.groupby(config.year_column):
        start_date = ee.Date.fromYMD(int(year), 1, 1)
        end_date = start_date.advance(1, "year")

        image = (
            ee.ImageCollection(config.alphaearth_collection)
            .filterDate(start_date, end_date)
            .filterBounds(
                ee.Geometry.MultiPoint(
                    year_fields[
                        [
                            config.longitude_column,
                            config.latitude_column,
                        ]
                    ].values.tolist()
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

        properties_list = [
            feature["properties"]
            for feature in sampled.getInfo()["features"]
        ]

        all_rows.append(
            _extract_feature_rows(
                properties_list,
                config.field_id_column,
                config.year_column,
                config.embedding_band_count,
            )
        )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def _build_usda_cdl_crop_mask(
    start_date: ee.Date,
    end_date: ee.Date,
) -> ee.Image:
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
        68, 69, 70, 71, 72, 74, 75, 76, 77,
    ]

    return (
        cdl.select("cropland")
        .remap(crop_classes, [1] * len(crop_classes), 0)
        .eq(1)
        .rename("crop_mask")
    )


def _build_ndvi_mask(
    region: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    config: PipelineConfig,
) -> ee.Image:
    s2 = (
        ee.ImageCollection(config.sentinel2_collection)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .median()
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

    return (
        ndvi.gte(config.ndvi_min)
        .And(ndvi.lte(config.ndvi_max))
        .rename("ndvi_mask")
    )


def _count_mask_pixels(
    mask: ee.Image,
    region: ee.Geometry,
    scale_meters: int,
    band_name: str,
) -> float:
    value = (
        mask.rename(band_name)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale_meters,
            maxPixels=1_000_000,
        )
        .getInfo()
        .get(band_name, 0)
    )

    return float(value or 0)


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

        if config.cropland_mask != "USDA_CDL":
            raise ValueError(f"Unsupported cropland mask: {config.cropland_mask}")

        crop_mask = _build_usda_cdl_crop_mask(start_date, end_date)

        final_mask = crop_mask
        mask_source = "USDA_CDL"
        ndvi_filter_applied = False

        if config.use_ndvi_filter:
            ndvi_mask = _build_ndvi_mask(
                region=region,
                start_date=start_date,
                end_date=end_date,
                config=config,
            )

            final_mask = crop_mask.And(ndvi_mask).rename("final_mask")
            mask_source = "USDA_CDL+NDVI"
            ndvi_filter_applied = True
        else:
            final_mask = final_mask.rename("final_mask")

        masked_alphaearth = alphaearth.updateMask(final_mask)

        if config.aggregation_method == "weighted_distance":
            lon_img = ee.Image.pixelLonLat().select("longitude")
            lat_img = ee.Image.pixelLonLat().select("latitude")

            dx_m = lon_img.subtract(lon).multiply(111_320)
            dy_m = lat_img.subtract(lat).multiply(110_540)

            distance_m = dx_m.pow(2).add(dy_m.pow(2)).sqrt()
            sigma = max(config.buffer_meters / 2, 1)

            distance_weight = (
                distance_m
                .pow(2)
                .divide(-2 * sigma * sigma)
                .exp()
            )

            weight = distance_weight.updateMask(final_mask).rename("weight")
            weighted_bands = alphaearth.multiply(weight)

            weighted_sums = weighted_bands.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=config.scale_meters,
                maxPixels=1_000_000,
            ).getInfo()

            weight_sum = weight.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=config.scale_meters,
                maxPixels=1_000_000,
            ).getInfo().get("weight", 0)

            embedding_stats: dict[str, object] = {}

            for band_name in _band_names(config.embedding_band_count):
                if weight_sum:
                    embedding_stats[band_name] = (
                        weighted_sums.get(band_name, 0) / weight_sum
                    )
                else:
                    embedding_stats[band_name] = None

        elif config.aggregation_method == "masked_mean":
            embedding_stats = masked_alphaearth.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=config.scale_meters,
                maxPixels=1_000_000,
            ).getInfo()

        else:
            raise ValueError(
                f"Unsupported aggregation method: {config.aggregation_method}"
            )

        crop_count = _count_mask_pixels(
            crop_mask,
            region,
            config.scale_meters,
            "crop_mask",
        )

        valid_count = _count_mask_pixels(
            final_mask,
            region,
            config.scale_meters,
            "valid_mask",
        )

        total_count = (
            ee.Image.constant(1)
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=region,
                scale=config.scale_meters,
                maxPixels=1_000_000,
            )
            .getInfo()
            .get("constant", 0)
        )

        total_count = float(total_count or 0)

        cropland_fraction = (
            crop_count / total_count
            if total_count
            else 0.0
        )

        valid_fraction = (
            valid_count / total_count
            if total_count
            else 0.0
        )

        confidence_score = valid_fraction

        if confidence_score >= 0.6:
            quality_flag = "good"
        elif confidence_score >= 0.3:
            quality_flag = "review"
        else:
            quality_flag = "bad"

        output_row: dict[str, object] = {
            config.field_id_column: field_id,
            config.year_column: year,
            "source_mode": "earth-engine-cropland-buffer",
            "cropland_fraction": cropland_fraction,
            "valid_fraction": valid_fraction,
            "confidence_score": confidence_score,
            "crop_pixel_count": crop_count,
            "valid_pixel_count": valid_count,
            "total_pixel_count": total_count,
            "buffer_meters": config.buffer_meters,
            "mask_source": mask_source,
            "cropland_mask": config.cropland_mask,
            "aggregation_method": config.aggregation_method,
            "use_ndvi_filter": config.use_ndvi_filter,
            "ndvi_filter_applied": ndvi_filter_applied,
            "ndvi_min": config.ndvi_min,
            "ndvi_max": config.ndvi_max,
            "quality_flag": quality_flag,
        }

        for band_index, band_name in enumerate(_band_names(config.embedding_band_count)):
            output_row[f"embedding_{band_index:02d}"] = embedding_stats.get(band_name)

        rows.append(output_row)

    return pd.DataFrame(rows)