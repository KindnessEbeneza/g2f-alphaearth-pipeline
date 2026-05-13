from __future__ import annotations

import argparse
import os
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd


ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BAND_COUNT = 64


def band_names(count: int = EMBEDDING_BAND_COUNT) -> list[str]:
    return [f"A{i:02d}" for i in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AlphaEarth embedding inside a selected FTW field polygon."
    )
    parser.add_argument(
        "--polygon",
        required=True,
        help="Path to selected field polygon GeoJSON.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to selected polygon metadata CSV.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Year to sample AlphaEarth annual embedding.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--scale-meters",
        type=int,
        default=30,
        help="Sampling scale in meters.",
    )
    return parser.parse_args()


def initialize_earth_engine() -> None:
    project = os.getenv("EE_PROJECT")
    if not project:
        raise RuntimeError(
            "EE_PROJECT is not set. Run: export EE_PROJECT=your-google-cloud-project-id"
        )

    ee.Initialize(project=project)


def polygon_to_ee_geometry(path: Path) -> ee.Geometry:
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"No geometry found in {path}")

    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.iloc[0]

    geojson = geom.__geo_interface__

    return ee.Geometry(geojson)


def main() -> None:
    args = parse_args()

    polygon_path = Path(args.polygon)
    metadata_path = Path(args.metadata)
    output_path = Path(args.out)

    if not polygon_path.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygon_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    initialize_earth_engine()

    region = polygon_to_ee_geometry(polygon_path)

    start_date = ee.Date.fromYMD(args.year, 1, 1)
    end_date = start_date.advance(1, "year")

    image = (
        ee.ImageCollection(ALPHAEARTH_COLLECTION)
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .mosaic()
        .select(band_names())
    )

    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=args.scale_meters,
        maxPixels=1_000_000,
    ).getInfo()

    pixel_count = image.select("A00").reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=args.scale_meters,
        maxPixels=1_000_000,
    ).getInfo().get("A00", 0)

    metadata = pd.read_csv(metadata_path).iloc[0].to_dict()

    row: dict[str, object] = {
        "field_id": metadata.get("field_id"),
        "year": args.year,
        "source_mode": "earth-engine-ftw-polygon",
        "polygon_source": str(polygon_path),
        "selected_polygon_id": metadata.get("selected_polygon_id"),
        "distance_to_point_m": metadata.get("distance_to_point_m"),
        "polygon_area_m2": metadata.get("polygon_area_m2"),
        "polygon_perimeter_m": metadata.get("polygon_perimeter_m"),
        "valid_pixel_count": pixel_count,
        "scale_meters": args.scale_meters,
    }

    for i, band in enumerate(band_names()):
        row[f"embedding_{i:02d}"] = stats.get(band)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, index=False)

    print(f"Saved polygon AlphaEarth embedding to {output_path}")
    print(f"Valid pixel count: {pixel_count}")
    print(f"Polygon area m²: {row['polygon_area_m2']}")
    print(f"Distance to point m: {row['distance_to_point_m']}")


if __name__ == "__main__":
    main()
