"""
Utility script to find the nearest field boundary polygon to a given coordinate.
Useful for matching approximate trial coordinates with detected crop fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the nearest polygon selection script."""
    parser = argparse.ArgumentParser(
        description="Select the nearest FTW polygon to a site coordinate."
    )
    parser.add_argument(
        "--polygons",
        required=True,
        help="Path to FTW polygons.parquet file.",
    )
    parser.add_argument(
        "--field-id",
        required=True,
        help="Field/site identifier.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Field/site year.",
    )
    parser.add_argument(
        "--longitude",
        required=True,
        type=float,
        help="Site longitude.",
    )
    parser.add_argument(
        "--latitude",
        required=True,
        type=float,
        help="Site latitude.",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=100.0,
        help="Maximum allowed distance between site point and selected polygon.",
    )
    parser.add_argument(
        "--out-geojson",
        required=True,
        help="Output path for selected polygon GeoJSON.",
    )
    parser.add_argument(
        "--out-metadata",
        required=True,
        help="Output path for selected polygon metadata CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution: Loads the full set of polygons, computes distances 
    to the given site coordinate, selects the nearest one, validates the 
    distance, and saves the selected polygon as GeoJSON.
    """
    args = parse_args()

    polygons_path = Path(args.polygons)
    out_geojson = Path(args.out_geojson)
    out_metadata = Path(args.out_metadata)

    if not polygons_path.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygons_path}")

    gdf = gpd.read_parquet(polygons_path)

    if gdf.empty:
        raise ValueError(f"No polygons found in {polygons_path}")

    if gdf.crs is None:
        raise ValueError(
            "Polygon file has no CRS. Cannot safely compute distances."
        )

    site_point = gpd.GeoDataFrame(
        {
            "field_id": [args.field_id],
            "year": [args.year],
        },
        geometry=[Point(args.longitude, args.latitude)],
        crs="EPSG:4326",
    )

    site_point_projected = site_point.to_crs(gdf.crs)

    gdf = gdf.copy()
    gdf["distance_to_point_m"] = gdf.distance(
        site_point_projected.geometry.iloc[0]
    )
    gdf["polygon_area_m2"] = gdf.geometry.area
    gdf["polygon_perimeter_m"] = gdf.geometry.length

    selected = gdf.sort_values("distance_to_point_m").head(1).copy()

    selected_distance = float(selected["distance_to_point_m"].iloc[0])

    if selected_distance <= args.max_distance_m:
        selection_status = "accepted"
    else:
        selection_status = "rejected_distance_too_large"

    selected["field_id"] = args.field_id
    selected["year"] = args.year
    selected["site_longitude"] = args.longitude
    selected["site_latitude"] = args.latitude
    selected["selection_status"] = selection_status
    selected["max_distance_m"] = args.max_distance_m

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)

    selected.to_crs("EPSG:4326").to_file(out_geojson, driver="GeoJSON")

    polygon_id_column = "id" if "id" in selected.columns else None

    metadata = {
        "field_id": args.field_id,
        "year": args.year,
        "longitude": args.longitude,
        "latitude": args.latitude,
        "selected_polygon_id": (
            selected[polygon_id_column].iloc[0]
            if polygon_id_column
            else None
        ),
        "distance_to_point_m": selected_distance,
        "polygon_area_m2": float(selected["polygon_area_m2"].iloc[0]),
        "polygon_perimeter_m": float(selected["polygon_perimeter_m"].iloc[0]),
        "max_distance_m": args.max_distance_m,
        "selection_status": selection_status,
        "source_polygons": str(polygons_path),
    }

    pd.DataFrame([metadata]).to_csv(out_metadata, index=False)

    print("Selected nearest polygon")
    print(f"Field ID: {args.field_id}")
    print(f"Year: {args.year}")
    print(f"Distance to point: {selected_distance:.2f} m")
    print(f"Polygon area: {metadata['polygon_area_m2']:.2f} m²")
    print(f"Status: {selection_status}")
    print(f"Saved polygon to: {out_geojson}")
    print(f"Saved metadata to: {out_metadata}")


if __name__ == "__main__":
    main()
