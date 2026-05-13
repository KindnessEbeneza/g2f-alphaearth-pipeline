from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


INPUT = Path("data/experiments/ftw_multisite/ftw_sites_sample.csv")

OUTPUT_ROOT = Path("data/experiments/ftw_multisite/outputs")

DELTA = 0.01

MODEL = "FTW_v2_3_Class_FULL_multiWindow"


def bbox_from_point(lon: float, lat: float) -> str:
    return (
        f"{lon - DELTA},"
        f"{lat - DELTA},"
        f"{lon + DELTA},"
        f"{lat + DELTA}"
    )


def run_ftw(
    field_id: str,
    year: int,
    lon: float,
    lat: float,
) -> int:
    out_dir = OUTPUT_ROOT / f"{field_id}_{year}"

    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = bbox_from_point(lon, lat)

    cmd = [
        "ftw",
        "inference",
        "all",
        f"--bbox={bbox}",
        f"--year={year}",
        f"--out={out_dir}",
        "--cloud_cover_max=60",
        "--buffer_days=60",
        "-m",
         MODEL,
        "--resize_factor=2",
        "--stac_host=earthsearch",
        "--overwrite",
    ]

    print("\nRunning FTW:")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    return result.returncode


def main() -> None:
    df = pd.read_csv(INPUT)

    successes = 0
    failures = 0

    for _, row in df.iterrows():
        field_id = row["field_id"]
        year = int(row["year"])
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        try:
            code = run_ftw(
                field_id=field_id,
                year=year,
                lon=lon,
                lat=lat,
            )

            if code == 0:
                successes += 1
            else:
                failures += 1

        except Exception as exc:
            print(f"FAILED: {field_id}_{year}: {exc}")
            failures += 1

    print("\nFinished FTW multisite inference")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
