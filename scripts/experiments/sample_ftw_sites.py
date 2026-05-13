from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT = Path("data/real/fields.csv")
OUTPUT = Path("data/experiments/ftw_multisite/ftw_sites_sample.csv")

N_SITES = 25
RANDOM_SEED = 42


def main() -> None:
    df = pd.read_csv(INPUT)

    required = [
        "field_id",
        "year",
        "longitude",
        "latitude",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sample = (
        df[required]
        .drop_duplicates()
        .sample(n=min(N_SITES, len(df)), random_state=RANDOM_SEED)
        .sort_values(["year", "field_id"])
        .reset_index(drop=True)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT, index=False)

    print(f"Saved {len(sample)} sampled sites to {OUTPUT}")
    print(sample.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

