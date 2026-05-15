from pathlib import Path

import pandas as pd


INPUT = Path("data/outputs/experiments/buffer_sensitivity_embeddings.csv")
OUTPUT = Path("data/outputs/experiments/confidence_summary.csv")


def main() -> None:
    df = pd.read_csv(INPUT)

    summary = (
        df.groupby("buffer_meters")
        .agg(
            rows=("confidence_score", "count"),
            mean_confidence=("confidence_score", "mean"),
            min_confidence=("confidence_score", "min"),
            max_confidence=("confidence_score", "max"),
            review_count=("quality_flag", lambda x: (x == "review").sum()),
            bad_count=("quality_flag", lambda x: (x == "bad").sum()),
        )
        .reset_index()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
