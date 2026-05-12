from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g2f_embeddings.config import load_config
from g2f_embeddings.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AlphaEarth embedding tables for Genomes to Fields."
    )
    parser.add_argument(
        "--config",
        default=os.getenv("PIPELINE_CONFIG", "configs/pipeline.yaml"),
        help="Path to the pipeline YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "earth-engine", "earth-engine-cropland-buffer"],
        help="Override the mode from the config file.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print the saved output preview instead of rerunning the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.mode:
        config.mode = args.mode

    if args.preview_only:
        import pandas as pd

        if not config.output_path.exists():
            raise FileNotFoundError(
                f"Output file not found at {config.output_path}. Run the pipeline first with --mode mock or --mode earth-engine."
            )
        preview = pd.read_csv(config.output_path)
        print(preview.head())
        return

    result = run_pipeline(config)
    print(f"Saved {len(result)} rows to {config.output_path}")
    print(result.head())


if __name__ == "__main__":
    main()
