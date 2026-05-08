from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g2f_embeddings.config import load_config
from g2f_embeddings.pipeline import run_pipeline


def test_mock_pipeline_creates_expected_output(tmp_path: Path) -> None:
    config = load_config("configs/pipeline.yaml")
    config.output_path = tmp_path / "embeddings.csv"
    config.mode = "mock"

    result = run_pipeline(config)

    assert len(result) == 2
    assert "embedding_00" in result.columns
    assert "embedding_63" in result.columns
    assert result["source_mode"].tolist() == ["mock", "mock"]

    saved = pd.read_csv(config.output_path)
    assert list(saved.columns) == list(result.columns)
