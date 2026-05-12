from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    fields_path: Path
    environment_path: Path
    output_path: Path
    mode: str
    field_id_column: str
    year_column: str
    longitude_column: str
    latitude_column: str
    scale_meters: int
    embedding_band_count: int
    alphaearth_collection: str
    buffer_meters: int = 100
    cropland_mask: str = "USDA_CDL"
    cropland_fraction_threshold: float = 0.5
    
    
    


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    raw = _load_yaml_like(path.read_text())
    

    return PipelineConfig(
        fields_path=Path(raw["input"]["fields_path"]),
        environment_path=Path(raw["input"]["environment_path"]),
        output_path=Path(raw["output"]["embeddings_path"]),
        mode=raw["pipeline"]["mode"],
        field_id_column=raw["pipeline"]["field_id_column"],
        year_column=raw["pipeline"]["year_column"],
        longitude_column=raw["pipeline"]["longitude_column"],
        latitude_column=raw["pipeline"]["latitude_column"],
        scale_meters=int(raw["pipeline"]["scale_meters"]),
        embedding_band_count=int(raw["pipeline"]["embedding_band_count"]),
        alphaearth_collection=raw["pipeline"]["alphaearth_collection"],
        buffer_meters=int(raw["pipeline"].get("buffer_meters", 100)),
        cropland_mask=raw["pipeline"].get("cropland_mask", "USDA_CDL"),
        cropland_fraction_threshold=float(raw["pipeline"].get("cropland_fraction_threshold", 0.5)),
    )


def _load_yaml_like(text: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            result[current_section] = {}
            continue
        if current_section is None:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        result[current_section][key] = _parse_scalar(value)

    return result


def _parse_scalar(value: str) -> object:
    if value.isdigit():
        return int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value
