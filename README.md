# G2F AlphaEarth Pipeline

A beginner-friendly geospatial embedding pipeline for evaluating whether large-scale Earth observation foundation model embeddings can represent agricultural environments and support downstream ML tasks for the Genomes to Fields (G2F) Initiative.

---

## Goal & Objectives

This project investigates the use of **Google AlphaEarth embeddings** for agricultural representation learning. The pipeline extracts geospatial embeddings from agricultural trial locations, applies spatial aggregation strategies, filters non-agricultural noise, and evaluates whether the resulting embeddings preserve meaningful agricultural structure.

The core question: *can Earth observation foundation models produce embeddings that are useful for crop classification, yield modeling, environmental similarity analysis, and field-level prediction?*

### Objectives
- Evaluate whether AlphaEarth embeddings meaningfully represent agricultural environments.
- Reduce contamination from non-agricultural land cover around site coordinates using spatial buffering and masking.
- Build a reproducible geospatial ML preprocessing pipeline that merges Earth Engine extractions with Genomes to Fields phenotype data.

---

## Implementation Details

The pipeline is implemented in **Python (>=3.12)** and orchestrates data extraction and merging processes using the Earth Engine API.

### Pipeline Modes
The main extraction engine (`src/g2f_embeddings/pipeline.py`) supports multiple execution modes defined via YAML configurations:
- **`mock`**: Fast testing mode that generates random mock embeddings locally to validate the data engineering pipeline without requiring Earth Engine credentials or internet access.
- **`earth-engine-cropland-buffer`**: Generates a circular buffer around each site coordinate and samples embeddings across the region, applying an agricultural land cover mask to retain only cropland pixels. This reduces contamination from urban areas, forests, water, and roads.

### Workflows
End-to-end workflows are wrapped in `scripts/workflows/run_clean_embeddings.py` which:
1. Validates the raw Genomes to Fields (G2F) CSV data.
2. Extracts necessary `fields.csv` (coordinates) and `environment.csv` (metadata) via preprocessing scripts.
3. Invokes the Earth Engine pipeline to compute the buffered and masked AlphaEarth embeddings.
4. Produces quality reports and a final merged CSV containing site years and their corresponding vector embeddings.

---

## Project Structure

```text
g2f-alphaearth-pipeline/
├── configs/             # Pipeline and workflow YAML configurations (e.g. real_pipeline.yaml)
├── data/                # Raw, processed, and output data directories (ignored by git)
├── scripts/             # CLI entrypoints and experimental scripts
│   ├── ftw/             # Experimental Field Delineation tools (Polygon extraction)
│   ├── preprocess/      # Data preprocessing tools (e.g., build_inputs_from_pheno.py)
│   ├── workflows/       # End-to-end workflow runner (run_clean_embeddings.py)
│   └── run_pipeline.py  # Main entrypoint to extract embeddings
├── src/                 # Core python package
│   └── g2f_embeddings/  # Library for Earth Engine integration and data merging
├── pyproject.toml       # Project metadata, dependencies, and build system
├── Makefile             # Shortcut commands for running tests and pipelines
└── uv.lock              # Lockfile for the `uv` package manager
```

---

## Setup & Installation

### 1. Prerequisites
- **Python >= 3.12**
- **Earth Engine Account**: You need to have an active Google Earth Engine account and project set up for API access.

### 2. Environment Setup

Using `pip` or `uv`:
```bash
# Clone the repository
git clone <repository_url>
cd g2f-alphaearth-pipeline

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
make install
# (Which runs: pip install -e ".[dev,ee]")
```

### 3. Earth Engine Authentication
Authenticate your local environment to access Google Earth Engine:
```bash
earthengine authenticate
```
*(Follow the instructions to log in and set your Google Cloud project).*

---

## Usage

The project includes a `Makefile` with helpful shortcuts. 

### 1. Preprocess Raw Data
Extract fields and environment definitions from a raw Genomes to Fields phenotype CSV:
```bash
make prep-pheno
```

### 2. Run Local Mock Pipeline
Verify the pipeline logic without calling Earth Engine:
```bash
make run-mock
```

### 3. Run Earth Engine Extraction
Run the pipeline to fetch actual AlphaEarth embeddings (configured via `configs/real_pipeline.yaml`):
```bash
make run-real-ee
```

### 4. End-to-End Clean Workflow
Execute the entire validated data ingestion, extraction, and reporting workflow configured in `configs/clean_embedding_workflow.yaml`:
```bash
make run-clean
```

---

## Experimental Tools
Scripts under `scripts/ftw/` explore field delineation boundaries (polygons) instead of point or buffer radii. These approaches are computationally heavier and can be explored directly:
```bash
python scripts/ftw/extract_polygon_alphaearth_embedding.py --help
```

---
*Research focus: Geospatial AI · Agricultural ML · Remote Sensing · Representation Learning · Earth Observation Foundation Models*
