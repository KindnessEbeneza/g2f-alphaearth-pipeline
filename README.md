# G2F AlphaEarth Pipeline

> **Cleaner agricultural representations from noisy trial coordinates — powered by Earth observation foundation model embeddings.**

A beginner-friendly, reproducible geospatial embedding pipeline that evaluates whether large-scale Earth observation foundation model embeddings can represent agricultural environments and support downstream ML tasks for the **Genomes to Fields (G2F) Initiative**.

---

## Table of Contents

- [Overview](#overview)
- [Core Question](#core-question)
- [Key Findings](#key-findings)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Bringing Your Own Dataset](#bringing-your-own-dataset)
- [Input Requirements](#input-requirements)
- [Output Files](#output-files)
- [Stable vs Experimental Components](#stable-vs-experimental-components)
- [Current Limitations](#current-limitations)
- [Future Directions](#future-directions)
- [Tech Stack](#tech-stack)
- [Research Context](#research-context)

---

## Overview

Agricultural trial datasets rarely come with precise field boundaries. What is commonly available instead are approximate GPS coordinates. Directly extracting Earth observation embeddings from those coordinates often introduces substantial non-agricultural noise from roads, forests, buildings, water bodies, and surrounding land cover.

This project investigates the use of **Google AlphaEarth embeddings** for agricultural representation learning. It builds a reusable, multi-stage geospatial embedding extraction pipeline that:

- Extracts geospatial embeddings from agricultural trial locations
- Applies spatial aggregation strategies and filters non-agricultural noise
- Evaluates whether the resulting embeddings preserve meaningful agricultural structure
- Merges Earth Engine extractions with Genomes to Fields phenotype data

The pipeline currently combines:

| Stage | Description |
|---|---|
| Spatial buffering | Approximates field area from noisy coordinates |
| Cropland masking | Removes non-agricultural pixels (USDA CDL) |
| NDVI-based filtering | Removes low-vegetation pixels via Sentinel-2 |
| Weighted aggregation | Distance-weighted embedding aggregation |
| Confidence estimation | Quality scoring per site-year |
| Workflow automation | Config-driven, fully reproducible end-to-end runs |

---

## Core Question

> *Can Earth observation foundation model embeddings produce meaningful agricultural representations from noisy real-world trial coordinates — without requiring exact field geometries?*

**Current evidence from the pipeline suggests: Yes** — with appropriate geospatial cleaning and quality estimation strategies.

---

## Key Findings

### Buffer Stability

Buffer sensitivity experiments across spatial scales showed highly stable embedding representations. Observed cosine similarity:

- **Mean ≈ 0.99** across buffer radii from **25 m → 250 m**

This suggests the current representation strategy is robust even under uncertain field geometry.

### Representation Quality Improvements

- Cropland masking measurably reduces non-agricultural contamination
- NDVI filtering improves agricultural purity beyond masking alone
- Distance-weighted aggregation improves embedding consistency
- Confidence scoring provides meaningful representation quality signals

---

## Pipeline Architecture

```
Raw Dataset
    │
    ▼
Validation & Preprocessing
    │
    ▼
Spatial Buffering (25 m / 50 m / 100 m / 250 m)
    │
    ▼
AlphaEarth Embedding Retrieval (Google Earth Engine)
    │
    ▼
Cropland Masking (USDA CDL)
    │
    ▼
NDVI Filtering (Sentinel-2)
    │
    ▼
Distance-Weighted Aggregation (Gaussian spatial weighting)
    │
    ▼
Confidence Estimation
    │
    ▼
ML-Ready Embeddings + Quality Reports
```

### Stage Details

#### 1. Dataset Validation & Preprocessing

Input datasets are validated and standardized before extraction. Validation checks include:

- Missing coordinates
- Invalid geographic values
- Duplicate records
- Formatting consistency

The preprocessing stage converts datasets into a standardized pipeline-ready format with required columns: `field_id`, `year`, `latitude`, `longitude`.

#### 2. Spatial Buffer Generation

Because exact field geometries are often unavailable, the system creates spatial buffers around each coordinate. Validated buffer radii: **25 m, 50 m, 100 m, 250 m**. These buffers approximate the surrounding agricultural area likely associated with the trial site.

```
coordinate  →  buffered candidate agricultural region
```

#### 3. AlphaEarth Embedding Retrieval

The pipeline retrieves per-pixel Earth observation embeddings using **Google Earth Engine** and **Google AlphaEarth embeddings** for all pixels inside each spatial buffer. At this stage the region still contains mixed land cover.

#### 4. Cropland Masking

To reduce non-agricultural contamination, **USDA CDL (Cropland Data Layer)** masks are applied before aggregation. The masking stage removes likely:

- Roads and buildings
- Forests and urban regions
- Water bodies
- Non-cultivated land

```
buffer  →  cropland-only candidate pixels
```

#### 5. NDVI-Based Filtering

A second filtering layer uses vegetation indices to further reduce noisy pixels. The pipeline computes **NDVI from Sentinel-2 imagery** and removes pixels outside expected vegetation thresholds:

| Threshold | Interpretation |
|---|---|
| NDVI < 0.15 | Likely roads / bare soil / built-up areas → removed |
| NDVI > 0.85 | Possible dense permanent canopy → removed |

```
buffer  →  cropland mask  →  NDVI filter  →  cleaner agricultural pixels
```

#### 6. Distance-Weighted Aggregation

Instead of simple averaging, the pipeline uses **Gaussian-style spatial weighting**:

1. Compute distance from center coordinate for each pixel
2. Assign larger weights to nearby pixels (more likely to belong to the target field)
3. Assign smaller weights to distant pixels
4. Compute weighted embedding averages

This reduces edge contamination while preserving local agricultural signal.

#### 7. Confidence Estimation

The pipeline estimates representation quality using retained agricultural pixel fractions:

```
confidence_score = (valid agricultural pixels after masking/filtering)
                 / (total pixels in spatial buffer)
```

| Category | Interpretation |
|---|---|
| `good` | High agricultural purity |
| `review` | Moderate contamination risk |
| `bad` | Heavily mixed / non-agricultural region |

This allows downstream ML systems to explicitly account for representation uncertainty instead of assuming all embeddings are equally reliable.

---

## Project Structure

```
g2f-alphaearth-pipeline/
├── configs/                    # Pipeline and workflow YAML configurations
│                               #   (e.g. real_pipeline.yaml, clean_embedding_workflow.yaml)
├── data/
│   ├── raw/                    # Input datasets
│   ├── processed/              # Intermediate outputs
│   ├── outputs/                # Final embeddings & reports
│   └── reports/                # Quality summaries
│   # (data/ is ignored by git)
├── scripts/
│   ├── workflows/              # End-to-end orchestration
│   │   └── run_clean_embeddings.py   # Main workflow runner
│   ├── preprocess/             # Data preprocessing tools
│   │   └── build_inputs_from_pheno.py
│   ├── experiments/            # Stable experiments
│   ├── experimental/           # Exploratory work
│   ├── ftw/                    # Experimental field delineation (polygon extraction)
│   └── run_pipeline.py         # Main entrypoint to extract embeddings
├── src/
│   └── g2f_embeddings/         # Core library for Earth Engine integration & data merging
├── notebooks/                  # Analysis notebooks
├── pyproject.toml              # Project metadata, dependencies, and build system
├── Makefile                    # Shortcut commands for running tests and pipelines
└── uv.lock                     # Lockfile for the `uv` package manager
```

---

## Setup & Installation

### Prerequisites

- **Python >= 3.12**
- **Earth Engine Account**: An active Google Earth Engine account and Google Cloud project set up for API access.

### 1. Clone the Repository

```bash
git clone https://github.com/KindnessEbeneza/g2f-alphaearth-pipeline.git
cd g2f-alphaearth-pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
make install
# Which runs: pip install -e ".[dev,ee]"
```

Or manually:

```bash
pip install -e ".[ee]"
```

### 4. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Follow the instructions to log in and set your Google Cloud project, then export your project ID:

```bash
export EE_PROJECT=your-project-id
```

Optionally load from a `.env` file:

```bash
source .env
```

---

## Usage

The project includes a `Makefile` with helpful shortcuts for all major pipeline stages.

### Pipeline Modes

The main extraction engine (`src/g2f_embeddings/pipeline.py`) supports multiple execution modes defined via YAML configurations:

| Mode | Description |
|---|---|
| `mock` | Fast local testing — generates random mock embeddings without requiring Earth Engine credentials or internet access |
| `earth-engine-cropland-buffer` | Generates a circular buffer around each site, samples embeddings, and applies a cropland mask to retain only agricultural pixels |

### Available Commands

#### Preprocess Raw Data

Extract fields and environment definitions from a raw Genomes to Fields phenotype CSV:

```bash
make prep-pheno
```

#### Run Local Mock Pipeline

Verify the pipeline logic without calling Earth Engine:

```bash
make run-mock
```

#### Run Earth Engine Extraction

Fetch actual AlphaEarth embeddings (configured via `configs/real_pipeline.yaml`):

```bash
make run-real-ee
```

#### End-to-End Clean Workflow

Execute the entire validated data ingestion, extraction, and reporting workflow (configured via `configs/clean_embedding_workflow.yaml`):

```bash
make run-clean
```

This executes:

```
validation
  → preprocessing
  → embedding extraction
  → cropland masking
  → NDVI filtering
  → weighted aggregation
  → confidence estimation
  → report generation
```

### Experimental Field Delineation Tools

Scripts under `scripts/ftw/` explore field delineation boundaries (polygons) instead of point or buffer radii. These approaches are computationally heavier:

```bash
python scripts/ftw/extract_polygon_alphaearth_embedding.py --help
```

---

## Bringing Your Own Dataset

You can easily pass your own geospatial agricultural dataset through the AlphaEarth embedding pipeline. The setup supports custom column headers, coordinate systems, and filtering.

### 1. Dataset Requirements

Your custom dataset must be a **CSV file** containing the following core attributes:

- **Field/Site Identifier**: A unique name or ID for each field/location (e.g., `location`, `site_id`, `Field 1`).
- **Year**: The year of the observation.

  > **⚠️ Important:** Because Google AlphaEarth annual embeddings start in **2017**, your dataset's years must be **≥ 2017**.

- **Latitude & Longitude**: Geospatial coordinates in decimal degrees (WGS84, EPSG:4326).
- **Metadata/Phenotypic Data (Optional)**: Any other numeric or categorical columns. Numeric columns are automatically averaged per site-year, and unique counts of categorical fields (like pedigrees/testers) are generated.

### 2. Run Options

You can process your dataset either using the **End-to-End Workflow Runner** (recommended) or the **Step-by-Step Manual Preprocessing**.

#### Method A: End-to-End Workflow (Recommended)

This approach validates, preprocesses, extracts, and generates reports in a single unified command.

**Step 1 — Place your dataset CSV** in the workspace:

```
data/my_custom_dataset.csv
```

**Step 2 — Create a new workflow configuration** (e.g., `configs/my_workflow.yaml`) and map your custom column names:

```yaml
input:
  raw_csv: data/my_custom_dataset.csv

columns:
  field_id: Your_Field_ID_Column
  year: Your_Year_Column
  latitude: Your_Latitude_Column
  longitude: Your_Longitude_Column

preprocess:
  output_dir: data/my_custom_preprocessed
  min_year: 2017

pipeline:
  config_path: configs/real_pipeline.yaml
  mode: earth-engine-cropland-buffer  # or "mock" for local testing

reports:
  output_dir: data/outputs/reports
```

**Step 3 — Run the workflow:**

```bash
python scripts/workflows/run_clean_embeddings.py --config configs/my_workflow.yaml
```

---

#### Method B: Step-by-Step Manual Ingestion

This approach gives you manual control over each ingestion stage.

**Step 1 — Preprocess and clean the input.**

Run `build_inputs_from_pheno.py` with custom column overrides to split your CSV into pipeline-ready `fields.csv` and `environment.csv`:

```bash
python scripts/preprocess/build_inputs_from_pheno.py \
  --input data/my_custom_dataset.csv \
  --output-dir data/my_custom_preprocessed \
  --min-year 2017 \
  --field-id-column Your_Field_ID_Column \
  --year-column Your_Year_Column \
  --latitude-column Your_Latitude_Column \
  --longitude-column Your_Longitude_Column
```

**Step 2 — Configure the pipeline.**

Create a pipeline configuration (e.g., `configs/my_pipeline.yaml`) pointing to your preprocessed data:

```yaml
input:
  fields_path: data/my_custom_preprocessed/fields.csv
  environment_path: data/my_custom_preprocessed/environment.csv

output:
  embeddings_path: data/outputs/my_custom_embeddings.csv

pipeline:
  mode: earth-engine
  field_id_column: field_id
  year_column: year
  longitude_column: longitude
  latitude_column: latitude
  scale_meters: 30
  embedding_band_count: 64
  alphaearth_collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
```

**Step 3 — Extract embeddings.**

Execute the pipeline directly using Earth Engine:

```bash
python scripts/run_pipeline.py --config configs/my_pipeline.yaml --mode earth-engine-cropland-buffer
```

### 3. Output Validation & Quality Reports

Once the run completes, the pipeline generates:

| File | Description |
|---|---|
| `data/outputs/reports/input_validation_report.csv` | Confirms all coordinates and fields were read properly; reports missing counts |
| `data/outputs/reports/embedding_quality_summary.csv` | Confidence scores and quality flags (`good`, `review`, `bad`) indicating pixel coverage and cloudiness |
| `data/outputs/my_custom_embeddings.csv` | Final output CSV merging your custom dataset with the 64-dimensional AlphaEarth embeddings |

---

## Input Requirements

### Required Columns

| Column | Required | Description |
|---|---|---|
| `field_id` | ✅ Yes | Unique trial identifier |
| `year` | ✅ Yes | Observation year |
| `latitude` | ✅ Yes | WGS84 decimal degrees |
| `longitude` | ✅ Yes | WGS84 decimal degrees |

### Optional Downstream Labels

| Column Type | Purpose |
|---|---|
| Yield variables | Downstream prediction targets |
| Environmental variables | Environmental context |
| Management variables | Agronomic metadata |
| Trait measurements | Downstream ML targets |

---

## Output Files

### Embedding Outputs

```
data/outputs/g2f_real_embeddings.csv
```

Contains: embedding vectors, confidence scores, aggregation metadata, quality flags, masking information.

### Quality Reports

```
data/outputs/reports/embedding_quality_summary.csv
```

Contains: confidence statistics, valid agricultural pixel fractions, quality distributions.

### Validation Reports

```
data/outputs/reports/input_validation_report.csv
```

Contains: dataset validation diagnostics, coordinate validation summaries, preprocessing checks.

---

## Stable vs Experimental Components

### ✅ Stable Components

Currently integrated into the stable `make run-clean` workflow:

- Preprocessing & validation
- AlphaEarth embedding extraction
- Cropland masking (USDA CDL)
- NDVI filtering (Sentinel-2)
- Distance-weighted aggregation
- Confidence estimation
- Quality reporting
- Workflow automation

### 🧪 Experimental Components

Exploratory components not yet integrated into the stable workflow:

- FTW field delineation
- DBSCAN embedding-space clustering
- Polygon-based field extraction
- Temporal embedding aggregation

---

## Current Limitations

- Buffering remains an approximation — exact field geometry is not yet solved globally
- USDA CDL is currently US-centric (limited global applicability)
- FTW field delineation is not integrated into the stable workflow
- Large-scale Earth Engine inference can be computationally intensive
- Coordinate noise cannot be fully eliminated

---

## Future Directions

- Embedding-space clustering (DBSCAN)
- Adaptive buffer sizing based on local land cover heterogeneity
- Temporal embedding aggregation across growing seasons
- Confidence-aware downstream ML integration
- Scalable field delineation workflows
- Global cropland mask support beyond USDA CDL

---

## Tech Stack

### Earth Observation

- Google Earth Engine
- Google AlphaEarth
- Sentinel-2
- USDA CDL

### Geospatial & Data Processing

- GeoPandas
- Pandas
- NumPy

### ML & Representation Analysis

- Scikit-learn
- UMAP

### Runtime

- Python >= 3.12
- `uv` (package management)

---

## Research Context

This project sits at the intersection of:

- **Geospatial AI** — leveraging spatial context for representation learning
- **Agricultural ML** — yield prediction, crop classification, environmental similarity analysis
- **Remote Sensing** — multispectral satellite imagery processing
- **Earth Observation Foundation Models** — AlphaEarth embeddings as general-purpose representations
- **Representation Learning** — evaluating embedding quality for downstream tasks

The project demonstrates that meaningful agricultural representations can be extracted from noisy coordinate-level trial data **without requiring exact field boundaries**, using scalable geospatial preprocessing and Earth observation foundation models.

### Downstream Applications

The generated embeddings support:

- Yield prediction
- Environmental similarity analysis
- Crop classification
- Agricultural representation learning
- Downstream geospatial ML workflows

---

## Reproducibility & Portability





---



---

*Research focus: Geospatial AI · Agricultural ML · Remote Sensing · Representation Learning · Earth Observation Foundation Models*
