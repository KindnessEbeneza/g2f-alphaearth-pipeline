# G2F AlphaEarth Pipeline

A geospatial embedding pipeline for evaluating whether large-scale Earth observation foundation model embeddings can represent agricultural environments and support downstream ML tasks.

---

## Overview

This project investigates the use of **Google AlphaEarth embeddings** for agricultural representation learning. The pipeline extracts geospatial embeddings from agricultural trial locations, applies spatial aggregation strategies, filters non-agricultural noise, and evaluates whether the resulting embeddings preserve meaningful agricultural structure.

The core question: *can Earth observation foundation models produce embeddings that are useful for crop classification, yield modeling, environmental similarity analysis, and field-level prediction?*

---

## Objectives

- Evaluate whether AlphaEarth embeddings meaningfully represent agricultural environments
- Reduce contamination from non-agricultural land cover around site coordinates
- Compare point-based vs. buffered spatial representations
- Experiment with cropland masking to better approximate real field conditions
- Explore field delineation approaches for more accurate agricultural boundary extraction
- Build a reproducible geospatial ML preprocessing pipeline for downstream modeling

---

## Pipeline Architecture

```
Raw Site Coordinates
        ↓
Preprocessing & Cleaning
        ↓
AlphaEarth Embedding Extraction
        ↓
Spatial Aggregation
  ├── Point-based embeddings
  ├── Buffered embeddings
  └── Cropland-masked buffered embeddings
        ↓
Embedding Evaluation
  ├── Similarity analysis
  ├── Spatial consistency checks
  └── UMAP visualization
        ↓
Future Downstream ML Tasks
```

---

## Spatial Aggregation Strategies

### 1. Point-Based Embeddings

Extracts embeddings directly at the geographic coordinate of each agricultural site.

**Pros:** Lightweight, scalable, preserves exact coordinate information.  
**Cons:** Sensitive to geolocation noise; may capture roads, buildings, forests, or neighboring land cover rather than actual field conditions.

### 2. Buffered Aggregation

Generates a circular buffer around each site coordinate, samples embeddings across the buffered region, and aggregates statistics from all sampled pixels.

Agricultural trial coordinates are often approximate and may not fall exactly inside a cultivated field. Buffering improves robustness by incorporating the surrounding agricultural landscape rather than relying on a single point.

### 3. Cropland-Masked Buffering *(recommended)*

Applies agricultural land cover masks to the buffered region, retaining only pixels classified as cropland before aggregating embeddings. This is the current stable approach.

Reduces contamination from urban areas, forests, water, roads, and bare soil outside cultivated areas.

---

## Datasets & Geospatial Resources

| Dataset | Purpose |
|---|---|
| **AlphaEarth Embeddings** | Primary representation source; environmental similarity and agricultural embedding extraction |
| **ESA WorldCereal** | Cropland mask for filtering agricultural pixels during spatial aggregation |
| **USDA CDL** | Agricultural land cover validation and cropland masking for U.S.-based experiments |
| **FTW (Field Delineation Workflow)** | Experimental; extracts approximate field boundaries to replace circular buffers |

> **Note on FTW:** Field boundary inference is computationally expensive on local hardware and not yet integrated into the stable pipeline. Treat FTW experiments as exploratory extensions only.

---

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd g2f-alphaearth-pipeline

# Create and activate environment
conda create -n g2f-alphaearth python=3.10
conda activate g2f-alphaearth

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
g2f-alphaearth-pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── scripts/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── experiments/
│   └── analysis/
├── notebooks/
├── results/
├── figures/
└── README.md
```

---

## Scripts

| Script | Purpose |
|---|---|
| `preprocess_dataset.py` | Cleans and standardizes phenotype/site-year datasets |
| `run_alphaearth_embeddings.py` | Extracts AlphaEarth embeddings from Google Earth Engine |
| `run_buffered_embeddings.py` | Generates buffered agricultural embeddings around site coordinates |
| `run_cropland_masked_embeddings.py` | Applies cropland filtering before embedding aggregation |
| `compute_similarity_metrics.py` | Computes cosine similarity across spatial aggregation methods |
| `generate_umap_visualizations.py` | Produces low-dimensional embedding visualizations |
| `run_ftw_multisite_inference.py` | *(Experimental)* FTW-based field boundary extraction |

---

## Example Workflow

**1. Preprocess input dataset**
```bash
python scripts/preprocessing/preprocess_dataset.py \
  --input data/raw/phenotype.csv \
  --output data/processed/site_years.csv
```

**2. Generate point-based embeddings**
```bash
python scripts/embeddings/run_alphaearth_embeddings.py \
  --input data/processed/site_years.csv \
  --output data/outputs/point_embeddings.csv
```

**3. Generate buffered embeddings**
```bash
python scripts/experiments/run_buffered_embeddings.py \
  --input data/processed/site_years.csv \
  --buffer_radius 100 \
  --output data/outputs/buffered_embeddings.csv
```

**4. Run cropland-masked aggregation**
```bash
python scripts/experiments/run_cropland_masked_embeddings.py \
  --input data/processed/site_years.csv \
  --buffer_radius 100 \
  --output data/outputs/cropland_embeddings.csv
```

**5. Compute similarity metrics**
```bash
python scripts/analysis/compute_similarity_metrics.py \
  --input data/outputs/cropland_embeddings.csv
```

**6. Generate UMAP visualizations**
```bash
python scripts/analysis/generate_umap_visualizations.py \
  --input data/outputs/cropland_embeddings.csv
```

---

## Experimental: FTW Field Delineation

```bash
# Single-site inference
python scripts/experiments/run_ftw_single_site.py

# Multi-site inference
python scripts/experiments/run_ftw_multisite_inference.py
```

> FTW inference is computationally intensive. Large multi-site runs may exhaust memory or overheat local hardware. This workflow is not part of the stable pipeline.

---

## Current Findings

- Point embeddings are highly sensitive to spatial noise
- Buffered aggregation improves embedding stability
- Cropland masking significantly reduces non-agricultural contamination
- Restricting aggregation to cropland-only pixels measurably improves agricultural representation quality

### Known Bottlenecks

- Mixed land cover persists even with buffering
- Many agricultural coordinates do not align precisely with field boundaries
- Accurate field delineation remains computationally expensive
- Large-scale geospatial inference requires significant compute resources

---

## Future Work

- Improved field delineation methods
- Better agricultural masking strategies
- Temporal embedding aggregation
- Integration with crop yield datasets

---

## Tech Stack

Python · Google Earth Engine · AlphaEarth · USDA CDL · ESA WorldCereal · GeoPandas · NumPy · Pandas · Scikit-learn · UMAP

---

*Research focus: Geospatial AI · Agricultural ML · Remote Sensing · Representation Learning · Earth Observation Foundation Models*
