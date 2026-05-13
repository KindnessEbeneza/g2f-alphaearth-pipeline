G2F AlphaEarth Pipeline

A geospatial embedding pipeline for evaluating whether large-scale Earth observation foundation model embeddings can represent agricultural environments and support downstream agricultural machine learning tasks.

Overview

This project investigates the use of Google AlphaEarth embeddings for agricultural representation learning. The pipeline extracts geospatial embeddings from agricultural trial locations, applies spatial aggregation strategies, filters non-agricultural noise, and evaluates whether the resulting embeddings preserve meaningful agricultural structure.

The project focuses on improving the quality of embedding representations before applying downstream machine learning tasks such as crop classification, yield modeling, environmental similarity analysis, or field-level prediction.

Project Objectives

The primary goals of this work are:

Evaluate whether AlphaEarth embeddings meaningfully represent agricultural environments.
Reduce contamination from non-agricultural land cover around site coordinates.
Compare point-based vs buffered spatial representations.
Experiment with cropland masking to better approximate real field conditions.
Explore field delineation approaches for more accurate agricultural boundary extraction.
Build a reproducible geospatial ML preprocessing pipeline for future downstream modeling tasks.
Current Pipeline Architecture
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
Core Methodology
1. Point-Based Embeddings

The simplest approach extracts embeddings directly at the geographic coordinate of each agricultural site.

Advantages
Computationally lightweight
Easy to scale
Preserves exact coordinate information
Limitations
Highly sensitive to geolocation noise
May capture roads, buildings, forests, or neighboring land cover
Often fails to represent actual field conditions
2. Buffered Spatial Aggregation

To better approximate field-scale environmental conditions, spatial buffers are generated around each site coordinate.

Instead of using a single pixel or point:

A circular buffer is created around each site
Embeddings are sampled across the buffered region
Aggregated statistics are computed from all sampled pixels

This approach helps:

Reduce coordinate-level noise
Capture surrounding agricultural context
Produce more stable environmental representations
Why Buffering Matters

Agricultural trial coordinates are often approximate and may not fall exactly inside the cultivated field.

Point extraction may therefore capture:

Roads
Tree lines
Water bodies
Buildings
Adjacent land uses

Buffering improves robustness by incorporating the surrounding agricultural landscape rather than relying on a single location.

3. Cropland-Masked Buffering

Even buffered regions can contain substantial non-agricultural pixels.

To address this, cropland masking was introduced.

Approach
Generate spatial buffers around site coordinates
Apply agricultural land cover masks
Retain only pixels classified as cropland
Aggregate embeddings only from agricultural pixels

This improves representation quality by reducing contamination from:

Urban areas
Forests
Water
Roads
Bare soil outside cultivated areas
Datasets and Geospatial Resources
AlphaEarth Embeddings

Primary representation source used throughout the pipeline.

Used for:

Environmental representation learning
Spatial similarity analysis
Agricultural embedding extraction
ESA WorldCereal

Used as a cropland mask source for filtering agricultural pixels during spatial aggregation.

Purpose
Separate agricultural from non-agricultural land cover
Improve field-level representation quality
Reduce spatial noise
USDA CDL (Cropland Data Layer)

Used for agricultural land cover validation and cropland masking in U.S.-based experiments.

Purpose
Validate agricultural regions
Improve masking accuracy
Compare cropland filtering strategies
FTW (Field Delineation Workflow)

Experimental component for extracting approximate field boundaries.

Intended Purpose
Replace simple circular buffers with field-aware boundaries
Improve agricultural pixel selection
Approximate true field geometry
Current Status

Experimental and not yet integrated into the stable pipeline.

Large-scale FTW inference runs proved computationally expensive on local hardware and require further optimization before production use.

Scripts Overview
Core Pipeline Scripts
Script	Purpose
preprocess_dataset.py	Cleans and standardizes phenotype/site-year datasets
run_alphaearth_embeddings.py	Extracts AlphaEarth embeddings from Google Earth Engine
run_buffered_embeddings.py	Generates buffered agricultural embeddings around site coordinates
run_cropland_masked_embeddings.py	Applies cropland filtering before embedding aggregation
compute_similarity_metrics.py	Computes cosine similarity across spatial aggregation methods
generate_umap_visualizations.py	Produces low-dimensional embedding visualizations
run_ftw_multisite_inference.py	Experimental FTW-based field boundary extraction
Installation
Clone Repository
git clone <repository_url>
cd g2f-alphaearth-pipeline
Create Environment
conda create -n g2f-alphaearth python=3.10
conda activate g2f-alphaearth
Install Dependencies
pip install -r requirements.txt
Project Structure
g2f-alphaearth-pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── scripts/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── experiments/
│   └── analysis/
│
├── notebooks/
│
├── results/
│
├── figures/
│
└── README.md
Example Pipeline Workflow
1. Preprocess Input Dataset

Prepare the raw phenotype or agricultural dataset.

python scripts/preprocessing/preprocess_dataset.py \
    --input data/raw/phenotype.csv \
    --output data/processed/site_years.csv
2. Generate Point-Based AlphaEarth Embeddings

Extract embeddings directly from site coordinates.

python scripts/embeddings/run_alphaearth_embeddings.py \
    --input data/processed/site_years.csv \
    --output data/outputs/point_embeddings.csv
3. Generate Buffered Embeddings

Create spatial buffers around each coordinate and aggregate embeddings.

python scripts/experiments/run_buffered_embeddings.py \
    --input data/processed/site_years.csv \
    --buffer_radius 100 \
    --output data/outputs/buffered_embeddings.csv
4. Run Cropland-Masked Aggregation

Restrict embedding aggregation to agricultural pixels only.

python scripts/experiments/run_cropland_masked_embeddings.py \
    --input data/processed/site_years.csv \
    --buffer_radius 100 \
    --output data/outputs/cropland_embeddings.csv
5. Compute Embedding Similarity Metrics

Evaluate spatial stability across extraction strategies.

python scripts/analysis/compute_similarity_metrics.py \
    --input data/outputs/cropland_embeddings.csv
6. Generate UMAP Visualizations

Visualize embedding distributions.

python scripts/analysis/generate_umap_visualizations.py \
    --input data/outputs/cropland_embeddings.csv
Experimental FTW Workflow
Single-Site FTW Inference
python scripts/experiments/run_ftw_single_site.py
Multisite FTW Inference
python scripts/experiments/run_ftw_multisite_inference.py
FTW Notes
FTW inference is computationally intensive.
Large multisite runs may overheat local machines or exhaust memory resources.
FTW remains experimental and is not yet part of the stable pipeline.
Current work focuses primarily on cropland-masked buffering as the stable preprocessing approach.
Current Findings
Observations
Point embeddings are highly sensitive to spatial noise.
Buffered aggregation improves embedding stability.
Cropland masking significantly reduces non-agricultural contamination.
Agricultural representation quality improves when restricting aggregation to cropland-only pixels.
Current Bottlenecks
1. Non-Agricultural Noise

Even with buffering, mixed land cover remains a major issue.

2. Approximate Site Coordinates

Many agricultural coordinates do not perfectly align with field boundaries.

3. Field Boundary Extraction

Accurate field delineation remains difficult and computationally expensive.

4. Computational Cost

Large-scale geospatial inference workflows require significant compute resources.

Future Work

Potential next steps include:

Improved field delineation methods
Better agricultural masking strategies
Temporal embedding aggregation
Integration with crop yield datasets



Preprocessing
    ↓
Point Embeddings
    ↓
Buffered Aggregation
    ↓
Cropland Masking
    ↓
Embedding Evaluation

FTW experiments should currently be treated as exploratory extensions.

Technologies Used
Python
Google Earth Engine
AlphaEarth embeddings
USDA CDL
GeoPandas
NumPy
Pandas
Scikit-learn
UMAP


Research focus:
Geospatial AI, Agricultural ML, Remote Sensing, Representation Learning, and Earth Observation Foundation Models.
