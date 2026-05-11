# Genomes to Fields Embedding Pipeline

This project builds a beginner-friendly pipeline for turning field or plot observations into compact vector features using AlphaEarth Foundations satellite embeddings.

## First goal

1. Load a field locations CSV.
2. Attach the target year for each field.
3. Pull the annual AlphaEarth embedding for that year from Google Earth Engine.
4. Sample the 64 embedding bands at each field location.
5. Join any environmental tabular features.
6. Save the final result as a CSV file for downstream modeling.


## Reusable Prep Stage

If you receive a new phenotype-style CSV, you do not need to realign it by hand each time.

Use the preprocessing script to build pipeline-ready tables:

```bash
python scripts/preprocess/build_inputs_from_pheno.py \
  --input path/to/your_file.csv \
  --output-dir data/real
