.PHONY: install run-mock run-ee run-real-ee preview test prep-pheno

install:
	pip install --upgrade pip
	pip install -e ".[dev,ee]"

run-mock:
	python scripts/run_pipeline.py --mode mock

run-ee:
	python scripts/run_pipeline.py --mode earth-engine

run-real-ee:
	python scripts/run_pipeline.py --config configs/real_pipeline.yaml --mode earth-engine

prep-pheno:
	python scripts/preprocess/build_inputs_from_pheno.py

preview:
	python scripts/run_pipeline.py --preview-only

test:
	pytest -q