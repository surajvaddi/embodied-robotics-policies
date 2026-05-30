.PHONY: test smoke-data smoke-grid

test:
	pytest

smoke-data:
	python scripts/create_synthetic_dataset.py --out data/smoke --episodes 8 --steps 16

smoke-grid:
	python -m ewpl.data.visualization --dataset data/smoke --out artifacts/plots/smoke_grid.png

