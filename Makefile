.PHONY: test smoke-data smoke-grid libero-dry-run libero-convert-smoke libero-grid robocasa-dry-run robocasa-convert-smoke robocasa-rollout-smoke

test:
	pytest

smoke-data:
	python scripts/create_synthetic_dataset.py --out data/smoke --episodes 8 --steps 16

smoke-grid:
	python -m ewpl.data.visualization --dataset data/smoke --out artifacts/plots/smoke_grid.png

libero-dry-run:
	python scripts/download_libero.py --suite libero_spatial --limit_tasks 2 --dry_run

libero-convert-smoke:
	python scripts/convert_to_canonical.py --source libero --config configs/data/libero.yaml --out data/canonical/libero_spatial_small --limit_episodes 20

libero-grid:
	python scripts/make_demo_grid.py --dataset data/canonical/libero_spatial_small --out artifacts/plots/libero_demo_grid.png

robocasa-dry-run:
	python scripts/download_robocasa.py --dry_run --limit_tasks 3

robocasa-convert-smoke:
	python scripts/convert_to_canonical.py --source robocasa --config configs/data/robocasa.yaml --out data/canonical/robocasa_small --limit_episodes 20

robocasa-rollout-smoke:
	python scripts/eval_rollouts.py --env robocasa --policy random --tasks 2 --episodes 5 --out artifacts/rollouts/robocasa_random
