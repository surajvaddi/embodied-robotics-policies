.PHONY: test smoke-data smoke-grid libero-dry-run libero-convert-smoke libero-grid robocasa-dry-run robocasa-convert-smoke robocasa-rollout-smoke lerobot-ingest-fake lerobot-ingest-real robocasa-hf-ingest-real openx-validate lerobot-validate-real robocasa-hf-validate-real lerobot-grid-real robocasa-hf-grid-real

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

lerobot-ingest-fake:
	python scripts/ingest_openx.py --config configs/data/openx_lerobot.yaml --subset fake --limit_episodes 2 --steps_per_episode 8 --out data/canonical/openx_small --fake_online

lerobot-ingest-real:
	.venv-lerobot/bin/python scripts/ingest_openx.py --config configs/data/openx_lerobot.yaml --subset small --limit_episodes 1 --steps_per_episode 16 --out data/canonical/lerobot_pusht_small

robocasa-hf-ingest-real:
	.venv-lerobot/bin/python scripts/ingest_openx.py --config configs/data/openx_lerobot.yaml --subset robocasa --limit_episodes 1 --steps_per_episode 16 --out data/canonical/robocasa_hf_small --allow_indexed_download

openx-validate:
	python -m ewpl.data.validation --dataset data/canonical/openx_small --out artifacts/reports/openx_validation.json

lerobot-validate-real:
	python3 -m ewpl.data.validation --dataset data/canonical/lerobot_pusht_small --out artifacts/reports/lerobot_pusht_validation.json

robocasa-hf-validate-real:
	python3 -m ewpl.data.validation --dataset data/canonical/robocasa_hf_small --out artifacts/reports/robocasa_hf_validation.json

lerobot-grid-real:
	python3 scripts/make_demo_grid.py --dataset data/canonical/lerobot_pusht_small --out artifacts/plots/lerobot_pusht_grid.png

robocasa-hf-grid-real:
	python3 scripts/make_demo_grid.py --dataset data/canonical/robocasa_hf_small --out artifacts/plots/robocasa_hf_grid.png
