# Embodied World Policy Lab — Full Project Plan

## 1. Project Thesis

**Embodied World Policy Lab** is a research-style robotics project for studying how language-conditioned robot policies improve when they are given richer world representations. The project compares classical imitation-learning policies, diffusion-based policies, post-trained vision-language-action models, learned world models, physics-aware reconstruction, and 3D Gaussian-splatting scene representations.

The central question is:

> **How well do different policy-learning approaches convert vision + language + robot state into robot actions, and how much do world-model, physics-aware, and Gaussian-splatting representations improve generalization, robustness, and failure prediction?**

This project is designed to demonstrate hands-on experience in:

- World-action foundation-model-style systems
- Diffusion policies
- Vision-language-action model post-training
- Multi-modal robot learning
- Physics-aware reconstruction
- Deformable/physical simulation reasoning
- Gaussian-splatting reconstruction
- Computer vision for embodied AI
- PyTorch robot-learning systems
- Distributed model training
- Research-grade experiment automation
- Statistical evaluation and LaTeX report generation

The project does **not** require a physical robot. The primary development loop uses simulation and public datasets. Real-robot data enters through offline demonstration datasets.

---

## 2. Main Experiment

The main experiment is a matrix comparison of **policy family** × **world representation family**.

### 2.1 Policy families

The base policies are:

1. **Behavior Cloning**
   - Supervised learning from demonstrations.
   - Input: image, language instruction, robot state.
   - Output: next robot action.

2. **Action-Chunking Transformer**
   - Transformer that predicts a sequence of future actions instead of a single action.
   - Better suited to smooth multi-step robot behaviors.

3. **Diffusion Policy**
   - Conditional diffusion model over future action trajectories.
   - Learns multi-modal action distributions and denoises action chunks conditioned on observations.

4. **Post-trained VLA model**
   - Fine-tune or adapt an open vision-language-action model.
   - Compares zero-shot VLA behavior against LoRA/QLoRA/OFT-style post-training.

### 2.2 World/representation families

Each policy will eventually be evaluated under multiple input-representation settings:

1. **2D observation baseline**
   - RGB image/video frame
   - Robot proprioceptive state
   - Language instruction

2. **Latent world-model representation**
   - Action-conditioned future latent prediction.
   - Predicts future scene embeddings, future frames, or failure risk.
   - Used as additional policy context or a rollout critic.

3. **Physics-aware reconstruction representation**
   - Reconstructs object states, contact relationships, occlusions, depth structure, support surfaces, and articulated/deformable state variables from RGB-D/sim state.
   - Produces object/contact/geometry tokens for the policy.

4. **Gaussian-splatting scene representation**
   - Builds 3D Gaussian scene reconstructions from multi-view observations.
   - Produces novel-view renderings, visibility/occlusion cues, and compact scene tokens for policies.

5. **Combined world representation**
   - Combines latent world model + physics-aware reconstruction + Gaussian-splatting features.
   - Used for the strongest final variants only, not for every early experiment.

### 2.3 Final comparison matrix

The final comparison should include:

| Policy | 2D Baseline | + World Model | + Physics-Aware Reconstruction | + Gaussian Splatting | + Combined Representation |
|---|---:|---:|---:|---:|---:|
| Behavior Cloning | yes | yes | yes | yes | optional |
| Action-Chunking Transformer | yes | yes | yes | yes | yes |
| Diffusion Policy | yes | yes | yes | yes | yes |
| Post-trained VLA | yes | yes | yes | yes | yes |

### 2.4 Primary metrics

The core metrics are:

- Task success rate
- Average return, if available
- Action prediction error on held-out demonstrations
- Trajectory smoothness
- Collision/contact violation rate
- Goal-condition satisfaction
- Language generalization success
- Scene/layout generalization success
- Object generalization success
- Occlusion robustness
- Deformable/contact-heavy task performance
- Inference latency
- GPU memory usage
- Training throughput
- Sample efficiency
- Failure-mode distribution
- Calibration of failure prediction, when applicable

### 2.5 Hypotheses

The project should test these hypotheses:

1. **H1: Policy class matters.**
   Action-chunking, diffusion, and VLA policies should outperform single-step behavior cloning on longer-horizon tasks.

2. **H2: Diffusion helps with action multi-modality.**
   Diffusion policies should perform better when multiple valid trajectories exist.

3. **H3: VLA post-training improves language generalization.**
   Post-trained VLA models should handle paraphrases and unseen instruction phrasings better than non-VLA policies.

4. **H4: World models improve failure prediction and planning under distribution shift.**
   Latent world models should be useful when evaluating future consequences of action chunks.

5. **H5: Physics-aware reconstruction improves contact-heavy tasks.**
   Reconstructed object/contact/geometry features should improve tasks involving occlusion, articulated objects, support surfaces, and deformable objects.

6. **H6: Gaussian-splatting reconstruction improves 3D spatial generalization.**
   3DGS-based scene tokens should help tasks with occluded objects, novel camera views, and rearranged scenes.

---

## 3. Data Sources and Their Uses

The project uses three realistic data sources.

### 3.1 Data Source 1: LIBERO

**Use:** controlled simulation benchmark for fast iteration, language-conditioned manipulation, spatial generalization, and lifelong/multitask evaluation.

LIBERO should be the first full simulation source integrated because it is controlled enough for rigorous experiments and small enough to iterate on.

Use LIBERO for:

- Initial imitation-learning experiments
- Language-conditioned policy training
- Spatial relationship tasks
- Object-position generalization
- Policy architecture comparisons
- Rollout success evaluation
- Main early-stage paper-style results

Expected data fields:

```text
instruction
rgb images
optional depth images
robot proprioceptive state
action sequence
sim state
success/failure signal
task metadata
```

### 3.2 Data Source 2: RoboCasa / RoboCasa365

**Use:** large-scale household/kitchen manipulation benchmark with diverse scenes and everyday tasks.

Use RoboCasa after LIBERO because it is more realistic and better aligned with household autonomy, but more complex to manage.

Use RoboCasa for:

- Kitchen and household task generalization
- Multi-task learning
- Scene diversity
- Foundation-model-style policy evaluation
- Gaussian reconstruction from richer environments
- Contact-heavy and articulated-object tasks
- Scaling experiments

Expected data fields:

```text
instruction
multi-view RGB
optional depth
robot state
action sequence
object state
scene/task metadata
binary task success from rollouts
```

### 3.3 Data Source 3: Open X-Embodiment / LeRobot-format datasets

**Use:** real-robot offline data for cross-embodiment transfer and real-world demonstration learning.

This data is not used to control a physical robot. It is used offline for:

- Pretraining encoders
- Learning action representations
- Evaluating action prediction on real robot demonstrations
- Testing whether sim-trained representations transfer to real data
- Testing whether real-data pretraining helps simulated rollout performance after fine-tuning

Use LeRobot as the canonical dataset interface wherever possible.

Expected data fields:

```text
videos or image sequences
robot state
actions
language/task labels
metadata about robot embodiment
episode-level metadata
```

---

## 4. Target Repository Structure

```text
embodied-world-policy-lab/
  README.md
  plan.md
  pyproject.toml
  uv.lock or poetry.lock
  Makefile
  Dockerfile
  .pre-commit-config.yaml

  configs/
    data/
      libero.yaml
      robocasa.yaml
      openx_lerobot.yaml
    train/
      bc_libero.yaml
      act_libero.yaml
      diffusion_libero.yaml
      vla_libero_lora.yaml
      world_model.yaml
      physics_recon.yaml
      gaussian_splatting.yaml
      fused_policy.yaml
    eval/
      rollout_libero.yaml
      rollout_robocasa.yaml
      generalization.yaml
      full_matrix.yaml
    report/
      dataset_audit.yaml
      bc_report.yaml
      act_report.yaml
      diffusion_report.yaml
      vla_report.yaml
      world_model_report.yaml
      reconstruction_report.yaml
      full_matrix_report.yaml

  scripts/
    setup_env.sh
    download_libero.py
    download_robocasa.py
    ingest_openx.py
    convert_to_canonical.py
    train.py
    eval_rollouts.py
    run_experiment.py
    generate_report.py
    make_demo_grid.py

  src/
    ewpl/
      __init__.py
      healthcheck.py

      config/
        schemas.py
        registry.py

      data/
        schemas.py
        episode.py
        storage.py
        canonical_dataset.py
        lerobot_adapter.py
        libero_adapter.py
        robocasa_adapter.py
        openx_adapter.py
        splits.py
        augmentations.py
        validation.py
        visualization.py

      sim/
        env_base.py
        libero_env.py
        robocasa_env.py
        scripted_policies.py
        rollout.py
        video.py

      models/
        encoders/
          vision.py
          language.py
          proprio.py
          fusion.py
        policies/
          base.py
          bc.py
          act.py
          diffusion_policy.py
          openvla_adapter.py
        world/
          latent_dynamics.py
          video_predictor.py
          failure_predictor.py
        reconstruction/
          depth_to_pointcloud.py
          physics_state.py
          contact_graph.py
          deformable_state.py
          gaussian_splatting.py
          scene_tokens.py
        fused/
          representation_conditioner.py
          policy_with_world.py

      training/
        loops.py
        losses.py
        optim.py
        schedulers.py
        checkpointing.py
        distributed.py
        mixed_precision.py
        logging.py

      eval/
        metrics.py
        rollout_metrics.py
        offline_metrics.py
        generalization.py
        failure_modes.py
        latency.py
        statistics.py

      experiments/
        runner.py
        manifest.py
        sweep.py
        ablations.py

      reports/
        plots.py
        tables.py
        latex.py
        compile.py
        templates/
          dataset_audit.tex.j2
          model_report.tex.j2
          full_matrix.tex.j2

      utils/
        seed.py
        io.py
        timing.py
        torch.py
        geometry.py

  tests/
    test_imports.py
    test_config_schema.py
    test_data_schema.py
    test_storage_roundtrip.py
    test_libero_adapter.py
    test_robocasa_adapter.py
    test_openx_adapter.py
    test_policy_shapes.py
    test_world_model_shapes.py
    test_reconstruction.py
    test_rollout_smoke.py
    test_statistics.py
    test_report_generation.py

  data/
    raw/
    canonical/
    processed/
    smoke/

  artifacts/
    checkpoints/
    rollouts/
    videos/
    plots/
    reconstructions/
    reports/
    logs/

  reports/
    paper.tex
    paper.pdf
    figures/
    tables/
```

---

# PHASE 0 — Repository, Environment, and Research Skeleton

## Goal

Create a reproducible research codebase that can be installed, tested, linted, and run on CPU or GPU.

## Steps

### Step 0.1 — Initialize package structure

Build:

- `pyproject.toml`
- `src/ewpl/__init__.py`
- `src/ewpl/healthcheck.py`
- `configs/`
- `scripts/`
- `tests/`
- `Makefile`

Implement:

```python
python -m ewpl.healthcheck
```

The healthcheck should print:

- Python version
- PyTorch version
- CUDA availability
- Number of visible GPUs
- Current working directory
- Whether optional robotics dependencies are installed

Tests:

```bash
pytest tests/test_imports.py
pytest tests/test_config_schema.py
ruff check src tests
```

Expected output:

```text
EWPL healthcheck
Python: 3.x
Torch: x.y.z
CUDA available: true/false
Optional deps: libero=?, robocasa=?, lerobot=?
Status: OK
```

### Step 0.2 — Add config system

Build:

- `src/ewpl/config/schemas.py`
- `src/ewpl/config/registry.py`
- base Hydra/OmegaConf config files

Config objects:

- `DataConfig`
- `ModelConfig`
- `TrainConfig`
- `EvalConfig`
- `ReportConfig`
- `ExperimentConfig`

Tests:

```bash
pytest tests/test_config_schema.py
```

Expected output:

```text
All config schemas validate against default YAML files.
```

### Step 0.3 — Add reproducibility tooling

Build:

- `src/ewpl/utils/seed.py`
- `src/ewpl/experiments/manifest.py`

Every run should save:

```text
experiment_id
config snapshot
git commit hash
hostname
timestamp
seed
package versions
```

Tests:

```bash
pytest tests/test_experiment_manifest.py
```

Expected output:

```text
artifacts/logs/<experiment_id>/manifest.json
```

## End-of-phase command

```bash
make test
python -m ewpl.healthcheck
```

## End-of-phase expected output

- Importable Python package
- Passing basic tests
- Working config validation
- Healthcheck report
- Experiment manifest support

---

# PHASE 1 — Canonical Robot Dataset Schema

## Goal

Create a single canonical dataset representation that can hold LIBERO, RoboCasa, and Open X/LeRobot data.

## Steps

### Step 1.1 — Define canonical episode schema

Build:

- `src/ewpl/data/schemas.py`
- `src/ewpl/data/episode.py`

Core objects:

```python
Observation:
    rgb: Tensor or path
    depth: Tensor or path | None
    proprio: Tensor
    language: str
    camera_intrinsics: dict | None
    camera_extrinsics: dict | None
    sim_state: dict | None

Action:
    vector: Tensor
    convention: str
    gripper: float | int | None

Step:
    t: int
    observation: Observation
    action: Action
    reward: float | None
    done: bool
    info: dict

Episode:
    episode_id: str
    source: Literal["libero", "robocasa", "openx", "lerobot", "synthetic"]
    task_id: str
    instruction: str
    steps: list[Step]
    success: bool | None
    metadata: dict
```

Tests:

```bash
pytest tests/test_data_schema.py
```

Expected output:

```text
Canonical Episode objects validate and serialize.
```

### Step 1.2 — Implement storage backend

Build:

- `src/ewpl/data/storage.py`
- `src/ewpl/data/canonical_dataset.py`

Storage should support:

- metadata in JSON/Parquet
- actions/proprio in Parquet or NPZ
- videos as MP4 or frame directories
- optional depth maps
- optional camera calibration files

Tests:

```bash
pytest tests/test_storage_roundtrip.py
```

Expected output:

```text
Synthetic episode saved and loaded with no schema loss.
```

### Step 1.3 — Synthetic smoke dataset

Build:

- `scripts/create_synthetic_dataset.py`
- `src/ewpl/data/visualization.py`

Generate small fake episodes with random images, states, actions, and language instructions.

Tests:

```bash
python scripts/create_synthetic_dataset.py --out data/smoke --episodes 8 --steps 16
pytest tests/test_storage_roundtrip.py
```

Expected output:

```text
data/smoke/
  metadata.parquet
  episodes/
  videos/
  dataset_card.json
artifacts/plots/smoke_contact_sheet.png
```

## End-of-phase command

```bash
python scripts/create_synthetic_dataset.py --out data/smoke --episodes 8 --steps 16
python -m ewpl.data.visualization --dataset data/smoke --out artifacts/plots/smoke_grid.png
pytest tests/test_data_schema.py tests/test_storage_roundtrip.py
```

## End-of-phase expected output

- Canonical dataset schema
- Dataset storage round-trip
- Synthetic dataset
- Contact sheet visualization

---

# PHASE 2 — LIBERO Data and Simulation Integration

## Goal

Integrate LIBERO as the first full robotics benchmark and convert its demonstrations into the canonical dataset format.

## Steps

### Step 2.1 — Add LIBERO environment adapter

Build:

- `src/ewpl/sim/env_base.py`
- `src/ewpl/sim/libero_env.py`
- `src/ewpl/data/libero_adapter.py`

The adapter should expose:

```python
reset(task_id: str, seed: int) -> Observation
step(action: Action) -> StepResult
render(camera: str) -> np.ndarray
is_success() -> bool
get_task_metadata() -> dict
```

Tests:

```bash
pytest tests/test_libero_adapter.py
```

Expected output:

```text
LIBERO adapter can reset one task and return canonical observation.
```

### Step 2.2 — Download or locate LIBERO demonstrations

Build:

- `scripts/download_libero.py`
- `configs/data/libero.yaml`

The script should:

- download or locate LIBERO datasets
- verify checksums when possible
- index tasks
- create a local dataset manifest

Tests:

```bash
python scripts/download_libero.py --suite libero_spatial --limit_tasks 2 --dry_run
```

Expected output:

```text
Found LIBERO suite: libero_spatial
Tasks indexed: 2
Demo files available: true/false
Manifest written: artifacts/logs/libero_manifest.json
```

### Step 2.3 — Convert LIBERO to canonical format

Build:

- `scripts/convert_to_canonical.py`
- `src/ewpl/data/libero_adapter.py::convert_episode`

Tests:

```bash
python scripts/convert_to_canonical.py \
  --source libero \
  --config configs/data/libero.yaml \
  --out data/canonical/libero_spatial_small \
  --limit_episodes 20

pytest tests/test_libero_adapter.py tests/test_storage_roundtrip.py
```

Expected output:

```text
data/canonical/libero_spatial_small/
  dataset_card.json
  metadata.parquet
  episodes/
  videos/
```

### Step 2.4 — LIBERO visualization

Build:

- `src/ewpl/data/visualization.py::render_episode_grid`
- `scripts/make_demo_grid.py`

Tests:

```bash
python scripts/make_demo_grid.py \
  --dataset data/canonical/libero_spatial_small \
  --out artifacts/plots/libero_demo_grid.png
```

Expected output:

- grid of episode frames
- instruction text overlay
- action magnitude plot
- success/failure label

## End-of-phase command

```bash
python scripts/convert_to_canonical.py --source libero --config configs/data/libero.yaml --out data/canonical/libero_spatial_small --limit_episodes 20
python scripts/make_demo_grid.py --dataset data/canonical/libero_spatial_small --out artifacts/plots/libero_demo_grid.png
pytest tests/test_libero_adapter.py tests/test_storage_roundtrip.py
```

## End-of-phase expected output

- LIBERO demos converted into canonical format
- LIBERO smoke visualization
- Basic LIBERO environment reset/step support

---

# PHASE 3 — RoboCasa / RoboCasa365 Integration

## Goal

Add RoboCasa as the richer household/kitchen manipulation benchmark for generalization, scale, and reconstruction experiments.

## Steps

### Step 3.1 — Add RoboCasa environment adapter

Build:

- `src/ewpl/sim/robocasa_env.py`
- `src/ewpl/data/robocasa_adapter.py`
- `configs/data/robocasa.yaml`

Expose the same environment interface used by LIBERO:

```python
reset(task_id: str, scene_id: str, seed: int) -> Observation
step(action: Action) -> StepResult
render(camera: str) -> np.ndarray
is_success() -> bool
get_task_metadata() -> dict
```

Tests:

```bash
pytest tests/test_robocasa_adapter.py
```

Expected output:

```text
RoboCasa adapter can reset one kitchen task and return canonical observation.
```

### Step 3.2 — Index RoboCasa tasks and scenes

Build:

- `scripts/download_robocasa.py`
- `src/ewpl/data/robocasa_adapter.py::index_tasks`

Tests:

```bash
python scripts/download_robocasa.py --dry_run --limit_tasks 3
```

Expected output:

```text
RoboCasa tasks indexed: 3
Scene variations indexed: N
Available demos: true/false
```

### Step 3.3 — Convert RoboCasa demonstrations

Build conversion into canonical format.

Tests:

```bash
python scripts/convert_to_canonical.py \
  --source robocasa \
  --config configs/data/robocasa.yaml \
  --out data/canonical/robocasa_small \
  --limit_episodes 20
```

Expected output:

```text
data/canonical/robocasa_small/
  dataset_card.json
  metadata.parquet
  videos/
  camera_calibration/
```

### Step 3.4 — RoboCasa rollout smoke test

Build a random or scripted policy smoke rollout.

Tests:

```bash
python scripts/eval_rollouts.py \
  --env robocasa \
  --policy random \
  --tasks 2 \
  --episodes 5 \
  --out artifacts/rollouts/robocasa_random
```

Expected output:

```text
rollout_metrics.csv
videos/*.mp4
success_rate: usually low
```

## End-of-phase command

```bash
python scripts/convert_to_canonical.py --source robocasa --config configs/data/robocasa.yaml --out data/canonical/robocasa_small --limit_episodes 20
python scripts/eval_rollouts.py --env robocasa --policy random --tasks 2 --episodes 5 --out artifacts/rollouts/robocasa_random
pytest tests/test_robocasa_adapter.py
```

## End-of-phase expected output

- RoboCasa task index
- RoboCasa canonical dataset
- Random rollout videos and metrics

---

# PHASE 4 — Open X-Embodiment / LeRobot Offline Dataset Integration

## Goal

Integrate real-robot offline datasets using LeRobot/Open X-style data pipelines.

## Steps

### Step 4.1 — Add LeRobot adapter

Build:

- `src/ewpl/data/lerobot_adapter.py`
- `configs/data/openx_lerobot.yaml`

Adapter responsibilities:

- load LeRobot-format datasets
- map videos/images to canonical observation format
- map robot states to canonical proprio format
- map actions to canonical action format
- preserve embodiment metadata

Tests:

```bash
pytest tests/test_openx_adapter.py
```

Expected output:

```text
LeRobot adapter validates dataset metadata and sample access.
```

### Step 4.2 — Add Open X ingestion script

Build:

- `scripts/ingest_openx.py`
- `src/ewpl/data/openx_adapter.py`

The ingestion script should support:

- selecting a small subset
- limiting number of episodes
- filtering by embodiment/task keywords
- converting to canonical format

Tests:

```bash
python scripts/ingest_openx.py \
  --config configs/data/openx_lerobot.yaml \
  --subset small \
  --limit_episodes 20 \
  --out data/canonical/openx_small
```

Expected output:

```text
data/canonical/openx_small/
  dataset_card.json
  metadata.parquet
  videos/
  embodiment_metadata.json
```

### Step 4.3 — Offline real-robot data QA

Build:

- data completeness checks
- action range histograms
- state dimension summaries
- language label coverage

Tests:

```bash
python -m ewpl.data.validation \
  --dataset data/canonical/openx_small \
  --out artifacts/reports/openx_validation.json
```

Expected output:

```text
openx_validation.json
openx_action_histograms.png
openx_language_distribution.png
```

## End-of-phase command

```bash
python scripts/ingest_openx.py --config configs/data/openx_lerobot.yaml --subset small --limit_episodes 20 --out data/canonical/openx_small
python -m ewpl.data.validation --dataset data/canonical/openx_small --out artifacts/reports/openx_validation.json
pytest tests/test_openx_adapter.py
```

## End-of-phase expected output

- Real-robot offline data loaded into the same schema as simulation data
- Action/state/language QA plots
- Embodiment metadata preserved

---

# PHASE 5 — Dataset Splits, Data QA, and Dataset Audit Report

## Goal

Create rigorous train/validation/test splits and generate the first LaTeX PDF report entirely from code.

## Steps

### Step 5.1 — Define split types

Build:

- `src/ewpl/data/splits.py`
- `configs/eval/generalization.yaml`

Splits:

1. **IID split**
   - random train/val/test episodes from same tasks.

2. **Object-position generalization**
   - train and test on different initial object positions.

3. **Scene/layout generalization**
   - train and test on different kitchen/layout variants.

4. **Language paraphrase generalization**
   - train on original instructions, test on paraphrased instructions.

5. **Task-composition generalization**
   - hold out combinations of object + relation + action.

6. **Source transfer split**
   - pretrain on Open X/LeRobot, fine-tune/evaluate on LIBERO/RoboCasa.

Tests:

```bash
pytest tests/test_splits.py
```

Expected output:

```text
No episode leakage across train/validation/test splits.
```

### Step 5.2 — Generate paraphrased instruction set

Build:

- `src/ewpl/data/augmentations.py::paraphrase_instruction`
- `data/processed/paraphrases/*.json`

The initial version can use manually specified paraphrase templates:

```text
"put the mug on the plate"
"place the cup onto the dish"
"move the mug so it rests on the plate"
```

Tests:

```bash
pytest tests/test_instruction_augmentations.py
```

Expected output:

```text
Paraphrase file validates and preserves task intent labels.
```

### Step 5.3 — Dataset statistics pipeline

Build:

- `src/ewpl/eval/statistics.py`
- `src/ewpl/reports/plots.py`
- `src/ewpl/reports/tables.py`

Generate:

- episodes per task
- steps per episode
- action magnitude distributions
- state dimension summaries
- language length distribution
- source/task coverage
- train/val/test split sizes

Tests:

```bash
pytest tests/test_statistics.py
```

Expected output:

```text
artifacts/plots/dataset_action_histogram.png
artifacts/plots/dataset_task_distribution.png
artifacts/tables/dataset_summary.csv
```

### Step 5.4 — Generate LaTeX dataset audit PDF

Build:

- `src/ewpl/reports/latex.py`
- `src/ewpl/reports/compile.py`
- `src/ewpl/reports/templates/dataset_audit.tex.j2`

Tests:

```bash
python scripts/generate_report.py \
  --type dataset_audit \
  --config configs/report/dataset_audit.yaml \
  --out reports/dataset_audit.pdf

pytest tests/test_report_generation.py
```

Expected output:

```text
reports/dataset_audit.pdf
reports/figures/*.png
reports/tables/*.tex
```

The report should include:

- dataset source descriptions
- split definitions
- data volume summary
- action/state distributions
- visual examples
- known limitations

## End-of-phase command

```bash
python -m ewpl.data.splits --dataset data/canonical/libero_spatial_small --config configs/eval/generalization.yaml --out data/processed/splits/libero_spatial
python scripts/generate_report.py --type dataset_audit --config configs/report/dataset_audit.yaml --out reports/dataset_audit.pdf
pytest tests/test_splits.py tests/test_statistics.py tests/test_report_generation.py
```

## End-of-phase expected output

- Formal train/val/test splits
- Dataset audit plots
- `reports/dataset_audit.pdf`
- The first code-generated LaTeX research artifact

---

# PHASE 6 — Unified Rollout and Evaluation Harness

## Goal

Build a model-agnostic evaluation system that can run any policy in LIBERO or RoboCasa and produce comparable metrics, rollout videos, and logs.

## Steps

### Step 6.1 — Define policy interface

Build:

- `src/ewpl/models/policies/base.py`

Policy API:

```python
class Policy(nn.Module):
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    def action_chunk(self, observation: Observation, horizon: int) -> Tensor: ...
```

Tests:

```bash
pytest tests/test_policy_shapes.py
```

Expected output:

```text
Random, scripted, and neural policies conform to the same API.
```

### Step 6.2 — Add rollout runner

Build:

- `src/ewpl/sim/rollout.py`
- `scripts/eval_rollouts.py`
- `src/ewpl/sim/video.py`

The runner should log:

- frames
- actions
- rewards
- success
- termination reason
- policy latency
- per-step metadata

Tests:

```bash
pytest tests/test_rollout_smoke.py
```

Expected output:

```text
Rollout smoke test completes with random policy.
```

### Step 6.3 — Add metric computation

Build:

- `src/ewpl/eval/metrics.py`
- `src/ewpl/eval/rollout_metrics.py`
- `src/ewpl/eval/failure_modes.py`

Metrics:

- success rate
- average episode length
- action L2 norm
- jerk/smoothness
- collision/contact violation rate, when available
- policy latency
- failure mode labels

Tests:

```bash
pytest tests/test_rollout_metrics.py
```

Expected output:

```text
Metric functions return deterministic values on synthetic rollouts.
```

## End-of-phase command

```bash
python scripts/eval_rollouts.py --env libero --policy random --tasks 2 --episodes 5 --out artifacts/rollouts/libero_random
python -m ewpl.eval.rollout_metrics --rollouts artifacts/rollouts/libero_random --out artifacts/reports/libero_random_metrics.csv
pytest tests/test_rollout_smoke.py tests/test_rollout_metrics.py
```

## End-of-phase expected output

- Random/scripted rollout support
- Rollout videos
- Metric CSV/Parquet files
- Baseline evaluation harness ready for learned policies

---

# PHASE 7 — Behavior Cloning Baseline

## Goal

Implement and evaluate the first learned policy: supervised behavior cloning.

## Steps

### Step 7.1 — Build encoders

Build:

- `src/ewpl/models/encoders/vision.py`
- `src/ewpl/models/encoders/language.py`
- `src/ewpl/models/encoders/proprio.py`
- `src/ewpl/models/encoders/fusion.py`

Encoders:

- small CNN for smoke tests
- ResNet/ViT option for RGB
- text embedding for instructions
- MLP for proprioceptive state
- fusion MLP/Transformer

Tests:

```bash
pytest tests/test_encoder_shapes.py
```

Expected output:

```text
Encoders produce fixed-size embeddings from canonical observations.
```

### Step 7.2 — Implement BC policy

Build:

- `src/ewpl/models/policies/bc.py`

Model:

```text
rgb/video frame + language embedding + proprio -> action vector
```

Tests:

```bash
pytest tests/test_policy_shapes.py::test_bc_policy_shape
```

Expected output:

```text
BC policy outputs action_dim tensor.
```

### Step 7.3 — Training loop

Build:

- `src/ewpl/training/loops.py`
- `src/ewpl/training/losses.py`
- `src/ewpl/training/checkpointing.py`
- `scripts/train.py`

Training features:

- train/val loop
- action MSE / Huber loss
- gradient clipping
- checkpoint saving
- metric logging

Tests:

```bash
python scripts/train.py --config configs/train/bc_libero.yaml --overrides trainer.max_steps=20 data.dataset=data/smoke
```

Expected output:

```text
artifacts/checkpoints/bc_smoke/latest.ckpt
artifacts/logs/bc_smoke/train_metrics.csv
```

### Step 7.4 — BC experiment with statistical report

Data collection:

- Train BC on LIBERO train split.
- Run N seeded evaluations on IID and generalization splits.
- Save rollouts and metrics.

Statistical assessment:

- mean success rate
- bootstrap 95% confidence interval
- action error mean/std
- latency mean/std

Plot generation:

- train/validation loss curve
- success-rate bar plot with confidence intervals
- failure-mode distribution
- latency histogram

Report generation:

```bash
python scripts/generate_report.py \
  --type model_report \
  --config configs/report/bc_report.yaml \
  --out reports/bc_report.pdf
```

Tests:

```bash
pytest tests/test_statistics.py tests/test_report_generation.py
```

## End-of-phase command

```bash
python scripts/train.py --config configs/train/bc_libero.yaml
python scripts/eval_rollouts.py --env libero --policy artifacts/checkpoints/bc_libero/latest.ckpt --config configs/eval/rollout_libero.yaml --out artifacts/rollouts/bc_libero
python scripts/generate_report.py --type model_report --config configs/report/bc_report.yaml --out reports/bc_report.pdf
pytest tests/test_policy_shapes.py tests/test_statistics.py tests/test_report_generation.py
```

## End-of-phase expected output

- Trained BC checkpoint
- Rollout videos
- Metrics CSV/Parquet
- Plots
- `reports/bc_report.pdf`

---

# PHASE 8 — Action-Chunking Transformer Policy

## Goal

Implement an action-chunking transformer that predicts multiple future actions per observation.

## Steps

### Step 8.1 — Implement action chunk dataset sampling

Build:

- `src/ewpl/data/canonical_dataset.py::sample_action_chunk`

Each training sample should include:

```text
observation_t
actions_t_to_t+k
mask_t_to_t+k
```

Tests:

```bash
pytest tests/test_action_chunk_sampling.py
```

Expected output:

```text
Action chunks have correct horizon and padding masks.
```

### Step 8.2 — Implement ACT policy

Build:

- `src/ewpl/models/policies/act.py`

Model components:

- image tokens
- language tokens
- proprio tokens
- Transformer encoder/decoder
- action chunk head
- optional CVAE latent variable

Tests:

```bash
pytest tests/test_policy_shapes.py::test_act_policy_shape
```

Expected output:

```text
ACT policy outputs [batch, horizon, action_dim].
```

### Step 8.3 — ACT training and rollout behavior

Build support for:

- chunk prediction loss
- temporal ensembling during inference
- closed-loop action execution

Tests:

```bash
python scripts/train.py --config configs/train/act_libero.yaml --overrides trainer.max_steps=20 data.dataset=data/smoke
pytest tests/test_policy_shapes.py::test_act_policy_shape
```

Expected output:

```text
ACT smoke checkpoint saved and can be loaded for rollout.
```

### Step 8.4 — ACT experiment report

Data collection:

- Train ACT on the same LIBERO splits as BC.
- Evaluate with identical rollout seeds.
- Save rollout videos and metrics.

Statistical assessment:

- paired comparison against BC where seeds/tasks match
- bootstrap confidence intervals
- temporal smoothness comparison

Plot generation:

- BC vs ACT success-rate plot
- action smoothness plot
- horizon-length ablation
- latency vs horizon plot

Report generation:

```bash
python scripts/generate_report.py \
  --type model_report \
  --config configs/report/act_report.yaml \
  --out reports/act_report.pdf
```

## End-of-phase command

```bash
python scripts/train.py --config configs/train/act_libero.yaml
python scripts/eval_rollouts.py --env libero --policy artifacts/checkpoints/act_libero/latest.ckpt --config configs/eval/rollout_libero.yaml --out artifacts/rollouts/act_libero
python scripts/generate_report.py --type model_report --config configs/report/act_report.yaml --out reports/act_report.pdf
pytest tests/test_action_chunk_sampling.py tests/test_policy_shapes.py tests/test_report_generation.py
```

## End-of-phase expected output

- Trained ACT checkpoint
- Rollout videos
- BC vs ACT comparative metrics
- `reports/act_report.pdf`

---

# PHASE 9 — Diffusion Policy

## Goal

Implement a conditional diffusion policy over future action sequences.

## Steps

### Step 9.1 — Diffusion scheduler and noise process

Build:

- `src/ewpl/models/policies/diffusion_policy.py`
- `src/ewpl/training/schedulers.py`

Implement:

- forward noising process
- denoising loss
- DDPM/DDIM-style sampling option
- conditioning on image/language/proprio embeddings

Tests:

```bash
pytest tests/test_diffusion_scheduler.py
```

Expected output:

```text
Noising and denoising tensors have expected shape and variance behavior.
```

### Step 9.2 — Conditional denoiser

Build denoiser with either:

- Transformer over action sequence tokens
- 1D U-Net over action horizon

Inputs:

```text
noisy action chunk
noise timestep
observation embedding
language embedding
proprio embedding
```

Tests:

```bash
pytest tests/test_policy_shapes.py::test_diffusion_policy_shape
```

Expected output:

```text
Diffusion policy predicts noise/action chunk with shape [batch, horizon, action_dim].
```

### Step 9.3 — Diffusion training and inference

Build:

- training loss for denoising
- inference sampler
- receding-horizon execution
- sampling temperature/steps config

Tests:

```bash
python scripts/train.py --config configs/train/diffusion_libero.yaml --overrides trainer.max_steps=20 data.dataset=data/smoke
```

Expected output:

```text
Diffusion smoke checkpoint saved.
```

### Step 9.4 — Diffusion policy experiment report

Data collection:

- Train on same LIBERO split.
- Evaluate IID/generalization splits.
- Evaluate diffusion sampling steps vs latency.

Statistical assessment:

- paired comparisons against BC and ACT
- bootstrap CIs
- success/latency Pareto analysis

Plot generation:

- success rate by model
- action trajectory smoothness
- inference latency vs diffusion steps
- failure modes

Report generation:

```bash
python scripts/generate_report.py \
  --type model_report \
  --config configs/report/diffusion_report.yaml \
  --out reports/diffusion_report.pdf
```

## End-of-phase command

```bash
python scripts/train.py --config configs/train/diffusion_libero.yaml
python scripts/eval_rollouts.py --env libero --policy artifacts/checkpoints/diffusion_libero/latest.ckpt --config configs/eval/rollout_libero.yaml --out artifacts/rollouts/diffusion_libero
python scripts/generate_report.py --type model_report --config configs/report/diffusion_report.yaml --out reports/diffusion_report.pdf
pytest tests/test_diffusion_scheduler.py tests/test_policy_shapes.py tests/test_report_generation.py
```

## End-of-phase expected output

- Trained diffusion policy checkpoint
- Rollout videos
- Sampling-step ablation
- Comparative plots
- `reports/diffusion_report.pdf`

---

# PHASE 10 — Post-Trained Vision-Language-Action Model

## Goal

Integrate and post-train an open VLA model or VLA-style adapter for language-conditioned robot control.

## Steps

### Step 10.1 — VLA data formatter

Build:

- `src/ewpl/models/policies/openvla_adapter.py`
- `src/ewpl/data/lerobot_adapter.py::to_vla_batch`

Formatter should produce:

```text
image input
language prompt
action target or action tokens
action normalization statistics
robot embodiment metadata
```

Tests:

```bash
pytest tests/test_vla_formatting.py
```

Expected output:

```text
VLA batch formatting preserves image, instruction, and normalized action targets.
```

### Step 10.2 — Zero-shot VLA evaluation wrapper

Build a wrapper that can call a pretrained VLA model and convert its predicted actions back into canonical action format.

Tests:

```bash
python scripts/eval_rollouts.py \
  --env libero \
  --policy openvla_zero_shot \
  --tasks 1 \
  --episodes 2 \
  --out artifacts/rollouts/vla_zero_shot_smoke
```

Expected output:

```text
VLA zero-shot wrapper runs without crashing and logs actions.
```

### Step 10.3 — Parameter-efficient VLA post-training

Implement one or more:

- LoRA
- QLoRA
- optimized fine-tuning recipe if supported
- frozen vision encoder ablation
- action-regression head ablation

Build:

- `configs/train/vla_libero_lora.yaml`
- `src/ewpl/training/distributed.py`
- `src/ewpl/training/mixed_precision.py`

Tests:

```bash
python scripts/train.py --config configs/train/vla_libero_lora.yaml --overrides trainer.max_steps=5 data.dataset=data/smoke
```

Expected output:

```text
VLA LoRA smoke checkpoint saved.
```

### Step 10.4 — VLA experiment report

Data collection:

- Evaluate pretrained VLA zero-shot.
- Fine-tune on LIBERO train split.
- Evaluate on IID, language paraphrase, spatial generalization, and object-position generalization splits.

Statistical assessment:

- zero-shot vs fine-tuned comparison
- VLA vs BC/ACT/Diffusion comparison
- language paraphrase success difference
- bootstrap CIs

Plot generation:

- zero-shot vs post-trained success
- language generalization chart
- training memory usage
- inference latency
- failure modes

Report generation:

```bash
python scripts/generate_report.py \
  --type model_report \
  --config configs/report/vla_report.yaml \
  --out reports/vla_report.pdf
```

## End-of-phase command

```bash
python scripts/train.py --config configs/train/vla_libero_lora.yaml
python scripts/eval_rollouts.py --env libero --policy artifacts/checkpoints/vla_libero_lora/latest.ckpt --config configs/eval/rollout_libero.yaml --out artifacts/rollouts/vla_libero_lora
python scripts/generate_report.py --type model_report --config configs/report/vla_report.yaml --out reports/vla_report.pdf
pytest tests/test_vla_formatting.py tests/test_report_generation.py
```

## End-of-phase expected output

- VLA adapter
- Zero-shot VLA rollout logs
- Post-trained VLA checkpoint
- Language generalization report
- `reports/vla_report.pdf`

---

# PHASE 11 — Action-Conditioned World Model

## Goal

Build a world model that predicts future latent states, future frames, and/or failure risk conditioned on observations and candidate actions.

## Steps

### Step 11.1 — Latent observation encoder

Build:

- `src/ewpl/models/world/latent_dynamics.py`
- `src/ewpl/models/world/video_predictor.py`

The encoder maps observations to latent state:

```text
z_t = encoder(rgb_t, proprio_t, language)
```

Tests:

```bash
pytest tests/test_world_model_shapes.py::test_latent_encoder_shape
```

Expected output:

```text
Latent encoder returns [batch, latent_dim].
```

### Step 11.2 — Action-conditioned dynamics model

Build:

```text
z_t, action_chunk_t:t+k -> z_t+1:t+k
```

Architectures:

- GRU dynamics baseline
- Transformer dynamics model
- optional video-prediction decoder

Tests:

```bash
pytest tests/test_world_model_shapes.py::test_dynamics_rollout_shape
```

Expected output:

```text
Predicted latent rollout shape matches horizon.
```

### Step 11.3 — Future-frame prediction head

Build optional decoder:

```text
z_future -> predicted RGB frame or low-resolution frame sequence
```

Losses:

- latent prediction MSE
- image reconstruction loss
- perceptual loss, optional
- success/failure prediction loss

Tests:

```bash
python scripts/train.py --config configs/train/world_model.yaml --overrides trainer.max_steps=20 data.dataset=data/smoke
```

Expected output:

```text
world_model_smoke/latest.ckpt
predicted_vs_actual_grid.png
```

### Step 11.4 — Failure prediction model

Build:

- `src/ewpl/models/world/failure_predictor.py`

Inputs:

```text
current latent state
candidate action chunk
predicted future latent states
```

Outputs:

```text
success_probability
failure_mode_logits
uncertainty estimate
```

Tests:

```bash
pytest tests/test_world_model_shapes.py::test_failure_predictor_shape
```

Expected output:

```text
Failure predictor returns calibrated probability tensor.
```

### Step 11.5 — World model experiment report

Data collection:

- Train world model on successful and failed rollouts.
- Collect rollouts from BC, ACT, Diffusion, and VLA policies.
- Store future prediction targets and actual outcomes.

Statistical assessment:

- one-step prediction error
- multi-step prediction error
- success/failure AUC
- calibration error
- correlation between prediction error and policy failure

Plot generation:

- predicted vs actual future frames
- latent prediction error over horizon
- ROC/PR curves for failure prediction
- calibration plots
- failure prediction examples

Report generation:

```bash
python scripts/generate_report.py \
  --type world_model_report \
  --config configs/report/world_model_report.yaml \
  --out reports/world_model_report.pdf
```

## End-of-phase command

```bash
python scripts/train.py --config configs/train/world_model.yaml
python scripts/run_experiment.py --config configs/eval/world_model_eval.yaml
python scripts/generate_report.py --type world_model_report --config configs/report/world_model_report.yaml --out reports/world_model_report.pdf
pytest tests/test_world_model_shapes.py tests/test_report_generation.py
```

## End-of-phase expected output

- Trained world model checkpoint
- Future latent/frame prediction examples
- Failure prediction metrics
- `reports/world_model_report.pdf`

---

# PHASE 12 — Physics-Aware Reconstruction and Deformable Simulation Track

## Goal

Build a physics-aware representation layer that reconstructs scene geometry, object state, contact relationships, occlusion, support relationships, and deformable/articulated state when available.

## Steps

### Step 12.1 — RGB-D to point cloud reconstruction

Build:

- `src/ewpl/models/reconstruction/depth_to_pointcloud.py`
- `src/ewpl/utils/geometry.py`

Inputs:

```text
RGB image
depth map
camera intrinsics
camera extrinsics
```

Outputs:

```text
colored point cloud
camera-space coordinates
world-space coordinates
```

Tests:

```bash
pytest tests/test_reconstruction.py::test_depth_to_pointcloud_plane
```

Expected output:

```text
Synthetic depth plane reconstructs to expected 3D plane.
```

### Step 12.2 — Object-state and contact graph extraction

Build:

- `src/ewpl/models/reconstruction/physics_state.py`
- `src/ewpl/models/reconstruction/contact_graph.py`

Use simulator ground truth first. Later add predicted versions from RGB-D.

Physics-aware features:

```text
object pose
object velocity
bounding boxes
support relationships
contact graph
relative spatial relationships
occlusion flags
articulated joint state
estimated mass/friction metadata if available
```

Tests:

```bash
pytest tests/test_reconstruction.py::test_contact_graph_schema
```

Expected output:

```text
Contact graph contains nodes, edges, edge types, and timestamps.
```

### Step 12.3 — Deformable/articulated task support

Build:

- `src/ewpl/models/reconstruction/deformable_state.py`
- `configs/data/deformable_tasks.yaml`

Use available simulated tasks involving:

- articulated doors/drawers/knobs
- cloth-like or soft-object tasks if available through selected simulator extensions
- flexible object proxies if full soft-body simulation is unavailable

Represent deformable/articulated state as:

```text
joint angle / drawer extension
surface keypoints
object deformation keypoints
contact patch estimate
```

Tests:

```bash
pytest tests/test_reconstruction.py::test_articulated_state_vector
```

Expected output:

```text
Articulated/deformable state vector validates for a smoke episode.
```

### Step 12.4 — Physics token encoder

Build:

- `src/ewpl/models/reconstruction/scene_tokens.py`

The encoder turns physics-aware state into policy tokens:

```text
object tokens
contact edge tokens
support relation tokens
articulation tokens
occlusion tokens
```

Tests:

```bash
pytest tests/test_reconstruction.py::test_physics_token_encoder_shape
```

Expected output:

```text
Physics token encoder returns [batch, num_tokens, dim].
```

### Step 12.5 — Physics-aware reconstruction experiment report

Data collection:

- Collect RGB-D/sim-state pairs from LIBERO and RoboCasa.
- Collect contact-heavy/articulated episodes.
- Generate point clouds, object states, contact graphs, and token embeddings.

Statistical assessment:

- point cloud reconstruction error where ground truth geometry is available
- object pose reconstruction error
- contact graph precision/recall
- correlation between contact graph errors and policy failures
- articulated/deformable state estimation error

Plot generation:

- 3D point-cloud renderings
- contact graph visualizations
- object-pose error distributions
- contact precision/recall plots
- example failure cases involving contact/occlusion

Report generation:

```bash
python scripts/generate_report.py \
  --type reconstruction_report \
  --config configs/report/physics_reconstruction_report.yaml \
  --out reports/physics_reconstruction_report.pdf
```

## End-of-phase command

```bash
python scripts/run_experiment.py --config configs/train/physics_recon.yaml
python scripts/generate_report.py --type reconstruction_report --config configs/report/physics_reconstruction_report.yaml --out reports/physics_reconstruction_report.pdf
pytest tests/test_reconstruction.py tests/test_report_generation.py
```

## End-of-phase expected output

- RGB-D point cloud reconstruction
- Object/contact/articulation state extraction
- Physics token encoder
- Contact/occlusion/deformation metrics
- `reports/physics_reconstruction_report.pdf`

---

# PHASE 13 — Gaussian-Splatting Scene Reconstruction

## Goal

Build a 3D Gaussian-splatting reconstruction pipeline and convert reconstructions into policy-usable scene features.

## Steps

### Step 13.1 — Multi-view data capture from simulation

Build:

- `src/ewpl/models/reconstruction/gaussian_splatting.py`
- `scripts/capture_multiview_scene.py`

For each selected scene/task, capture:

```text
multi-view RGB frames
camera intrinsics
camera extrinsics
optional depth maps
object/task metadata
```

Tests:

```bash
python scripts/capture_multiview_scene.py \
  --env libero \
  --task_id smoke_task \
  --views 8 \
  --out artifacts/reconstructions/smoke_multiview
```

Expected output:

```text
images/*.png
cameras.json
scene_metadata.json
```

### Step 13.2 — Gaussian reconstruction backend wrapper

Build wrapper for chosen backend:

- official 3DGS implementation, or
- `gsplat`, or
- another maintained PyTorch-compatible Gaussian splatting backend

Interface:

```python
fit_gaussians(images, cameras, config) -> GaussianScene
render_view(scene, camera) -> image
save_scene(scene, path)
load_scene(path) -> GaussianScene
```

Tests:

```bash
pytest tests/test_gaussian_splatting.py::test_gaussian_backend_smoke
```

Expected output:

```text
Gaussian backend can optimize or load a tiny synthetic scene.
```

### Step 13.3 — Novel-view rendering metrics

Build:

- PSNR
- SSIM
- LPIPS, optional
- depth/geometry consistency if depth is available
- render latency

Tests:

```bash
pytest tests/test_gaussian_splatting.py::test_novel_view_metric_shapes
```

Expected output:

```text
Novel-view metrics return scalar values for held-out views.
```

### Step 13.4 — Gaussian scene token encoder

Build:

- `src/ewpl/models/reconstruction/scene_tokens.py::GaussianSceneEncoder`

Possible scene features:

```text
rendered policy-view embedding
visibility/occlusion scores
object-centric Gaussian clusters
camera-conditioned scene token
occupancy/query features around end effector
```

Tests:

```bash
pytest tests/test_gaussian_splatting.py::test_gaussian_scene_encoder_shape
```

Expected output:

```text
Gaussian scene encoder returns [batch, num_tokens, dim].
```

### Step 13.5 — Gaussian reconstruction experiment report

Data collection:

- Capture multi-view scenes from LIBERO and RoboCasa.
- Build 3DGS reconstructions for selected task scenes.
- Hold out views for evaluation.

Statistical assessment:

- PSNR/SSIM/LPIPS across scenes
- reconstruction quality vs number of views
- render latency
- occlusion visibility accuracy
- relation between reconstruction quality and downstream policy success

Plot generation:

- input views
- novel rendered views
- reconstruction quality histograms
- view-count ablation plots
- latency plots
- scene-token visualization

Report generation:

```bash
python scripts/generate_report.py \
  --type reconstruction_report \
  --config configs/report/gaussian_reconstruction_report.yaml \
  --out reports/gaussian_reconstruction_report.pdf
```

## End-of-phase command

```bash
python scripts/capture_multiview_scene.py --env libero --config configs/train/gaussian_splatting.yaml --out artifacts/reconstructions/libero_multiview
python scripts/run_experiment.py --config configs/train/gaussian_splatting.yaml
python scripts/generate_report.py --type reconstruction_report --config configs/report/gaussian_reconstruction_report.yaml --out reports/gaussian_reconstruction_report.pdf
pytest tests/test_gaussian_splatting.py tests/test_report_generation.py
```

## End-of-phase expected output

- Multi-view scene captures
- 3D Gaussian reconstructions
- Novel-view renderings
- Gaussian scene tokens
- `reports/gaussian_reconstruction_report.pdf`

---

# PHASE 14 — Representation Fusion: Policies with World/Physics/Gaussian Context

## Goal

Integrate world-model, physics-aware, and Gaussian-splatting representations into the policy families.

## Steps

### Step 14.1 — Unified representation conditioner

Build:

- `src/ewpl/models/fused/representation_conditioner.py`

The conditioner should expose:

```python
conditioner(observation, candidate_action_chunk=None) -> RepresentationBundle
```

RepresentationBundle:

```text
base_obs_embedding
world_latent_tokens
future_prediction_tokens
failure_risk_token
physics_tokens
gaussian_scene_tokens
combined_tokens
```

Tests:

```bash
pytest tests/test_fusion.py::test_representation_bundle_schema
```

Expected output:

```text
RepresentationBundle validates with optional token groups.
```

### Step 14.2 — Add fused policy wrapper

Build:

- `src/ewpl/models/fused/policy_with_world.py`

Wrapper modes:

```text
policy=bc, representation=2d
policy=bc, representation=world
policy=bc, representation=physics
policy=bc, representation=gaussian
policy=bc, representation=combined
...
```

Tests:

```bash
pytest tests/test_fusion.py::test_fused_policy_shapes
```

Expected output:

```text
Fused BC/ACT/Diffusion/VLA policies return valid action outputs.
```

### Step 14.3 — Integrate world model as critic or context

Two modes:

1. **Context mode**
   - predicted future latents become input tokens.

2. **Critic mode**
   - policy proposes candidate action chunks.
   - world model predicts failure risk.
   - choose candidate with lower risk.

Tests:

```bash
pytest tests/test_fusion.py::test_world_model_critic_ranking
```

Expected output:

```text
World critic ranks synthetic safe action above unsafe action.
```

### Step 14.4 — Integrate physics-aware tokens

Add physics tokens to:

- BC fusion MLP
- ACT transformer input tokens
- Diffusion conditioning vector
- VLA prompt/action adapter as auxiliary scene tokens or side-channel features

Tests:

```bash
pytest tests/test_fusion.py::test_physics_tokens_in_all_policy_families
```

Expected output:

```text
All policy classes accept physics tokens.
```

### Step 14.5 — Integrate Gaussian scene tokens

Add Gaussian tokens as:

- rendered view embedding
- occupancy queries around end effector
- visibility/occlusion features
- object-centric 3D scene tokens

Tests:

```bash
pytest tests/test_fusion.py::test_gaussian_tokens_in_all_policy_families
```

Expected output:

```text
All policy classes accept Gaussian scene tokens.
```

## End-of-phase command

```bash
python scripts/train.py --config configs/train/fused_policy.yaml --overrides policy=bc representation=world trainer.max_steps=20 data.dataset=data/smoke
python scripts/train.py --config configs/train/fused_policy.yaml --overrides policy=act representation=physics trainer.max_steps=20 data.dataset=data/smoke
python scripts/train.py --config configs/train/fused_policy.yaml --overrides policy=diffusion representation=gaussian trainer.max_steps=20 data.dataset=data/smoke
pytest tests/test_fusion.py tests/test_policy_shapes.py
```

## End-of-phase expected output

- Fused policy abstraction
- All policy families can consume additional world/physics/3DGS tokens
- Smoke checkpoints for fused variants

---

# PHASE 15 — Full Policy × Representation Matrix Experiment

## Goal

Run the central research experiment comparing all major policy families and world-representation variants.

## Steps

### Step 15.1 — Experiment manifest and grid generation

Build:

- `src/ewpl/experiments/sweep.py`
- `configs/eval/full_matrix.yaml`

Generate experiment cells:

```text
BC-2D
BC-World
BC-Physics
BC-Gaussian
ACT-2D
ACT-World
ACT-Physics
ACT-Gaussian
ACT-Combined
Diffusion-2D
Diffusion-World
Diffusion-Physics
Diffusion-Gaussian
Diffusion-Combined
VLA-ZeroShot
VLA-PostTrained-2D
VLA-PostTrained-World
VLA-PostTrained-Physics
VLA-PostTrained-Gaussian
VLA-PostTrained-Combined
```

Tests:

```bash
pytest tests/test_experiment_grid.py
```

Expected output:

```text
Experiment grid validates and creates unique experiment IDs.
```

### Step 15.2 — Data collection

For each experiment cell:

- train or load checkpoint
- evaluate on fixed tasks
- run multiple seeds
- save rollout videos
- save per-episode metrics
- save per-step logs
- save model resource metrics

Required splits:

```text
IID
object-position generalization
language paraphrase generalization
scene/layout generalization
occlusion-heavy tasks
contact/articulated tasks
```

Run:

```bash
python scripts/run_experiment.py --config configs/eval/full_matrix.yaml --stage collect
```

Expected output:

```text
artifacts/experiments/full_matrix/
  metrics.parquet
  per_step_logs.parquet
  rollouts/
  checkpoints/
  manifests/
```

### Step 15.3 — Statistical assessment

Build:

- `src/ewpl/eval/statistics.py`

Include:

- bootstrap 95% confidence intervals
- paired model comparisons where same task/seed pairs exist
- mixed-effects logistic regression for binary success
- Holm-Bonferroni correction for multiple comparisons
- effect sizes
- rank stability across seeds
- correlation between reconstruction quality and downstream success

Run:

```bash
python scripts/run_experiment.py --config configs/eval/full_matrix.yaml --stage stats
```

Expected output:

```text
artifacts/experiments/full_matrix/stats/
  model_ranking.csv
  pairwise_tests.csv
  confidence_intervals.csv
  mixed_effects_summary.txt
```

### Step 15.4 — Plot generation

Build:

- `src/ewpl/reports/plots.py`

Generate:

- model success-rate matrix heatmap
- success with confidence intervals
- success vs latency Pareto plot
- language generalization plot
- scene generalization plot
- occlusion/contact task plot
- failure-mode stacked bars
- world-model calibration plot
- Gaussian reconstruction quality vs downstream success plot
- physics contact accuracy vs downstream success plot

Run:

```bash
python scripts/run_experiment.py --config configs/eval/full_matrix.yaml --stage plots
```

Expected output:

```text
reports/figures/full_matrix_success_heatmap.png
reports/figures/success_latency_pareto.png
reports/figures/language_generalization.png
reports/figures/reconstruction_vs_success.png
```

### Step 15.5 — Full LaTeX research report

Build:

- `src/ewpl/reports/templates/full_matrix.tex.j2`

Report sections:

1. Abstract
2. Introduction
3. Related work summary
4. Dataset sources
5. Methodology
6. Policy families
7. World and reconstruction representations
8. Experimental setup
9. Results
10. Statistical analysis
11. Failure analysis
12. Limitations
13. Future work
14. Appendix with configs and hyperparameters

Run:

```bash
python scripts/generate_report.py \
  --type full_matrix \
  --config configs/report/full_matrix_report.yaml \
  --out reports/full_policy_matrix.pdf
```

Expected output:

```text
reports/full_policy_matrix.pdf
reports/full_policy_matrix.tex
reports/figures/*.png
reports/tables/*.tex
```

## End-of-phase command

```bash
python scripts/run_experiment.py --config configs/eval/full_matrix.yaml --stage collect
python scripts/run_experiment.py --config configs/eval/full_matrix.yaml --stage stats
python scripts/run_experiment.py --config configs/eval/full_matrix.yaml --stage plots
python scripts/generate_report.py --type full_matrix --config configs/report/full_matrix_report.yaml --out reports/full_policy_matrix.pdf
pytest tests/test_experiment_grid.py tests/test_statistics.py tests/test_report_generation.py
```

## End-of-phase expected output

- Full policy × representation metrics
- Statistical tests
- Research plots
- Full LaTeX PDF report
- Clear answer to the main research question

---

# PHASE 16 — Cross-Source Transfer Experiments

## Goal

Test whether real-robot offline data and large-scale simulated household data improve downstream manipulation performance.

## Steps

### Step 16.1 — Pretraining pipeline

Build configs for:

```text
pretrain on Open X/LeRobot subset
fine-tune on LIBERO
fine-tune on RoboCasa
train from scratch on LIBERO
train from scratch on RoboCasa
```

Tests:

```bash
python scripts/train.py --config configs/train/pretrain_openx.yaml --overrides trainer.max_steps=20 data.dataset=data/smoke
```

Expected output:

```text
Pretraining smoke checkpoint saved.
```

### Step 16.2 — Transfer evaluation

Evaluate:

- Open X pretraining -> LIBERO fine-tuning
- Open X pretraining -> RoboCasa fine-tuning
- LIBERO training -> RoboCasa evaluation/fine-tuning
- RoboCasa training -> LIBERO evaluation/fine-tuning

Data collection:

- same tasks and seeds where possible
- offline action prediction on real-robot heldout data
- rollout success in simulation

Statistical assessment:

- sample efficiency curves
- transfer gain over scratch training
- confidence intervals across seeds
- action-space normalization sensitivity

Plot generation:

- success vs number of fine-tuning demos
- transfer matrix heatmap
- action prediction error by embodiment
- language coverage comparison

Report generation:

```bash
python scripts/generate_report.py \
  --type transfer_report \
  --config configs/report/transfer_report.yaml \
  --out reports/cross_source_transfer.pdf
```

## End-of-phase command

```bash
python scripts/run_experiment.py --config configs/eval/cross_source_transfer.yaml --stage collect
python scripts/run_experiment.py --config configs/eval/cross_source_transfer.yaml --stage stats
python scripts/run_experiment.py --config configs/eval/cross_source_transfer.yaml --stage plots
python scripts/generate_report.py --type transfer_report --config configs/report/transfer_report.yaml --out reports/cross_source_transfer.pdf
pytest tests/test_openx_adapter.py tests/test_statistics.py tests/test_report_generation.py
```

## End-of-phase expected output

- Cross-source transfer metrics
- Sample-efficiency curves
- Transfer matrix heatmap
- `reports/cross_source_transfer.pdf`

---

# PHASE 17 — Distributed Training and Scaling Infrastructure

## Goal

Make the project credible as a scalable ML/robotics training system.

## Steps

### Step 17.1 — Add DDP/FSDP-compatible training

Build:

- `src/ewpl/training/distributed.py`
- support for `torchrun`
- rank-aware logging
- distributed checkpointing
- distributed sampler support

Tests:

```bash
python -m torch.distributed.run --nproc_per_node=1 scripts/train.py --config configs/train/bc_libero.yaml --overrides trainer.max_steps=10 data.dataset=data/smoke
```

Expected output:

```text
Single-process distributed smoke training completes.
```

### Step 17.2 — Mixed precision and memory controls

Build:

- AMP/bfloat16 support
- gradient accumulation
- gradient checkpointing
- activation checkpointing for VLA/diffusion
- memory logging

Tests:

```bash
python scripts/train.py --config configs/train/diffusion_libero.yaml --overrides trainer.precision=bf16 trainer.max_steps=10 data.dataset=data/smoke
```

Expected output:

```text
Mixed-precision smoke training completes.
```

### Step 17.3 — Scaling benchmark report

Data collection:

- train small workloads on 1 GPU and multi-GPU if available
- collect examples/sec
- GPU memory usage
- checkpoint time
- dataloader throughput

Statistical assessment:

- throughput mean/std over repeated runs
- scaling efficiency
- bottleneck attribution

Plot generation:

- throughput by model
- memory by model
- scaling efficiency
- dataloader vs model time

Report generation:

```bash
python scripts/generate_report.py \
  --type scaling_report \
  --config configs/report/scaling_report.yaml \
  --out reports/scaling_report.pdf
```

## End-of-phase command

```bash
python -m torch.distributed.run --nproc_per_node=1 scripts/train.py --config configs/train/bc_libero.yaml --overrides trainer.max_steps=10 data.dataset=data/smoke
python scripts/run_experiment.py --config configs/eval/scaling.yaml
python scripts/generate_report.py --type scaling_report --config configs/report/scaling_report.yaml --out reports/scaling_report.pdf
pytest tests/test_distributed.py tests/test_report_generation.py
```

## End-of-phase expected output

- Distributed training support
- Mixed precision training
- Resource metrics
- `reports/scaling_report.pdf`

---

# PHASE 18 — Demo Dashboard and Research README

## Goal

Create a polished repo that reads like a research project and visually demonstrates robot policies, reconstructions, and results.

## Steps

### Step 18.1 — README rewrite

The README should include:

```text
Project title
Research question
Method overview diagram
Data sources
Policy families
World/reconstruction modules
Experiment matrix
Main results table
Failure-mode examples
Rollout GIFs
Gaussian reconstruction examples
How to reproduce
How to run a small smoke experiment
How to generate the paper PDF
```

Expected README command examples:

```bash
make setup
make test
make smoke-data
make train-bc-smoke
make eval-smoke
make report-smoke
```

Tests:

```bash
python scripts/validate_readme_commands.py
```

Expected output:

```text
All README smoke commands are valid.
```

### Step 18.2 — Demo dashboard

Build:

- `dashboard/streamlit_app.py` or a lightweight React dashboard

Dashboard features:

- select experiment
- compare model success rates
- view rollout videos
- inspect failure modes
- view future predictions from world model
- view point clouds/contact graphs
- view Gaussian-splatting novel views

Tests:

```bash
python dashboard/streamlit_app.py --smoke_test
```

Expected output:

```text
Dashboard loads experiment metadata and renders core panels.
```

### Step 18.3 — Final paper compilation target

Build Make targets:

```bash
make paper
make full-matrix
make demo-assets
```

Tests:

```bash
make paper
```

Expected output:

```text
reports/paper.pdf
```

## End-of-phase command

```bash
python scripts/validate_readme_commands.py
make paper
python dashboard/streamlit_app.py --smoke_test
pytest tests/test_report_generation.py
```

## End-of-phase expected output

- Research-style README
- Demo dashboard
- Final paper PDF
- Reproducible commands

---

# PHASE 19 — Final Reproducibility, Packaging, and Release

## Goal

Make the project clean enough to share publicly and discuss in interviews.

## Steps

### Step 19.1 — Reproducibility package

Build:

- Dockerfile
- pinned dependency lockfile
- seed registry
- dataset cards
- model cards
- experiment manifests
- result artifact index

Tests:

```bash
make reproducibility-check
```

Expected output:

```text
All required artifacts exist.
All configs resolve.
All reported experiments have manifests.
All paper figures map to source result files.
```

### Step 19.2 — Model and dataset cards

For each trained model, save:

```text
model name
training data
policy family
representation family
action convention
normalization stats
evaluation tasks
known limitations
checkpoint path
```

For each dataset split, save:

```text
source
tasks
episodes
splits
known biases
license notes
```

Tests:

```bash
pytest tests/test_model_cards.py tests/test_dataset_cards.py
```

Expected output:

```text
All model cards and dataset cards validate.
```

### Step 19.3 — Final release script

Build:

- `scripts/package_release.py`

Package:

```text
README.md
plan.md
reports/paper.pdf
reports/full_policy_matrix.pdf
selected rollout GIFs
selected reconstruction images
configs
model cards
small smoke dataset or synthetic demo data
```

Tests:

```bash
python scripts/package_release.py --out artifacts/release/ewpl_release.zip
```

Expected output:

```text
artifacts/release/ewpl_release.zip
```

## End-of-phase command

```bash
make reproducibility-check
python scripts/package_release.py --out artifacts/release/ewpl_release.zip
pytest
```

## End-of-phase expected output

- Clean final release package
- Reproducible paper and plots
- Public-facing research README
- Interview-ready robotics foundation-model project

---

# 5. Suggested Development Order

Even though this is not an MVP plan, implementation should still proceed in dependency order:

```text
0. Environment and repo skeleton
1. Canonical dataset schema
2. LIBERO integration
3. RoboCasa integration
4. Open X/LeRobot integration
5. Dataset splits and audit report
6. Rollout/evaluation harness
7. Behavior cloning
8. Action-chunking transformer
9. Diffusion policy
10. VLA post-training
11. World model
12. Physics-aware reconstruction
13. Gaussian splatting
14. Representation fusion
15. Full policy × representation experiment
16. Cross-source transfer
17. Distributed training
18. Dashboard and README
19. Reproducibility release
```

Do not begin the full comparison matrix until the individual model reports and reconstruction reports are working. The full matrix should compose earlier working modules, not introduce new infrastructure.

---

# 6. Core Commands to Maintain Throughout Development

Every phase should preserve these commands:

```bash
make test
make lint
python -m ewpl.healthcheck
python scripts/create_synthetic_dataset.py --out data/smoke --episodes 8 --steps 16
python scripts/train.py --config configs/train/bc_libero.yaml --overrides trainer.max_steps=20 data.dataset=data/smoke
python scripts/eval_rollouts.py --env libero --policy random --tasks 2 --episodes 5 --out artifacts/rollouts/smoke
python scripts/generate_report.py --type dataset_audit --config configs/report/dataset_audit.yaml --out reports/dataset_audit.pdf
```

If any future phase breaks one of these commands, the phase is not complete.

---

# 7. What the Final Project Should Prove

At completion, the project should prove that you can:

1. Build a real robot-learning software stack.
2. Ingest and normalize robotics datasets from simulation and real-world demonstrations.
3. Train behavior cloning, action-chunking, diffusion, and VLA policies.
4. Add world-model and reconstruction representations to embodied policies.
5. Evaluate robot policies through real rollouts, not just training loss.
6. Generate statistically meaningful research reports from code.
7. Work with 3D reconstruction, physics state, and Gaussian splatting in an autonomy context.
8. Run scalable PyTorch training workflows.
9. Present results like a serious research project.

The final README should make the project look like a small research lab artifact, not a class assignment.

