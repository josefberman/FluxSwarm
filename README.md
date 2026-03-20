# FluxSwarm

FluxSwarm is a physics-informed reinforcement learning project for controlling a swarm in a 2D fluid environment.  
It combines:

- fluid simulation (via PhiFlow),
- multi-agent swarm dynamics,
- and policy optimization (MOMAPPO / PPO-style training).

The project is geared toward experiments where swarm behavior is optimized under multiple objectives such as:

- center-of-mass location progress,
- cohesion,
- and action smoothness.

## Features

- **Custom Gymnasium environment** for swarm-in-fluid control (`SwarmEnv`).
- **Multi-objective PPO training loop** with separate value heads per objective.
- **Parallelized rollout collection** using vectorized environments.
- **Reward weighting support** during optimization with unweighted objective plotting.
- **TensorBoard logging** and run-folder artifact tracking.
- **Post-run plotting utilities** for rewards, objectives, kinematics, and fields.

## Repository Structure

- `main.py` - entry point; builds simulation + swarm + inflow and launches training.
- `RL.py` - environment implementation and RL training logic (`run_MOMAPPO`, `run_PPO`).
- `simulation.py` - fluid/swarm stepping and simulation utilities.
- `data_structures.py` - core data containers (`Simulation`, `Swarm`, `Inflow`, `Fluid`, etc.).
- `plotting.py` - run artifact visualization and export helpers.
- `logs.py` - run folder creation and parameter/hyperparameter logging.
- `requirements.txt` - Python dependencies.
- `fluxswarm/rl/env.py` - additional environment implementation variant.
- `test_*.py` - utility/validation scripts.

## Requirements

- Python 3.10+ recommended
- Linux (or compatible environment with required native dependencies)
- CUDA-capable GPU recommended for training speed (project defaults to GPU in places)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run training:

```bash
python main.py
```

At startup, the script asks whether to create a new run folder:

- `y` -> creates a new timestamped folder under `run/`
- `n` -> prompts for an existing folder name

### Example With Explicit Hyperparameters

```bash
python main.py \
  --num-envs 8 \
  --total-time 100.0 \
  --dt 0.05 \
  --n-steps 256 \
  --batch-size 32 \
  --update-epochs 10 \
  --gamma 0.95 \
  --clip-coef 0.2 \
  --ent-coef 0.01 \
  --lr 3e-4 \
  --swarm-num-x 4 \
  --swarm-num-y 4 \
  --swarm-max-force 3700 \
  --inflow-velocity 162
```

### CLI Options

`main.py` exposes flags for:

- simulation timing (`--total-time`, `--dt`, `--dt-substeps`)
- swarm layout and force limits (`--swarm-num-x`, `--swarm-num-y`, `--swarm-max-force`)
- inflow profile (`--inflow-velocity`)
- RL training (`--num-envs`, `--n-steps`, `--batch-size`, `--update-epochs`, `--gamma`, `--clip-coef`, `--ent-coef`, `--lr`)
- field saving (`--save-fields`, `--no-save-fields`)

Run `python main.py --help` for the full command reference.

## Training and Rewards

The environment computes three objective components:

- `location_progress`
- `cohesion`
- `smoothness`

The scalar reward used for environment reward accumulation is weighted by objective weights in `SwarmEnv` (for example `w_loc_prog`, `w_cohesion`, `w_smooth`).

Current workflow in this repo:

- **Optimization** (PPO updates) uses weighted reward components.
- **Objective plotting** can use unweighted objective values for easier interpretation.

## Outputs and Artifacts

Training and logs are written under `run/<folder_name>/...`.

Common artifacts include:

- `configuration.txt` - simulation/swarm/flow settings snapshot
- `hyperparameters_*.txt` - training hyperparameters
- TensorBoard logs in `MOMAPPO_tb`
- model checkpoints in `MOMAPPO/models`
- plots:
  - `rewards.jpg`
  - `rewards_objectives.jpg`
  - `locations.jpg`
  - `velocities.jpg`
  - plus CSV exports for corresponding series

## TensorBoard

The training loop may launch TensorBoard automatically.  
If needed, run manually:

```bash
tensorboard --logdir run/<folder_name>/MOMAPPO_tb --port 6006 --host 127.0.0.1
```

Then open [http://127.0.0.1:6006](http://127.0.0.1:6006).

## Notes and Caveats

- The code currently includes interactive prompts in `main.py`; for batch automation you may want to replace prompts with explicit flags.
- Some defaults are tuned for experimentation rather than production packaging.
- Large simulations can be memory/compute heavy; start with smaller `--total-time` and fewer environments when validating setup.

## Development

Recommended basic workflow:

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Run a short training job to verify end-to-end execution.
4. Inspect TensorBoard and generated plots.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

