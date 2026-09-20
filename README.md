# FluxSwarm

Physics-informed multi-objective multi-agent RL for controlling a microrobot swarm in a 2D fluid channel.

## Highlights

- **Batched PhiFlow solver** — all environments share one process and a GPU batch dimension (no `SubprocVecEnv` CUDA contention).
- **Two-way fluid–swarm coupling** by default (`--coupling one-way` for faster runs).
- **MOMAPPO** — CTDE multi-objective PPO (progress, energy, smoothness) with corrected tanh-Gaussian log-probs, per-agent ratios, and actor-only PCGrad.
- **Position-blind local observations** with an observability ladder (`--obs-localization`).
- **Baselines**: brute-upstream, brute-wall, and standard single-objective PPO.
- **Outputs** under `runs_new/` (legacy `run/` is never touched).

## Install

```bash
conda activate fluxswarm   # or your env with phiflow + torch
pip install -e .
```

## Train

```bash
python -m fluxswarm.cli train \
  --batch-envs 64 \
  --coupling two-way \
  --total-timesteps 200000 \
  --obs-localization none \
  --progress-reward potential \
  --tag momappo_exp1
```

## Baselines

```bash
python -m fluxswarm.cli baseline --policy upstream --max-steps 5000 --coupling one-way
python -m fluxswarm.cli baseline --policy wall --wall-policy-mode static
python -m fluxswarm.cli baseline --policy ppo --total-timesteps 200000
```

## Figures

```bash
python -m fluxswarm.cli figures --run runs_new/<run_id>
python -m fluxswarm.cli figures --run runs_new/<with_pcgrad> --compare-pcgrad runs_new/<without>
```

Single-run figures land in `runs_new/<run_id>/figures/`. Comparisons go to `runs_new/_comparisons/<slug>/`.

## Key flags

| Flag | Meaning |
|------|---------|
| `--coupling {two-way,one-way}` | Obstacle coupling in pressure solve |
| `--obs-localization {none,imu,displacement,absolute-y,full}` | Localization ablation |
| `--progress-reward {potential,fluid-relative,legacy}` | Progress objective |
| `--no-pcgrad` | Disable gradient surgery |
| `--output-root` | Default `runs_new` |
| `--batch-envs` | Env batch size (default 64) |

See `python -m fluxswarm.cli train --help` for the full list.

## Layout

```
fluxswarm/
  config.py          CLI + dataclasses
  physics/           batched PhiFlow + swarm mechanics
  envs/              BatchedSwarmEnv, observations, rewards
  agents/            MOMAPPO, networks, PCGrad
  baselines/         brute + PPO
  runs/              recorder → runs_new/
  analysis/plots/    figure generators
  cli/               train / evaluate / baseline / figures
```

## Citation

See `CITATION.cff`. License: Apache-2.0.
