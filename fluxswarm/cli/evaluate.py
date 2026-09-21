"""Evaluate a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fluxswarm.agents.networks import ActorCriticMO
from fluxswarm.config import Config, build_argparser, parse_config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv
from fluxswarm.perf import configure_threading


def main(argv: list[str] | None = None) -> None:
    configure_threading()
    parser = build_argparser("Evaluate a FluxSwarm checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = Config.load(Path(args.checkpoint).parent.parent / "config.yaml") if (
        Path(args.checkpoint).parent.parent / "config.yaml"
    ).exists() else parse_config([])
    # Apply overrides from remaining known flags via a second parse is awkward; use defaults + checkpoint
    cfg.train.batch_envs = 1

    env = BatchedSwarmEnv(cfg, batch=1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCriticMO(ckpt["num_members"], ckpt["obs_local_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    obs, _ = env.reset()
    successes = 0
    for ep in range(args.episodes):
        done = False
        while not done:
            with torch.no_grad():
                o = torch.tensor(obs)
                dist, _ = model.actor.dist(o)
                action = torch.tanh(dist.mean)
            obs, _, terms, truncs, info = env.step(action.numpy())
            done = bool(terms[0] or truncs[0])
            if done and info["infos"][0].get("termination_reason") == "success":
                successes += 1
                obs, _ = env.reset()
    print(f"Successes: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
