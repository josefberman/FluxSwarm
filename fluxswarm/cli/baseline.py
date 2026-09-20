"""Run baselines."""
from __future__ import annotations

import argparse

from fluxswarm.baselines.brute import run_brute
from fluxswarm.baselines.ppo import train_ppo_baseline
from fluxswarm.config import build_argparser, parse_config
from fluxswarm.perf import configure_threading


def main(argv: list[str] | None = None) -> None:
    configure_threading()
    parser = build_argparser("Run a FluxSwarm baseline")
    parser.add_argument(
        "--policy",
        choices=["upstream", "wall", "ppo"],
        default="upstream",
        help="Baseline policy",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    # parse_config needs full argv without --policy
    filtered = []
    skip = False
    for a in (argv or []):
        if skip:
            skip = False
            continue
        if a in ("--policy", "--max-steps"):
            skip = True
            continue
        if a.startswith("--policy=") or a.startswith("--max-steps="):
            continue
        filtered.append(a)
    cfg = parse_config(filtered)

    if args.policy == "ppo":
        run_dir = train_ppo_baseline(cfg)
    else:
        run_dir = run_brute(cfg, policy=args.policy, max_steps=args.max_steps)
    print(f"Baseline complete. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
