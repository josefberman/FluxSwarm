"""Train MOMAPPO."""
from __future__ import annotations

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.config import build_argparser, parse_config
from fluxswarm.perf import configure_threading


def main(argv: list[str] | None = None) -> None:
    configure_threading()
    # Re-parse with full argparser (parse_config already builds one)
    import sys

    args = argv if argv is not None else sys.argv[1:]
    cfg = parse_config(args)
    run_dir = train_momappo(cfg)
    print(f"Training complete. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
