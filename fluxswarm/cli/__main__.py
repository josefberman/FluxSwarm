"""python -m fluxswarm.cli / fluxswarm entrypoint."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="fluxswarm", description="FluxSwarm research CLI")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "baseline", "figures"],
        help="Subcommand",
    )
    if not argv:
        parser.print_help()
        return
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "train":
        from fluxswarm.cli.train import main as train_main

        train_main(rest)
    elif cmd == "evaluate":
        from fluxswarm.cli.evaluate import main as eval_main

        eval_main(rest)
    elif cmd == "baseline":
        from fluxswarm.cli.baseline import main as base_main

        base_main(rest)
    elif cmd == "figures":
        from fluxswarm.cli.figures import main as fig_main

        fig_main(rest)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
