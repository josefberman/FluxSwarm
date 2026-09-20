"""Figure CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from fluxswarm.analysis.plots import generate_all_for_run, plot_pcgrad_comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate FluxSwarm figures")
    parser.add_argument("--run", type=str, required=True, help="Path to a run directory under runs_new/")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    parser.add_argument("--compare-pcgrad", type=str, default=None, help="Second run (without PCGrad)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if args.compare_pcgrad:
        out = plot_pcgrad_comparison(run_dir, Path(args.compare_pcgrad), Path(args.out) if args.out else None)
        print(f"Wrote {out}")
    else:
        paths = generate_all_for_run(run_dir)
        if args.out:
            # copy/rewrite into override — regenerate with out as run figures dir surrogate
            out_root = Path(args.out)
            out_root.mkdir(parents=True, exist_ok=True)
            for p in paths:
                target = out_root / p.name
                target.write_bytes(p.read_bytes())
                side = p.with_suffix(".json")
                if side.exists():
                    (out_root / side.name).write_bytes(side.read_bytes())
        for p in paths:
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
