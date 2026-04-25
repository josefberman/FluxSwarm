"""
Training CLI and shared simulation geometry defaults for main.py.

Import this module without pulling in PhiFlow or GPU checks, so tools like
animate_locations.py can mirror main's defaults.
"""
from __future__ import annotations

import argparse

# --- Geometry / domain (used by main.main() and animate_locations defaults) ---
SIM_LENGTH_X = 100.0
SIM_LENGTH_Y = 2.0
MEMBER_RADIUS = 0.25


def build_training_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FluxSwarm reinforcement learning training loop.\n\n"
            "This script sets up the fluid simulation, swarm layout, and inflow profile, "
            "then trains a MOMAPPO agent using multiple parallel environments. "
            "Use the flags below to override key simulation, swarm, and inflow parameters."
        )
    )
    parser.add_argument(
        '--save-fields',
        dest='save_fields',
        action='store_true',
        default=False,
        help='Save velocity and pressure fields to npz files (default: False)',
    )
    parser.add_argument(
        '--no-save-fields',
        dest='save_fields',
        action='store_false',
        help='Disable saving fields to npz files',
    )

    parser.add_argument(
        '--total-time',
        type=float,
        default=100.0,
        help='Total simulation time in seconds for each episode (default: 100.0)',
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.001,
        help='Base simulation timestep dt in seconds (default: 0.001)',
    )
    parser.add_argument(
        '--dt-substeps',
        type=int,
        default=28,
        help='Number of substeps per dt used in the simulator (default: 28)',
    )

    parser.add_argument(
        '--sim-length-x',
        type=float,
        default=SIM_LENGTH_X,
        help=f'Simulation domain extent in x (mm) (default: {SIM_LENGTH_X})',
    )
    parser.add_argument(
        '--sim-length-y',
        type=float,
        default=SIM_LENGTH_Y,
        help=f'Simulation domain extent in y (mm) (default: {SIM_LENGTH_Y})',
    )
    parser.add_argument(
        '--member-radius',
        type=float,
        default=MEMBER_RADIUS,
        help=f'Radius of each swarm member (mm) (default: {MEMBER_RADIUS})',
    )

    parser.add_argument(
        '--swarm-num-x',
        type=int,
        default=8,
        help='Number of swarm members along the x-direction (default: 5)',
    )
    parser.add_argument(
        '--swarm-num-y',
        type=int,
        default=2,
        help='Number of swarm members along the y-direction (default: 3)',
    )
    parser.add_argument(
        '--swarm-max-force',
        type=float,
        # default=1491,  # The force induced by a 0.020T/m magnetic field on a magnetite sphere.
        # help='Maximum propulsion force per swarm member in mg*mm/s^2 (default: 1491)',
        default=1491/2,  # The force induced by a 0.010T/m magnetic field on a magnetite sphere.
        help='Maximum propulsion force per swarm member in mg*mm/s^2 (default: 1491/2)',
    )

    parser.add_argument(
        '--inflow-velocity',
        type=float,
        default=100,
        help='Peak inflow centerline velocity in mm/s (default: 100)',
    )

    parser.add_argument(
        '--num-envs',
        type=int,
        default=16,
        help='Number of parallel environments (default: 16)',
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=128,
        help='Number of steps per environment (default: 128)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)',
    )
    parser.add_argument(
        '--update-epochs',
        type=int,
        default=4,
        help='Number of update epochs (default: 4)',
    )
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.01,
        help='Entropy coefficient (default: 0.01)',
    )
    parser.add_argument(
        '--clip-coef',
        type=float,
        default=0.2,
        help='Clip coefficient (default: 0.2)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='Discount factor (default: 0.95)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)',
    )
    return parser


def training_cli_defaults() -> argparse.Namespace:
    """All training CLI defaults as a namespace (equivalent to ``parse_args([])``)."""
    return build_training_argparser().parse_args([])
