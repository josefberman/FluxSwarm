"""Configuration dataclasses and CLI for FluxSwarm."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

SIM_LENGTH_X = 100.0
SIM_LENGTH_Y = 2.0
MEMBER_RADIUS = 0.25


@dataclass
class SimConfig:
    length_x: float = SIM_LENGTH_X
    length_y: float = SIM_LENGTH_Y
    resolution_scale: float = 10.0  # cells per mm
    dt: float = 0.005
    total_time: float = 100.0
    dt_substeps: int = 20
    viscosity: float = 3.0
    inflow_velocity: float = 400.0
    inflow_period: float = 1.0
    coupling: Literal["two-way", "one-way"] = "two-way"
    fluid_density: float = 1.06

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            max(2, int(self.length_x * self.resolution_scale)),
            max(2, int(self.length_y * self.resolution_scale)),
        )

    @property
    def dx(self) -> float:
        return self.length_x / self.resolution[0]

    @property
    def dy(self) -> float:
        return self.length_y / self.resolution[1]

    @property
    def time_steps(self) -> int:
        return int(self.total_time / self.dt)

    @property
    def substeps(self) -> int:
        return max(1, self.dt_substeps)


@dataclass
class SwarmConfig:
    num_x: int = 8
    num_y: int = 2
    member_radius: float = MEMBER_RADIUS
    member_density: float = 15.12
    member_max_force: float = 1491.0 / 2.0
    use_centered_layout: bool = True
    left_location: Optional[float] = None
    bottom_location: Optional[float] = None
    member_interval_x: Optional[float] = None
    member_interval_y: Optional[float] = None

    @property
    def num_members(self) -> int:
        return self.num_x * self.num_y


@dataclass
class TaskConfig:
    episode_duration: float = 10.0
    success_x: float = 20.0
    failure_x: float = 80.0
    progress_reward: Literal["potential", "fluid-relative", "legacy"] = "potential"
    w_progress: float = 9.0
    w_energy: float = 1.0
    w_smooth: float = 1.0


@dataclass
class ObsConfig:
    preset: Literal["rich", "legacy"] = "rich"
    localization: Literal["none", "imu", "displacement", "absolute-y", "full"] = "none"
    ring_points: int = 8
    ring_radius_factor: float = 3.0
    history: int = 4
    neighbor_k: int = 3
    neighbor_radius: float = 2.0
    velocity_frame: Literal["fluid", "lab"] = "fluid"
    imu_bias: float = 1.0
    imu_noise: float = 0.1


@dataclass
class TrainConfig:
    batch_envs: int = 64
    n_steps: int = 16
    batch_size: int = 4
    update_epochs: int = 4
    ent_coef: float = 0.01
    clip_coef: float = 0.2
    gamma: float = 0.95
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    lr: float = 1e-3
    total_timesteps: int = 200_000
    use_pcgrad: bool = True
    use_action_x_prior: bool = True
    action_x_prior_warmup_fraction: float = 0.2
    actor: Literal["mlp", "recurrent"] = "mlp"
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_init: float = -1.0
    max_grad_norm: float = 0.5
    device: str = "cuda"
    device_split: Literal["gpu", "cpu-shard"] = "gpu"
    cpu_shard_fraction: float = 0.25
    seed: int = 0
    save_fields: bool = False
    tensorboard_port: int = 6006
    resume: bool = True


@dataclass
class RunConfig:
    output_root: str = "runs_new"
    tag: str = ""
    wall_policy_mode: Literal["static", "dynamic"] = "static"


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        def _section(section_cls, key: str):
            raw = d.get(key, {}) or {}
            valid = {f.name for f in fields(section_cls)}
            return section_cls(**{k: v for k, v in raw.items() if k in valid})

        return cls(
            sim=_section(SimConfig, "sim"),
            swarm=_section(SwarmConfig, "swarm"),
            task=_section(TaskConfig, "task"),
            obs=_section(ObsConfig, "obs"),
            train=_section(TrainConfig, "train"),
            run=_section(RunConfig, "run"),
        )

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        with Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    mapping = {
        "sim_length_x": ("sim", "length_x"),
        "sim_length_y": ("sim", "length_y"),
        "resolution_scale": ("sim", "resolution_scale"),
        "dt": ("sim", "dt"),
        "total_time": ("sim", "total_time"),
        "dt_substeps": ("sim", "dt_substeps"),
        "inflow_velocity": ("sim", "inflow_velocity"),
        "coupling": ("sim", "coupling"),
        "swarm_num_x": ("swarm", "num_x"),
        "swarm_num_y": ("swarm", "num_y"),
        "member_radius": ("swarm", "member_radius"),
        "swarm_max_force": ("swarm", "member_max_force"),
        "episode_duration": ("task", "episode_duration"),
        "success_x": ("task", "success_x"),
        "failure_x": ("task", "failure_x"),
        "progress_reward": ("task", "progress_reward"),
        "w_progress": ("task", "w_progress"),
        "w_energy": ("task", "w_energy"),
        "w_smooth": ("task", "w_smooth"),
        "obs_preset": ("obs", "preset"),
        "obs_localization": ("obs", "localization"),
        "obs_ring_points": ("obs", "ring_points"),
        "obs_history": ("obs", "history"),
        "neighbor_radius": ("obs", "neighbor_radius"),
        "neighbor_k": ("obs", "neighbor_k"),
        "velocity_frame": ("obs", "velocity_frame"),
        "imu_bias": ("obs", "imu_bias"),
        "imu_noise": ("obs", "imu_noise"),
        "batch_envs": ("train", "batch_envs"),
        "n_steps": ("train", "n_steps"),
        "batch_size": ("train", "batch_size"),
        "update_epochs": ("train", "update_epochs"),
        "ent_coef": ("train", "ent_coef"),
        "clip_coef": ("train", "clip_coef"),
        "gamma": ("train", "gamma"),
        "lr": ("train", "lr"),
        "total_timesteps": ("train", "total_timesteps"),
        "device_split": ("train", "device_split"),
        "seed": ("train", "seed"),
        "save_fields": ("train", "save_fields"),
        "tensorboard_port": ("train", "tensorboard_port"),
        "actor": ("train", "actor"),
        "output_root": ("run", "output_root"),
        "tag": ("run", "tag"),
        "wall_policy_mode": ("run", "wall_policy_mode"),
    }

    for flag, (section, attr) in mapping.items():
        if not hasattr(args, flag):
            continue
        val = getattr(args, flag)
        if val is None:
            continue
        setattr(getattr(cfg, section), attr, val)

    if getattr(args, "no_pcgrad", False):
        cfg.train.use_pcgrad = False
    if getattr(args, "use_action_x_prior", None) is not None:
        cfg.train.use_action_x_prior = args.use_action_x_prior
    if getattr(args, "action_x_prior_warmup_fraction", None) is not None:
        cfg.train.action_x_prior_warmup_fraction = args.action_x_prior_warmup_fraction
    if getattr(args, "no_save_fields", False):
        cfg.train.save_fields = False

    return cfg


def build_argparser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
        or "FluxSwarm: multi-objective multi-agent PPO for swarms in fluid channels.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file (CLI overrides it).")

    g = parser.add_argument_group("simulation")
    g.add_argument("--sim-length-x", type=float, default=None)
    g.add_argument("--sim-length-y", type=float, default=None)
    g.add_argument("--resolution-scale", type=float, default=None)
    g.add_argument("--dt", type=float, default=None)
    g.add_argument("--total-time", type=float, default=None)
    g.add_argument("--dt-substeps", type=int, default=None)
    g.add_argument("--inflow-velocity", type=float, default=None)
    g.add_argument("--coupling", choices=["two-way", "one-way"], default=None)

    g = parser.add_argument_group("swarm")
    g.add_argument("--swarm-num-x", type=int, default=None)
    g.add_argument("--swarm-num-y", type=int, default=None)
    g.add_argument("--member-radius", type=float, default=None)
    g.add_argument("--swarm-max-force", type=float, default=None)

    g = parser.add_argument_group("task")
    g.add_argument("--episode-duration", type=float, default=None)
    g.add_argument("--success-x", type=float, default=None)
    g.add_argument("--failure-x", type=float, default=None)
    g.add_argument("--progress-reward", choices=["potential", "fluid-relative", "legacy"], default=None)
    g.add_argument("--w-progress", type=float, default=None)
    g.add_argument("--w-energy", type=float, default=None)
    g.add_argument("--w-smooth", type=float, default=None)

    g = parser.add_argument_group("observations")
    g.add_argument("--obs-preset", choices=["rich", "legacy"], default=None)
    g.add_argument(
        "--obs-localization",
        choices=["none", "imu", "displacement", "absolute-y", "full"],
        default=None,
    )
    g.add_argument("--obs-ring-points", type=int, default=None)
    g.add_argument("--obs-history", type=int, default=None)
    g.add_argument("--neighbor-radius", type=float, default=None)
    g.add_argument("--neighbor-k", type=int, default=None)
    g.add_argument("--velocity-frame", choices=["fluid", "lab"], default=None)
    g.add_argument("--imu-bias", type=float, default=None)
    g.add_argument("--imu-noise", type=float, default=None)

    g = parser.add_argument_group("training")
    g.add_argument("--batch-envs", "--num-envs", dest="batch_envs", type=int, default=None)
    g.add_argument("--n-steps", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--update-epochs", type=int, default=None)
    g.add_argument("--ent-coef", type=float, default=None)
    g.add_argument("--clip-coef", type=float, default=None)
    g.add_argument("--gamma", type=float, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--total-timesteps", type=int, default=None)
    g.add_argument("--no-pcgrad", action="store_true", default=False)
    g.add_argument("--no-action-x-prior", dest="use_action_x_prior", action="store_false", default=None)
    g.add_argument("--action-x-prior-warmup-fraction", type=float, default=None)
    g.add_argument("--actor", choices=["mlp", "recurrent"], default=None)
    g.add_argument("--device-split", choices=["gpu", "cpu-shard"], default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--save-fields", dest="save_fields", action="store_true", default=None)
    g.add_argument("--no-save-fields", action="store_true", default=False)
    g.add_argument("--tensorboard-port", type=int, default=None)

    g = parser.add_argument_group("run")
    g.add_argument("--output-root", type=str, default=None)
    g.add_argument("--tag", type=str, default=None)
    g.add_argument("--wall-policy-mode", choices=["static", "dynamic"], default=None)

    return parser


def parse_config(argv: list[str] | None = None) -> Config:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = Config.load(args.config) if args.config else Config()
    return _apply_cli_overrides(cfg, args)
