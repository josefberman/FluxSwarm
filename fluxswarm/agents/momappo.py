"""Multi-objective multi-agent PPO (CTDE) with corrected log-probs and PCGrad."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from fluxswarm.agents.networks import ActorCriticMO, tanh_log_prob, tanh_sample
from fluxswarm.agents.pcgrad import pcgrad_merge_actor
from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv
from fluxswarm.runs.recorder import RunRecorder


class RunningNorm:
    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        batch_mean = flat.mean(0)
        batch_var = flat.var(0, unbiased=False)
        batch_count = flat.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta.pow(2) * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


def blend_preset_x(raw: torch.Tensor, relax: float, preset_x: float = -1.0) -> torch.Tensor:
    """Blend toward -x thrust prior, then project to unit disk via tanh already applied."""
    r = float(np.clip(relax, 0.0, 1.0))
    out = raw.clone()
    out[..., 0] = (1 - r) * preset_x + r * raw[..., 0]
    norms = torch.linalg.norm(out, dim=-1, keepdim=True).clamp_min(1e-8)
    return out / torch.clamp(norms, min=1.0) * torch.clamp(norms, max=1.0)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with per-step done masking.

    rewards/values: (T, B, N), dones: (T, B), last_*: (B, N) / (B,)
    """
    T, B, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, N, device=rewards.device)
    next_values = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = last_value
        else:
            next_v = next_values[t]
        # dones[t] True => episode ended; do not bootstrap across the boundary
        mask = (1.0 - dones[t].float()).unsqueeze(-1)
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


class RolloutBuffer:
    def __init__(self, T: int, B: int, N: int, obs_dim: int, device: torch.device):
        self.T, self.B, self.N, self.obs_dim = T, B, N, obs_dim
        self.device = device
        self.obs = torch.zeros(T, B, N, obs_dim, device=device)
        self.actions = torch.zeros(T, B, N, 2, device=device)
        self.logprobs = torch.zeros(T, B, N, device=device)  # per-agent
        self.rewards = torch.zeros(T, B, N, 3, device=device)
        self.values = torch.zeros(T, B, N, 3, device=device)
        self.dones = torch.zeros(T, B, device=device)
        self.ptr = 0

    def add(self, obs, actions, logprobs, rewards, values, dones):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones.float()
        self.ptr += 1

    def get(self, adv, ret):
        T, B, N = self.T, self.B, self.N
        return {
            "obs": self.obs.reshape(T * B, N, self.obs_dim),
            "actions": self.actions.reshape(T * B, N, 2),
            "logprobs": self.logprobs.reshape(T * B, N),
            "adv": adv.reshape(T * B, N, 3),
            "ret": ret.reshape(T * B, N, 3),
            "values": self.values.reshape(T * B, N, 3),
        }


def train_momappo(
    cfg: Config,
    env: Optional[BatchedSwarmEnv] = None,
    recorder: Optional[RunRecorder] = None,
) -> Path:
    from concurrent.futures import ThreadPoolExecutor

    from fluxswarm.perf import resolve_device, split_batch_sizes

    device = torch.device(resolve_device(cfg.train.device))
    gpu_n, cpu_n = split_batch_sizes(cfg)
    if env is None:
        # Primary env on GPU (or CPU if no CUDA). CPU shard is optional and
        # stepped after the GPU batch when device_split=cpu-shard.
        env = BatchedSwarmEnv(cfg, batch=gpu_n, device=str(device))
    cpu_env = None
    cpu_pool = None
    if cpu_n > 0 and device.type == "cuda":
        cpu_env = BatchedSwarmEnv(cfg, batch=cpu_n, device="cpu")
        cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-shard")
    if recorder is None:
        recorder = RunRecorder(cfg, algorithm="momappo")
    recorder.save_config(cfg)

    B = env.batch
    N = env.n
    obs_dim = env.obs_dim
    model = ActorCriticMO(
        num_members=N,
        obs_local_dim=obs_dim,
        hidden_sizes=cfg.train.hidden_sizes,
        log_std_init=cfg.train.log_std_init,
        recurrent=(cfg.train.actor == "recurrent"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    obs_norm = RunningNorm(obs_dim, device)

    models_dir = recorder.run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    latest = models_dir / "model_latest.pt"
    if cfg.train.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    writer = SummaryWriter(log_dir=str(recorder.run_dir / "tb"))
    obs_np, _ = env.reset(seed=cfg.train.seed)
    cpu_obs_np = None
    if cpu_env is not None:
        cpu_obs_np, _ = cpu_env.reset(seed=cfg.train.seed + 1)
    obs = torch.tensor(obs_np, device=device)
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)

    n_steps = cfg.train.n_steps
    # Count true env-steps across all shards
    envs_per_step = B + (cpu_n if cpu_env is not None else 0)
    total = cfg.train.total_timesteps_per_env * envs_per_step
    num_updates = max(1, total // (n_steps * envs_per_step))
    global_steps = 0
    prior_warmup_rows = max(1, int(num_updates * n_steps * cfg.train.action_x_prior_warmup_fraction))

    pbar = trange(num_updates, desc="MOMAPPO")
    for update in pbar:
        buf = RolloutBuffer(n_steps, B, N, obs_dim, device)
        for t in range(n_steps):
            with torch.no_grad():
                dist, _ = model.actor.dist(obs)
                raw_action, logp_pre = tanh_sample(dist)
                logp = logp_pre.sum(dim=-1)
                relax = 1.0
                if cfg.train.use_action_x_prior:
                    row = update * n_steps + t
                    relax = min(1.0, (row + 1) / prior_warmup_rows)
                    action = blend_preset_x(raw_action, relax)
                    logp = tanh_log_prob(dist, action).sum(dim=-1)
                else:
                    action = raw_action

                joint = obs.reshape(B, N * obs_dim)
                values = model.critic(joint).permute(0, 2, 1)

            action_np = action.detach().cpu().numpy()
            cpu_fut = None
            if cpu_env is not None and cpu_obs_np is not None:
                with torch.no_grad():
                    c_obs = obs_norm.normalize(torch.tensor(cpu_obs_np, device=device))
                    c_dist, _ = model.actor.dist(c_obs)
                    c_act = torch.tanh(c_dist.mean).detach().cpu().numpy()
                cpu_fut = cpu_pool.submit(cpu_env.step, c_act)
            next_obs_np, scalar_r, terms, truncs, info = env.step(action_np)
            pbar.set_postfix(rollout=f"{t + 1}/{n_steps}", steps=global_steps + B, refresh=True)
            if cpu_fut is not None:
                cpu_obs_np, c_scalar, _, _, c_info = cpu_fut.result()
                recorder.log_step(global_steps + B, cpu_env, c_scalar, c_info)

            reward_matrix = torch.tensor(info["reward_matrix"], device=device, dtype=torch.float32)
            dones = torch.tensor(terms | truncs, device=device)

            buf.add(obs, action.detach(), logp.detach(), reward_matrix, values, dones)

            recorder.log_step(
                global_steps,
                env,
                scalar_r,
                info,
                save_fields=cfg.train.save_fields and (global_steps % 50 == 0),
            )
            if B > 0:
                writer.add_scalar("objectives/progress", float(reward_matrix[0, :, 0].mean()), global_steps)
                writer.add_scalar("objectives/energy", float(reward_matrix[0, :, 1].mean()), global_steps)
                writer.add_scalar("objectives/smoothness", float(reward_matrix[0, :, 2].mean()), global_steps)
                writer.add_scalar("training/action_x_prior_relax", relax, global_steps)

            next_obs = torch.tensor(next_obs_np, device=device)
            obs_norm.update(next_obs)
            obs = obs_norm.normalize(next_obs)
            global_steps += envs_per_step

        # Bootstrap
        with torch.no_grad():
            joint = obs.reshape(B, N * obs_dim)
            last_val = model.critic(joint).permute(0, 2, 1)
            last_done = torch.zeros(B, device=device)

        advs, rets = [], []
        for k in range(3):
            adv_k, ret_k = compute_gae(
                buf.rewards[..., k],
                buf.values[..., k],
                buf.dones,
                last_val[..., k],
                last_done,
                cfg.train.gamma,
                cfg.train.gae_lambda,
            )
            adv_k = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)
            advs.append(adv_k)
            rets.append(ret_k)
        adv = torch.stack(advs, dim=-1)
        ret = torch.stack(rets, dim=-1)

        data = buf.get(adv, ret)
        n_samples = data["obs"].shape[0]
        batch_size = min(cfg.train.batch_size, n_samples)
        idxs = np.arange(n_samples)

        for _epoch in range(cfg.train.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_samples, batch_size):
                mb = idxs[start : start + batch_size]
                b_obs = data["obs"][mb]
                b_act = data["actions"][mb]
                b_old_logp = data["logprobs"][mb]
                b_adv = data["adv"][mb]
                b_ret = data["ret"][mb]

                dist, _ = model.actor.dist(b_obs)
                new_logp = tanh_log_prob(dist, b_act).sum(dim=-1)
                ratio = torch.exp(new_logp - b_old_logp)

                loss_pis = []
                for k in range(3):
                    adv_k = b_adv[..., k]
                    surr1 = ratio * adv_k
                    surr2 = torch.clamp(ratio, 1 - cfg.train.clip_coef, 1 + cfg.train.clip_coef) * adv_k
                    loss_pis.append(-torch.min(surr1, surr2).mean())

                joint = b_obs.reshape(b_obs.shape[0], N * obs_dim)
                values = model.critic(joint).permute(0, 2, 1)
                loss_v = 0.5 * ((values - b_ret) ** 2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()

                optimizer.zero_grad(set_to_none=True)
                if cfg.train.use_pcgrad:
                    pcgrad_merge_actor(model.actor, loss_pis)
                else:
                    sum(loss_pis).backward(retain_graph=True)

                aux = cfg.train.vf_coef * loss_v - cfg.train.ent_coef * entropy
                aux.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "num_members": N,
            "obs_local_dim": obs_dim,
            "total_timesteps": global_steps,
            "use_pcgrad": cfg.train.use_pcgrad,
        }
        torch.save(ckpt, latest)
        torch.save(ckpt, models_dir / f"model_{update:05d}.pt")
        pbar.set_postfix(steps=global_steps, ent=float(entropy.detach()))

    writer.close()
    recorder.close()
    if cpu_pool is not None:
        cpu_pool.shutdown(wait=True)
    return recorder.run_dir
