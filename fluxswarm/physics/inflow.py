"""Inflow temporal waveforms and Poiseuille spatial profiles."""
from __future__ import annotations

import math as pymath
from typing import Protocol

import torch


class Waveform(Protocol):
    def __call__(self, t: float | torch.Tensor, amplitude: float) -> float | torch.Tensor: ...


def beat_waveform(t: float | torch.Tensor, amplitude: float, period: float = 1.0) -> float | torch.Tensor:
    """Cardiac-like triple-Gaussian pulsatile beat (period in seconds).

    Matches the legacy ``auxiliary.beat_waveform`` when ``period=1``.
    """
    # Shift so the main lobe sits early in the cycle (legacy used +0.5).
    if isinstance(t, torch.Tensor):
        inner = torch.remainder(t + 0.5 * period, period) / period
        a = amplitude
        return (
            a * torch.exp(-((inner - 0.14) ** 2) / (2 * 0.04 ** 2))
            - 0.1 * a * torch.exp(-((inner - 0.32) ** 2) / (2 * 0.035 ** 2))
            + 0.05 * a * torch.exp(-((inner - 0.45) ** 2) / (2 * 0.05 ** 2))
        )
    inner = ((t + 0.5 * period) % period) / period
    a = amplitude
    return (
        a * pymath.exp(-((inner - 0.14) ** 2) / (2 * 0.04 ** 2))
        - 0.1 * a * pymath.exp(-((inner - 0.32) ** 2) / (2 * 0.035 ** 2))
        + 0.05 * a * pymath.exp(-((inner - 0.45) ** 2) / (2 * 0.05 ** 2))
    )


def poiseuille_profile(y: torch.Tensor, length_y: float) -> torch.Tensor:
    """Unit-peak parabolic profile across the channel height."""
    R = length_y / 2.0
    return 1.0 - ((y - R) / R) ** 2


def poiseuille_on_grid(ny: int, length_y: float, device: torch.device | None = None) -> torch.Tensor:
    """Parabolic profile sampled at staggered-u cell centers (ny samples)."""
    y = (torch.arange(ny, device=device, dtype=torch.float64) + 0.5) * (length_y / ny)
    return poiseuille_profile(y, length_y)
