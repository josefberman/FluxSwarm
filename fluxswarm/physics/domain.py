"""Domain geometry, grid helpers, and units (mm, s, mg)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fluxswarm.config import SimConfig


@dataclass
class Domain:
    length_x: float
    length_y: float
    nx: int
    ny: int
    dx: float
    dy: float

    @classmethod
    def from_config(cls, sim: SimConfig) -> "Domain":
        nx, ny = sim.resolution
        return cls(
            length_x=sim.length_x,
            length_y=sim.length_y,
            nx=nx,
            ny=ny,
            dx=sim.dx,
            dy=sim.dy,
        )

    def clamp_positions(
        self,
        pos: torch.Tensor,
        radii: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clamp centers into [r, L-r]. Returns (clamped_pos, hit_mask)."""
        lo = radii.unsqueeze(-1)
        hi_x = self.length_x - radii
        hi_y = self.length_y - radii
        hi = torch.stack([hi_x, hi_y], dim=-1)
        clamped = torch.max(torch.min(pos, hi), lo)
        hit = (clamped - pos).abs().sum(dim=-1) > 1e-12
        return clamped, hit
