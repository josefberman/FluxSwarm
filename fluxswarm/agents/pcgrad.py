"""PCGrad: project conflicting gradients (actor parameters only)."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def pcgrad_merge_actor(
    actor: nn.Module,
    losses: Sequence[torch.Tensor],
) -> None:
    """Backward each loss, project conflicting grads on actor params, write mean.

    Critic parameters are untouched. Callers should then backward critic/entropy
    losses separately.
    """
    params = [p for p in actor.parameters() if p.requires_grad]
    if not params:
        return

    grads = []
    for loss in losses:
        actor.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g = torch.cat([
            (p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.numel(), device=p.device))
            for p in params
        ])
        grads.append(g)

    # PCGrad projection
    projected = []
    for i, gi in enumerate(grads):
        g = gi.clone()
        for j, gj in enumerate(grads):
            if i == j:
                continue
            dot = torch.dot(g, gj)
            if dot < 0:
                g = g - dot / (gj.norm() ** 2 + 1e-8) * gj
        projected.append(g)
    merged = torch.stack(projected).mean(dim=0)

    # Write back
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = merged[offset : offset + n].view_as(p).clone()
        offset += n
