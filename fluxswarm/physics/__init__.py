"""Physics package: domain, inflow, swarm, solver."""

from fluxswarm.physics.domain import Domain
from fluxswarm.physics.solver import BatchedFluidSolver

__all__ = ["Domain", "BatchedFluidSolver"]
