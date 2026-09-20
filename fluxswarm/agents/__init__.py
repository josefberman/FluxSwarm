"""Agents package: MOMAPPO and networks."""

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.agents.networks import ActorCriticMO

__all__ = ["train_momappo", "ActorCriticMO"]
