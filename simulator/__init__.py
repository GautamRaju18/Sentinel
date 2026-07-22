"""A simulated production environment for Sentinel to investigate.

Deterministic and seeded: the same scenario always produces the same logs,
metrics and deploys. That is what makes the eval suite in evals/ meaningful —
otherwise "did the agent find the root cause" is unanswerable.
"""

from simulator.world import SimulatedWorld, get_world

__all__ = ["SimulatedWorld", "get_world"]
