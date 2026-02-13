"""Reinforcement Learning module for retrosynthesis models.

Implements PPO fine-tuning with chemical rewards:
  - Validity reward (RDKit parseable)
  - Round-trip reward (forward model produces original product)
  - Tanimoto reward (fingerprint similarity)
  - Exact match bonus (matches ground truth)
"""

__all__ = ["ChemicalRewardFunction"]


def __getattr__(name):
    if name == "ChemicalRewardFunction":
        from rasyn.rl.rewards import ChemicalRewardFunction
        return ChemicalRewardFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
