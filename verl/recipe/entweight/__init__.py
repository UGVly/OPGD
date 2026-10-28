"""Entropy-weighted GRPO recipe.

Plain GRPO whose per-token advantage is reweighted by the min-max normalized
negative token entropy of the current policy. The weighting math lives in
``verl.trainer.ppo.core_algos.compute_entropy_token_weights`` because the actor
worker applies it; this recipe owns the config, the launcher, and the contracts.
See README.md.
"""
