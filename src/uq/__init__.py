"""Uncertainty quantification: perturbations + epistemic/aleatoric decomposition."""

from src.uq.perturb import (
    PerturbSpec,
    apply_perturbation,
    cat_syn,
    context_shuffle,
    evidence_mask,
    miss_inject,
    num_jitter,
)

__all__ = [
    "PerturbSpec",
    "apply_perturbation",
    "cat_syn",
    "context_shuffle",
    "evidence_mask",
    "miss_inject",
    "num_jitter",
]
