"""
R-PCS naming module.

This module keeps backward compatibility with existing PCCS naming.
"""

from Data.pccs import (
    compute_pccs_source_scores,
    compute_rpcs_source_scores,
)

__all__ = [
    "compute_rpcs_source_scores",
    "compute_pccs_source_scores",
]

