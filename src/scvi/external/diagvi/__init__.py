"""DIAGVI model for multi-modal integration with guidance graphs."""

from ._model import DIAGVI
from ._module import DIAGVAE
from ._task import DiagTrainingPlan
from ._utils import (
    add_gene_coords_from_gtf,
    construct_peak_gene_mapping,
    propagate_highly_variable,
)

__all__ = [
    "DIAGVI",
    "DIAGVAE",
    "DiagTrainingPlan",
    "add_gene_coords_from_gtf",
    "construct_peak_gene_mapping",
    "propagate_highly_variable",
]
