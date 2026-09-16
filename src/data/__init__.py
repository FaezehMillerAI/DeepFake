"""Datasets, sharding and the pseudo-mask pipeline."""
from .pseudo_masks import generate_pseudo_mask, validate_pseudo_masks, PseudoMaskResult
from .datasets import PairedEditDataset, MaskSupervisedDataset, split_dataset_indices
from .acquire import inspect_dfbench_metadata, update_manifest, find_dfbench_pairs

__all__ = [
    "generate_pseudo_mask",
    "validate_pseudo_masks",
    "PseudoMaskResult",
    "PairedEditDataset",
    "MaskSupervisedDataset",
    "split_dataset_indices",
    "inspect_dfbench_metadata",
    "update_manifest",
    "find_dfbench_pairs",
]


