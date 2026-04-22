"""Safe batch processing utilities.

This module provides utilities for filtering batches of aligned arrays
based on sample-wise validity criteria, typically derived from an
anchor array (e.g. centroid coordinates).

The core functionality ensures that all dataset components (images,
masks, metadata, etc.) remain aligned when invalid samples (NaNs, Infs)
are removed. This is critical for preventing silent misalignment bugs
in downstream processing.

It also provides lightweight logging helpers to track which samples
were filtered, improving traceability and debugging.

These utilities are intended for use during data preparation stages,
particularly after feature extraction steps such as centroid estimation.

Author(s): Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import numpy as np
from typing import Any, Dict, Tuple


def _compute_valid_mask(anchor: np.ndarray) -> np.ndarray:
    """
    Compute validity mask from anchor array (e.g. centroids).
    """
    if anchor.ndim == 1:
        return np.isfinite(anchor)
    return np.isfinite(anchor).all(axis=1)


def safe_batch_builder(
    anchor: np.ndarray,
    **arrays: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Filter a batch of aligned arrays using a validity mask derived from anchor.

    Parameters
    ----------
    anchor:
        Array used to compute validity (typically centroids).
    **arrays:
        All arrays that must remain aligned (images, masks, icovs, ids, etc.)

    Returns
    -------
    mask:
        Boolean mask of valid samples.
    filtered:
        Dictionary of filtered arrays.
    """

    mask = _compute_valid_mask(anchor)

    if mask is None or len(mask) == 0:
        return

    n = len(mask)

    filtered = {}

    for key, arr in arrays.items():
        if arr is None:
            filtered[key] = None
            continue

        if isinstance(arr, np.ndarray) and len(arr) == n:
            filtered[key] = arr[mask]
        else:
            # broadcast-like metadata (lists, scalars, etc.)
            filtered[key] = arr

    return mask, filtered

def log_filtered_objects(mask, obj_ids, logger, context=""):
    """
    Log identifiers of samples removed by a validity mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask indicating valid samples.
    obj_ids : array-like
        Identifiers corresponding to each sample.
    logger : logging.Logger
        Logger instance used for reporting.
    context : str, optional
        Additional context (e.g. CCD quadrant, dither).
    """
    n_removed = (~mask).sum()

    if n_removed > 0:
        logger.warning(
            f"{n_removed} samples removed {context}"
        )
        logger.debug(
            f"Removed object_ids: {obj_ids[~mask].tolist()}"
        )