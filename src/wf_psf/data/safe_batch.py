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

from collections.abc import Sequence
import numbers
import numpy as np
from typing import Any


def _compute_valid_mask(anchor: np.ndarray) -> np.ndarray:
    """Return a boolean mask indicating which samples contain only finite values.

    A sample is considered valid if all its values are finite
    (i.e., not NaN or ±Inf), as determined by ``np.isfinite``.

    Parameters
    ----------
    anchor : np.ndarray
        Input array of shape (N,) or (N, D), where N is the number of samples.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (N,). For 1D input, each element indicates whether
        the corresponding value is finite. For 2D input, each element indicates
        whether all values in the corresponding row are finite.
    """
    if anchor.ndim == 1:
        return np.isfinite(anchor)
    return np.isfinite(anchor).all(axis=1)


def safe_batch_builder(
    anchor: np.ndarray,
    **arrays: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter aligned arrays using a validity mask derived from an anchor array.

    This strict version enforces that all array-like inputs are aligned along
    their first dimension. Any mismatch raises an error.

    Parameters
    ----------
    anchor : np.ndarray
        Array used to compute validity (typically centroids), of shape (N,) or (N, D).
    **arrays : dict of str to Any
        Arrays associated with each sample. All NumPy arrays must have length N.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (N,) indicating valid samples.
    filtered : dict
        Dictionary containing filtered arrays.

    Raises
    ------
    ValueError
        If any array has a length different from N.
    TypeError
        If an unsupported type is passed.
    """
    mask = _compute_valid_mask(anchor)

    if mask.ndim != 1:
        raise ValueError("Validity mask must be 1D.")

    n = len(mask)

    if n == 0:
        raise ValueError("Empty anchor array.")

    if not mask.any():
        raise ValueError("All samples were filtered out.")

    filtered = {}

    for key, arr in arrays.items():
        if arr is None:
            filtered[key] = None
            continue

        # --- Array-like objects ---
        if hasattr(arr, "__array__"):
            arr = np.asarray(arr)

            if len(arr) != n:
                raise ValueError(...)

            if arr.dtype == object:
                raise TypeError(
                    f"Array '{key}' has dtype=object, which is not supported."
                )

            filtered[key] = arr[mask]

        # --- Sequences (lists, tuples, etc.) ---
        elif isinstance(arr, Sequence) and not isinstance(arr, (str, bytes)):
            if len(arr) == n:
                # Treat as aligned → filter
                filtered[key] = [arr[i] for i in range(n) if mask[i]]
            else:
                raise ValueError(
                    f"Sequence '{key}' has length {len(arr)} but expected {n}."
                )

        # --- Scalars ---
        elif isinstance(arr, numbers.Number):
            filtered[key] = arr

        else:
            raise TypeError(f"Unsupported type for '{key}': {type(arr)}.")

    return mask, filtered


def log_filtered_objects(mask, obj_ids, logger, context=""):
    """Log identifiers of samples removed by a validity mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of shape (N,) indicating valid samples.
    obj_ids : array-like
        Identifiers aligned with the samples (length N).
    logger : logging.Logger
        Logger instance used for reporting.
    context : str, optional
        Additional context string appended to log messages.
    """
    n_removed = (~mask).sum()

    if n_removed > 0:
        logger.warning(f"{n_removed} samples removed ({context})")
        logger.debug(f"Removed object_ids: {obj_ids[~mask].tolist()}")
