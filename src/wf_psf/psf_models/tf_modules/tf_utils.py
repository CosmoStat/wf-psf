"""TensorFlow Utilities Module.

Provides lightweight utility functions for safely converting and managing data types 
within TensorFlow-based workflows.

Includes:
- `ensure_tensor`: ensures inputs are TensorFlow tensors with specified dtype

These tools are designed to support PSF model components, including lazy property evaluation, 
data input validation, and type normalization.

This module is intended for internal use in model layers and inference components to enforce 
TensorFlow-compatible inputs.

Authors: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import tensorflow as tf
import numpy as np


@tf.function
def find_position_indices(obs_pos, batch_positions):
    """Find indices of batch positions within observed positions using vectorized operations.
    
    This function locates the indices of multiple query positions within a 
    reference set of observed positions using broadcasting and vectorized operations.
    Each position in the batch must have an exact match in the observed positions.

    Parameters
    ----------
    obs_pos : tf.Tensor
        Reference positions tensor of shape (n_obs, 2), where n_obs is the number of 
        observed positions. Each row contains [x, y] coordinates.
    batch_positions : tf.Tensor
        Query positions tensor of shape (batch_size, 2), where batch_size is the number 
        of positions to look up. Each row contains [x, y] coordinates.
    
    Returns
    -------
    indices : tf.Tensor
        Tensor of shape (batch_size,) containing the indices of each batch position 
        within obs_pos. The dtype is tf.int64.
    
    Raises
    ------
    tf.errors.InvalidArgumentError
        If any position in batch_positions is not found in obs_pos.
    
    Notes
    -----
    Uses exact equality matching - positions must match exactly. More efficient than 
    iterative lookups for multiple positions due to vectorized operations.
    """
    # Shape: obs_pos (n_obs, 2), batch_positions (batch_size, 2)
    # Expand for broadcasting: (1, n_obs, 2) and (batch_size, 1, 2)
    obs_expanded = tf.expand_dims(obs_pos, 0)
    pos_expanded = tf.expand_dims(batch_positions, 1)
    
    # Compare all positions at once: (batch_size, n_obs)
    matches = tf.reduce_all(tf.equal(obs_expanded, pos_expanded), axis=2)
    
    # Find the index of the matching position for each batch item
    # argmax returns the first True value's index along axis=1 
    indices = tf.argmax(tf.cast(matches, tf.int32), axis=1)
    
    # Verify all positions were found
    tf.debugging.assert_equal(
        tf.reduce_all(tf.reduce_any(matches, axis=1)),
        True,
        message="Some positions not found in obs_pos"
    )
    
    return indices

def ensure_tensor(input_array, dtype=tf.float32):
    """
    Ensure the input is a TensorFlow tensor of the specified dtype.
    
    Parameters
    ----------
    input_array : array-like, tf.Tensor, or np.ndarray
        The input to convert.
    dtype : tf.DType, optional
        The desired TensorFlow dtype (default: tf.float32).
    
    Returns
    -------
    tf.Tensor
        A TensorFlow tensor with the specified dtype.
    """
    if tf.is_tensor(input_array):
        # If already a tensor, optionally cast dtype if different
        if input_array.dtype != dtype:
            return tf.cast(input_array, dtype)
        return input_array
    else:
        # Convert numpy arrays or other types to tensor
        return tf.convert_to_tensor(input_array, dtype=dtype)

