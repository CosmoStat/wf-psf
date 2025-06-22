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

