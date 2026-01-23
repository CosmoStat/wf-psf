"""Optimizer utilities for WF-PSF.

This module provides utility functions to create optimizers for training or evaluation of PSF models.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import tensorflow as tf


def is_optimizer_instance(obj):
    return hasattr(obj, "apply_gradients") and hasattr(obj, "get_config")

def get_optimizer(optimizer_config=None, **overrides):
    """
    Return a compiled optimizer instance based on configuration or name.

    Parameters
    ----------
    optimizer_config : RecursiveNamespace, dict, or str, optional
        Optimizer configuration (from YAML or programmatically), or string name.
    **overrides : keyword arguments
        Optional hyperparameters to override values in optimizer_config
        (e.g., learning_rate, beta_1, beta_2, epsilon, amsgrad).

    Returns
    -------
    tf.keras.optimizers.Optimizer
    """
    # Detect TensorFlow version
    version = tuple(map(int, tf.__version__.split(".")[:2]))
    is_legacy = version < (2, 11)

    # --- Normalize input to a dictionary
    if isinstance(optimizer_config, str):
        optimizer_name = optimizer_config.lower()
        optimizer_params = {}
    elif isinstance(optimizer_config, dict):
        optimizer_name = optimizer_config.get("name", "adam").lower()
        optimizer_params = dict(optimizer_config)
    elif hasattr(optimizer_config, "__dict__"):  # RecursiveNamespace
        optimizer_name = getattr(optimizer_config, "name", "adam").lower()
        optimizer_params = {
            k: getattr(optimizer_config, k) for k in optimizer_config.__dict__
        }
    else:
        optimizer_name = "adam"
        optimizer_params = {}

    # Apply any overrides
    optimizer_params.update(overrides)

    # Extract learning_rate
    learning_rate = optimizer_params.pop("learning_rate", 1e-3)

    # --- Rectified Adam (TensorFlow Addons)
    if optimizer_name in ["rectified_adam", "radam"]:
        try:
            import tensorflow_addons as tfa
        except ImportError:
            raise ImportError(
                "TensorFlow Addons not found. Install with `pip install wf_psf[addons]`."
            )
        optimizer_params.pop("name", None)
        return tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    # --- Standard Adam (Legacy or Current)
    if optimizer_name == "adam":
        opt_cls = (
            tf.keras.optimizers.legacy.Adam if is_legacy else tf.keras.optimizers.Adam
        )
        return opt_cls(
            learning_rate=learning_rate,
            beta_1=optimizer_params.get("beta_1", 0.9),
            beta_2=optimizer_params.get("beta_2", 0.999),
            epsilon=optimizer_params.get("epsilon", 1e-07),
            amsgrad=optimizer_params.get("amsgrad", False),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
