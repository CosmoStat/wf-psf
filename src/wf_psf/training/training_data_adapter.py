"""Training Data Adapter.

A module containing training data adapter methods.

Author(s): Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from wf_psf.data.data_adapter import DataAdapter
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class TrainingDataAdapter:
    """TrainingDataAdapter.

    Wraps a generic DataAdapter to prepare training-specific inputs and targets
    for TensorFlow models.

    Responsibilities:
    - Stack sources and masks if loss requires it.
    - Return train / validation inputs and targets separately.
    - Keep loss-specific logic localized.
    """

    def __init__(self, base_adapter: DataAdapter, loss_type: str = "mse"):
        self.base = base_adapter
        self.loss_type = loss_type

    # ---- Inputs ----
    @property
    def train_inputs(self):
        """Return train inputs.

        Train inputs as a list [positions, seds].
        """
        positions = self.base.train_data.get("positions")
        seds = self.base.train_data.get("seds")
        return [positions, seds] if seds is not None else [positions]

    @property
    def validation_inputs(self):
        """Return validation inputs.

        Validation inputs as a list [positions, seds].
        """
        positions = self.base.test_data.get("positions")
        seds = self.base.test_data.get("seds")
        return [positions, seds] if seds is not None else [positions]

    # ---- Targets ----
    @property
    def train_targets(self):
        """Return train targets.

        Train targets for the model, stacking masks if needed
        """
        sources = self.base.train_data.get("sources")
        if self.loss_type == "mask_mse":
            logger.info("Stacking sources and masks...")
            masks = self.base.train_data.get("masks")
            if masks is None:
                raise ValueError("mask_mse requires masks for training.")
            return tf.stack([sources, masks], axis=-1)
        return sources

    @property
    def validation_targets(self):
        """Return Validation targets.

        Validation targets, stacking masks if needed
        """
        sources = self.base.test_data.get("sources")
        if self.loss_type == "mask_mse":
            masks = self.base.test_data.get("masks")
            if masks is None:
                raise ValueError("mask_mse requires masks for validation.")
            return tf.stack([sources, masks], axis=-1)
        return sources
