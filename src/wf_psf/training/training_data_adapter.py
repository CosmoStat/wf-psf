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
        self.loss_type = loss_type

        # --- Extract data ---
        train_data = base_adapter.train_data
        val_data = base_adapter.test_data

        # --- Materialize everything ---
        logger.debug("Materializing training data snapshot...")

        self._train_inputs = self._prepare_inputs(train_data, split="train")
        self._validation_inputs = self._prepare_inputs(val_data, split="validation")

        self._train_targets = self._prepare_targets(train_data, split="train")
        self._validation_targets = self._prepare_targets(val_data, split="validation")

        logger.debug("Training data snapshot ready.")

    # ---- Helpers ----
    def _prepare_inputs(self, data, split: str) -> list[tf.Tensor]:
        """Prepare Inputs."""
        positions = data.get("positions")
        seds = data.get("seds")

        if positions is None:
            raise ValueError(f"Missing positions for {split} inputs.")

        return [positions, seds] if seds is not None else [positions]

    def _prepare_targets(self, data, split: str) -> tf.Tensor:
        """Prepare Targets."""
        sources = data.get("sources")

        if sources is None:
            raise ValueError(f"Missing sources for {split} targets.")

        if self.loss_type == "mask_mse":
            logger.info(
                f"Stacking sources and masks for {split} (data preparation phase)..."
            )

            masks = data.get("masks")
            if masks is None:
                raise ValueError(f"mask_mse requires masks for {split}.")

            return tf.stack([sources, masks], axis=-1)

        return sources

    # ------------------------------------------------------------------
    # Public API (pure accessors, no computation)
    # ------------------------------------------------------------------

    # ---- Inputs ----
    @property
    def train_inputs(self) -> list[tf.Tensor]:
        """Return train inputs.

        Train inputs as a list [positions, seds].
        """
        return self._train_inputs

    @property
    def validation_inputs(self) -> list[tf.Tensor]:
        """Return validation inputs.

        Validation inputs as a list [positions, seds].
        """
        return self._validation_inputs

    # ---- Targets ----
    @property
    def train_targets(self) -> tf.Tensor:
        """Return train targets.

        Train targets for the model
        """
        return self._train_targets

    @property
    def validation_targets(self) -> tf.Tensor:
        """Return Validation targets.

        Validation targets
        """
        return self._validation_targets
