"""TensorFlow dataset converter for PSF datasets.

This module provides the TensorFlowDatasetConverter class, which handles the conversion of PSF datasets (both dataclass-based and dict-based) into TensorFlow tensors suitable for training, evaluation, and inference. It includes methods for processing SEDs using a PSF simulator and converting various dataset formats into a consistent TensorFlow format.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import tensorflow as tf
from typing import Union
from wf_psf.data.schemas import DatasetMode, SCHEMAS
from wf_psf.data.data_utils import (
    DatasetContainer, 
    ConversionContext,
    SEDContext
)
from wf_psf.psf_models.psf_models import PSFSimulator
from wf_psf.psf_models.tf_modules.tf_utils import ensure_tensor
from wf_psf.utils.utils import generate_SED_elems_in_tensorflow
import logging

logger = logging.getLogger(__name__)

class TensorFlowDatasetConverter:
    """
    Convert structured dataset fields into TensorFlow-compatible tensors.

    This converter applies schema-driven preprocessing and validation to
    dataset fields used throughout the wf-psf pipeline. Conversion behavior
    is controlled by a :class:`DatasetSchema` selected through the
    corresponding :class:`DatasetMode`.

    Required and optional dataset fields are processed according to the
    active schema. Field-specific transformations are delegated to handler
    functions registered in the schema, while generic fields are converted
    directly to TensorFlow tensors.

    Runtime dependencies required by specialized handlers (e.g. SED
    processing) are provided through a :class:`ConversionContext`.

    Notes
    -----
    - Required fields may raise exceptions or warnings depending on schema
      strictness.
    - Optional fields are processed only if present in the dataset.
    - Tensor conversion defaults to ``tf.float32`` unless overridden by
      a field-specific handler.
    - This converter currently targets TensorFlow workflows and produces
      TensorFlow tensor outputs suitable for training and inference.
    """

    def _process_field(
        self,
        dataset,
        result,
        key,
        schema,
        context,
        required,
    ):
        """
        Process a single dataset field according to schema rules.

        This method applies validation, handler dispatch, and TensorFlow
        tensor conversion for an individual dataset field. Specialized
        preprocessing is delegated to schema-registered handlers when
        available; otherwise, values are converted directly to tensors.

        Parameters
        ----------
        dataset : DatasetContainer or dict
            Input dataset containing raw field values.
        result : dict
            Mutable output dictionary storing converted dataset fields.
        key : str
            Canonical dataset field name to process.
        schema : DatasetSchema
            Active dataset schema defining required fields, optional
            fields, strictness behavior, and field handlers.
        context : ConversionContext
            Runtime conversion context containing optional domain-specific
            processing dependencies.
        required : bool
            Whether the field is required under the active schema.

        Raises
        ------
        ValueError
            Raised if:
            
            - A required field is missing while schema strictness is enabled.
            - A handler requires a domain-specific context that is absent.

        Notes
        -----
        - Missing optional fields are silently ignored.
        - Missing required fields generate warnings when schema strictness
          is disabled.
        - Field handlers are resolved dynamically from the active schema.
        """
        MISSING = object()

        v = dataset.get(key, MISSING)

        if v is MISSING:
            if required and schema.strict:
                raise ValueError(
                    f"Dataset field '{key}' required for "
                    f"{schema.id} is missing."
                )

            if required:
                logger.warning(
                    f"Dataset field '{key}' required for "
                    f"{schema.id} is missing."
                )

            return

        handler = schema.handlers.get(key)
        
        if handler is not None:
            domain_ctx = getattr(context, key, None)

            if domain_ctx is None:
                raise ValueError(
                    f"Missing context for domain '{key}'"
                )

            result[key] = handler(self, v, domain_ctx)

        else:
            result[key] = ensure_tensor(v, dtype=tf.float32)

    def convert_dataset(
        self,
        dataset: Union[DatasetContainer, dict],
        simPSF: PSFSimulator,
        n_bins_lambda: int,
        mode : DatasetMode = DatasetMode.TRAIN
    ):
        """
        Convert a dataset into TensorFlow-compatible tensors.

        Dataset conversion is performed according to the schema associated
        with the selected :class:`DatasetMode`. Required and optional fields
        are processed independently, and field-specific preprocessing is
        delegated to registered schema handlers when applicable.

        A :class:`ConversionContext` is constructed internally and passed
        through the conversion pipeline to provide runtime dependencies
        required by specialized handlers.

        Parameters
        ----------
        dataset : DatasetContainer or dict
            Input dataset containing raw arrays, metadata, and optional
            preprocessing fields.
        simPSF : PSFSimulator
            PSF simulator used for SED preprocessing operations.
        n_bins_lambda : int
            Number of wavelength bins used during SED discretization.
        mode : DatasetMode, default=DatasetMode.TRAIN
            Dataset conversion mode defining the active schema and
            validation behaviour.

        Returns
        -------
        dict
            Dictionary containing TensorFlow tensors keyed by canonical
            dataset field names.

        Raises
        ------
        ValueError
            Raised if:
            
            - A required dataset field is missing under strict schema
              validation.
            - A required domain-specific context is unavailable for a
              registered field handler.

        Notes
        -----
        - Required fields are validated according to schema strictness.
        - Optional fields are processed only if present.
        - Generic fields are converted using ``ensure_tensor``.
        - Specialized preprocessing (e.g. SED conversion) is delegated
          to registered schema handlers.
        """
        
        schema = SCHEMAS[mode]
        req_keys = schema.required_keys
        opt_keys = schema.optional_keys 

        logger.info(
            f"Using dataset schema '{schema.id}' "
            f"(required_fields={len(req_keys)}, "
            f"optional_fields={len(opt_keys)})"
        )

        result = dict(dataset)

        context = ConversionContext(
            seds=SEDContext(simPSF=simPSF, n_bins_lambda=n_bins_lambda)
        )

        # Handle required keys
        for k in req_keys:
            self._process_field(
                dataset=dataset,
                result=result,
                key=k,
                schema=schema,
                context=context,
                required=True,
            )
            
        # Handle optional keys
        for k in opt_keys:
            self._process_field(
                dataset=dataset,
                result=result,
                key=k,
                schema=schema,
                context=context,
                required=False,
            )
                    
        return result

    @staticmethod
    def process_seds(sed_data, simPSF, n_bins_lambda):
        """
        Process SEDs using simPSF and convert to TensorFlow tensors.

        This is a core operation that must be performed on all SED data before
        use in training or inference. Converts raw SED arrays into wavelength-
        binned TensorFlow tensors.

        Parameters
        ----------
        sed_data : array_like
            Array of SEDs, shape (N, n_wavelengths) or similar
        simPSF : PSFSimulator
            PSF simulator used for SED processing.
        n_bins_lambda : int
            Number of wavelength bins for SED processing.

        Returns
        -------
        tf.Tensor
            Processed SED tensor, shape (N, n_bins_lambda, n_components)

        Raises
        ------
        ValueError
            If sed_data is None

        Notes
        -----
        - Uses tf.float64 internally for precision during generation
        - Returns tf.float32 for training efficiency
        - Transposes to shape (N, n_bins_lambda, n_components)
        """
        processed = [
            generate_SED_elems_in_tensorflow(
                sed, simPSF, n_bins=n_bins_lambda, tf_dtype=tf.float64
            )
            for sed in sed_data
        ]
        sed_tensor = ensure_tensor(processed, dtype=tf.float32)

        return tf.transpose(sed_tensor, perm=[0, 2, 1])
