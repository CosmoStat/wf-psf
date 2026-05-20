"""
Field-specific dataset preprocessing handlers.

This module defines specialized transformation handlers used during
schema-driven dataset conversion. Handlers encapsulate field-level
preprocessing logic that cannot be performed through generic tensor
conversion alone.

Handlers are registered through :class:`DatasetSchema` definitions and
invoked dynamically by dataset converters during training, evaluation,
or inference workflows.

Typical responsibilities include:

- Spectral energy distribution (SED) preprocessing
- Instrument-specific tensor transformations
- Physics-aware feature conversion
- Domain-specific data normalization

Handlers receive both raw dataset values and runtime conversion
contexts, allowing preprocessing operations to access external
dependencies such as PSF simulators or instrument configuration
objects without coupling these concerns to the converter itself.

Notes
-----
The handler system is intentionally extensible and designed to support
additional scientific domains and instrument pipelines beyond the
current Euclid-specific workflows.

Authors
-------
Jennifer Pollack <jennifer.pollack@cea.fr>
"""

def process_seds_handler(converter, dataset, sed_context):
    """Process SEDS handler.
    
    Parameters
    ----------
    converter : TensorFlowDataSetConverter
        TensorFlow dataset converter instance 
    dataset : array_like
        Array of SEDS, shape (N, n_wavelengths) or similar
    sed_context : SEDContext
        Context object containing parameters required for SED processing
    """
    return converter.process_seds(
        dataset,
        simPSF=sed_context.simPSF,
        n_bins_lambda=sed_context.n_bins_lambda,
    )


# Handler registry
handlers = {
    "seds": process_seds_handler
}