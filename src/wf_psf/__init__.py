import importlib

# Dynamically import modules to trigger side effects when wf_psf is imported
importlib.import_module("wf_psf.psf_models.psf_models")
importlib.import_module("wf_psf.psf_models.models.psf_model_semiparametric")
importlib.import_module("wf_psf.psf_models.models.psf_model_physical_polychromatic")
importlib.import_module("wf_psf.psf_models.tf_modules.tf_psf_field")

# Register config handlers
importlib.import_module("wf_psf.training.training_config_handler")
importlib.import_module("wf_psf.metrics.metrics_config_handler")
importlib.import_module("wf_psf.plotting.plotting_config_handler")
importlib.import_module("wf_psf.data.data_config_handler")
