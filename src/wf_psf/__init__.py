import importlib

# Dynamically import modules to trigger side effects when wf_psf is imported
importlib.import_module("wf_psf.psf_models.psf_models")
importlib.import_module("wf_psf.psf_models.psf_model_semiparametric")
importlib.import_module("wf_psf.psf_models.psf_model_physical_polychromatic")
importlib.import_module("wf_psf.psf_models.tf_psf_field")
