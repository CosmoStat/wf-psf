import importlib

# Dynamically import modules to trigger side effects when wf_psf is imported
<<<<<<< HEAD
importlib.import_module("wf_psf.psf_models.psf_models")
importlib.import_module("wf_psf.psf_models.models.psf_model_semiparametric")
importlib.import_module("wf_psf.psf_models.models.psf_model_physical_polychromatic")
importlib.import_module("wf_psf.psf_models.tf_modules.tf_psf_field")
=======
importlib.import_module('wf_psf.psf_models.psf_models')
importlib.import_module('wf_psf.psf_models.models.psf_model_semiparametric')
importlib.import_module('wf_psf.psf_models.models.psf_model_physical_polychromatic')
importlib.import_module('wf_psf.psf_models.tf_modules.tf_psf_field') 
>>>>>>> 8ada04a (Refactor: Encapsulate logic in psf_models package with subpackages: models and tf_modules, add/rm modules, update import statements and tests)
