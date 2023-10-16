# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os

sys.path.insert(0, os.path.abspath("src/wf_psf"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "wf-psf"
copyright = "2023, Tobias Liaudat, Jennifer Pollack, Sam Farrens"
author = "Tobias Liaudat, Jennifer Pollack, Sam Farrens"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []
intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
