# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os
from datetime import datetime

current_year = datetime.now().year

sys.path.insert(0, os.path.abspath("src/wf_psf"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
start_year = 2023
current_year = datetime.now().year

project = "wf-psf"
if current_year > start_year:
    copyright = f"{start_year}–{current_year}, CosmoStat"
else:
    copyright = f"{start_year}, CosmoStat"
author = "CosmoStat"
release = "3.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "numpydoc",
    "sphinx_rtd_theme",
]

extensions += ["sphinx.ext.autosummary"]
autosummary_generate = True
autosummary_generate_overwrite = True
# Ignore inherited members to reduce stub warnings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,  # This helps reduce warnings
}

templates_path = ["_templates"]
exclude_patterns = []
intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
sphinxemoji_style = "twemoji"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# Add MyST enable extensions
myst_enable_extensions = [
    "colon_fence",
]

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "imgs/cosmostat_logo.png"
html_theme_options = {
    "analytics_id": "G-XXXXXXXXXX",  #  Provided by Google in your dashboard
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#ffb400",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- BibTeX Setting  ----------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- Mock imports for documentation ------------------------------------------
autodoc_mock_imports = [
    "tensorflow",
    "tensorflow_addons",
]
