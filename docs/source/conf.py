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
copyright = "2023, CosmoStat"
author = "CosmoStat"
release = "2.0.1"

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

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
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
