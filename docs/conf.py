# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import doctest

# -- Project information -----------------------------------------------------

project = 'Psifr'
copyright = '2020, Neal Morton'
author = 'Neal Morton'

# The full version, including alpha/beta/rc tags
release = 'v0.7.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['references.bib']

doctest_default_flags = (
    doctest.ELLIPSIS
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.DONT_ACCEPT_TRUE_FOR_1
    | doctest.NORMALIZE_WHITESPACE
)

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'seaborn': ('https://seaborn.pydata.org', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
html_logo = "_static/logo.png"

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    "github_url": "https://github.com/mortonne/psifr",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']