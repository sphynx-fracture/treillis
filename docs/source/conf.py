# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import sys
import asyncio

sys.path.insert(0, r'../../src/treillis')
sys.path.insert(0, r'../../cea/scripts')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'treillis'
year = datetime.now().year
copyright = f'2019-{year}, SPEC CEA Iramis'
author = 'Antoine Montiel'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'myst_parser',
              'numpydoc', 'sphinx.ext.intersphinx',
              'sphinx.ext.autosummary', 'sphinx_copybutton',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive','sphinx_design',
              'sphinx.ext.mathjax', 'sphinx_gallery.gen_gallery',
              
              ]

templates_path = ['_templates']

numpydoc_use_plots = False
numpydoc_xref_param_type = True

try:
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
        )
except:
    42
 
# MYST ------------------------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'myst',
    '.md': 'myst',
    # '.ipynb': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# autodoc ---------------------------------------------------------------------
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'imported-members': True,
}

copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"


# autosummary of the modules
autosummary_generate = True
# Source - https://stackoverflow.com/q
# Posted by Jacob Marble, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-08, License - CC BY-SA 3.0

autoclass_content = 'both'

# copy button
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

exclude_patterns = []#'auto_example/']

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_theme_options = {
    "logo": {
        "text": "treillis",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "show_nav_level": 2,
    "show_toc_level": 3,
}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

html_sidebars = {
    'install': [],
    'concept/concept': [],
    }

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scikit-sparse': ('https://scikit-sparse.readthedocs.io/en/stable/', None)
}


# Sphinx gallery --------------------------------------------------------------
import sphinx_gallery

sphinx_gallery_conf = {
     'examples_dirs': ['../../examples'],#, '../../cea/cea_examples'],   # path to your example scripts
     'gallery_dirs':['auto_examples'],#, 'cea/examples'],  # path to where to save gallery generated output
     'subsection_order': ['../../examples/api','../../examples/complex'],
     'ignore_pattern': 'no_output',
     'abort_on_example_error': False,
     'plot_gallery': 'True',
     'only_warn_on_example_error': True,
     'parallel': False,
     'min_reported_time': 1,
     'remove_config_comments': True,
     
}


# Epub generation -------------------------------------------------------------
