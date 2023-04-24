import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "graphtik.sphinxext",
]

latex_documents = [
    (
        "index",
        "test.tex",
        "The basic Sphinx documentation for testing",
        "Sphinx",
        "report",
    )
]

# FIXME: htm5 dosn't work!??
html_experimental_html5_writer = True
