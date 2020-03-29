# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import importlib
import io
import logging
import os
import os.path as osp
import re
import subprocess
import sys

from docutils import core as dcore
from readme_renderer import rst

proj_path = osp.join(osp.dirname(__file__), "..")

########################
## Copied from Twine

# Regular expression used to capture and reformat docutils warnings into
# something that a human can understand. This is loosely borrowed from
# Sphinx: https://github.com/sphinx-doc/sphinx/blob
# /c35eb6fade7a3b4a6de4183d1dd4196f04a5edaf/sphinx/util/docutils.py#L199
_REPORT_RE = re.compile(
    r"^<string>:(?P<line>(?:\d+)?): "
    r"\((?P<level>DEBUG|INFO|WARNING|ERROR|SEVERE)/(\d+)?\) "
    r"(?P<message>.*)",
    re.DOTALL | re.MULTILINE,
)


class _WarningStream:
    def __init__(self):
        self.output = io.StringIO()

    def write(self, text):
        matched = _REPORT_RE.search(text)

        if not matched:
            self.output.write(text)
            return

        self.output.write(
            "line {line}: {level_text}: {message}\n".format(
                level_text=matched.group("level").capitalize(),
                line=matched.group("line"),
                message=matched.group("message").rstrip("\r\n"),
            )
        )

    def __str__(self):
        return self.output.getvalue()


def test_README_as_PyPi_landing_page(monkeypatch):
    long_desc = subprocess.check_output(
        "python setup.py --long-description".split(), cwd=proj_path
    )
    assert long_desc

    err_stream = _WarningStream()
    result = rst.render(
        long_desc,
        # The specific options are a selective copy of:
        # https://github.com/pypa/readme_renderer/blob/master/readme_renderer/rst.py
        stream=err_stream,
        halt_level=2,  # 2=WARN, 1=INFO
    )
    if not result:
        raise AssertionError(str(err_stream))


def test_sphinx_html(monkeypatch):
    # Fail on warnings, but don't rebuild all files (no `-a`),
    # rm -r ./build/sphinx/ # to fully test site.
    site_args = list("setup.py build_sphinx -W".split())
    if logging.getLogger(__name__).isEnabledFor(logging.INFO):
        site_args.append("-v")
    monkeypatch.setattr(sys, "argv", site_args)
    monkeypatch.chdir(proj_path)
    importlib.import_module("setup")
