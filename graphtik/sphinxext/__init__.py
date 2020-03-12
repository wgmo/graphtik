# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Extends Sphinx with :rst:dir:`graphtik` directive rendering plots from doctest code."""
import collections.abc as cabc
import re
from pathlib import Path
from shutil import copyfileobj
from typing import List, Union

import sphinx
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst import roles as rst_roles
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.ext import doctest as extdoctest
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.writers.html import HTMLTranslator

from .. import __version__

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Use backported to PY<3.7 `importlib_resources` lib.
    import importlib_resources as pkg_resources


log = logging.getLogger(__name__)


class graphtik_node(nodes.General, nodes.Element):
    """Non-writtable node wrapping (literal-block + figure(dynaimage)) nodes.

    The figure gets the :name: & :align: options, the contained dynaimage gets
    the :height", :width:, :scale:, :classes: options.
    """


def _ignore_node_but_process_children(
    self: HTMLTranslator, node: graphtik_node
) -> None:
    raise nodes.SkipDeparture


class dynaimage(nodes.General, nodes.Inline, nodes.Element):
    """Writes a tag in `tag` attr (``<img>`` for PNGs, ``<object>`` for SVGs/PDFs). """


def _html_visit_dynaimage(self: HTMLTranslator, node: dynaimage):
    # See sphinx/writers/html5:visit_image()
    tag = getattr(node, "tag", None)
    if not tag:
        # Probably couldn't find :graphvar: in doctest globals.
        raise nodes.SkipNode

    assert tag in ["img", "object"], (tag, node)

    atts = dict(node.attlist())

    self.body.append(self.emptytag(node, tag, "", **atts))
    self.body.append(f"</{tag}>")
    self.body.append("\n")
    self.body.append(getattr(node, "cmap", ""))

    raise nodes.SkipNode


_image_formats = ("png", "svg", "svgz", "pdf", None)


def _valid_format_option(argument: str) -> Union[str, None]:
    # None choice ignored, choice() would scream due to upper().
    if not argument:
        return None
    return directives.choice(argument, _image_formats)


_graphtik_options = {
    "caption": directives.unchanged,
    "graph-format": _valid_format_option,
    "graphvar": directives.unchanged_required,
}
_doctest_options = extdoctest.DoctestDirective.option_spec
_img_options = {
    k: v
    for k, v in directives.images.Image.option_spec.items()
    if k not in ("name", "align", "target")
}
_fig_options = {
    k: v
    for k, v in directives.images.Figure.option_spec.items()
    if k in ("name", "figclass", "align", "figwidth")
}
_option_spec = {
    **_doctest_options,
    **_img_options,
    **_fig_options,
    **_graphtik_options,
}


class _GraphtikTestDirective(extdoctest.TestDirective):
    """A doctest-derrived directive embedding graphtik plots into a sphinx site."""

    _real_name: str
    _con_name: str

    def run(self) -> List[nodes.Node]:
        """Con :class:`.TestDirective` it's some subclass, and append custom options in return node."""
        options = self.options

        ## Allow empty directives,
        #  just to print any graphvar in globals.
        #
        # if not "\n".join(self.content).strip():
        #     # Add dummy code before kicking the doctest directive.
        #     self.content = ["pass"]

        self.name = self._con_name
        try:
            original_nodes = super().run()
        finally:
            self.name = self._real_name

        location = self.state_machine.get_source_and_line(self.lineno)

        img_format = self._decide_img_format(options)
        log.debug(
            "decided `graph-format`: %r", img_format, location=location,
        )
        if not img_format:
            # Bail out, probably unknown builder.
            return original_nodes

        node = graphtik_node(graphvar=options.get("graphvar"), img_format=img_format)
        node.source, node.line = location
        node += original_nodes

        figure = nodes.figure()
        figure.source, figure.line = location
        self.add_name(figure)
        align = options.get("align")
        if align:
            align = f"align-{align}"
            figure["classes"].append(align)
        figure["classes"] += options.get("figclass", "").split()
        node += figure

        img_attrs = {k: v for k, v in options.items() if k in _img_options}
        # TODO: emulate sphinx-processing for image width & height attrs.
        image = dynaimage(**img_attrs)
        dynaimage.source, dynaimage.line = location
        image["classes"] += options.get("class", "").split()
        figure += image

        ## See sphinx.ext.graphviz:figure_wrapper(),
        #  and <sphinx.git>/tests/roots/test-add_enumerable_node/enumerable_node.py:MyFigure
        #
        caption = options.get("caption")
        if caption:
            inodes, messages = self.state.inline_text(caption, self.lineno)
            caption_node = nodes.caption(caption, "", *inodes)
            self.set_source_info(caption_node)
            caption_node += messages
            node += caption_node

        return [node]

    def _decide_img_format(self, options):
        img_format = None
        if "graph-format" in options:
            img_format = options["graph-format"]
        else:
            img_format = self.config.graphtik_default_graph_format
        if not img_format:
            builder_name = self.env.app.builder.name
            for regex, fmt in self.config.graphtik_graph_formats_by_builder.items():
                if re.search(regex, builder_name):
                    img_format = fmt
                    break
            else:
                log.debug(
                    "builder-name %r did not match any key in `graphtik_graph_formats_by_builder`"
                    ", no plot will happen",
                    builder_name,
                )

        return img_format


class GraphtikDoctestDirective(_GraphtikTestDirective):
    """Embeds plots from doctest code (see :rst:dir:`graphtik`). """

    option_spec = _option_spec
    _real_name = "graphkit"
    _con_name = "doctest"


class GraphtikTestoutputDirective(_GraphtikTestDirective):
    """Like :rst:dir:`graphtik` directive, but  emulates doctest :rst:dir:`testoutput` blocks. """

    option_spec = _option_spec
    _real_name = "graphtik-output"
    _con_name = "testoutput"


def _should_work(app: Sphinx):
    """Avoid event-callbacks if not producing HTML/Latex."""
    return isinstance(
        app.builder,
        (
            sphinx.builders.html.StandaloneHTMLBuilder,
            sphinx.builders.latex.LaTeXBuilder,
        ),
    )


def _run_doctests_on_graphtik_document(app: Sphinx, doctree: nodes.Node, docname: str):
    """Callback of `doctree-resolved`` event. """
    from ._graphtikbuilder import get_graphtik_builder

    if _should_work(app) and any(doctree.traverse(graphtik_node)):
        log.info(__("Graphtik-ing document %r..."), docname)
        graphtik_builder = get_graphtik_builder(app)
        graphtik_builder.test_doc(docname, doctree)


def _stage_my_pkg_resource(inp_fname, out_fpath):
    with pkg_resources.open_binary(__package__, inp_fname) as inp, open(
        out_fpath, "wb"
    ) as out:
        copyfileobj(inp, out)


_css_fname = "graphtik.css"


def _copy_graphtik_static_assets(app: Sphinx, exc: Exception) -> None:
    """Callback of `build-finished`` event. """
    if not exc and _should_work(app):
        dst = Path(app.outdir, "_static", _css_fname)
        if not dst.exists():
            _stage_my_pkg_resource(_css_fname, dst)


def _validate_and_apply_configs(app: Sphinx, config: Config):
    """Callback of `config-inited`` event. """
    config.graphtik_default_graph_format is None or _valid_format_option(
        config.graphtik_default_graph_format
    )


def setup(app: Sphinx):
    app.require_sphinx("2.0")
    app.setup_extension("sphinx.ext.doctest")

    app.add_config_value(
        "graphtik_graph_formats_by_builder",
        {"html": "svg", "latex": "pdf"},
        "html",
        [cabc.Mapping],
    )
    app.add_config_value("graphtik_default_graph_format", None, "html", [str, None])
    # TODO: impl sphinx-config --> graphtik-configs
    app.add_config_value("graphtik_configurations", {}, "html", [cabc.Mapping])
    # TODO: impl sphinx-config --> plot keywords
    app.add_config_value("graphtik_plot_keywords", {}, "html", [cabc.Mapping])

    app.add_node(graphtik_node, html=(_ignore_node_but_process_children, None))
    # TODO: implement visitor to support LaTex.
    app.add_node(dynaimage, html=(_html_visit_dynaimage, None))
    app.add_directive("graphtik", GraphtikDoctestDirective)
    app.add_directive("graphtik-output", GraphtikTestoutputDirective)
    app.connect("config-inited", _validate_and_apply_configs)
    app.connect("doctree-resolved", _run_doctests_on_graphtik_document)
    app.connect("build-finished", _copy_graphtik_static_assets)
    app.add_css_file(_css_fname)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
