"""A builder that Render graphtik plots from doctest-runner's globals."""
from hashlib import sha1
from pathlib import Path
from typing import Union

import pydot
from boltons.iterutils import first
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.ext import doctest as extdoctest
from sphinx.locale import _, __
from sphinx.util import logging

from ..base import Plotter
from . import graphtik_node, img_options
from . import doctestglobs


Plottable = Union[None, Plotter, pydot.Dot]

log = logging.getLogger(__name__)


class GraphtikPlotsBuilder(doctestglobs.ExposeGlobalsDocTestBuilder):
    """Retrieve *plottable* from doctests globals and render them. """

    run_empty_code = True

    def _globals_updated(self, code: extdoctest.TestCode, globs: dict):
        """Collect plottable from doctest-runner globals and render graphtik plot. """
        node: nodes.Element = code.node.parent
        if isinstance(node, graphtik_node):
            plottable = self._retrieve_graphvar_plottable(
                globs, node["graphvar"], code.filename, code.lineno
            )
            if plottable:
                if not isinstance(plottable, pydot.Dot):
                    plottable = plottable.plot()
                rel_img_path = self._render_dot_image(node, plottable)
                dot_str = str(plottable)
                node += self._make_image_node(node, rel_img_path, dot_str=dot_str)

                node.replace_self(node.children)

    def _is_plottable(self, value):
        return isinstance(value, (Plotter, pydot.Dot))

    def _retrieve_graphvar_plottable(
        self, globs: dict, graphvar, filename, line_number,
    ) -> Plottable:
        plottable: Plottable = None

        if graphvar is None:
            ## Pick last plottable from globals.
            #
            for i, var in enumerate(reversed(list(globs))):
                value = globs[var]
                if self._is_plottable(value):
                    log.debug(
                        __(
                            "picked plottable %r from doctest globals (num %s from the end0)"
                        ),
                        var,
                        i,
                        location=(filename, line_number),
                    )
                    plottable = value
                    break
            else:
                log.error(
                    __("could not find any plottable in doctest globals"),
                    location=(filename, line_number),
                )

        else:
            ## Pick named var from globals.
            #
            try:
                value = globs[graphvar]
                if not self._is_plottable(value):
                    log.warning(
                        __(
                            "value of graphvar %r in doctest globals is not plottable but %r"
                        ),
                        graphvar,
                        type(value).__name__,
                        location=(filename, line_number),
                    )
                else:
                    plottable = value
            except KeyError:
                log.warning(
                    __("could not find graphvar %r in doctest globals"),
                    graphvar,
                    location=(filename, line_number),
                )

        return plottable

    def _render_dot_image(self, node: nodes.Node, dot: pydot.Dot) -> Path:
        """
        Ensure png(+usemap)|svg|svgz|pdf file exist, and return its path.
        """

        img_format = node["img_format"]

        ## Derrive image-filename from graph contents.
        #
        hasher = sha1()
        hasher.update(str(dot).encode())
        fname = f"graphtik-{hasher.hexdigest()}.{img_format}"
        abs_fpath = Path(self.outdir, self.imagedir, fname)

        ## Don not re-write images, thay have content-named path,
        #  so they are never out-of-date.
        #
        if not abs_fpath.is_file():
            abs_fpath.parent.mkdir(parents=True, exist_ok=True)
            # active builder (not me...).
            builder_name = self.app.builder.name
            if builder_name == "latex":
                dot.write(abs_fpath, format="pdf")
            else:  # "html" in builder_name:
                dot.write(abs_fpath, format=img_format)

                if img_format == "png":
                    dot.write(abs_fpath.with_suffix(".cmap"), format="cmapx")

        ## XXX: used to work till active-builder attributes were transfered to self.
        # rel_fpath = Path(self.imgpath, fname)
        rel_fpath = Path(self.imagedir, fname)

        return rel_fpath

    def _make_image_node(
        self, node: nodes.Element, rel_img_path: Path, dot_str: str, **img_kw,
    ) -> nodes.image:
        img_kw.update((k, node[k]) for k in img_options if k != "align" and k in node)
        img_format = node["img_format"]

        if 1 or img_format == "png":
            img_node = nodes.image(
                uri=str(rel_img_path),
                alt=dot_str,
                title=dot_str,
                candidates={},
                **img_kw,
            )
            img_node["usemap"] = rel_img_path.with_suffix(".cmap")
        else:
            img_node = nodes.graphtik_svg(
                data=str(rel_img_path),
                alt=dot_str,
                title=dot_str,
                candidates={},
                **img_kw,
            )
            img_node["usemap"] = rel_img_path.with_suffix(".cmap")

        # Wrap image-node in a figure-node.
        #
        # TODO: support RST standard figure-options (figwidth, figclass)
        figure_node = nodes.figure()
        if "align" in node:
            figure_node["align"] = node["align"]
        # TODO: make sphinx-SVGs zoomable.
        figure_node += img_node
        ## Use caption-node, if prepared by directive.
        #
        caption_node = first(node.traverse(nodes.caption))
        if caption_node:
            caption_node.parent.remove(caption_node)
            figure_node += caption_node

        return figure_node


_graphtik_builder = None


def get_graphtik_builder(app: Sphinx) -> GraphtikPlotsBuilder:
    """Initialize a singleton patched doctest-builder"""
    global _graphtik_builder
    if _graphtik_builder is None:
        _graphtik_builder = GraphtikPlotsBuilder(app)
        _graphtik_builder.set_environment(app.env)
        _graphtik_builder.imgpath = app.builder.imgpath
        _graphtik_builder.imagedir = app.builder.imagedir
        _graphtik_builder.init()
    return _graphtik_builder
