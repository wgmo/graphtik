# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
Plotting of graphtik graphs.

Separate from `graphtik.base` to avoid too many imports too early.
from contextlib import contextmanager
"""
import copy
import html
import inspect
import io
import json
import logging
import os
import re
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import networkx as nx
import pydot
from boltons.iterutils import remap

from .base import PlotArgs, func_name, func_source
from .modifiers import optional
from .network import _EvictInstruction
from .op import Operation

log = logging.getLogger(__name__)


#: A nested dictionary controlling the rendering of graph-plots in Jupyter cells,
#:
#: as those returned from :meth:`.Plottable.plot()` (currently as SVGs).
#: Either modify it in place, or pass another one in the respective methods.
#:
#: The following keys are supported.
#:
#: :param svg_pan_zoom_json:
#:     arguments controlling the rendering of a zoomable SVG in
#:     Jupyter notebooks, as defined in https://github.com/ariutta/svg-pan-zoom#how-to-use
#:     if `None`, defaults to string (also maps supported)::
#:
#:             "{controlIconsEnabled: true, zoomScaleSensitivity: 0.4, fit: true}"
#:
#: :param svg_element_styles:
#:     mostly for sizing the zoomable SVG in Jupyter notebooks.
#:     Inspect & experiment on the html page of the notebook with browser tools.
#:     if `None`, defaults to string (also maps supported)::
#:
#:         "width: 100%; height: 300px;"
#:
#: :param svg_container_styles:
#:     like `svg_element_styles`, if `None`, defaults to empty string (also maps supported).
default_jupyter_render = {
    "svg_pan_zoom_json": "{controlIconsEnabled: true, fit: true}",
    "svg_element_styles": "width: 100%; height: 300px;",
    "svg_container_styles": "",
}


def _parse_jupyter_render(dot) -> Tuple[str, str, str]:
    jupy_cfg: Mapping[str, Any] = getattr(dot, "_jupyter_render", None)
    if jupy_cfg is None:
        jupy_cfg = default_jupyter_render

    def parse_value(key: str, parser: Callable) -> str:
        if key not in jupy_cfg:
            return parser(default_jupyter_render.get(key, ""))

        val: Union[Mapping, str] = jupy_cfg.get(key)
        if not val:
            val = ""
        elif not isinstance(val, str):
            val = parser(val)
        return val

    def styles_parser(d: Mapping) -> str:
        return "".join(f"{key}: {val};\n" for key, val in d)

    svg_container_styles = parse_value("svg_container_styles", styles_parser)
    svg_element_styles = parse_value("svg_element_styles", styles_parser)
    svg_pan_zoom_json = parse_value("svg_pan_zoom_json", json.dumps)

    return svg_pan_zoom_json, svg_element_styles, svg_container_styles


def _dot2svg(dot):
    """
    Monkey-patching for ``pydot.Dot._repr_html_()` for rendering in jupyter cells.

    Original ``_repr_svg_()`` trick was suggested in https://github.com/pydot/pydot/issues/220.

    .. Note::
        Had to use ``_repr_html_()`` and not simply ``_repr_svg_()`` because
        (due to https://github.com/jupyterlab/jupyterlab/issues/7497)


    .. TODO:
        Render in jupyter cells fully on client-side without SVG, using lib:
        https://visjs.github.io/vis-network/docs/network/#importDot
        Or with plotly https://plot.ly/~empet/14007.embed

    """
    pan_zoom_json, element_styles, container_styles = _parse_jupyter_render(dot)
    svg_txt = dot.create_svg().decode()
    html_txt = f"""
        <div class="svg_container">
            <style>
                .svg_container {{
                    {container_styles}
                }}
                .svg_container SVG {{
                    {element_styles}
                }}
            </style>
            <script src="https://ariutta.github.io/svg-pan-zoom/dist/svg-pan-zoom.min.js"></script>
            <script type="text/javascript">
                var scriptTag = document.scripts[document.scripts.length - 1];
                var parentTag = scriptTag.parentNode;
                var svg_el = parentTag.querySelector(".svg_container svg");
                svgPanZoom(svg_el, {pan_zoom_json});
            </script>
            {svg_txt}
        </</>
    """
    return html_txt


def _monkey_patch_for_jupyter(pydot):
    """Ensure Dot instance render in Jupyter notebooks. """
    if not hasattr(pydot.Dot, "_repr_html_"):
        pydot.Dot._repr_html_ = _dot2svg

        import dot_parser

        def parse_dot_data(s):
            """Patched to fix pydot/pydot#171 by letting ex bubble-up."""
            global top_graphs

            top_graphs = list()
            graphparser = dot_parser.graph_definition()
            graphparser.parseWithTabs()
            tokens = graphparser.parseString(s)
            return list(tokens)

        dot_parser.parse_dot_data = parse_dot_data


_monkey_patch_for_jupyter(pydot)


def _is_class_value_in_list(lst, cls, value):
    return any(isinstance(i, cls) and i == value for i in lst)


def _merge_conditions(*conds):
    """combines conditions as a choice in binary range, eg, 2 conds --> [0, 3]"""
    return sum(int(bool(c)) << i for i, c in enumerate(conds))


def quote_dot_word(word: Any):
    """
    Workaround *pydot* parsing of node-id & labels by encoding as HTML.

    - `pydot` library does not quote DOT-keywords anywhere (pydot#111).
    - Char ``:`` denote port/compass-points and break IDs (pydot#224).
    - Non-strings are not quoted_if_necessary by pydot.

    .. Attention::
        It does not correctly handle ``ID:port:compass-point`` format.

    See https://www.graphviz.org/doc/info/lang.html)
    """
    if word is None:
        return word
    word = str(word)
    if not word:
        return word

    if word[0] != "<" or word[-1] != ">":
        word = f"<{html.escape(word)}>"
    word = word.replace(":", "&#58;")

    return word


def as_identifier(s):
    """
    Convert string into a valid ID, both for html & graphviz.

    It must not rely on Graphviz's HTML-like string,
    because it would not be a valid HTML-ID.

    - Adapted from https://stackoverflow.com/a/3303361/548792,
    - HTML rule from https://stackoverflow.com/a/79022/548792
    - Graphviz rules: https://www.graphviz.org/doc/info/lang.html
    """
    s = s.strip()
    # Remove invalid characters
    s = re.sub("[^0-9a-zA-Z_]", "_", s)
    # Remove leading characters until we find a letter
    # (HTML-IDs cannot start with underscore)
    s = re.sub("^[^a-zA-Z]+", "", s)
    if s in pydot.dot_keywords:
        s = f"{s}_"

    return s


def _pub_props(*d, **kw) -> None:
    """Keep kv-pairs from dictionaries whose keys do not start with underscore(``_``)."""
    return {k: v for k, v in dict(*d, *kw).items() if not str(k).startswith("_")}


def _convey_pub_props(src: dict, dst: dict) -> None:
    """Pass all keys from `src` not starting with underscore(``_``) into `dst`."""
    dst.update((k, v) for k, v in src.items() if not str(k).startswith("_"))


class Ref:
    """Cross-reference (by default) to :class:`.Style` attribute values."""

    __slots__ = ("ref", "_base")

    def __init__(self, ref, base=None):
        self.ref = ref
        self._base = base

    @property
    def base(self):
        return getattr(self, "_base", None) or Style

    def rebased(self, base):
        """Makes re-based clone to the given class/object."""
        return Ref(self.ref, base)

    def __repr__(self):
        return f"Ref({self.base}, '{self.ref}')"

    def __str__(self):
        return getattr(self.base, self.ref, "!missed!")


class Style:
    """Values applied like a theme - patch class or pass instance to plotter."""

    resched_thickness = 4
    fill_color = "wheat"
    failed_color = "LightCoral"
    cancel_color = "Grey"
    broken_color = "Red"
    overwrite_color = "SkyBlue"
    steps_color = "#009999"
    in_plan = "#990000"
    #: the url to the architecture section explaining *graphtik* glossary,
    #: linked by legend.
    arch_url = "https://graphtik.readthedocs.io/en/latest/arch.html"

    kw_graph = {
        "graph_type": "digraph",
        "fontname": "italic",
        # Whether to plot `curved/polyline edges
        # <https://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:splines>`_
        "splines": "ortho",
    }
    kw_data = {}
    kw_op = {}
    kw_op_url = {
        # See :meth:`.Pltter.annotate_plot_args()`
        # Union[str, Callable[[str], str]]
        # "url_format": None,
        # "target": None,
    }
    kw_op_executed = {"fillcolor": Ref("fill_color"), "style": "filled"}
    kw_op_failed = {"fillcolor": Ref("failed_color"), "style": "filled"}
    kw_op_canceled = {"fillcolor": Ref("cancel_color"), "style": "filled"}
    kw_edge = {}

    #: If ``'URL'``` key missing/empty, no legend icon will be plotted.
    kw_legend = {
        "name": "legend",
        "shape": "component",
        "style": "filled",
        "fillcolor": "yellow",
        "URL": "https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg",
    }

    def __init__(self, **kw):
        """
        Deep-copy class-attributes (to avoid sideffects) and apply user-overrides,

        retargeting any :class:`Ref` on my self (from class's 'values).
        """
        values = _pub_props(vars(type(self)))
        values.update(kw)
        self.resolve_refs(values)

    def resolve_refs(self, values: dict = None) -> None:
        """Rebase any refs in styles, and deep copy (for free)."""
        if values is None:
            values = vars(self)
        vars(self).update(
            remap(
                values,
                lambda _p, k, v: (k, str(v.rebased(self)))
                if isinstance(v, Ref)
                else True,
            )
        )


class Plotter:
    """
    a :term:`plotter` renders diagram images of :term:`plottable`\\s.

    .. attribute:: Plotter.style

        The :ref:`customizable <plot-customizations>` :class:`.Style` instance
        controlling theming values & dictionaries for plots.
    """

    def __init__(self, style=None):
        self.style = style or Style()

    def copy(self) -> "Plotter":
        """deep copy of all styles"""
        return copy.deepcopy(self)

    def plot(self, plot_args: PlotArgs):
        plot_args = self.annotate_plot_args(plot_args)
        dot = self.build_pydot(**plot_args.kw_build_pydot)
        return self.render_pydot(dot, **plot_args.kw_render_pydot)

    def build_pydot(
        self,
        graph: nx.Graph,
        *,
        name=None,
        steps=None,
        inputs=None,
        outputs=None,
        solution=None,
        clusters=None,
    ) -> pydot.Dot:
        """
        Build a |pydot.Dot|_ out of a Network graph/steps/inputs/outputs and return it

        to be fed into `Graphviz`_ to render.

        See :meth:`.Plottable.plot()` for the arguments, sample code, and
        the legend of the plots.
        """
        if graph is None:
            raise ValueError("At least `graph` to plot must be given!")

        style = self.style

        new_clusters = {}

        def append_or_cluster_node(dot, nx_node, node):
            if not clusters or not nx_node in clusters:
                dot.add_node(node)
            else:
                cluster_name = clusters[nx_node]
                node_cluster = new_clusters.get(cluster_name)
                if not node_cluster:
                    node_cluster = new_clusters[cluster_name] = pydot.Cluster(
                        cluster_name, label=cluster_name
                    )
                node_cluster.add_node(node)

        def append_any_clusters(dot):
            for cluster in new_clusters.values():
                dot.add_subgraph(cluster)

        def get_node_name(a):
            if isinstance(a, Operation):
                a = a.name
            return quote_dot_word(a)

        kw = style.kw_graph.copy()
        _convey_pub_props(graph.graph, kw)
        dot = pydot.Dot(**kw)

        if name:
            dot.set_name(as_identifier(name))

        # draw nodes
        for nx_node, data in graph.nodes(data=True):
            if isinstance(nx_node, str):
                # SHAPE change if with inputs/outputs.
                # tip: https://graphviz.gitlab.io/_pages/doc/info/shapes.html
                choice = _merge_conditions(
                    inputs and nx_node in inputs, outputs and nx_node in outputs
                )
                shape = "rect invhouse house hexagon".split()[choice]

                kw = {"name": quote_dot_word(nx_node), "shape": shape}

                # FrameColor change by step type
                if steps and nx_node in steps:
                    kw["color"] = style.in_plan

                # LABEL change with solution.
                if solution and nx_node in solution:
                    kw["style"] = "filled"
                    kw["fillcolor"] = (
                        style.overwrite_color
                        if nx_node in getattr(solution, "overwrites", ())
                        else style.fill_color
                    )

            else:  # Operation
                kw = {
                    "shape": "oval",
                    "name": quote_dot_word(nx_node.name),
                    "fontname": "italic",
                }

                if nx_node.rescheduled:
                    kw["penwidth"] = style.resched_thickness
                if hasattr(solution, "is_failed") and solution.is_failed(nx_node):
                    kw["style"] = "filled"
                    kw["fillcolor"] = style.failed_color
                elif nx_node in getattr(solution, "executed", ()):
                    kw["style"] = "filled"
                    kw["fillcolor"] = style.fill_color
                elif nx_node in getattr(solution, "canceled", ()):
                    kw["style"] = "filled"
                    kw["fillcolor"] = style.cancel_color

            _convey_pub_props(data, kw)
            node = pydot.Node(**kw)
            append_or_cluster_node(dot, nx_node, node)

        append_any_clusters(dot)

        # draw edges
        for src, dst, data in graph.edges(data=True):
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)

            kw = {}
            if data.get("optional"):
                kw["style"] = "dashed"
            if data.get("sideffect"):
                kw["color"] = "blue"

            if getattr(src, "rescheduled", None) or getattr(src, "endured", None):
                kw["style"] = "dashed"
                if solution and dst not in solution and dst not in steps:
                    kw["color"] = style.broken_color

            _convey_pub_props(data, kw)
            edge = pydot.Edge(src=src_name, dst=dst_name, **kw)
            dot.add_edge(edge)

        # draw steps sequence
        if steps and len(steps) > 1:
            it1 = iter(steps)
            it2 = iter(steps)
            next(it2)
            for i, (src, dst) in enumerate(zip(it1, it2), 1):
                src_name = get_node_name(src)
                dst_name = get_node_name(dst)
                edge = pydot.Edge(
                    src=src_name,
                    dst=dst_name,
                    label=str(i),
                    style="dotted",
                    color=style.steps_color,
                    fontcolor=style.steps_color,
                    fontname="bold",
                    fontsize=18,
                    arrowhead="vee",
                    splines=True,
                )
                dot.add_edge(edge)

        if style.kw_legend.get("URL"):
            dot.add_node(pydot.Node(**style.kw_legend))

        return dot

    def annotate_plot_args(self, plot_args: PlotArgs) -> None:
        """
        Customize nodes, e.g. add doc URLs/tooltips, solution tooltips.

        :param plot_args:
            must have, at least, a `graph`, as returned by :class:`.Plottable.prepare_plot_args()`

        :return:
            the updated `plot_args`

        Currently it populates the following *networkx* node-properties:

        1. Merge :attr:`Style.kw_op_url` dictionary on operation-nodes,
           **only** if that style attribute contains a truthy value for the key
           ``'url_format'``;  that key-value may be either:

           - a callable ``format_url(fn_dot_path: str) -> str``, or
           - an ``%s``-format string accepting the same function dot-path.

            The above result is placed in a ``'URL'`` key, understood by `Graphviz`_,
            and the ``'url_format'`` key is discarded before merge.

           .. Note::
               - SVG tooltips may not work without URL on PDFs:
                 https://gitlab.com/graphviz/graphviz/issues/1425

               - Browsers & Jupyter lab are blocking local-urls (e.g. on SVGs),
                 see tip in :term:`plottable`.

        2. Set tooltips with the fn-code for operation-nodes.

        3. Set tooltips with the solution-values for data-nodes.
        """
        from .op import Operation

        kw_op_url = self.style.kw_op_url
        if not kw_op_url or not kw_op_url.get("url_format"):
            kw_op_url = None

        for nx_node, node_attrs in plot_args.graph.nodes.data():
            tooltip = None
            if isinstance(nx_node, Operation):
                if kw_op_url and "URL" not in node_attrs:
                    fn_path = func_name(nx_node.fn, None, mod=1, fqdn=1, human=0)
                    if fn_path:
                        url_fmt = kw_op_url["url_format"]
                        url = (
                            url_fmt(fn_path) if callable(url_fmt) else url_fmt % fn_path
                        )

                        kw = kw_op_url.copy()
                        del kw["url_format"]
                        kw["URL"] = graphviz_html_string(url)
                        node_attrs.update(kw)
                if "tooltip" not in node_attrs:
                    fn_source = func_source(nx_node.fn, None, human=1)
                    if fn_source:
                        node_attrs["tooltip"] = graphviz_html_string(fn_source)
            else:  # DATA node
                sol = plot_args.solution
                if sol is not None and "tooltip" not in node_attrs:
                    val = sol.get(nx_node)
                    tooltip = "None" if val is None else f"({type(val).__name__}) {val}"
                    node_attrs["tooltip"] = graphviz_html_string(tooltip)

        return plot_args

    def render_pydot(
        self, dot: pydot.Dot, filename=None, show=False, jupyter_render: str = None
    ):
        """
        Render a |pydot.Dot|_ instance with `Graphviz`_ in a file and/or in a matplotlib window.

        :param dot:
            the pre-built |pydot.Dot|_ instance
        :param str filename:
            Write diagram into a file.
            Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
            call :func:`.supported_plot_formats()` for more.
        :param show:
            If it evaluates to true, opens the  diagram in a  matplotlib window.
            If it equals `-1`, it returns the image but does not open the Window.
        :param jupyter_render:
            a nested dictionary controlling the rendering of graph-plots in Jupyter cells.
            If `None`, defaults to :data:`default_jupyter_render`;  you may modify those
            in place and they will apply for all future calls (see :ref:`jupyter_rendering`).

            You may increase the height of the SVG cell output with
            something like this::

                netop.plot(jupyter_render={"svg_element_styles": "height: 600px; width: 100%"})

        :return:
            the matplotlib image if ``show=-1``, or the given `dot` annotated with any
            jupyter-rendering configurations given in `jupyter_render` parameter.

        See :meth:`.Plottable.plot()` for sample code.
        """
        # Save plot
        #
        if filename:
            formats = supported_plot_formats()
            _basename, ext = os.path.splitext(filename)
            if not ext.lower() in formats:
                raise ValueError(
                    "Unknown file format for saving graph: %s"
                    "  File extensions must be one of: %s" % (ext, " ".join(formats))
                )

            dot.write(filename, format=ext.lower()[1:])

        ## Display graph via matplotlib
        #
        if show:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            png = dot.create_png()
            sio = io.BytesIO(png)
            img = mpimg.imread(sio)
            if show != -1:
                plt.imshow(img, aspect="equal")
                plt.show()

            return img

        ## Propagate any properties for rendering in Jupyter cells.
        dot._jupyter_render = jupyter_render

        return dot

    def legend(self, filename=None, show=None, jupyter_render: Mapping = None):
        """
        Generate a legend for all plots (see :meth:`.Plottable.plot()` for args)

        See :meth:`Plotter.render_pydot` for the rest arguments.
        """

        ## From https://stackoverflow.com/questions/3499056/making-a-legend-key-in-graphviz
        # Render it manually with these python commands, and remember to update result in git:
        #
        #   from graphtik.plot import legend
        #   legend('docs/source/images/GraphtikLegend.svg')
        dot_text = """
        digraph {
            rankdir=LR;
            subgraph cluster_legend {
            label="Graphtik Legend";

            operation   [shape=oval fontname=italic
                        tooltip="A function with needs & provides."
                        URL="%(arch_url)s#term-operation"];
            insteps     [label="execution step" fontname=italic
                        tooltip="Either an operation or ean eviction-instruction."
                        URL="%(arch_url)s#term-execution-steps"];
            executed    [shape=oval style=filled fillcolor=wheat fontname=italic
                        tooltip="Operation executed successfully."
                        URL="%(arch_url)s#term-solution"];
            failed      [shape=oval style=filled fillcolor=LightCoral fontname=italic
                        tooltip="Failed operation - downstream ops will cancel."
                        URL="%(arch_url)s#term-endurance"];
            rescheduled  [shape=oval penwidth=4 fontname=italic
                        tooltip="Operation may fail / provide partial outputs so `net` must reschedule."
                        URL="%(arch_url)s#term-reschedulling"];
            canceled    [shape=oval style=filled fillcolor=Grey fontname=italic
                        tooltip="Canceled operation due to failures or partial outputs upstream."
                        URL="%(arch_url)s#term-reschedule"];
            operation -> insteps -> executed -> failed -> rescheduled -> canceled [style=invis];

            data    [shape=rect
                    tooltip="Any data not given or asked."
                    URL="%(arch_url)s#term-graph"];
            input   [shape=invhouse
                    tooltip="Solution value given into the computation."
                    URL="%(arch_url)s#term-inputs"];
            output  [shape=house
                    tooltip="Solution value asked from the computation."
                    URL="%(arch_url)s#term-outputs"];
            inp_out [shape=hexagon label="inp+out"
                    tooltip="Data both given and asked."
                    URL="%(arch_url)s#term-netop"];
            evicted [shape=rect color="%(in_plan)s"
                    tooltip="Instruction step to erase data from solution, to save memory."
                    URL="%(arch_url)s#term-evictions"];
            sol     [shape=rect style=filled fillcolor=wheat label="in solution"
                    tooltip="Data contained in the solution."
                    URL="%(arch_url)s#term-solution"];
            overwrite [shape=rect style=filled fillcolor=SkyBlue
                    tooltip="More than 1 values exist in solution with this name."
                    URL="%(arch_url)s#term-overwrites"];
            data -> input -> output -> inp_out -> evicted -> sol -> overwrite [style=invis];

            e1          [style=invis];
            e1          -> requirement;
            requirement [color=invis
                        tooltip="Source operation --> target `provides` OR source `needs` --> target operation."
                        URL="%(arch_url)s#term-needs"];
            requirement -> optional     [style=dashed];
            optional    [color=invis
                        tooltip="Target operation may run without source `need` OR source operation may not `provide` target data."
                        URL="%(arch_url)s#term-needs"];
            optional    -> sideffect    [color=blue];
            sideffect   [color=invis
                        tooltip="Fictive data not consumed/produced by operation functions."
                        URL="%(arch_url)s#term-sideffects"];
            sideffect   -> broken       [color="red" style=dashed]
            broken      [color=invis
                        tooltip="Target data was not `provided` by source operation due to failure / partial-outs."
                        URL="%(arch_url)s#term-partial outputs"];
            broken   -> sequence        [color="%(steps_color)s" penwidth=4 style=dotted
                                        arrowhead=vee label=1 fontcolor="%(steps_color)s"];
            sequence    [color=invis penwidth=4 label="execution sequence"
                        tooltip="Sequence of execution steps."
                        URL="%(arch_url)s#term-execution-steps"];
            }
        }
        """ % {
            "arch_url": self.style.arch_url,
            **vars(self.style),
        }

        dot = pydot.graph_from_dot_data(dot_text)[0]
        # cluster = pydot.Cluster("Graphtik legend", label="Graphtik legend")
        # dot.add_subgraph(cluster)

        # nodes = dot.Node()
        # cluster.add_node("operation")

        return self.render_pydot(dot, filename=filename, show=show)


def legend(
    filename=None, show=None, jupyter_render: Mapping = None, plotter: Plotter = None,
):
    """
    Generate a legend for all plots (see :meth:`.Plottable.plot()` for args)

    :param plotter:
        override the :term:`installed plotter`

    See :meth:`Plotter.render_pydot` for the rest arguments.
    """
    plotter = plotter or get_installed_plotter()
    return plotter.legend(filename, show, jupyter_render)


def supported_plot_formats() -> List[str]:
    """return automatically all `pydot` extensions"""
    return [".%s" % f for f in pydot.Dot().formats]


def graphviz_html_string(s):
    import html

    if s:
        s = html.escape(s).replace("\n", "&#10;")
        s = f"<{s}>"
    return s


_installed_plotter: ContextVar[Plotter] = ContextVar(
    "installed_plotter", default=Plotter()
)


@contextmanager
def installed_plotter(plotter: Plotter) -> None:
    """Like :func:`set_installed_plotter()` as a context-manager to reset old value. """
    if not isinstance(plotter, Plotter):
        raise ValueError(f"Cannot install invalid plotter: {plotter}")
    resetter = _installed_plotter.set(plotter)
    try:
        yield
    finally:
        _installed_plotter.reset(resetter)


def set_installed_plotter(plotter: Plotter):
    """
    The default instance to render :term:`plottable`\\s,

    unless overridden with a `plotter` argument in :meth:`.Plottable.plot()`.

    :param plotter:
        the :class:`plotter` instance to install
    """
    if not isinstance(plotter, Plotter):
        raise ValueError(f"Cannot install invalid plotter: {plotter}")
    return _installed_plotter.set(plotter)


def get_installed_plotter() -> Plotter:
    """Get the previously installed  :class:`.Plotter` instance or default one."""
    plotter = _installed_plotter.get()
    if not isinstance(plotter, Plotter):
        raise ValueError(f"Missing or invalid installed plotter: {plotter}")

    return plotter
