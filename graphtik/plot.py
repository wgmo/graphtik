# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
:term:`plotting` handled by the :term:`active plotter` & :term:`current theme`.

.. doctest::
    :hide:

    .. Workaround sphinx-doc/sphinx#6590

    >>> from graphtik.plot import *
    >>> __name__ = "graphtik.plot"
"""
import html
import inspect
import io
import json
import logging
import os
import re
import textwrap
from collections import abc, namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from itertools import chain, count
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import jinja2
import markupsafe
import networkx as nx
import pydot
from boltons.iterutils import default_enter, default_exit, get_path, remap
from boltons.setutils import IndexedSet as iset
from jinja2.filters import do_truncate

from . import __version__
from .base import Operation, PlotArgs, Plottable, first_solid, func_name, func_source
from .config import (
    is_debug,
    is_endure_operations,
    is_marshal_tasks,
    is_parallel_tasks,
    is_reschedule_operations,
)
from .modifier import get_jsonp, is_sfx, is_sfxed
from .planning import yield_node_names

log = logging.getLogger(__name__)

## Gracefully import numpy & pandas to check for null values
#  (probable mistakes in function definitions).
#
_null_checks = []
try:
    import numpy as np

    def is_empty_array(val):
        return isinstance(val, np.ndarray) and not np.any()

    _null_checks.append(is_empty_array)
except ImportError:
    pass
try:
    from pandas.core.generic import NDFrame

    def is_empty_frame(val):
        return isinstance(val, NDFrame) and val.empty

    _null_checks.append(is_empty_frame)
except ImportError:
    pass

#: A nested dictionary controlling the rendering of graph-plots in Jupyter cells,
#:#: as those returned from :meth:`.Plottable.plot()` (currently as SVGs).
#: Either modify it in place, or pass another one in the respective methods.
#:
#: Any keys ommitted, are taken from :data:`default_jupyter_render` - pass
#: an empty dict(``{}``) if that is not desirable
#: (eg. to revert to panZoom's library's defaults).
#:
#: The following keys are supported.
#:
#: :param svg_pan_zoom_json:
#:     arguments controlling the rendering of a zoomable SVG in
#:     Jupyter notebooks, as defined in https://github.com/bumbu/svg-pan-zoom#how-to-use
#:     if `None`, defaults to this map (json strings also supported)::
#:
#:             {"controlIconsEnabled": True, "fit": True}"
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
#:
#: .. note::
#:      referred also by :rst:dir:`graphtik`'s :confval:`graphtik_zoomable_options`
#:      default configuration value.
default_jupyter_render = {
    "svg_pan_zoom_json": {"controlIconsEnabled": True, "fit": True},
    "svg_element_styles": "width: 100%; height: 300px;",
    "svg_container_styles": "",
}


def _parse_jupyter_render(dot) -> Tuple[str, str, str]:
    jupy_cfg: Mapping[str, Any] = getattr(dot, "_jupyter_render", None)
    if jupy_cfg is None:
        jupy_cfg = default_jupyter_render

    def parse_value(key: str, parser: Callable) -> str:
        val: Union[Mapping, str] = jupy_cfg.get(key, default_jupyter_render.get(key))

        if not val:
            val = ""
        elif not isinstance(val, str):
            val = parser(val)
        return val

    def styles_parser(d: Mapping) -> str:
        try:
            return "".join(f"{key}: {val};\n" for key, val in d)
        except Exception as ex:
            breakpoint()
            raise ValueError(f"Failed to parse styles {d!r} due to: {ex}") from ex

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
    # Load js-script only once from: https://stackoverflow.com/a/57572107/548792

    import random

    pan_zoom_json, element_styles, container_styles = _parse_jupyter_render(dot)
    svg_txt = dot.create_svg().decode()
    svg_id = f"graphtik-plot-{random.randrange(int(1<<32)):08x}"
    html_txt = f"""
<div id="{svg_id}" class="svg_container">
    <style>
        #{svg_id} {{
            {container_styles}
        }}
        #{svg_id} SVG {{
            {element_styles}
        }}
    </style>
    <script type="text/javascript">
    function graphtik_renderPlot() {{
        var svg_el = document.querySelector("#{svg_id} svg");
        svgPanZoom(svg_el, {pan_zoom_json});
    }}

    function graphtik_load_panZoom_script_n_render() {{
        let id = 'graphtik_panZoomScript'
        if(document.getElementById(id) === null) {{
            let script = document.createElement('script')
            script.setAttribute(
                'src',
                'https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.5.0/dist/svg-pan-zoom.min.js')
            script.setAttribute('id', id)
            document.body.appendChild(script)

            script.onload = graphtik_renderPlot
        }} else {{
            graphtik_renderPlot()
        }}
    }}

    graphtik_load_panZoom_script_n_render()
    </script>
    {svg_txt}
</div>
    """
    return html_txt


def _monkey_patch_for_jupyter(pydot):
    """Ensure Dot instance render in Jupyter notebooks."""
    if not hasattr(pydot.Dot, "_repr_html_"):
        pydot.Dot._repr_html_ = _dot2svg

        import dot_parser

        def parse_dot_data(s):
            """Patched to fix pydot/pydot#171 by letting ex bubble-up."""
            global top_graphs  # pylint: disable=variable-not-defined-globally

            top_graphs = list()
            graphparser = dot_parser.graph_definition()
            graphparser.parseWithTabs()
            tokens = graphparser.parseString(s)
            return list(tokens)

        dot_parser.parse_dot_data = parse_dot_data


_monkey_patch_for_jupyter(pydot)


def is_nx_node_dependent(graph, nx_node):
    """Return true if node's edges are not :term:`subdoc` only."""
    return any(
        1
        for _src, _dst, subdoc in chain(
            graph.in_edges(nx_node, data="subdoc"),
            graph.out_edges(nx_node, data="subdoc"),
        )
        if not subdoc
    )


# TODO: move to base.py, to reduce fan-in imports (and be frank with module diagram).
def graphviz_html_string(
    s,
    *,
    repl_nl=None,
    repl_colon=None,
    xmltext=None,
):
    """
    Workaround *pydot* parsing of node-id & labels by encoding as HTML.

    - `pydot` library does not quote DOT-keywords anywhere (pydot#111).
    - Char ``:`` on node-names denote port/compass-points and break IDs (pydot#224).
    - Non-strings are not quote_if_necessary by pydot.
    - NLs im tooltips of HTML-Table labels `need substitution with the XML-entity
      <see https://stackoverflow.com/a/27448551/548792>`_.
    - HTML-Label attributes (``xmlattr=True``) need both html-escape & quote.

    .. Attention::
        It does not correctly handle ``ID:port:compass-point`` format.

    See https://www.graphviz.org/doc/info/lang.html)
    """
    import html

    if s:
        s = html.escape(s)

        if repl_nl:
            s = s.replace("\n", "&#10;").replace("\t", "&#9;")
        if repl_colon:
            s = s.replace(":", "&#58;")
        if not xmltext:
            s = f"<{s}>"
    return s


def quote_html_tooltips(s):
    """Graphviz HTML-Labels ignore NLs & TABs."""
    if s:
        s = html.escape(s.strip()).replace("\n", "&#10;").replace("\t", "&#9;")
    return s


def quote_node_id(s):
    """See :func:`graphviz_html_string()`"""
    return graphviz_html_string(s, repl_colon=True)


def get_node_name(nx_node, raw=False):
    if isinstance(nx_node, Operation):
        nx_node = nx_node.name
    return nx_node if raw else quote_node_id(nx_node)


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


class Ref:
    """Deferred attribute reference :meth:`resolve`\\d  on a some object(s)."""

    __slots__ = ("ref", "default")

    def __init__(self, ref, default=...):
        self.ref = ref
        self.default = default

    def resolve(self, *objects, default=...):
        """Makes re-based clone to the given class/object."""
        ref = self.ref
        for b in objects:
            if hasattr(b, ref):
                return getattr(b, ref)
        if default is ...:
            default = self.default
        if default is ...:
            raise AttributeError(f"Reference {ref!r} not found in {objects}")
        return default

    def __repr__(self):
        return f"Ref('{self.ref}')"


def _drop_gt_lt(x):
    """SVGs break even with &gt;."""
    return x and re.sub('[<>"]', "_", str(x))


def _escape_or_none(context: jinja2.environment.EvalContext, x, escaper):
    """Do not markup Nones/empties, so `xmlattr` filter does not include them."""
    return x and markupsafe.Markup(escaper(str(x)))


def _format_exception(ex):
    """Printout ``type(msg)``."""
    if ex:
        return f"{type(ex).__name__}: {ex}"


def _vector_info(val):
    """Printout ``val.info()`` or fallback to ``val.shape``."""
    buf = io.StringIO()
    try:
        val.info(buf=buf)
        return buf.getvalue()
    except Exception:
        try:
            return f"shape: {val.shape}, dtype: {val.dtype}"
        except Exception:
            try:
                return len(val)
            except Exception:
                pass


@jinja2.pass_environment
def _reversing_truncate(
    env, s, length=255, killwords=False, end="...", leeway=None, reverse=None
) -> str:
    """
    Like :class:`jinja2.filters.do_truncate` but from the end (keeping the suffix).
    """
    if not reverse:
        return do_truncate(env, s, length, killwords, end, leeway)

    s = s[::-1]
    ret = do_truncate(env, s, length, killwords, end, leeway)
    return ret[::-1]


def _make_jinja2_environment() -> jinja2.Environment:
    env = jinja2.Environment()

    env.filters["ee"] = jinja2.pass_eval_context(
        partial(_escape_or_none, escaper=html.escape)
    )
    env.filters["eee"] = jinja2.pass_eval_context(
        partial(_escape_or_none, escaper=quote_html_tooltips)
    )
    env.filters["vector_info"] = _vector_info
    env.filters["slug"] = jinja2.pass_eval_context(
        partial(_escape_or_none, escaper=as_identifier)
    )
    env.filters["hrefer"] = _drop_gt_lt
    env.filters["ex"] = _format_exception
    env.filters["truncate"] = _reversing_truncate
    env.filters["sideffected"] = lambda x: is_sfx(x) or None
    env.filters["sfxed"] = env.filters["sfx_list"] = lambda x: is_sfxed(x) or None
    env.filters["jsonp"] = get_jsonp

    return env


#: Environment to append our own jinja2 filters.
_jinja2_env = _make_jinja2_environment()


def make_template(s):
    """
    Makes dedented jinja2 templates supporting extra escape filters for `Graphviz`_:

    ``ee``
        Like default escape filter ``e``, but Nones/empties evaluate to false.
        Needed because the default `escape` filter breaks `xmlattr` filter with Nones .
    ``eee``
        Escape for when writting inside HTML-strings.
        Collapses nones/empties (unlike default ``e``).
    ``hrefer``
        Dubious escape for when writting URLs inside Graphviz attributes.
        Does NOT collapse nones/empties (like default ``e``)
    ``ex``
        format exceptions
    ``truncate``
        reversing truncate (keep tail) if `truncate` arg is true
    ``sideffected``
        return the `sideffected` part of an `sfxed` or none
    ``sfx_list``
        return the `sfx_list` part of an `sfxed` or none
    ``jsonp``
        return the `jsonp` list of a dependency or none
    """
    return _jinja2_env.from_string(textwrap.dedent(s).strip())


def _render_template(tpl: jinja2.Template, **kw) -> str:
    """Ignore falsy values, to skip attributes in template all together."""
    return tpl.render(**{k: v for k, v in kw.items() if v})


def make_data_value_tooltip(plot_args: PlotArgs):
    """Called on datanodes, when solution exists."""
    node = plot_args.nx_item
    assert node in plot_args.solution
    val = plot_args.solution[node]
    if val is None:
        tooltip = "(None)"
    elif hasattr(val, "shape"):
        if hasattr(val, "dtype"):
            tooltip = (
                f"({type(val).__name__}, shape: {val.shape}, dtype: {val.dtype}) {val}"
            )
        else:
            tooltip = f"({type(val).__name__}, shape: {val.shape}) {val}"
    else:
        tooltip = f"({type(val).__name__}) {val}"
    return quote_html_tooltips(tooltip[:4095])


def make_overwrite_tooltip(plot_args: PlotArgs):
    """Called on datanodes, withmultiple overwrite values."""
    node = plot_args.nx_item
    assert node in plot_args.solution
    val = plot_args.solution.overwrites[node]
    val_str = "\n  ".join(f"{i}. {v}" for i, v in reversed(list(enumerate(val))))
    tooltip = f"(x{len(val)} overwrites) {val_str}"
    return quote_html_tooltips(tooltip[:4095])


def make_op_prune_comment(plot_args: PlotArgs):
    op = plot_args.nx_item
    sol = plot_args.solution
    plottable = plot_args.plottable

    comments = ()
    if hasattr(sol, "plan") and op in sol.plan.comments:
        comments = sol.plan.comments
    elif op in getattr(plottable, "comments", ()):
        comments = plottable.comments

    if comments:
        return f"(pruned due to {comments[op]})"


def make_op_tooltip(plot_args: PlotArgs):
    """the string-representation of an operation (name, needs, provides)"""
    return plot_args.nx_attrs.get("_op_tooltip", str(plot_args.nx_item))


def make_fn_tooltip(plot_args: PlotArgs):
    """the sources of the operation-function"""
    if "_fn_tooltip" in plot_args.nx_attrs:
        return plot_args.nx_attrs["_fn_tooltip"]
    return func_source(plot_args.nx_item.fn, None, human=1)


class Theme:
    """
    The poor man's css-like :term:`plot theme` (see also :class:`.StyleStack`).

    Tip: `Graphviz node-attributes <https://graphviz.org/doc/info/attrs.html>`_

    To use the values contained in theme-instances, stack them in a :class:`.StylesStack`,
    and :meth:`.StylesStack.merge` them with :term:`style expansion`\\s
    (read it from :meth:`.StyleStack.expand()`).

    .. theme-warn-start
    .. Attention::
        It is recommended to use other means for :ref:`plot-customizations`
        instead of modifying directly theme's class-attributes.

        All :class:`Theme` *class-attributes* are deep-copied when constructing
        new instances, to avoid modifications by mistake, while attempting to update
        *instance-attributes* instead
        (*hint:* allmost all its attributes are containers i.e. dicts).
        Therefore any class-attributes modification will be ignored, until a new
        ``Theme`` instance from the patched class is used .

    .. theme-warn-end

    """

    ##########
    ## VARIABLES

    #: args for jinja2 patched `truncate` filter, above.
    truncate_args = ((23, True), {"reverse": True})
    fill_color = "wheat"
    null_color = "#ffa9cd"
    sideffect_color = "blue"
    subdoc_color = "#8B4513"  # SaddleBrown
    pruned_color = "#d3d3d3"  # LightGrey
    canceled_color = "#a9a9a9"  # DarkGray
    failed_color = "LightCoral"
    resched_thickness = 4
    broken_color = "Red"
    overwrite_color = "SkyBlue"
    steps_color = "#00bbbb"
    vector_color = "#7193ff"  # pandas logo
    evicted_color = "#006666"
    #: the url to the architecture section explaining *graphtik* glossary,
    #: linked by legend.
    arch_url = "https://graphtik.readthedocs.io/en/latest/arch.html"

    ##########
    ## GRAPH

    kw_graph = {
        "graph_type": "digraph",
        "fontname": "italic",
        ## Whether to plot `curved/polyline edges
        #  <https://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:splines>`_
        #  BUT disabled due to crashes:
        #  https://gitlab.com/graphviz/graphviz/issues/1408
        # "splines": "ortho",
    }
    #: styles per plot-type
    kw_graph_plottable_type = {
        "FnOp": {},
        "Pipeline": {},
        "Network": {},
        "ExecutionPlan": {},
        "Solution": {},
    }
    #: For when type-name of :attr:`PlotArgs.plottable` is not found
    #: in :attr:`.kw_plottable_type` ( ot missing altogether).
    kw_graph_plottable_type_unknown = {}

    #: Attributes applying to all nodes with ``node [...]`` graphviz construct,
    #: append in graph only if non-empty.
    node_defaults = {"style": ["filled"], "fillcolor": "white"}
    #: Attributes applying to all edges with ``edge [...]`` graphviz construct,
    #: appended in graph only if non-empty.
    edge_defaults = {}

    ##########
    ## DATA node
    ##

    #: Keys to ignore from data styles & node-attrs,
    #: because they are handled internally by HTML-Label, and/or
    #: interact badly with that label.
    data_bad_html_label_keys = {"label"}

    #: Jinja2 params for the HTML-Table label
    kw_data_label = {}
    #: Reduce margins, since sideffects take a lot of space
    #: (default margin: x=0.11, y=0.055O)
    kw_data = {
        "shape": "rect",
        "fixedsize": "shape",
    }
    kw_data_inp = {}
    kw_data_out = {}
    kw_data_inp_only = {"shape": "invhouse", "tooltip": ["(input)"]}
    kw_data_out_only = {"shape": "house", "tooltip": ["(output)"]}
    kw_data_io = {
        "shape": "hexagon",
        "tooltip": ["(input+output)"],
    }
    kw_data_sideffect = {
        "color": Ref("sideffect_color"),
        "fontcolor": Ref("sideffect_color"),
    }
    kw_data_sideffected = {}
    kw_data_to_evict = {
        "color": Ref("evicted_color"),
        "style": ["filled", "dashed"],
        "tooltip": ["(to evict)"],
    }
    ##
    ## data STATE
    ##
    kw_data_pruned = {
        "fontcolor": Ref("pruned_color"),
        "color": Ref("pruned_color"),
        "tooltip": ["(pruned)"],
    }
    kw_data_in_solution = {
        "style": ["filled"],
        "fillcolor": Ref("fill_color"),
        "tooltip": [make_data_value_tooltip],
    }
    kw_data_in_solution_null = {
        "fillcolor": Ref("null_color"),
        "tooltip": ["(null-result)"],
    }
    kw_data_evicted = {"penwidth": "3", "tooltip": ["(evicted)"]}
    kw_data_overwritten = {
        "style": ["filled"],
        "fillcolor": Ref("overwrite_color"),
        "tooltip": [make_overwrite_tooltip],
    }
    kw_data_missing = {
        "fontcolor": Ref("canceled_color"),
        "color": Ref("canceled_color"),
        "tooltip": ["(missing-optional or canceled)"],
    }
    data_template = make_template(
        """\
        <<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
            <TR>
                {%- if solution and nx_item in solution and solution[nx_item]|attr('shape') -%}
                {%- set val = solution[nx_item] %}
                <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                    TITLE="{{- val | vector_info | eee -}}"
                    TARGET="_top"
                ><FONT FACE="monospace" COLOR="{{ vector_color | eee }}"><B>#</B></FONT></TD>
                {%- endif -%}

                {%- if val is defined and val.index is defined %}
                <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                    TITLE="{{- val.index | string | eee -}}"
                    TARGET="_top"
                ><FONT FACE="monospace" COLOR="{{ vector_color | eee }}"><B>R</B></FONT></TD>
                {%- endif -%}

                {%- if val is defined and val.columns is defined %}
                <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                    TITLE="{{- val.columns | string | eee -}}"
                    TARGET="_top"
                ><FONT FACE="monospace" COLOR="{{ vector_color | eee }}"><B>C</B></FONT></TD>
                {%- endif -%}

                {%- if steps and nx_item in steps %}
                <TD STYLE="rounded" CELLSPACING="2" CELLPADDING="4"
                {{- {
                    'BGCOLOR': step_bgcolor | eee,
                    'TITLE': step_tooltip | eee,
                    'HREF': step_url | hrefer | ee,
                    'TARGET': step_target | eee
                    } | xmlattr }}
                ><FONT FACE="monospace" COLOR="{{ step_color | eee }}"><B>
                        {{- steps.index(nx_item) -}}
                    </B></FONT></TD>
                {%- endif -%}

                <TD>
                {%- if nx_item | jsonp -%}
                    {%- for js_step in nx_item | jsonp -%}
                        {%- if loop.first -%}
                            {{- js_step | truncate -}}
                            /
                        {%- else -%}
                            {{- '\n' -}}
                            {{- '  ' * (loop.index - 1) -}}
                            +--
                            {{- js_step | truncate -}}
                            {%- if not loop.last -%}
                                /
                            {%- endif -%}
                        {%- endif -%}
                        <BR ALIGN="LEFT"/>
                    {%- endfor -%}
                {%- elif nx_item | sfxed -%}
                    {{- nx_item | sideffected | truncate -}}
                {%- else -%}
                    {{- nx_item | truncate -}}
                {%- endif -%}

                {%- if nx_item | sfxed -%}
                    <BR/>(<I>sfxed:</I> {{ nx_item | sfx_list | join(', ') | eee }})
                {%- endif -%}

            </TD>
            </TR>
        </TABLE>>
        """
    )

    ##########
    ## OPERATION node
    ##

    #: Keys to ignore from operation styles & node-attrs,
    #: because they are handled internally by HTML-Label, and/or
    #: interact badly with that label.
    op_bad_html_label_keys = {"shape", "label", "style"}
    #: props for operation node (outside of label))
    kw_op = {
        "name": lambda pa: quote_node_id(pa.nx_item.name),
        "shape": "plain",  # dictated by Graphviz docs
        # Set some base tooltip, or else, "TABLE" shown...
        "tooltip": [lambda pa: graphviz_html_string(pa.nx_item.name)],
    }

    kw_op_executed = {"fillcolor": Ref("fill_color")}
    kw_op_endured = {
        "penwidth": Ref("resched_thickness"),
        "style": ["dashed"],
        "tooltip": ["(endured)"],
        "badges": ["!"],
    }
    kw_op_rescheduled = {
        "penwidth": Ref("resched_thickness"),
        "style": ["dashed"],
        "tooltip": ["(rescheduled)"],
        "badges": ["?"],
    }
    kw_op_parallel = {"badges": ["|"]}
    kw_op_marshalled = {"badges": ["&"]}
    kw_op_returns_dict = {"badges": ["}"]}
    ##
    ## op STATE
    ##
    kw_op_pruned = {"color": Ref("pruned_color"), "fontcolor": Ref("pruned_color")}
    kw_op_prune_comment = {"op_tooltip": [make_op_prune_comment]}
    kw_op_failed = {
        "fillcolor": Ref("failed_color"),
        "tooltip": [make_template("{{ solution.executed[nx_item] if solution | ex }}")],
    }
    kw_op_canceled = {"fillcolor": Ref("canceled_color"), "tooltip": ["(canceled)"]}
    #: Operation styles may specify one or more "letters"
    #: in a `badges` list item, as long as the "letter" is contained in the dictionary
    #: below.
    op_badge_styles = {
        "badge_styles": {
            "!": {
                "tooltip": "endured",
                "bgcolor": "#04277d",
                "color": "white",
                "URL": "https://graphtik.readthedocs.io/en/latest/arch.html#term-endured",
                "target": "_top",
            },
            "?": {
                "tooltip": "rescheduled",
                "bgcolor": "#fc89ac",
                "color": "white",
                "URL": "https://graphtik.readthedocs.io/en/latest/arch.html#term-partial-outputs",
                "target": "_top",
            },
            "|": {
                "tooltip": "parallel",
                "bgcolor": "#b1ce9a",
                "color": "white",
                "URL": "https://graphtik.readthedocs.io/en/latest/arch.html#term-parallel-execution",
                "target": "_top",
            },
            "&": {
                "tooltip": "marshalled",
                "bgcolor": "#4e3165",
                "color": "white",
                "URL": "https://graphtik.readthedocs.io/en/latest/arch.html#term-marshalling",
                "target": "_top",
            },
            "}": {
                "tooltip": "returns_dict",
                "bgcolor": "#cc5500",
                "color": "white",
                "URL": "https://graphtik.readthedocs.io/en/latest/arch.html#term-returns-dictionary",
                "target": "_top",
            },
        }
    }
    #: Jinja2 params for the HTML-Table label, applied 1ST.
    kw_op_label = {
        "op_name": lambda pa: pa.nx_item.name,
        "fn_name": lambda pa: pa.nx_item
        and func_name(pa.nx_item.fn, mod=1, fqdn=1, human=1),
        "op_truncate": Ref("truncate_args"),
        "fn_truncate": Ref("truncate_args"),
        "op_url": Ref("op_url", default=None),
        "op_link_target": "_top",
        "fn_url": Ref("fn_url", default=None),
        "fn_link_target": "_top",
    }
    #: Jinja2 params for the HTML-Table label applied AT THE END.
    kw_op_label2 = {
        "op_tooltip": [make_op_tooltip],  # ensure
        "fn_tooltip": [make_fn_tooltip],
    }
    #: Try to mimic a regular `Graphviz`_ node attributes
    #: (see examples in ``test.test_plot.test_op_template_full()`` for params).
    #: TODO: fix jinja2 template is un-picklable!
    op_template = make_template(
        """\
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded"
          {{- {
          'BORDER': penwidth | ee,
          'COLOR': color | ee,
          'BGCOLOR': fillcolor | ee
          } | xmlattr -}}>
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left"
                  {{- {
                  'TOOLTIP': op_tooltip | truncate | eee,
                  'HREF': op_url | hrefer | ee,
                  'TARGET': op_link_target | eee
                  } | xmlattr }}
                >
                    {%- if fontcolor -%}<FONT COLOR="{{ fontcolor }}">{%- endif -%}
                    {{- '<B>OP:</B> <I>%s</I>' % op_name | truncate(*op_truncate[0], **op_truncate[1]) | ee if op_name -}}
                    {%- if fontcolor -%}</FONT>{%- endif -%}
                </TD>
                <TD BORDER="1" SIDES="b" ALIGN="right">
                {%- if badges or (steps and op_name in steps) -%}
                    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2" ALIGN="right">
                        <TR>
                        {%- if steps and op_name in steps %}
                            <TD STYLE="rounded" HEIGHT="22" WIDTH="14" FIXEDSIZE="true" VALIGN="BOTTOM"
                            {{- {
                            'BGCOLOR': step_bgcolor | eee,
                            'TITLE': step_tooltip | eee,
                            'HREF': step_url | hrefer | ee,
                            'TARGET': step_target | eee
                            } | xmlattr -}}
                            ><FONT FACE="monospace" COLOR="{{ step_color | eee }}"><B>
                                {{- steps.index(op_name) | eee -}}
                            </B></FONT></TD>
                        {%- endif -%}
                        {%- for badge in badges %}
                            <TD STYLE="rounded" HEIGHT="22" WIDTH="12" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="{{ badge_styles[badge].bgcolor
                                }}" TITLE="{{ badge_styles[badge].tooltip | eee
                                }}" HREF="{{ badge_styles[badge].URL | hrefer | ee
                                }}" TARGET="{{ badge_styles[badge].target }}"
                            ><FONT FACE="monospace" COLOR="{{ badge_styles[badge].color }}"><B>
                                {{- badge | eee -}}
                            </B></FONT></TD>
                        {%- endfor %}
                        </TR>
                    </TABLE>
                {%- endif -%}
                </TD>
            </TR>
            {%- if fn_name %}
            <TR>
                <TD COLSPAN="2" ALIGN="left"
                  {{- {
                  'TOOLTIP': fn_tooltip | truncate | eee,
                  'HREF': fn_url | hrefer | ee,
                  'TARGET': fn_link_target | eee
                  } | xmlattr }}
                  >
                    {%- if fontcolor -%}
                    <FONT COLOR="{{ fontcolor }}">
                    {%- endif -%}
                    <B>FN:</B> {{ fn_name | truncate(*fn_truncate[0], **fn_truncate[1]) | eee }}
                    {%- if fontcolor -%}
                    </FONT>
                    {%- endif -%}
                </TD>
            </TR>
            {%- endif %}
        </TABLE>>
        """
    )

    ##########
    ## EDGE
    ##

    kw_edge = {"tailport": "s", "headport": "n"}
    kw_edge_tail_op = {}
    kw_edge_head_op = {"arrowtail": "inv", "dir": "back"}
    kw_edge_optional = {"style": ["dashed"], "tooltip": ["(optional)"]}
    kw_edge_sideffect = {"color": Ref("sideffect_color")}
    kw_edge_implicit = {
        lambda pa: (
            "arrowtail" if isinstance(pa.nx_item[0], Operation) else "arrowhead"
        ): "obox",
        "dir": "both",
        "tooltip": ["(implicit)"],
        "fontcolor": Ref("sideffect_color"),
    }
    #: Added conditionally if `alias_of` found in edge-attrs.
    kw_edge_alias = {
        "fontsize": 11,  # default: 14
        "label": make_template(
            "<<I>(alias of)</I><BR/>{{ nx_attrs['alias_of'] | eee }}>"
        ),
    }
    #: Rendered if ``keyword`` exists in `nx_attrs`.
    kw_edge_mapping_keyword = {
        "fontsize": 11,  # default: 14
        "fontname": "italic",
        "tooltip": ["(mapped-fn-keyword)"],
        "label": make_template(
            "**{{ nx_attrs['keyword'] | eee if nx_attrs['keyword'] != nx_item[0] }}"
        ),
    }
    kw_edge_subdoc = {
        "color": Ref("subdoc_color"),
        "arrowtail": "odot",
        "dir": "back",
        "tailport": "se",
        "headport": "nw",
        "tooltip": ["(subdoc)"],
    }

    kw_edge_pruned = {"color": Ref("pruned_color"), "fontcolor": Ref("pruned_color")}
    kw_edge_rescheduled = {"style": ["dashed"]}
    kw_edge_endured = {"style": ["dashed"]}
    kw_edge_broken = {"tooltip": ["(partial-broken)"], "color": Ref("broken_color")}
    kw_edge_null_result = {"color": Ref("null_color"), "tooltip": ["(null-result)"]}

    ##########
    ## Other
    ##

    #: `None`:
    #:      plot just a badge with the order (a number) of each op/data
    #:      in steps (if contained);
    #: true:
    #:      plot also :term:`execution steps`, linking operations and evictions
    #:      with green dotted lines labeled with numbers denoting the execution order;
    #: false:
    #:      hide even op/data step order badges.
    show_steps = None
    #: `None`:
    #:  hide any parent/subdoc not related directly to some operation;
    #: true:
    #:  plot also :term:`hierarchical data` nodes not directly linked
    #:  to operations;
    #: false:
    #:  hide also parent-subdoc relation edges.
    show_chaindocs = None
    #: Available as jinja2 params for both data & operation templates.
    kw_step_badge = {
        "step_bgcolor": Ref("steps_color"),
        "step_color": "white",
        "step_url": "https://graphtik.readthedocs.io/en/latest/arch.html#term-steps",
        "step_target": "_top",
        "step_tooltip": "computation order",
        "vector_color": Ref("vector_color"),
    }
    #: step edges
    kw_step = {
        "style": "dotted",  # Note: Step styles are not *remerged*.`
        "color": Ref("steps_color"),
        "fontcolor": Ref("steps_color"),
        "fontname": "bold",
        "fontsize": 18,
        "arrowhead": "vee",
        "splines": True,
    }
    #: If ``'URL'``` key missing/empty, no legend icon included in plots.
    kw_legend = {
        "name": "legend",
        "shape": "component",
        "style": "filled",
        "fillcolor": "yellow",
        "URL": "https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg",
        "target": "_blank",
    }

    def __init__(self, *, _prototype: "Theme" = None, **kw):
        """
        Deep-copy public class-attributes of prototype and apply user-overrides,

        :param _prototype:
            Deep-copy its :func:`vars()`, and apply on top any `kw`

        Don't forget to resolve any :class:`Ref` on my self when used.
        """
        if _prototype is None:
            _prototype = type(self)
        else:
            assert isinstance(_prototype, Theme), _prototype

        class_attrs = Theme.theme_attributes(type(self))
        class_attrs.update(kw)
        vars(self).update(remap(class_attrs))

    @staticmethod
    def theme_attributes(obj) -> dict:
        """Extract public data attributes of a :class:`Theme` instance."""
        return {
            k: v
            for k, v in vars(obj).items()
            if not callable(v) and not k.startswith("_")
        }

    def withset(self, **kw) -> "Theme":
        """Returns a deep-clone modified by `kw`."""
        return type(self)(_prototype=self, **kw)


def remerge(*containers, source_map: list = None):
    """
    Merge recursively dicts and extend lists with :func:`boltons.iterutils.remap()` ...

    screaming on type conflicts, ie, a list needs a list, etc, unless one of them
    is None, which is ignored.

    :param containers:
        a list of dicts or lists to merge; later ones take precedence
        (last-wins).
        If `source_map` is given, these must be 2-tuples of ``(name: container)``.
    :param source_map:
        If given, it must be a dictionary, and `containers` arg must be 2-tuples
        like ``(name: container)``.
        The `source_map` will be populated with mappings between path and the name
        of the container it came from.

        .. Warning::
            if source_map given, the order of input dictionaries is NOT preserved
            is the results  (important if your code rely on PY3.7 stable dictionaries).

    :return:
        returns a new, merged top-level container.

    - Adapted from https://gist.github.com/mahmoud/db02d16ac89fa401b968
      but for lists and dicts only, ignoring Nones and screams on incompatible types.
    - Discusson in: https://gist.github.com/pleasantone/c99671172d95c3c18ed90dc5435ddd57


    **Example**

    >>> defaults = {
    ...     'subdict': {
    ...         'as_is': 'hi',
    ...         'overridden_key1': 'value_from_defaults',
    ...         'overridden_key1': 2222,
    ...         'merged_list': ['hi', {'untouched_subdict': 'v1'}],
    ...     }
    ... }

    >>> overrides = {
    ...     'subdict': {
    ...         'overridden_key1': 'overridden value',
    ...         'overridden_key2': 5555,
    ...         'merged_list': ['there'],
    ...     }
    ... }

    >>> from graphtik.plot import remerge
    >>> source_map = {}
    >>> remerge(
    ...     ("defaults", defaults),
    ...     ("overrides", overrides),
    ...     source_map=source_map)
     {'subdict': {'as_is': 'hi',
                  'overridden_key1': 'overridden value',
                  'merged_list': ['hi', {'untouched_subdict': 'v1'}, 'there'],
                  'overridden_key2': 5555}}
    >>> source_map
    {('subdict', 'as_is'): 'defaults',
     ('subdict', 'overridden_key1'): 'overrides',
     ('subdict', 'merged_list'):  ['defaults', 'overrides'],
     ('subdict',): 'overrides',
     ('subdict', 'overridden_key2'): 'overrides'}
    """

    ret = None

    def remerge_enter(path, key, old_parent):
        new_parent, new_items = default_enter(path, key, old_parent)
        if new_items is False:
            # Drop strings and non-iterables.
            return new_parent, new_items

        if ret and not path and key is None:
            new_parent = ret
        try:
            # TODO: type check?
            new_parent = get_path(ret, path + (key,))
        except KeyError:
            pass

        if new_parent is not None:
            if (isinstance(old_parent, list) ^ isinstance(new_parent, list)) or (
                isinstance(old_parent, dict) ^ isinstance(new_parent, dict)
            ):
                raise TypeError(
                    f"Incompatible types {type(old_parent)} <-- {type(new_parent)}!"
                )
            if isinstance(old_parent, list):
                new_parent.extend(old_parent)
                # lists are purely additive, stop recursion.
                new_items = ()
        else:
            if isinstance(old_parent, list):
                new_parent = [*old_parent]
                # lists are purely additive, stop recursion.
                new_items = ()

        return new_parent, new_items

    def remerge_exit(path, key, old_parent, new_parent, items):
        if new_parent is None:
            return old_parent  # FIXME: not cloned?
        return default_exit(path, key, old_parent, new_parent, items)

    if source_map is not None:

        def remerge_visit(path, key, value):
            full_path = path + (key,)
            if isinstance(value, list):
                old = source_map.get(full_path)
                if old:
                    old.append(t_name)
                else:
                    source_map[full_path] = [t_name]
            else:
                source_map[full_path] = t_name
            return True

        for t_name, cont in containers:
            ret = remap(
                cont, enter=remerge_enter, visit=remerge_visit, exit=remerge_exit
            )

    else:
        for cont in containers:
            ret = remap(cont, enter=remerge_enter, exit=remerge_exit)

    return ret


#: Any nx-attributes starting with this prefix
#: are appended verbatim as graphviz attributes,
#: by :meth:`.stack_user_style()`.
USER_STYLE_PREFFIX = "graphviz."


class StylesStack(NamedTuple):
    """
    A mergeable stack of dicts preserving provenance and :term:`style expansion`.

    The :meth:`.merge()` method joins the collected stack of styles into a single
    dictionary, and if DEBUG (see :func:`.remerge()`) insert their provenance in
    a ``'tooltip'`` attribute;
    Any lists are merged (important for multi-valued `Graphviz`_ attributes
    like ``style``).

    Then they are :meth:`expanded <expand>`.
    """

    #: current item's plot data with at least :attr:`.PlotArgs.theme` attribute. ` `
    plot_args: PlotArgs
    #: A list of 2-tuples: (name, dict) containing the actual styles
    #: along with their provenance.
    named_styles: List[Tuple[str, dict]]
    #: When true, keep merging despite expansion errors.
    ignore_errors: bool = False

    def add(self, name, kw=...):
        """
        Adds a style by name from style-attributes, or provenanced explicitly, or fail early.

        :param name:
            Either the provenance name when the `kw` styles is given,
            OR just an existing attribute of :attr:`style` instance.
        :param kw:
            if given and is None/empty, ignored.
        """
        if kw is ...:
            kw = getattr(self.plot_args.theme, name)  # will scream early
        if kw:
            self.named_styles.append((name, kw))

    def stack_user_style(self, nx_attrs: dict, skip=()):
        """
        Appends keys in `nx_attrs` starting with :data:`USER_STYLE_PREFFIX` into the stack.
        """
        if nx_attrs:
            kw = {
                k[len(USER_STYLE_PREFFIX) :]: v
                for k, v in nx_attrs.items()
                if k.startswith(USER_STYLE_PREFFIX) and k not in skip
            }

            self.add("user-overrides", kw)

    def expand(self, style: dict) -> dict:
        """
        Apply :term:`style expansion`\\s on an already merged style.

        .. theme-expansions-start

        - Call any *callables* found as keys, values or the whole style-dict,
          passing in the current :class:`plot_args <.PlotArgs>`, and replace those
          with the callable's result (even more flexible than templates).

        - Resolve any :class:`.Ref` instances, first against the current *nx_attrs*
          and then against the attributes of the current theme.

        - Render jinja2 templates with template-arguments all attributes of :class:`plot_args
          <.PlotArgs>` instance in use, (hence much more flexible than :class:`.Ref`).

        - Any Nones results above are discarded.

        - Workaround pydot/pydot#228 pydot-cstor not supporting styles-as-lists.

        - Merge tooltip & tooltip lists.

        .. theme-expansions-end
        """

        def expand_visitor(path, k, v):
            """
            A :func:`.remap()` visit-cb to resolve :class:`.Ref`, render templates & call callables.
            """
            visit_type = type(v).__name__
            try:
                expanded = False
                if callable(k):
                    k = k(self.plot_args)
                    expanded = True

                if isinstance(v, Ref):
                    v = v.resolve(self.plot_args.nx_attrs, self.plot_args.theme)
                    expanded = True
                elif isinstance(v, jinja2.Template):
                    v = v.render(**self.plot_args._asdict())
                    expanded = True
                elif callable(v):
                    v = v(self.plot_args)
                    expanded = True

                return (
                    False
                    if k is None or v in (..., None)
                    else (k, v)
                    if expanded
                    else True
                )
            except Exception as ex:
                path = f'{"/".join(path)}/{k}'
                msg = f"Failed expanding {visit_type} @ '{path}' = {v!r} due to: {type(ex).__name__}({ex})"
                if self.ignore_errors:
                    log.warning(msg)
                    return False
                else:
                    raise ValueError(msg) from ex

        style = remap(style, visit=expand_visitor)

        ## Merge Graphviz-style props
        #
        graphviz_style = style.get("style")
        if isinstance(graphviz_style, (list, tuple)):
            ## FIXME: support only plain-strings as graphviz-styles.
            graphviz_style = ",".join(str(i) for i in iset(graphviz_style))
            if "," in graphviz_style:
                graphviz_style = f'"{graphviz_style}"'
            style["style"] = graphviz_style

        ## Merge tooltips
        #
        for k in style.keys():
            if k.endswith("tooltip"):
                tooltips = style[k]
                if isinstance(tooltips, (list, tuple)):
                    style[k] = "\n".join(tooltips)

        return style

    def merge(self, debug=None) -> dict:
        """
        Recursively merge :attr:`named_styles` and :meth:`.expand` the result style.

        :param debug:
            When not `None`, override :ref:`debug`;  when enabled,
            tooltips are overridden with provenance & nx_attrs.

        :return:
            the merged styles
        """
        # FIXME: too clever, avoided re-writting own merge, and now errors cannot debug :-(
        if not self.named_styles:
            return {}

        if (debug is None and is_debug()) or debug:
            from itertools import count
            from pprint import pformat

            styles_provenance = {}
            style = remerge(*self.named_styles, source_map=styles_provenance)

            ## Append debug info
            #
            provenance_str = pformat(
                {
                    ".".join(
                        # resolve any callable theme-keys
                        str(k_step(self.plot_args)) if callable(k_step) else k_step
                        for k_step in k
                    ): v
                    for k, v in styles_provenance.items()
                },
                indent=2,
            )
            tooltip = f"- styles: {provenance_str}\n- extra_attrs: {pformat(self.plot_args.nx_attrs)}"
            style["tooltip"] = graphviz_html_string(tooltip)

        else:
            style = remerge(*(style_dict for _name, style_dict in self.named_styles))
            style = remerge(
                *(
                    ## Expand also callables producing the whole style-dict
                    style_dict(self.plot_args) if callable(style_dict) else style_dict
                    for _name, style_dict in self.named_styles
                )
            )
        assert isinstance(style, dict), (style, self.named_styles)

        return self.expand(style)


class Plotter:
    """
    a :term:`plotter` renders diagram images of :term:`plottable`\\s.

    .. attribute:: Plotter.default_theme

        The :ref:`customizable <plot-customizations>` :class:`.Theme` instance
        controlling theme values & dictionaries for plots.
    """

    def __init__(self, theme: Theme = None, **styles_kw):
        self.default_theme: Theme = theme or Theme(**styles_kw)

    def with_styles(self, **kw) -> "Plotter":
        """
        Returns a cloned plotter with a deep-copied theme modified as given.

        See also :meth:`Theme.withset()`.
        """
        return type(self)(self.default_theme.withset(**kw))

    def _new_styles_stack(self, plot_args: PlotArgs, ignore_errors=None):
        return StylesStack(plot_args, [], ignore_errors=ignore_errors)

    def plot(self, plot_args: PlotArgs):
        dot = self.build_pydot(plot_args)
        return self.render_pydot(dot, **plot_args.kw_render_pydot)

    def build_pydot(self, plot_args: PlotArgs) -> pydot.Dot:
        """
        Build a |pydot.Dot|_ out of a Network graph/steps/inputs/outputs and return it

        to be fed into `Graphviz`_ to render.

        See :meth:`.Plottable.plot()` for the arguments, sample code, and
        the legend of the plots.
        """
        if plot_args.graph is None:
            raise ValueError("At least `graph` to plot must be given!")

        plot_args = plot_args.with_defaults(theme=self.default_theme)
        theme = plot_args.theme

        graph, steps = self._skip_no_plot_nodes(plot_args.graph, plot_args.steps)
        plot_args = plot_args._replace(graph=graph, steps=steps)

        styles = self._new_styles_stack(plot_args._replace(nx_attrs=graph.graph))

        styles.add("kw_graph")

        plottable_type = type(plot_args.plottable).__name__.split(".")[-1]
        styles.add(
            f"kw_graph_plottable_type-{plottable_type}",
            theme.kw_graph_plottable_type.get(
                plottable_type, theme.kw_graph_plottable_type_unknown
            ),
        )

        styles.stack_user_style(graph.graph)

        dot = pydot.Dot(**styles.merge())

        if plot_args.name:
            dot.set_name(as_identifier(plot_args.name))

        ## Item-args for nodes, edges & steps spring off of this.
        plot_args = plot_args._replace(dot=dot, clustered={})

        ## Graph's node/edge defaults
        #
        if theme.node_defaults:
            node_styles = self._new_styles_stack(plot_args)
            node_styles.add("node_defaults")
            dot.add_node(pydot.Node("node", **node_styles.merge()))
        if theme.edge_defaults:
            edge_styles = self._new_styles_stack(plot_args)
            edge_styles.add("edge_defaults")
            dot.add_node(pydot.Node("edge", **edge_styles.merge()))

        # To hide edges from unused chain-docs.
        hidden = set()

        ## NODES
        #
        for nx_node, data in graph.nodes.data(True):
            if not theme.show_chaindocs and not is_nx_node_dependent(graph, nx_node):
                hidden.add(nx_node)
                continue

            node_plot_args = plot_args._replace(nx_item=nx_node, nx_attrs=data)
            dot_node = self._make_node(node_plot_args)
            node_plot_args = node_plot_args._replace(dot_item=dot_node)

            self._append_or_cluster_node(node_plot_args)

        ## EDGES
        #
        for src, dst, data in graph.edges.data(True):
            if (
                src in hidden
                or dst in hidden
                or (
                    data.get("subdoc")
                    and not theme.show_chaindocs
                    and theme.show_chaindocs is not None
                )
            ):
                continue

            edge_plot_args = plot_args._replace(nx_item=(src, dst), nx_attrs=data)
            dot.add_edge(self._make_edge(edge_plot_args))

        ## Draw steps sequence, if it's worth it.
        #
        if steps and theme.show_steps and len(steps) > 1:
            it1 = iter(steps)
            it2 = iter(steps)
            next(it2)
            for i, (src, dst) in enumerate(zip(it1, it2), 1):
                src_name = get_node_name(src)
                dst_name = get_node_name(dst)
                styles = self._new_styles_stack(plot_args)

                styles.add("kw_step")
                edge = pydot.Edge(
                    src=src_name, dst=dst_name, label=str(i), **styles.merge()
                )
                dot.add_edge(edge)

        self._add_legend_icon(plot_args)

        return dot

    def _make_node(self, plot_args: PlotArgs) -> pydot.Node:
        """
        Override it to customize nodes, e.g. add doc URLs/tooltips, solution tooltips.

        :param plot_args:
            must have, at least, a `graph`, `nx_item` & `nx-attrs`, as returned
            by :class:`.Plottable.prepare_plot_args()`

        :return:
            the update `plot_args` with the new :attr:`.PlotArgs.dot_item`

        Currently it does the folllowing on operations:

        1. Set fn-link to `fn` documentation url (from edge_props['fn_url'] or discovered).

           .. Note::
               - SVG tooltips may not work without URL on PDFs:
                 https://gitlab.com/graphviz/graphviz/issues/1425

               - Browsers & Jupyter lab are blocking local-urls (e.g. on SVGs),
                 see tip in :term:`plottable`.

        2. Set tooltips with the fn-code for operation-nodes.

        3. Set tooltips with the solution-values for data-nodes.
        """
        theme: Theme = plot_args.theme
        graph = plot_args.graph
        nx_node = plot_args.nx_item
        node_attrs = plot_args.nx_attrs
        (plottable, _, _, steps, inputs, outputs, solution, *_) = plot_args

        label_styles = self._new_styles_stack(plot_args)
        if isinstance(nx_node, str):  # DATA
            label_styles.add("kw_data_label")
            label_styles.stack_user_style(node_attrs)
            label_styles.add("kw_step_badge")

            styles = self._new_styles_stack(plot_args)

            styles.add("kw_data")

            ## Data-kind
            #
            styles.add("node-name", {"name": quote_node_id(nx_node)})

            is_inp = inputs and nx_node in inputs
            is_out = outputs and nx_node in outputs
            if is_inp:
                styles.add("kw_data_inp")
                if not is_out:
                    styles.add("kw_data_inp_only")
            if is_out:
                styles.add("kw_data_out")
                if not is_inp:
                    styles.add("kw_data_out_only")
            if is_inp and is_out:
                styles.add("kw_data_io")

            if is_sfx(nx_node):
                styles.add("kw_data_sideffect")
                if is_sfxed(nx_node):
                    styles.add("kw_data_sideffected")

            ## Data-state
            #
            is_pruned = (
                hasattr(plottable, "dag") and nx_node not in plottable.dag.nodes
            ) or (hasattr(solution, "dag") and nx_node not in solution.dag.nodes)
            if is_pruned:
                graph.nodes[nx_node]["_pruned"] = True  # Signal to edge-plotting.
                styles.add("kw_data_pruned")
                #
                #  Note that Pruned "sibling" data due to asked outputs,
                #  are also evicted afterwards.
                #  So the cases below must still run.

            if steps and nx_node in steps:
                styles.add("kw_data_to_evict")

            if solution is not None:
                if nx_node in solution:
                    styles.add("kw_data_in_solution")
                    val = solution[nx_node]
                    for check in _null_checks:
                        try:
                            if check(val):
                                styles.add("kw_data_in_solution_null")
                                break
                        except Exception:
                            log.debug(
                                "Skipping null-check error on %s", val, exc_info=1
                            )

                    if nx_node in getattr(solution, "overwrites", ()):
                        styles.add("kw_data_overwritten")

                elif nx_node in steps:
                    styles.add("kw_data_evicted")
                elif not is_pruned and not is_sfx(nx_node):
                    styles.add("kw_data_missing")

            styles.stack_user_style(node_attrs, skip=theme.data_bad_html_label_keys)

            styles.add(
                "data_label_template",
                {
                    "label": _render_template(
                        theme.data_template,
                        **plot_args._asdict(),
                        **label_styles.merge(),
                    )
                },
            )

        else:  # OPERATION
            label_styles.add("kw_op_label")

            ## Op-kind
            #
            if first_solid(
                is_reschedule_operations(), getattr(nx_node, "rescheduled", None)
            ):
                label_styles.add("kw_op_rescheduled")
            if first_solid(is_endure_operations(), getattr(nx_node, "endured", None)):
                label_styles.add("kw_op_endured")
            if first_solid(is_parallel_tasks(), getattr(nx_node, "parallel", None)):
                label_styles.add("kw_op_parallel")
            if first_solid(is_marshal_tasks(), getattr(nx_node, "marshalled", None)):
                label_styles.add("kw_op_marshalled")
            if getattr(nx_node, "returns_dict", None):
                label_styles.add("kw_op_returns_dict")

            ## Op-state
            #

            if steps and nx_node not in steps:
                label_styles.add("kw_op_pruned")
            label_styles.add("kw_op_prune_comment")

            if solution:
                if hasattr(solution, "is_failed") and solution.is_failed(nx_node):
                    label_styles.add("kw_op_failed")
                elif nx_node in getattr(solution, "executed", ()):
                    label_styles.add("kw_op_executed")
                elif nx_node in getattr(solution, "canceled", ()):
                    label_styles.add("kw_op_canceled")

            label_styles.stack_user_style(node_attrs)

            # TODO: Optimize and merge badge_styles once!
            label_styles.add("op_badge_styles")
            label_styles.add("kw_step_badge")
            label_styles.add("kw_op_label2")

            styles = self._new_styles_stack(plot_args)
            styles.add("kw_op")

            # Decide if steps badge would be shown.
            #
            if not steps or not theme.show_steps and theme.show_steps is not None:
                steps = None
            else:
                steps = steps and list(yield_node_names(steps))

            styles.add(
                "op_label_template",
                {
                    "label": _render_template(
                        theme.op_template, steps=steps, **label_styles.merge()
                    )
                },
            )

            # Exclude Graphviz node attributes interacting badly with HTML-Labels.
            styles.stack_user_style(node_attrs, skip=theme.op_bad_html_label_keys)

        return pydot.Node(**styles.merge())

    def _build_cluster_path(self, plot_args: PlotArgs, *path) -> None:
        """Return the last cluster in the path created"""
        clustered = plot_args.clustered
        root = plot_args.dot
        for step in path:
            cluster = clustered.get(step)
            if not cluster:
                cluster = clustered[step] = pydot.Cluster(
                    step, label=graphviz_html_string(step)
                )
                root.add_subgraph(cluster)
            root = cluster

        return root

    def _append_or_cluster_node(self, plot_args: PlotArgs) -> None:
        """Add dot-node in dot now, or "cluster" it, to be added later."""

        root = plot_args.dot
        clusters = plot_args.clusters
        nx_node = plot_args.nx_item

        # By default, node-name clustering is enabled.
        #
        if clusters is None:
            clusters = True

        ## First check if user has asked explicit clusters,
        #  otherwise split by node's name parts.
        #
        cluster_path = None
        if clusters:
            if isinstance(clusters, abc.Mapping):
                cluster_path = clusters.get(nx_node)
                if cluster_path:
                    cluster_path = clusters[nx_node].split(".")
            else:
                cluster_path = get_node_name(nx_node, raw=1).split(".")[:-1]

            root = self._build_cluster_path(plot_args, *cluster_path or ())
        root.add_node(plot_args.dot_item)

    def _make_edge(self, plot_args: PlotArgs) -> pydot.Edge:
        """Override it to customize edge appearance."""
        graph, solution = plot_args.graph, plot_args.solution
        (src, dst), edge_attrs = plot_args.nx_item, plot_args.nx_attrs
        src_name, dst_name = get_node_name(src), get_node_name(dst)

        ## Edge-kind
        #
        styles = self._new_styles_stack(plot_args)

        styles.add("kw_edge")
        if isinstance(src, Operation):
            styles.add("kw_edge_tail_op")
        if isinstance(dst, Operation):
            styles.add("kw_edge_head_op")
        if edge_attrs.get("optional"):
            styles.add("kw_edge_optional")
        if edge_attrs.get("sideffect"):
            styles.add("kw_edge_sideffect")
        if edge_attrs.get("implicit"):
            styles.add("kw_edge_implicit")
        if edge_attrs.get("alias_of"):
            styles.add("kw_edge_alias")
        if edge_attrs.get("keyword"):
            styles.add("kw_edge_mapping_keyword")
        if edge_attrs.get("subdoc"):
            styles.add("kw_edge_subdoc")

        if getattr(src, "rescheduled", None):
            styles.add("kw_edge_rescheduled")
        if getattr(src, "endured", None):
            styles.add("kw_edge_endured")

        ## Edge-state
        #
        if graph.nodes[src].get("_pruned") or graph.nodes[dst].get("_pruned"):
            styles.add("kw_edge_pruned")
        if (
            hasattr(solution, "dag")
            and hasattr(solution, "plan")
            and (src, dst) not in solution.dag.edges
            and (src, dst) in solution.plan.dag.edges
        ):
            styles.add("kw_edge_broken")
        if (
            isinstance(src, Operation)
            and not is_sfx(dst)
            and hasattr(solution, "executed")
            and src in solution.executed
        ):
            result = solution.executed.get(src)
            try:
                if dst in result and result[dst] is None:
                    styles.add("kw_edge_null_result")
            except Exception:
                pass  # was exception

        styles.stack_user_style(edge_attrs)
        edge = pydot.Edge(src=src_name, dst=dst_name, **styles.merge())

        return edge

    def _add_legend_icon(self, plot_args: PlotArgs):
        """Optionally add an icon to diagrams linking to legend (if url given)."""
        styles = self._new_styles_stack(plot_args)
        styles.add("kw_legend")
        kw = styles.merge()
        if kw.get("URL"):
            plot_args.dot.add_node(pydot.Node(**kw))

    def _skip_no_plot_nodes(
        self, graph: nx.Graph, steps: Collection
    ) -> Tuple[nx.Graph, Collection]:
        """
        Drop any nodes, steps & edges with "no_plot" attribute.

        :param graph:
            modifies it(!) by removing those items
        """
        nodes_to_del = {n for n, no_plot in graph.nodes.data("no_plot") if no_plot}
        graph.remove_nodes_from(nodes_to_del)
        if steps:
            steps = [s for s in steps if s not in nodes_to_del]
        graph.remove_edges_from(
            [(src, dst) for src, dst, no_plot in graph.edges.data("no_plot") if no_plot]
        )

        return graph, steps

    def render_pydot(self, dot: pydot.Dot, filename=None, jupyter_render: str = None):
        """
        Render a |pydot.Dot|_ instance with `Graphviz`_ in a file and/or in a matplotlib window.

        :param dot:
            the pre-built |pydot.Dot|_ instance
        :param str filename:
            Write a file or open a `matplotlib` window.

            - If it is a string or file, the diagram is written into the file-path

              Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
              call :func:`.plot.supported_plot_formats()` for more.

            - If it IS `True`, opens the  diagram in a  matplotlib window
              (requires `matplotlib` package to be installed).

            - If it equals `-1`, it mat-plots but does not open the window.

            - Otherwise, just return the ``pydot.Dot`` instance.

            :seealso: :attr:`.PlotArgs.filename`
        :param jupyter_render:
            a nested dictionary controlling the rendering of graph-plots in Jupyter cells.
            If `None`, defaults to :data:`default_jupyter_render`;  values for
            ommitted  sub-keys are taken also from the above dict - pass an empty dict(``{}``)
            if that is not desirable (eg. to revert to panZoom's library's defaults).

            You may modify those
            in place and they will apply for all future calls (see :ref:`jupyter_rendering`).

            You may increase the height of the SVG cell output with
            something like this::

                from graphtik.plot import default_jupyter_render
                plottable.plot(jupyter_render={
                    **default_jupyter_render,
                    "svg_element_styles": "height: 600px; width: 100%"
                })
        :return:
            the matplotlib image if ``filename=-1``, or the given `dot` annotated with any
            jupyter-rendering configurations given in `jupyter_render` parameter.

        See :meth:`.Plottable.plot()` for sample code.
        """
        if isinstance(filename, (bool, int)):
            ## Display graph via matplotlib
            #
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt

            png = dot.create_png()
            sio = io.BytesIO(png)
            img = mpimg.imread(sio)
            if filename != -1:
                plt.imshow(img, aspect="equal")
                plt.show()

            return img

        elif filename:
            ## Save plot
            #
            formats = supported_plot_formats()
            _basename, ext = os.path.splitext(filename)
            if not ext.lower() in formats:
                raise ValueError(
                    "Unknown file format for saving graph: %s"
                    "  File extensions must be one of: %s" % (ext, " ".join(formats))
                )

            dot.write(filename, format=ext.lower()[1:])

        ## Propagate any properties for rendering in Jupyter cells.
        dot._jupyter_render = jupyter_render

        return dot

    def legend(
        self, filename=None, jupyter_render: Mapping = None, theme: Theme = None
    ):
        """
        Generate a legend for all plots (see :meth:`.Plottable.plot()` for args)

        See :meth:`Plotter.render_pydot` for the rest arguments.
        """
        if theme is None:
            theme = self.default_theme

        # Expand all Theme styles
        #
        plot_args = PlotArgs(
            name="legend", plotter=self, theme=theme, solution={}, nx_attrs={}
        )
        ss = self._new_styles_stack(plot_args, ignore_errors=True)
        theme_styles = {
            attr: ss.expand(d) if isinstance(d, dict) else d
            for attr, d in Theme.theme_attributes(theme).items()
        }

        ss = self._new_styles_stack(plot_args)
        ss.add("kw_data_to_evict")
        ss.add("kw_data_evicted")
        ss.add(
            "manual",
            {
                "shape": "rect",
                "tooltip": "Instruction step to erase data from solution, to save memory.",
                "URL": graphviz_html_string(f"{theme.arch_url}#term-evictions"),
            },
        )
        theme_styles["evicted"] = str(pydot.Node(**ss.merge()))

        ss = self._new_styles_stack(plot_args)
        ss.add("kw_data_sideffect")
        ss.add(
            "manual",
            {
                "shape": "rect",
                "label": "sideffect",
                "tooltip": "Fictive data not consumed/produced by operation functions.",
                "URL": graphviz_html_string(f"{theme.arch_url}#term-sideffects"),
            },
        )
        theme_styles["token"] = str(pydot.Node(**ss.merge()))
        theme_styles["ver"] = __version__

        ## From https://stackoverflow.com/questions/3499056/making-a-legend-key-in-graphviz
        # Render it manually with these python commands, and remember to update result in git:
        #
        #   from graphtik import plot
        #   plot.legend('docs/source/images/GraphtikLegend.svg')
        dot_text = (
            """
        digraph {
            rankdir=TB;
            subgraph cluster_legend {
            label="Graphtik Legend %(ver)s";
            edge [target=_top];
            node [target=_top];

            Dependencies   [shape=plaintext fontsize=16 fontname="bold italic"];
            Dependencies -> needs [dir=back arrowtail=inv
                                tooltip="Compulsory dependency from src `needs` data --> dst Operation."
                                URL="%(arch_url)s#term-needs"];
            needs [color=invis];
            needs -> provides     [tooltip="Compulsory dependency from src Operation --> dst `provides` data"];
            provides [color=invis];
            provides -> optional     [style=dashed label="**kw" tooltip="A `need` passed in as kw-argument underlying fn."];
            optional    [color=invis label="optional\npartial out"
                        tooltip="Target operation may run without source `need` OR source operation may not `provide` target data."
                        URL="%(arch_url)s#term-needs"];
            optional    -> implicit [dir=both arrowtail=inv arrowhead=obox];
            implicit    [color=invis label="implicit"
                        tooltip="A `need` not fed into/out of the function but expected to exist in the `solution`."
                        URL="%(arch_url)s#term-implicit"];
            implicit    -> sideffect    [color=blue];
            sideffect   [color=invis
                        tooltip="Fictive data not consumed/produced by operation functions."
                        URL="%(arch_url)s#term-sideffects"];
            sideffect   -> null [color="%(null_color)s"]
            null       [color=invis
                        tooltip="Result for this output was null (probably error)"
                        URL="#"];
            null    -> broken       [color="red" style=dashed]
            broken      [color=invis
                        tooltip="Target data was not `provided` by source operation due to failure / partial-outs."
                        URL="%(arch_url)s#term-partial outputs"];
            broken   -> subdoc [color="%(subdoc_color)s" arrowtail=odot dir=back]
            subdoc      [color=invis
                        tooltip="Parent --> child document, accessed with `json pointer` dependencies."
                        URL="%(arch_url)s#term-jsonp"];
            subdoc  -> sequence [color="%(steps_color)s" penwidth=4 style=dotted
                                        arrowhead=vee label=1 fontcolor="%(steps_color)s"];

            sequence    [color=invis penwidth=4 label="execution\nsequence"
                        tooltip="Sequence of execution steps."
                        URL="%(arch_url)s#term-execution-steps"];


            DataLabel   [shape=plaintext fontsize=16 fontname="bold italic"
                            label=Data];
            data    [shape=rect fixedsize=shape
                    tooltip="Any data not given or asked."
                    URL="%(arch_url)s#term-graph"];
            input   [shape=invhouse fixedsize=shape
                    tooltip="Solution value given into the computation."
                    URL="%(arch_url)s#term-inputs"];
            output  [shape=house fixedsize=shape
                    tooltip="Solution value asked from the computation."
                    URL="%(arch_url)s#term-outputs"];
            inp_out [shape=hexagon label="inp+out" fixedsize=shape
                    tooltip="Data both given and asked."
                    URL="%(arch_url)s#term-dependency"];
            evicted %(evicted)s
            token %(token)s
            sol     [shape=rect fixedsize=shape
                    style=filled fillcolor="%(fill_color)s" label="with value"
                    tooltip="Data contained in the solution."
                    URL="%(arch_url)s#term-solution"];
            nullval [shape=rect fixedsize=shape
                    style=filled fillcolor="%(null_color)s" label="null"
                    tooltip="Data in solution but is None (probable error)."
                    URL="%(arch_url)s#term-solution"];
            overwrite [shape=rect fixedsize=shape
                    style=filled fillcolor=SkyBlue
                    tooltip="More than 1 values exist in solution with this name."
                    URL="%(arch_url)s#term-overwrite"];
            DataLabel -> data -> input -> output -> inp_out -> sfx -> evicted -> sol -> nullval -> overwrite [style=invis];



            Operations   [shape=plaintext fontsize=16 fontname="bold italic"];
            operation   [shape=plain
                        label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="ROUNDED">
                          <TR><TD BORDER="1" SIDES="B">operation</TD></TR>
                          <TR><TD><I>function</I></TD></TR>
                        </TABLE>>
                        tooltip="A function with needs & provides."
                        URL="%(arch_url)s#term-operation"];
            pruned    [shape=plain color="%(pruned_color)s" fontcolor="%(pruned_color)s"
                        label=<<TABLE CELLBORDER="0" STYLE="ROUNDED">
                          <TR><TD>pruned</TD></TR></TABLE>>
                        tooltip="Operation excluded from plan."
                        URL="%(arch_url)s#term-prune"];
            executed    [shape=plain
                        label=<<TABLE CELLBORDER="0" STYLE="ROUNDED" BGCOLOR="wheat">
                          <TR><TD>executed</TD></TR></TABLE>>
                        tooltip="Operation executed successfully."
                        URL="%(arch_url)s#term-solution"];
            failed      [shape=plain
                        label=<<TABLE CELLBORDER="0" STYLE="ROUNDED" BGCOLOR="LightCoral">
                          <TR><TD>failed</TD></TR></TABLE>>
                        tooltip="Failed operation - downstream ops will cancel."
                        URL="%(arch_url)s#term-endurance"];
            rescheduled [shape=plain
                        label=<<TABLE border="4" CELLBORDER="0" STYLE="ROUNDED">
                          <TR><TD>endured<br/>rescheduled</TD></TR></TABLE>>
                        tooltip="Operation may fail or provide partial outputs so `net` must reschedule."
                        URL="%(arch_url)s#term-reschedulling"];
            canceled    [shape=plain
                        label=<<TABLE CELLBORDER="0" STYLE="ROUNDED" BGCOLOR="Grey">
                          <TR><TD>canceled</TD></TR></TABLE>>
                        tooltip="Canceled operation due to failures or partial outputs upstream."
                        URL="%(arch_url)s#term-reschedule"];
            Operations -> operation -> pruned -> executed -> failed -> rescheduled -> canceled [style=invis];
            }
        }
        """
            % theme_styles
        )

        dot = pydot.graph_from_dot_data(dot_text)[0]
        # cluster = pydot.Cluster("Graphtik legend", label="Graphtik legend")
        # dot.add_subgraph(cluster)

        # nodes = dot.Node()
        # cluster.add_node("operation")

        return self.render_pydot(dot, filename=filename, jupyter_render=jupyter_render)


def legend(
    filename=None,
    show=None,
    jupyter_render: Mapping = None,
    plotter: Plotter = None,
):
    """
    Generate a legend for all plots (see :meth:`.Plottable.plot()` for args)

    :param plotter:
        override the :term:`active plotter`
    :param show:
        .. deprecated:: v6.1.1
            Merged with `filename` param (filename takes precedence).

    See :meth:`Plotter.render_pydot` for the rest arguments.
    """
    if show:
        import warnings

        warnings.warn(
            "Argument `plot` has merged with `filename` and will be deleted soon.",
            DeprecationWarning,
        )
        if not filename:
            filename = show

    plotter = plotter or get_active_plotter()
    return plotter.legend(filename, jupyter_render)


def save_plot_file_by_sha1(plottable: Plottable, dir_prefix: Path):
    """Save :term:`plottable` in a fpath generated from sha1 of the `dot`."""
    from hashlib import sha1

    img_format = "svg"

    dot = plottable.plot()

    ## Derrive image-filename from graph contents.
    hasher = sha1()
    hasher.update(str(dot).encode())
    fname = f"{hasher.hexdigest()}.{img_format}"
    fpath = dir_prefix / fname

    log.info("Rendering errored %s into file '%s'...", plottable, fpath)
    dir_prefix.mkdir(parents=True, exist_ok=True)
    dot.write(fpath, format=img_format)

    return fpath


def supported_plot_formats() -> List[str]:
    """return automatically all `pydot` extensions"""
    return [".%s" % f for f in pydot.Dot().formats]


_active_plotter: ContextVar[Plotter] = ContextVar("active_plotter", default=Plotter())


@contextmanager
def active_plotter_plugged(plotter: Plotter) -> None:
    """
    Like :func:`set_active_plotter()` as a context-manager, resetting back to old value.
    """
    if not isinstance(plotter, Plotter):
        raise TypeError(f"Cannot install invalid plotter: {plotter}")
    resetter = _active_plotter.set(plotter)
    try:
        yield
    finally:
        _active_plotter.reset(resetter)


def set_active_plotter(plotter: Plotter):
    """
    The default instance to render :term:`plottable`\\s,

    unless overridden with a `plotter` argument in :meth:`.Plottable.plot()`.

    :param plotter:
        the :class:`plotter` instance to install
    """
    if not isinstance(plotter, Plotter):
        raise TypeError(f"Cannot install invalid plotter: {plotter}")
    return _active_plotter.set(plotter)


def get_active_plotter() -> Plotter:
    """Get the previously active  :class:`.plotter` instance or default one."""
    plotter = _active_plotter.get()
    assert isinstance(plotter, Plotter), plotter

    return plotter
