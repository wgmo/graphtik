# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
Plotting of graph graphs handled by :term:`active plotter` (see also :mod:`.base`).

Separate from `graphtik.base` to avoid too many imports too early.
from contextlib import contextmanager

# Workaround sphinx-doc/sphinx#6590
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
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
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
import networkx as nx
import pydot
from boltons.iterutils import default_enter, default_visit, get_path, remap

from .base import PlotArgs, func_name, func_source
from .config import is_debug
from .modifiers import optional
from .netop import NetworkOperation
from .network import ExecutionPlan, Network, Solution, _EvictInstruction
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


def graphviz_html_string(
    s, *, repl_nl=None, repl_colon=None, xmltext=None,
):
    """
    Workaround *pydot* parsing of node-id & labels by encoding as HTML.

    - `pydot` library does not quote DOT-keywords anywhere (pydot#111).
    - Char ``:`` on node-names denote port/compass-points and break IDs (pydot#224).
    - Non-strings are not quoted_if_necessary by pydot.
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


def get_node_name(nx_node):
    if isinstance(nx_node, Operation):
        nx_node = nx_node.name
    return quote_node_id(nx_node)


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


class NodeArgs(NamedTuple):
    """All the args for :meth:`.Plotter._make_node()` call. """

    #: Where to add graphviz nodes & stuff.
    dot: pydot.Dot
    #: The node (data(str) or :class:`Operation`) as gotten from nx-graph.
    nx_node: Any = None
    #: Attributes gotten from nx-graph fot the given node.
    node_attrs: dict = None
    #: The pydot-node created
    dot_node: Any = None
    #: Collect the actual clustered `dot_nodes` among the given nodes.
    clustered: dict = None


class Ref:
    """Deferred attribute reference, :meth:`resolve`\\d when a `base` is given."""

    __slots__ = ("ref",)

    def __init__(self, ref, base=None):
        self.ref = ref

    def resolve(self, base):
        """Makes re-based clone to the given class/object."""
        return getattr(base, self.ref, "!missed!")

    def __repr__(self):
        return f"Ref('{self.ref}')"


def _drop_gt_lt(x):
    """SVGs break even with &gt;."""
    return x and re.sub('[<>"]', "_", str(x))


def _escape_or_none(context: jinja2.environment.EvalContext, x, escaper):
    """Do not markup Nones/empties, so `xmlattr` filter does not include them."""
    return x and jinja2.Markup(escaper(str(x)))


_jinja2_env = jinja2.Environment()
# Default `escape` filter breaks Nones for xmlattr.

_jinja2_env.filters["ee"] = jinja2.evalcontextfilter(
    partial(_escape_or_none, escaper=html.escape)
)
_jinja2_env.filters["eee"] = jinja2.evalcontextfilter(
    partial(_escape_or_none, escaper=quote_html_tooltips)
)
_jinja2_env.filters["slug"] = jinja2.evalcontextfilter(
    partial(_escape_or_none, escaper=as_identifier)
)
_jinja2_env.filters["hrefer"] = _drop_gt_lt


def _render_template(tpl: jinja2.Template, **kw) -> str:
    """Ignore falsy values, to skip attributes in template all together. """
    return tpl.render(**{k: v for k, v in kw.items() if v})


class Theme:
    """
    The poor man's css-like :term:`plot theme` applied like a theme.

    .. NOTE::
        Changing class attributes AFTER the module has loaded WON'T change themes;
        Either patch directly the :attr:`Plotter.theme` of :term:`active plotter`),
        or pass a new styles to a new plotter, as described in :ref:`plot-customizations`.

    """

    ##########
    ## VARIABLES

    fill_color = "wheat"
    pruned_color = "#d3d3d3"  # LightGrey
    canceled_color = "#a9a9a9"  # DarkGray
    failed_color = "LightCoral"
    resched_thickness = 4
    broken_color = "Red"
    overwrite_color = "SkyBlue"
    steps_color = "#00bbbb"
    evicted = "#006666"
    #: See :meth:`.Plotter._make_py_item_link()`.
    py_item_url_format: Union[str, Callable[[str], str]] = None
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
        "NetworkOperation": {},
        "Network": {},
        "ExecutionPlan": {},
        "Solution": {},
    }
    #: For when type-name of :attr:`PlotArgs.plottable` is not found
    #: in :attr:`.kw_plottable_type` ( ot missing altogether).
    kw_graph_plottable_type_unknown = {}

    ##########
    ## DATA node

    #: Reduce margins, since sideffects take a lost of space
    #: (default margin: x=0.11, y=0.055O)
    kw_data = {"margin": "0.02,0.02"}
    #: SHAPE change if with inputs/outputs,
    #: see https://graphviz.gitlab.io/_pages/doc/info/shapes.html
    kw_data_io_choice = {
        0: {"shape": "rect"},  # no IO
        1: {"shape": "invhouse"},  # Inp
        2: {"shape": "house"},  # Out
        3: {"shape": "hexagon"},  # Inp/Out
    }
    kw_data_pruned = {
        "fontcolor": Ref("pruned_color"),
        "color": Ref("pruned_color"),
        "tooltip": "(pruned)",
    }
    kw_data_sideffect = {"color": "blue", "fontcolor": "blue"}
    kw_data_to_evict = {
        "color": Ref("evicted"),
        "fontcolor": Ref("evicted"),
        "style": ["dashed"],
        "tooltip": "(to evict)",
    }
    kw_data_in_solution = {"style": ["filled"], "fillcolor": Ref("fill_color")}
    kw_data_evicted = {"penwidth": "3", "tooltip": "(evicted)"}
    kw_data_overwritten = {"style": ["filled"], "fillcolor": Ref("overwrite_color")}
    kw_data_canceled = {
        "fillcolor": Ref("canceled_color"),
        "style": ["filled"],
        "tooltip": "(canceled)",
    }

    ##########
    ## OPERATION node

    #: Keys to ignore from operation styles & node-attrs,
    #: because they are handled internally by HTML-Label, and/or
    #: interact badly with that label.
    op_bad_html_label_keys = {"shape", "label", "style"}
    op_link_target = fn_link_target = "_top"
    #: props for operation node (outside of label))
    kw_op = {}
    #: props only for HTML-Table label
    kw_op_label = {}
    kw_op_pruned = {"color": Ref("pruned_color"), "fontcolor": Ref("pruned_color")}
    kw_op_executed = {"fillcolor": Ref("fill_color")}
    kw_op_endured = {
        "penwidth": Ref("resched_thickness"),
        "style": ["dashed"],
        "tooltip": "(rescheduled)",
    }
    kw_op_rescheduled = {
        "penwidth": Ref("resched_thickness"),
        "style": ["dashed"],
        "tooltip": "(rescheduled)",
    }
    kw_op_failed = {"fillcolor": Ref("failed_color"), "tooltip": "(failed)"}
    kw_op_canceled = {"fillcolor": Ref("canceled_color"), "tooltip": "(canceled)"}
    #: Applied only if :attr:`py_item_url_format` defined, or
    #: ``"_op_link_target"`` in nx_node-attributes.

    #: Try to mimic a regular `Graphviz`_ node attributes
    #: (see examples in ``test.test_plot.test_op_template_full()`` for params).
    #: TODO: fix jinja2 template is un-picklable!
    op_template = _jinja2_env.from_string(
        textwrap.dedent(
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
                      'TOOLTIP': (tooltip or op_tooltip) | truncate | eee,
                      'HREF': op_url | hrefer | ee,
                      'TARGET': op_link_target | e
                      } | xmlattr }}
                    >
                        {%- if fontcolor -%}<FONT COLOR="{{ fontcolor }}">{%- endif -%}
                        {{- '<B>OP:</B> <I>%s</I>' % op_name |ee if op_name -}}
                        {%- if fontcolor -%}</FONT>{%- endif -%}
                    </TD>
                </TR>
                {%- if fn_name -%}
                <TR>
                    <TD ALIGN="left"
                      {{- {
                      'TOOLTIP': fn_tooltip | truncate | eee,
                      'HREF': fn_url | hrefer | ee,
                      'TARGET': fn_link_target | e
                      } | xmlattr }}
                      >
                        {%- if fontcolor -%}
                        <FONT COLOR="{{ fontcolor }}">
                        {%- endif -%}
                        <B>FN:</B> {{ fn_name | eee }}
                        {%- if fontcolor -%}
                        </FONT>
                        {%- endif -%}
                    </TD>
                </TR>
                {%- endif %}
            </TABLE>>
            """
        ).strip()
    )

    ##########
    ## EDGE

    kw_edge = {}
    kw_edge_optional = {"style": ["dashed"]}
    kw_edge_sideffect = {"color": "blue"}
    kw_edge_pruned = {"color": Ref("pruned_color")}
    kw_edge_rescheduled = {"style": ["dashed"]}
    kw_edge_endured = {"style": ["dashed"]}
    kw_edge_broken = {"color": Ref("broken_color")}

    ##########
    ## Other

    include_steps = False
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
        "target": "_top",
    }

    def __init__(self, _prototype: "Theme" = None, **kw):
        """
        Deep-copy class-attributes (to avoid sideffects) and apply user-overrides,

        resolving any :class:`Ref` on my self (from class's 'values).

        :param _prototype:
            If given, class-attributes are ignored, deep-copying "public" properties
            only from this instance (and apply on top any `kw`).
        """
        if _prototype is None:
            _prototype = type(self)
        else:
            assert isinstance(_prototype, Theme), _prototype

        values = {
            k: v
            for k, v in vars(type(self)).items()
            if not callable(v) and not k.startswith("_")
        }
        values.update(kw)
        self.resolve_refs(values)

    def resolve_refs(self, values: dict = None) -> None:
        """
        Resolve style attribute refs, and deep copy (for free) as instance attributes.

        :param values:
            if not given, resolve and clone my class's attributes

        :raises:
            Nothing(!), not to CRASH :term:`default active plotter` on import-time.
            Ref-errors are log-ERROR reported, and the item with the ref is skipped.
        """

        def remap_item(path, k, v):
            if isinstance(v, Ref):
                try:
                    return (k, v.resolve(self))
                except Exception as ex:
                    log.error(
                        "Invalid theme-ref '%s.%s': '%r' --> '%s' due to: %s",
                        ".".join(path),
                        k,
                        v,
                        v,
                        ex,
                    )
                    return False
            return True

        if values is None:
            values = vars(self)
        vars(self).update(remap(values, remap_item))

    def with_set(self, **kw) -> "Theme":
        """Returns a deep-clone modified by `kw`."""
        return type(self)(_prototype=self, **kw)


def remerge(*containers, source_map: list = None):
    """
    Merge recursively dicts or lists with :func:`boltons.iterutils.remap()`.

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

    if source_map is None:
        containers = [(id(t), t) for t in containers]

    ret = None

    def remerge_enter(path, key, value):
        new_parent, new_items = default_enter(path, key, value)
        if ret and not path and key is None:
            new_parent = ret
        try:
            cur_val = get_path(ret, path + (key,))
        except KeyError:
            pass
        else:
            # TODO: type check?
            new_parent = cur_val

        if isinstance(value, list):
            # lists are purely additive. See https://github.com/mahmoud/boltons/issues/81
            new_parent.extend(value)
            new_items = []

        return new_parent, new_items

    for t_name, cont in containers:
        if source_map is not None:

            def remerge_visit(path, key, value):
                key = path + (key,)
                if isinstance(value, list):
                    old = source_map.get(key)
                    if old:
                        old.append(t_name)
                    else:
                        source_map[key] = [t_name]
                else:
                    source_map[key] = t_name
                return True

        else:
            remerge_visit = default_visit

        ret = remap(cont, enter=remerge_enter, visit=remerge_visit)

    return ret


class StylesStack(NamedTuple):
    """A stack of mergeable dicts and their provenance, auto-deduced from a :class:`Theme`."""

    #: The style instance to read style-attributes from,
    #: when a style is given as string.
    theme: Theme
    #: A list of 2-tuples: (name, dict) containing the actual styles
    #: along with their provenance.
    named_styles: List[Tuple[str, dict]]

    def add(self, name, kw=None):
        """
        Adds a style by name from style-attributes, or provenanced explicitly, or fail early.

        :param name:
            Either the provenance name when the `kw` styles is given,
            OR just an existing attribute of :attr:`style` instance.
        """
        if kw is None:
            kw = getattr(self.theme, name)  # will scream early
        self.named_styles.append((name, kw))

    def asdict(self, nx_attrs: dict, debug=None) -> dict:
        """
        Optionally include style-merging debug-info and workaround pydot/pydot#228 ...

        pydot-cstor not supporting styles-as-lists.

        :param nx_attrs:
            a dictionary to be included in the debug tooltip
        :param debug:
            When not `None`, override :func:`config.is_debug` flag.
        """

        if (debug is None and is_debug()) or debug:
            from pprint import pformat
            from itertools import count

            styles_provenance = {}
            d = remerge(*self.named_styles, source_map=styles_provenance)
            provenance_str = pformat(
                {".".join(k): v for k, v in styles_provenance.items()}, indent=2
            )
            tooltip = f"- styles: {provenance_str}\n- extra_attrs: {pformat(nx_attrs, depth=2)}"
            d["tooltip"] = graphviz_html_string(tooltip)
        else:
            d = remerge(*(style_dict for _name, style_dict in self.named_styles))
        assert isinstance(d, dict), (d, self.named_styles)

        style = d.get("style")
        if isinstance(style, (list, tuple)):
            d["style"] = ",".join(str(i) for i in set(style))

        return d


class Plotter:
    """
    a :term:`plotter` renders diagram images of :term:`plottable`\\s.

    .. attribute:: Plotter.theme

        The :ref:`customizable <plot-customizations>` :class:`.Theme` instance
        controlling theme values & dictionaries for plots.
    """

    def __init__(self, theme: Theme = None, **styles_kw):
        self.theme: Theme = theme or Theme(**styles_kw)

    def with_styles(self, **kw) -> "Plotter":
        """
        Returns a cloned plotter with deep-coped theme modified as given.

        See also :meth:`Theme.with_set()`.
        """
        return type(self)(self.theme.with_set(**kw))

    def _new_styles_stack(self):
        return StylesStack(self.theme, [])

    def plot(self, plot_args: PlotArgs):
        if isinstance(plot_args.plottable, Solution):
            plot_args = plot_args.with_defaults(solution=plot_args.plottable)
        dot = self.build_pydot(plot_args)
        return self.render_pydot(dot, **plot_args.kw_render_pydot)

    def build_pydot(self, plot_args: PlotArgs) -> pydot.Dot:
        """
        Build a |pydot.Dot|_ out of a Network graph/steps/inputs/outputs and return it

        to be fed into `Graphviz`_ to render.

        See :meth:`.Plottable.plot()` for the arguments, sample code, and
        the legend of the plots.
        """
        graph, steps, solution = plot_args.graph, plot_args.steps, plot_args.solution
        theme = self.theme

        if graph is None:
            raise ValueError("At least `graph` to plot must be given!")

        graph, steps = self._skip_nodes(graph, steps)
        styles = self._new_styles_stack()

        styles.add("kw_graph")

        plottable_type = type(plot_args.plottable).__name__.split(".")[-1]
        styles.add(
            f"kw_graph_plottable_type-{plottable_type}",
            theme.kw_graph_plottable_type.get(
                plottable_type, theme.kw_graph_plottable_type_unknown
            ),
        )

        styles.add("user-overrides", _pub_props(graph.graph))
        dot = pydot.Dot(**styles.asdict(graph.graph))

        if plot_args.name:
            dot.set_name(as_identifier(plot_args.name))

        ## NODES
        #
        base_node_args = node_args = NodeArgs(dot=dot, clustered={})
        for nx_node, data in graph.nodes.data(True):
            node_args = base_node_args._replace(nx_node=nx_node, node_attrs=data)

            dot_node = self._make_node(plot_args, node_args)
            node_args = node_args._replace(dot_node=dot_node)

            self._append_or_cluster_node(plot_args, node_args)
        self._append_any_clustered_nodes(plot_args, node_args)

        ## EDGES
        #
        for src, dst, data in graph.edges.data(True):
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)

            ## Edge-kind
            #
            styles = self._new_styles_stack()

            styles.add("kw_edge")
            if data.get("optional"):
                styles.add("kw_edge_optional")
            if data.get("sideffect"):
                styles.add("kw_edge_sideffect")

            if getattr(src, "rescheduled", None):
                styles.add("kw_edge_rescheduled")
            if getattr(src, "endured", None):
                styles.add("kw_edge_endured")

            ## Edge-state
            #
            if graph.nodes[src].get("_pruned") or graph.nodes[dst].get("_pruned"):
                styles.add("kw_edge_pruned")
            if (
                solution is not None
                and (src, dst) not in solution.dag.edges
                and (src, dst) in solution.plan.dag.edges
            ):
                styles.add("kw_edge_broken")

            styles.add("user-overrides", _pub_props(data))
            edge = pydot.Edge(src=src_name, dst=dst_name, **styles.asdict(data))
            dot.add_edge(edge)

        ## Draw steps sequence, if it's worth it.
        #
        if steps and theme.include_steps and len(steps) > 1:
            it1 = iter(steps)
            it2 = iter(steps)
            next(it2)
            for i, (src, dst) in enumerate(zip(it1, it2), 1):
                src_name = get_node_name(src)
                dst_name = get_node_name(dst)
                edge = pydot.Edge(
                    src=src_name, dst=dst_name, label=str(i), **theme.kw_step
                )
                dot.add_edge(edge)

        self._add_legend_icon(plot_args, node_args)

        return dot

    def _make_node(self, plot_args: PlotArgs, node_args: NodeArgs) -> pydot.Node:
        """
        Customize nodes, e.g. add doc URLs/tooltips, solution tooltips.

        :param plot_args:
            must have, at least, a `graph`, as returned by :class:`.Plottable.prepare_plot_args()`

        :return:
            the updated `plot_args`

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
        from .op import Operation

        theme = self.theme
        graph = plot_args.graph
        nx_node = node_args.nx_node
        node_attrs = node_args.node_attrs
        (plottable, _, _, steps, inputs, outputs, solution, *_,) = plot_args

        if isinstance(nx_node, str):  # DATA
            styles = self._new_styles_stack()

            styles.add("kw_data")
            styles.add("node-name", {"name": quote_node_id(nx_node)})

            io_choice = _merge_conditions(
                inputs and nx_node in inputs, outputs and nx_node in outputs
            )
            styles.add(
                f"kw_data_io_choice: {io_choice}", theme.kw_data_io_choice[io_choice]
            )

            ## Data-kind
            #
            if any(d for _, _, d in graph.out_edges(nx_node, "sideffect")) or any(
                d for _, _, d in graph.in_edges(nx_node, "sideffect")
            ):
                styles.add("kw_data_sideffect")

            ## Data-state
            #
            if (
                (
                    isinstance(plottable, ExecutionPlan)
                    and nx_node not in plottable.dag.nodes
                )
                or (
                    isinstance(plottable, Solution)
                    and nx_node not in plottable.dag.nodes
                )
                or (solution is not None and nx_node not in solution.dag.nodes)
            ):
                assert (
                    not steps or nx_node not in steps
                ), f"Given `steps` missmatch `plan` and/or `solution`!\n  {plot_args}"
                styles.add("kw_data_pruned")
                graph.nodes[nx_node]["_pruned"] = True  # Signal to edge-plotting.
            else:
                if steps and nx_node in steps:
                    styles.add("kw_data_to_evict")

                if solution is not None:
                    if nx_node in solution:
                        data_tooltip = self._make_data_value_tooltip(
                            plot_args, node_args
                        )
                        if data_tooltip:
                            styles.add("node-code", {"tooltip": data_tooltip})

                        styles.add("kw_data_in_solution")
                        if nx_node in solution.overwrites:
                            styles.add("kw_data_overwritten")

                    elif nx_node not in steps:
                        styles.add("kw_data_canceled")
                    else:
                        styles.add("kw_data_evicted")

            styles.add("user-overrides", _pub_props(node_attrs))

        else:  # OPERATION
            op_name = nx_node.name
            label_styles = self._new_styles_stack()

            label_styles.add("kw_op_label")
            label_styles.add(
                "node-code",
                {
                    "op_name": op_name,
                    "fn_name": func_name(nx_node.fn, mod=1, fqdn=1, human=1),
                    "op_tooltip": self._make_op_tooltip(plot_args, node_args),
                    "fn_tooltip": self._make_fn_tooltip(plot_args, node_args),
                },
            )

            ## Op-kind
            #
            if nx_node.rescheduled:
                label_styles.add("kw_op_rescheduled")
            if nx_node.endured:
                label_styles.add("kw_op_endured")

            ## Op-state
            #
            if steps and nx_node not in steps:
                label_styles.add("kw_op_pruned")
            if solution:
                if solution.is_failed(nx_node):
                    label_styles.add("kw_op_failed")
                elif nx_node in solution.executed:
                    label_styles.add("kw_op_executed")
                elif nx_node in solution.canceled:
                    label_styles.add("kw_op_canceled")

            (op_url, op_link_target) = self._make_op_link(plot_args, node_args)
            (fn_url, fn_link_target) = self._make_fn_link(plot_args, node_args)
            label_styles.add(
                "tooltip-code",
                {
                    "op_url": op_url,
                    "op_link_target": op_link_target,
                    "fn_url": fn_url,
                    "fn_link_target": fn_link_target,
                },
            )

            label_styles.add("user-overrides", _pub_props(node_attrs))

            styles = self._new_styles_stack()
            styles.add(
                "init",
                {
                    "name": quote_node_id(nx_node.name),
                    "shape": "plain",
                    "label": _render_template(
                        self.theme.op_template, **label_styles.asdict(node_attrs),
                    ),
                    # Set some base tooltip, or else, "TABLE" shown...
                    "tooltip": graphviz_html_string(op_name),
                },
            )

            # Graphviz node attributes interacting badly with HTML-Labels.
            #
            bad_props = theme.op_bad_html_label_keys
            styles.add(
                "user-overrides",
                {k: v for k, v in _pub_props(node_attrs).items() if k not in bad_props},
            )

        return pydot.Node(**styles.asdict(node_attrs))

    def _make_op_link(
        self, plot_args: PlotArgs, node_args: NodeArgs
    ) -> Tuple[Optional[str], Optional[str]]:
        return self._make_py_item_link(plot_args, node_args, node_args.nx_node, "op")

    def _make_fn_link(
        self, plot_args: PlotArgs, node_args: NodeArgs
    ) -> Tuple[Optional[str], Optional[str]]:
        return self._make_py_item_link(plot_args, node_args, node_args.nx_node.fn, "fn")

    def _make_py_item_link(
        self, plot_args: PlotArgs, node_args: NodeArgs, item, prefix
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Deduce fn's url (e.g. docs) from theme, or from override in  `node_attrs`.

        :return:
            Search and return, in this order, any pair with truthy "url" element:

            1. node-attrs: ``(_{prefix}_url, _{prefix}_link_target)``
            2. theme-attributes: ``({prefix}_url, {prefix}_link_target)``
            3. fallback: ``(None, None)``

            An existent link-target from (1) still applies even if (2) is selected.
        """
        node_attrs = node_args.node_attrs
        if f"_{prefix}_url" in node_attrs:
            return (
                node_attrs[f"_{prefix}_url"],
                node_attrs.get(f"_{prefix}_link_target"),
            )

        fn_link = (None, None)
        url_format = self.theme.py_item_url_format
        if url_format:
            dot_path = func_name(node_args.nx_node.fn, None, mod=1, fqdn=1, human=0)
            if dot_path:
                url_data = {
                    "dot_path": dot_path,
                    "posix_path": dot_path.replace(".", "/"),
                }

                fn_url = (
                    url_format(url_data)
                    if callable(url_format)
                    else url_format % url_data
                )
                fn_link = (
                    fn_url,
                    node_attrs.get(f"_{prefix}_link_target", self.theme.fn_link_target),
                )

        return fn_link

    def _make_data_value_tooltip(self, plot_args: PlotArgs, node_args: NodeArgs):
        """Called on datanodes, when solution exists. """
        node = node_args.nx_node
        if node in plot_args.solution:
            val = plot_args.solution.get(node)
            tooltip = "(None)" if val is None else f"({type(val).__name__}) {val}"
            return quote_html_tooltips(tooltip)

    def _make_op_tooltip(self, plot_args: PlotArgs, node_args: NodeArgs):
        """the string-representation of an operation (name, needs, provides)"""
        return node_args.node_attrs.get("_op_tooltip", str(node_args.nx_node))

    def _make_fn_tooltip(self, plot_args: PlotArgs, node_args: NodeArgs):
        """the sources of the operation-function"""
        if "_fn_tooltip" in node_args.node_attrs:
            return node_args.node_attrs["_fn_tooltip"]

        fn_source = func_source(node_args.nx_node.fn, None, human=1)
        if fn_source:
            fn_source = fn_source

        return fn_source

    def _append_or_cluster_node(self, plot_args: PlotArgs, node_args: NodeArgs) -> None:
        """Add dot-node in dot now, or "cluster" it, to be added later. """
        # TODO remap nested plot-clusters:
        clusters = plot_args.clusters
        clustered = node_args.clustered
        nx_node = node_args.nx_node

        if not clusters or not nx_node in clusters:
            node_args.dot.add_node(node_args.dot_node)
        else:
            cluster_name = clusters[nx_node]
            node_cluster = clustered.get(cluster_name)
            if not node_cluster:
                node_cluster = clustered[cluster_name] = pydot.Cluster(
                    cluster_name, label=cluster_name
                )
            node_cluster.add_node(node_args.dot_node)

    def _append_any_clustered_nodes(
        self, plot_args: PlotArgs, node_args: NodeArgs
    ) -> None:
        # TODO remap nested plot-clusters:
        dot = node_args.dot
        for cluster in node_args.clustered.values():
            dot.add_subgraph(cluster)

    def _add_legend_icon(self, plot_args: PlotArgs, node_args: NodeArgs):
        """Optionally add an icon to diagrams linking to legend (if url given)."""
        kw_legend = self.theme.kw_legend
        if kw_legend and self.theme.kw_legend.get("URL"):
            node_args.dot.add_node(pydot.Node(**kw_legend))

    def _skip_nodes(
        self, graph: nx.Graph, steps: Collection
    ) -> Tuple[nx.Graph, Collection]:
        ## Drop any nodes, steps & edges with "_no_plot" attribute.
        #
        nodes_to_del = {n for n, no_plot in graph.nodes.data("_no_plot") if no_plot}
        graph.remove_nodes_from(nodes_to_del)
        if steps:
            steps = [s for s in steps if s not in nodes_to_del]
        graph.remove_edges_from(
            [
                (src, dst)
                for src, dst, no_plot in graph.edges.data("_no_plot")
                if no_plot
            ]
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
            If `None`, defaults to :data:`default_jupyter_render`;  you may modify those
            in place and they will apply for all future calls (see :ref:`jupyter_rendering`).

            You may increase the height of the SVG cell output with
            something like this::

                netop.plot(jupyter_render={"svg_element_styles": "height: 600px; width: 100%"})
        :return:
            the matplotlib image if ``filename=-1``, or the given `dot` annotated with any
            jupyter-rendering configurations given in `jupyter_render` parameter.

        See :meth:`.Plottable.plot()` for sample code.
        """
        if isinstance(filename, (bool, int)):
            ## Display graph via matplotlib
            #
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

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

    def legend(self, filename=None, jupyter_render: Mapping = None):
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
            rescheduled [shape=oval penwidth=4 fontname=italic label=<endured/rescheduled>
                        tooltip="Operation may fail or provide partial outputs so `net` must reschedule."
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
            evicted [shape=rect color="%(evicted)s"
                    tooltip="Instruction step to erase data from solution, to save memory."
                    URL="%(arch_url)s#term-evictions"];
            sol     [shape=rect style=filled fillcolor=wheat label="in solution"
                    tooltip="Data contained in the solution."
                    URL="%(arch_url)s#term-solution"];
            overwrite [shape=rect theme=filled fillcolor=SkyBlue
                    tooltip="More than 1 values exist in solution with this name."
                    URL="%(arch_url)s#term-overwrites"];
            data -> input -> output -> inp_out -> evicted -> sol -> overwrite [theme=invis];

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
            **vars(self.theme),
        }

        dot = pydot.graph_from_dot_data(dot_text)[0]
        # cluster = pydot.Cluster("Graphtik legend", label="Graphtik legend")
        # dot.add_subgraph(cluster)

        # nodes = dot.Node()
        # cluster.add_node("operation")

        return self.render_pydot(dot, filename=filename, jupyter_render=jupyter_render)


def legend(
    filename=None, show=None, jupyter_render: Mapping = None, plotter: Plotter = None,
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
        raise ValueError(f"Cannot install invalid plotter: {plotter}")
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
        raise ValueError(f"Cannot install invalid plotter: {plotter}")
    return _active_plotter.set(plotter)


def get_active_plotter() -> Plotter:
    """Get the previously active  :class:`.plotter` instance or default one."""
    plotter = _active_plotter.get()
    if not isinstance(plotter, Plotter):
        raise ValueError(f"Missing or invalid active plotter: {plotter}")

    return plotter
