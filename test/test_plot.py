# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import abc
import pickle
import sys
from functools import partial
from operator import add

import dill
import networkx as nx
import pydot
import pytest
from jinja2 import Template

from graphtik import base, compose, operation, planning, plot
from graphtik.fnop import PlotArgs
from graphtik.modifier import optional
from graphtik.pipeline import Pipeline
from graphtik.plot import (
    Plotter,
    Ref,
    StylesStack,
    Theme,
    active_plotter_plugged,
    get_active_plotter,
)

from .helpers import oneliner


@pytest.fixture
def pipeline():
    return compose(
        "pipeline",
        operation(name="add", needs=["a", "b1"], provides=["ab1"])(add),
        operation(name="sub", needs=["a", optional("b2")], provides=["ab2"])(
            lambda a, b=1: a - b
        ),
        operation(name="abb", needs=["ab1", "ab2"], provides=["asked"])(add),
    )


@pytest.fixture(params=[{"a": 1}, {"a": 1, "b1": 2}])
def inputs(request):
    return {"a": 1, "b1": 2}


@pytest.fixture(params=[None, ("a", "b1")])
def input_names(request):
    return request.param


@pytest.fixture(params=[None, ["asked", "b1"]])
def outputs(request):
    return request.param


@pytest.fixture(params=[None, 1])
def solution(pipeline, inputs, outputs, request):
    return request.param and pipeline(inputs, outputs)


###### TEST CASES #######
##


def test_plotting_docstring():
    common_formats = ".png .dot .jpg .jpeg .pdf .svg".split()
    for ext in common_formats:
        assert ext in Pipeline.plot.__doc__
        assert ext in planning.Network.plot.__doc__


def _striplines(s):
    if not s:
        return s
    return "/n".join(i.strip() for i in s.strip().splitlines())


def test_op_label_template_full():
    theme_op_badges = plot.Theme.op_badge_styles
    avail_badges = theme_op_badges["badge_styles"]
    kw = dict(
        op_name="the op",
        fn_name="the fn",
        penwidth="44",
        color="red",
        fontcolor="blue",
        fillcolor="wheat",
        op_truncate=plot.Theme.truncate_args,
        fn_truncate=plot.Theme.truncate_args,
        op_url="http://op_url.com<label>",
        op_tooltip='<op " \t tooltip>',
        op_link_target="_self",
        fn_url='http://fn_url.com/quoto"and',
        fn_tooltip="<fn\ntooltip>",
        fn_link_target="_top",
        badges=list(avail_badges),  # cspell: disable-line
        **theme_op_badges,
    )
    got = plot._render_template(plot.Theme.op_template, **kw)
    print(got)
    exp = """
    <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BORDER="44" COLOR="red" BGCOLOR="wheat">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="&lt;op &quot; &#9; tooltip&gt;" HREF="http://op_url.com_label_" TARGET="_self"
            ><FONT COLOR="blue"><B>OP:</B> <I>the op</I></FONT></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2" ALIGN="right">
                    <TR>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="12" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#04277d" TITLE="endured" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-endured" TARGET="_top"
                        ><FONT FACE="monospace" COLOR="white"><B>!</B></FONT></TD>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="12" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#fc89ac" TITLE="rescheduled" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-partial-outputs" TARGET="_top"
                        ><FONT FACE="monospace" COLOR="white"><B>?</B></FONT></TD>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="12" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#b1ce9a" TITLE="parallel" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-parallel-execution" TARGET="_top"
                        ><FONT FACE="monospace" COLOR="white"><B>|</B></FONT></TD>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="12" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#4e3165" TITLE="marshalled" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-marshalling" TARGET="_top"
                        ><FONT FACE="monospace" COLOR="white"><B>&amp;</B></FONT></TD>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="12" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#cc5500" TITLE="returns_dict" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-returns-dictionary" TARGET="_top"
                        ><FONT FACE="monospace" COLOR="white"><B>}</B></FONT></TD>
                    </TR>
                </TABLE></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TOOLTIP="&lt;fn&#10;tooltip&gt;" HREF="http://fn_url.com/quoto_and" TARGET="_top"
            ><FONT COLOR="blue"><B>FN:</B> the fn</FONT></TD>
        </TR>
    </TABLE>>
    """

    assert _striplines(got) == _striplines(exp)

    ## Check
    for k, v in [
        (k, v)
        for k, v in kw.items()
        if not any(i in k for i in "truncate tooltip url badge".split())
    ]:
        assert v in got, (k, v)


def test_op_label_template_empty():
    got = plot._render_template(plot.Theme.op_template)
    print(got)
    exp = """
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left"
                ></TD>
                <TD BORDER="1" SIDES="b" ALIGN="right"></TD>
            </TR>
        </TABLE>>
        """
    assert _striplines(got) == _striplines(exp)


def test_op_label_template_fn_empty():
    got = plot._render_template(
        plot.Theme.op_template,
        op_name="op",
        fn_name="fn",
        op_truncate=plot.Theme.truncate_args,
        fn_truncate=plot.Theme.truncate_args,
    )
    print(got)
    exp = """
    <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left"
            ><B>OP:</B> <I>op</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left"
            ><B>FN:</B> fn</TD>
        </TR>
    </TABLE>>
    """
    assert _striplines(got) == _striplines(exp)


def test_op_label_template_nones():
    kw = dict(
        op_name=None,
        fn_name=None,
        penwidth=None,
        color=None,
        fillcolor=None,
        op_url=None,
        op_tooltip=None,
        op_link_target=None,
        fn_url=None,
        fn_tooltip=None,
        fn_link_target=None,
    )
    got = plot._render_template(plot.Theme.op_template, **kw)
    print(got)
    exp = """
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left"
                ></TD>
                <TD BORDER="1" SIDES="b" ALIGN="right"></TD>
            </TR>
        </TABLE>>
        """

    assert _striplines(got) == _striplines(exp)


@pytest.mark.slow
def test_plot_formats(pipeline, tmp_path):
    ## Generate all formats  (not needing to save files)

    inputs = {"a": 1, "b1": 2}
    outputs = ["asked", "b1"]
    solution = pipeline.compute(inputs, outputs)

    # The 1st list does not working on my PC, or travis.
    # NOTE: maintain the other lists manually from the Exception message.
    failing_formats = ".dia .hpgl .mif .mp .pcl .pic .vtx .xlib".split()
    # The subsequent format names producing the same dot-file.
    dupe_formats = [
        ".cmapx_np",  # .cmapx
        ".imap_np",  # .imap
        ".jpeg",  # .jpe
        ".jpg",  # .jpe
        ".plain-ext",  # .plain
    ]
    null_formats = ".cmap .ismap".split()
    forbidden_formats = set(failing_formats + dupe_formats + null_formats)
    formats_to_check = sorted(set(plot.supported_plot_formats()) - forbidden_formats)

    # Collect old dots to detect dupes.
    prev_renders = {}
    dupe_errs = []
    for ext in formats_to_check:
        # Check Network.
        #
        render = pipeline.plot(solution=solution).create(format=ext[1:])
        if not render:
            dupe_errs.append("\n  null: %s" % ext)

        elif render in prev_renders.values():
            dupe_errs.append(
                "\n  dupe: %s <--> %s"
                % (ext, [pext for pext, pdot in prev_renders.items() if pdot == render])
            )
        else:
            prev_renders[ext] = render

    if dupe_errs:
        raise AssertionError("Failed pydot formats: %s" % "".join(sorted(dupe_errs)))


def _check_common_end(s1, s2, clip_index):
    __tracebackhide__ = True  # pylint: disable=unused-variable
    s1, s2 = s1[clip_index:], s2[clip_index:]
    ## Locate the index where they start to differ.
    #
    for i, (c1, c2) in enumerate(zip(reversed(s1), reversed(s2))):
        if c1 != c2:
            s1, s2 = s1[-i - 16 :], s2[-i - 16 :]
            assert s1 == s2


@active_plotter_plugged(Plotter(show_steps=True))
def test_plotters_hierarchy(pipeline: Pipeline, inputs, outputs):
    # Plotting original network, no plan.
    base_dot = str(pipeline.plot(inputs=inputs, outputs=outputs))
    assert base_dot
    assert f"digraph {pipeline.name} {{" in str(base_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" in str(base_dot)  # graph-label

    # Just smoke-test plotting of operations
    str(pipeline.ops[0].plot())

    sol = pipeline.compute(inputs, outputs)

    # Plotting of pipeline must remains the same.
    pipeline_dot = str(pipeline.plot(inputs=inputs, outputs=outputs))
    assert pipeline_dot == base_dot

    # Plotting plan alone has no label.
    plan_dot = str(sol.plan.plot(inputs=inputs, outputs=outputs))
    assert plan_dot
    assert plan_dot != base_dot
    assert plan_dot != pipeline_dot
    assert f"digraph {pipeline.name} {{" not in str(plan_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" not in str(plan_dot)  # graph-label

    # Plot a pipeline + solution, which must be different from all before.
    sol_pipeline_dot = str(pipeline.plot(inputs=inputs, outputs=outputs, solution=sol))
    assert sol_pipeline_dot != base_dot
    assert sol_pipeline_dot != pipeline_dot
    assert sol_pipeline_dot != plan_dot
    assert f"digraph {pipeline.name} {{" in str(pipeline_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" in str(pipeline_dot)  # graph-label

    # Plot a solution, which must not equal anything so far.
    sol_dot = str(sol.plot())
    sol_dot.startswith("digraph solution_x9_nodes {")
    assert sol_dot != str(pipeline.plot(inputs=inputs, outputs=outputs, solution=sol))
    assert sol_dot != str(pipeline.plot(solution=sol))

    # Calling plot directly on plan misses pipeline name
    raw_plan_dot = str(sol.plan.plot(inputs=inputs, outputs=outputs))
    assert f"digraph {pipeline.name} {{" not in str(raw_plan_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" not in str(raw_plan_dot)  # graph-label

    # Check plan does not contain solution, unless given.
    raw_sol_plan_dot = str(sol.plan.plot(inputs=inputs, outputs=outputs, solution=sol))
    assert raw_sol_plan_dot != raw_plan_dot


def test_plot_bad_format(pipeline, tmp_path):
    with pytest.raises(ValueError, match="Unknown file format") as exinfo:
        pipeline.plot(filename="bad.format")

    ## Check help msg lists all supported formats
    for ext in plot.supported_plot_formats():
        assert exinfo.match(ext)


def test_plot_write_file(pipeline, tmp_path):
    # Try saving a file from one format.

    fpath = tmp_path / "network.png"
    dot1 = pipeline.plot(str(fpath))
    assert fpath.exists()
    assert dot1


def _check_plt_img(img):
    assert img is not None
    assert len(img) > 0


def test_plot_matpotlib(pipeline, tmp_path):
    ## Try matplotlib Window, but # without opening a Window.

    # do not open window in headless travis
    img = pipeline.plot(filename=-1)
    _check_plt_img(img)


def test_plot_jupyter(pipeline, tmp_path):
    ## Try returned  Jupyter SVG.

    dot = pipeline.plot()
    s = dot._repr_html_()
    assert "<svg" in s.lower()


def test_plot_legend(pipeline, tmp_path):
    ## Try returned  Jupyter SVG.

    dot = plot.legend()
    assert dot

    img = plot.legend(filename=-1)
    _check_plt_img(img)


def test_style_Ref():
    s = plot.Ref("arch_url")
    assert s.resolve(Theme) == "https://graphtik.readthedocs.io/en/latest/arch.html"
    assert repr(s) == "Ref('arch_url')"

    class C:
        arch_url = "1"

    class C2:
        arch_url = "2"

    ss = s.resolve(C)
    assert str(ss) == "1"
    ss = s.resolve(C, C2)
    assert str(ss) == "1"
    ss = s.resolve(C2, C)
    assert str(ss) == "2"

    r = plot.Ref("resched_thickness")  # int target
    str(r)  # should not scream
    assert r.resolve(Theme) == 4

    ## Test default
    s = plot.Ref("bad")
    ss = s.resolve(C, C2, default=4)
    assert ss == 4
    s = plot.Ref("bad", default=5)
    ss = s.resolve(C, C2, default=4)
    assert ss == 4
    ss = s.resolve(C, C2)
    assert ss == 5


def test_Theme_withset():
    p = Theme()
    p2 = p.withset()
    assert p is not p2

    p3 = p.withset(show_steps=True)
    assert p3.show_steps
    assert not p.show_steps


def test_Plotter_with_styles():
    p = Plotter()
    p2 = p.with_styles()
    assert p is not p2
    assert p.default_theme is not p2.default_theme

    p3 = p.with_styles(show_steps=True)
    assert p3.default_theme.show_steps
    assert not p.default_theme.show_steps


def test_StylesStack_expansion_ok():
    plot_args = PlotArgs(
        name="James",
        theme=Theme(
            surname="Bond",
            test_style1={"a_list": [{"ref": Ref("surname")}]},
            test_style2={
                "a_list": [{"callable": lambda plot_args: f"{plot_args.name}"}]
            },
            test_style3={"a_list": [Template("{{ name }} {{theme.surname}}")]},
        ),
    )
    styles = StylesStack(plot_args, [])
    styles.add("test_style1")
    styles.add("test_style2")
    styles.add("test_style3")
    kw = styles.merge()

    print(kw["a_list"])
    assert kw["a_list"] == [
        {"ref": "Bond"},
        {"callable": "James"},
        "James Bond",
    ]


def test_StylesStack_expansion_skip_item():
    plot_args = PlotArgs(theme=Theme(test_style={"foo": lambda pa: ...}))
    styles = StylesStack(plot_args, [])
    styles.add("test_style")
    kw = styles.merge()
    assert kw == {}


@pytest.mark.parametrize(
    "style, exp",
    [
        (Ref("BAD_REF"), "BAD_REF"),
        (Template("{{ BAD_VAR.WORSE }}"), "BAD_VAR"),
        (lambda pa: BAD_CALL, "BAD_CALL"),  # pylint: disable=undefined-variable
    ],
)
def test_StylesStack_expansion_BAD(style, exp):
    plot_args = PlotArgs(theme=Theme(test_style={"foo": style}))
    styles = StylesStack(plot_args, [])
    styles.add("test_style")
    with pytest.raises(ValueError, match=exp):
        styles.merge()


def test_plotter_customizations(pipeline, monkeypatch):
    ## default URL
    #
    url = Theme.kw_legend["URL"]
    dot = str(pipeline.plot())
    assert url in dot, dot

    ## New active_plotter
    #
    with active_plotter_plugged(Plotter(theme=Theme(kw_legend={"URL": None}))):
        dot = str(pipeline.plot())
        assert "legend" not in dot, dot
        assert url not in dot, dot

        ## URL --> plotter in args
        #
        url1 = "http://example.1.org"
        dot = str(pipeline.plot(plotter=Plotter(theme=Theme(kw_legend={"URL": url1}))))
        assert url1 in dot, dot
        assert url not in dot, dot
    dot = str(pipeline.plot())
    assert url in dot, dot

    url2 = "http://example.2.org"
    with active_plotter_plugged(Plotter(theme=Theme(kw_legend={"URL": url2}))):
        dot = str(pipeline.plot())
        assert url2 in dot, dot
        assert url not in dot, dot
    dot = str(pipeline.plot())
    assert url in dot, dot
    assert url1 not in dot, dot

    ## URL --> plotter in args
    #
    dot = str(pipeline.plot(plotter=Plotter(theme=Theme(kw_legend={"URL": None}))))
    assert "legend" not in dot, dot

    dot = str(pipeline.plot(plotter=Plotter(theme=Theme(kw_legend={"URL": url2}))))
    assert url2 in dot, dot
    assert url not in dot, dot


def test_plotter_customizations_ignore_class(pipeline, monkeypatch):
    # Class patches ignored
    url = Theme.kw_legend["URL"]
    url_ignore = "http://foo.com"
    monkeypatch.setitem(Theme.kw_legend, "URL", url_ignore)
    dot = str(pipeline.plot())
    assert url in dot, dot
    assert url_ignore not in dot, dot


@pytest.mark.xfail(reason="jinja2 template fails pickling")
def test_plotter_pickle():
    plotter = Plotter()
    pickle.dumps(plotter)


@pytest.mark.xfail(reason="jinja2 template fails dill")
def test_plotter_dill():
    plotter = Plotter()
    dill.dumps(plotter)


def func(a, b):
    pass


@pytest.fixture()
def dot_str_pipeline():
    return compose(
        "graph",
        operation(name="node", needs=["edge", "digraph: strict"], provides=["<graph>"])(
            add
        ),
        operation(
            name="cu:sto:m", needs=["edge", "digraph: strict"], provides=["<graph>"]
        )(func),
    )


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="PY3.6- have different docstrings for builtins."
)
def test_node_dot_str0(dot_str_pipeline):
    dot_str = str(dot_str_pipeline.plot())
    print(dot_str)
    exp = """
    digraph graph_ {
    fontname=italic;
    label=<graph>;
    node [fillcolor=white, style=filled];
    <edge> [fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD>edge</TD>
                </TR>
            </TABLE>>, shape=invhouse, tooltip="(input)"];
    <digraph&#58; strict> [fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD>digraph: strict</TD>
                </TR>
            </TABLE>>, shape=invhouse, tooltip="(input)"];
    <node> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FnOp(name=&#x27;node&#x27;, needs=[&#x27;edge&#x27;, &#x27;digraph: strict&#x27;], provides=[&#x27;&lt;graph&gt;&#x27;], fn=&#x27;add&#x27;)" TARGET="_top"
            ><B>OP:</B> <I>node</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TOOLTIP="Same as a + b." TARGET="_top"
            ><B>FN:</B> &lt;built-in function add&gt;</TD>
        </TR>
    </TABLE>>, shape=plain, tooltip=<node>];
    <&lt;graph&gt;> [fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD><graph></TD>
                </TR>
            </TABLE>>, shape=house, tooltip="(output)"];
    <cu&#58;sto&#58;m> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FnOp(name=&#x27;cu:sto:m&#x27;, needs=[&#x27;edge&#x27;, &#x27;digraph: strict&#x27;], provides=[&#x27;&lt;graph&gt;&#x27;], fn=&#x27;func&#x27;)" TARGET="_top"
            ><B>OP:</B> <I>cu:sto:m</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TOOLTIP="def func(a, b):&#10;    pass" TARGET="_top"
            ><B>FN:</B> test.test_plot.func</TD>
        </TR>
    </TABLE>>, shape=plain, tooltip=<cu:sto:m>];
    <edge> -> <node>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <edge> -> <cu&#58;sto&#58;m>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <digraph&#58; strict> -> <node>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <digraph&#58; strict> -> <cu&#58;sto&#58;m>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <node> -> <&lt;graph&gt;>  [headport=n, tailport=s];
    <cu&#58;sto&#58;m> -> <&lt;graph&gt;>  [headport=n, tailport=s];
    legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled, target=_blank];
    }
    """

    assert _striplines(dot_str) == _striplines(exp)


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="PY3.6 - has different docstrings for builtins."
)
def test_node_dot_str1(dot_str_pipeline, monkeypatch):
    ## Test node-hidding & Graph-overlaying.
    #
    overlay = nx.DiGraph()
    hidden_op = dot_str_pipeline.net.find_op_by_name("node")
    overlay.add_node(hidden_op, no_plot=True)
    overlay.graph["graphviz.splines"] = "ortho"

    exp = r"""
    digraph solution_x5_nodes {
    fontname=italic;
    splines=ortho;
    node [fillcolor=white, style=filled];
    subgraph "cluster_after pruning" {
    label=<after pruning>;
    <edge> [fillcolor=wheat, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD>edge</TD>
                </TR>
            </TABLE>>, shape=invhouse, style=filled, tooltip="(input)\n(int) 1"];
    <digraph&#58; strict> [fillcolor=wheat, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD>digraph: strict</TD>
                </TR>
            </TABLE>>, shape=invhouse, style=filled, tooltip="(input)\n(int) 2"];
    <&lt;graph&gt;> [fillcolor=SkyBlue, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD><graph></TD>
                </TR>
            </TABLE>>, shape=house, style=filled, tooltip="(output)\n(None)\n(x2 overwrites) 1. 3&#10;  0. None"];
    <cu&#58;sto&#58;m> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BGCOLOR="wheat">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FnOp(name=&#x27;cu:sto:m&#x27;, needs=[&#x27;edge&#x27;, &#x27;digraph: strict&#x27;], provides=[&#x27;&lt;graph&gt;&#x27;], fn=&#x27;func&#x27;)" TARGET="_top"
            ><B>OP:</B> <I>cu:sto:m</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2" ALIGN="right">
                    <TR>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="14" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#00bbbb" TITLE="computation order" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-steps" TARGET="_top"><FONT FACE="monospace" COLOR="white"><B>0</B></FONT></TD>
                    </TR>
                </TABLE></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TOOLTIP="def func(a, b):&#10;    pass" TARGET="_top"
            ><B>FN:</B> test.test_plot.func</TD>
        </TR>
    </TABLE>>, shape=plain, tooltip=<cu:sto:m>];
    }

    <edge> -> <cu&#58;sto&#58;m>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <digraph&#58; strict> -> <cu&#58;sto&#58;m>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <cu&#58;sto&#58;m> -> <&lt;graph&gt;>  [color="#ffa9cd", headport=n, tailport=s, tooltip="(null-result)"];
    legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled, target=_blank];
    }
    """

    ## Theme-param.
    #
    sol = dot_str_pipeline.compute({"edge": 1, "digraph: strict": 2})
    dot_str = str(
        sol.plot(
            graph=overlay,
            theme=Theme(
                op_link_target="_self",
                fn_link_target="bad",
            ),
        )
    )

    print(dot_str)
    assert _striplines(dot_str) == _striplines(exp)
    assert "<node>" not in dot_str

    ## Plotter-param
    #
    dot_str = str(
        sol.plot(
            graph=overlay,
            plotter=get_active_plotter().with_styles(
                op_link_target="_self",
                fn_link_target="bad",
            ),
        )
    )
    assert _striplines(dot_str) == _striplines(exp)
    assert "<node>" not in dot_str


def test_step_badge():
    p = compose(
        "",
        operation(str, "0", provides="b"),
        operation(str, "1", needs="b", provides="c"),
    )
    sol = p.compute(outputs="c")
    dot_str = str(sol.plot())
    print(dot_str)
    exp = r"""
    digraph solution_x4_nodes {
    fontname=italic;
    node [fillcolor=white, style=filled];
    <0> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BGCOLOR="wheat">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FnOp(name=&#x27;0&#x27;, provides=[&#x27;b&#x27;], fn=&#x27;str&#x27;)" TARGET="_top"
            ><B>OP:</B> <I>0</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2" ALIGN="right">
                    <TR>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="14" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#00bbbb" TITLE="computation order" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-steps" TARGET="_top"><FONT FACE="monospace" COLOR="white"><B>0</B></FONT></TD>
                    </TR>
                </TABLE></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TARGET="_top"
            ><B>FN:</B> builtins.str</TD>
        </TR>
    </TABLE>>, shape=plain, tooltip=<0>];
    <b> [color="#006666", fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR>
                    <TD STYLE="rounded" CELLSPACING="2" CELLPADDING="4" BGCOLOR="#00bbbb" TITLE="computation order" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-steps" TARGET="_top"
                    ><FONT FACE="monospace" COLOR="white"><B>2</B></FONT></TD><TD>b</TD>
                </TR>
            </TABLE>>, penwidth=3, shape=rect, style="filled,dashed", tooltip="(to evict)\n(evicted)"];
    <1> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BGCOLOR="wheat">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FnOp(name=&#x27;1&#x27;, needs=[&#x27;b&#x27;], provides=[&#x27;c&#x27;], fn=&#x27;str&#x27;)" TARGET="_top"
            ><B>OP:</B> <I>1</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2" ALIGN="right">
                    <TR>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="14" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#00bbbb" TITLE="computation order" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-steps" TARGET="_top"><FONT FACE="monospace" COLOR="white"><B>1</B></FONT></TD>
                    </TR>
                </TABLE></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TARGET="_top"
            ><B>FN:</B> builtins.str</TD>
        </TR>
    </TABLE>>, shape=plain, tooltip=<1>];
    <c> [fillcolor=wheat, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR><TD>c</TD>
                </TR>
            </TABLE>>, shape=house, style=filled, tooltip="(output)\n(str)"];
    <0> -> <b>  [headport=n, tailport=s];
    <b> -> <1>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <1> -> <c>  [headport=n, tailport=s];
    legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled, target=_blank];
    }
    """
    assert _striplines(dot_str) == _striplines(exp)


def test_node_edge_defaults():
    dot = compose("defs", operation(str, "a", "b")).plot(
        theme={"node_defaults": {"color": "red"}}
    )
    assert "color=red" in str(dot)

    dot = compose("defs", operation(str, "a", "b")).plot(
        theme={"edge_defaults": {"color": "red"}}
    )
    assert "color=red" in str(dot)


def test_combine_clusters():
    p1 = compose(
        "op1",
        operation(lambda a, b: None, name="op1", needs=["a", "b"], provides=["ab"]),
        operation(lambda a, b: None, name="op2", needs=["a", "ab"], provides="c"),
        operation(lambda a: None, name="op3", needs="c", provides="C"),
    )

    p2 = compose(
        "op2",
        operation(lambda a, b: None, name="op1", needs=["a", "b"], provides=["ab"]),
        operation(lambda a, b: None, name="op2", needs=["c", "ab"], provides=["cab"]),
    )

    merged_graph = compose("m", p1, p2, nest=True)
    dot: pydot.Dot = merged_graph.plot()
    assert dot.get_subgraph(f"cluster_{p1.name}")
    assert dot.get_subgraph(f"cluster_{p2.name}")


def test_degenerate_pipeline():
    compose("defs", operation(str, "a")).plot()


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="PY3.7 - has different docstrings for multiindex series.",
)
def test_plot_pandas_numpy():
    import numpy as np
    import pandas as pd

    pipe = compose(
        "a", operation(lambda a, c: np.array([1.0, 2]), "A", ["i1", "i2"], "o")
    )
    sol = pipe.compute(
        {
            "i1": pd.DataFrame(
                [1, 2],
                columns=pd.Index(["A"], name="hi"),
                index=pd.Index(["r1", "r2"], name="n0"),
            ),
            "i2": pd.Series(
                [1],
                index=pd.MultiIndex.from_arrays([["a"], ["b"]], names=["l1", "l2"]),
            ),
        }
    )
    got = str(sol.plot())
    print(got)
    exp = r"""
    digraph solution_x4_nodes {
    fontname=italic;
    node [fillcolor=white, style=filled];
    <i1> [fillcolor=wheat, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR>
                    <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                        TITLE="&lt;class &#x27;pandas.core.frame.DataFrame&#x27;&gt;&#10;Index: 2 entries, r1 to r2&#10;Data columns (total 1 columns):&#10; #   Column  Non-Null Count  Dtype&#10;---  ------  --------------  -----&#10; 0   A       2 non-null      int64&#10;dtypes: int64(1)&#10;memory usage: 32.0+ bytes"
                        TARGET="_top"
                    ><FONT FACE="monospace" COLOR="#7193ff"><B>#</B></FONT></TD>
                    <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                        TITLE="Index([&#x27;r1&#x27;, &#x27;r2&#x27;], dtype=&#x27;object&#x27;, name=&#x27;n0&#x27;)"
                        TARGET="_top"
                    ><FONT FACE="monospace" COLOR="#7193ff"><B>R</B></FONT></TD>
                    <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                        TITLE="Index([&#x27;A&#x27;], dtype=&#x27;object&#x27;, name=&#x27;hi&#x27;)"
                        TARGET="_top"
                    ><FONT FACE="monospace" COLOR="#7193ff"><B>C</B></FONT></TD><TD>i1</TD>
                </TR>
            </TABLE>>, shape=invhouse, style=filled, tooltip="(input)\n(DataFrame, shape: (2, 1)) hi  A&#10;n0   &#10;r1  1&#10;r2  2"];
    <i2> [fillcolor=wheat, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR>
                    <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                        TITLE="&lt;class &#x27;pandas.core.series.Series&#x27;&gt;&#10;MultiIndex: 1 entries, (&#x27;a&#x27;, &#x27;b&#x27;) to (&#x27;a&#x27;, &#x27;b&#x27;)&#10;Series name: None&#10;Non-Null Count  Dtype&#10;--------------  -----&#10;1 non-null      int64&#10;dtypes: int64(1)&#10;memory usage: 344.0+ bytes"
                        TARGET="_top"
                    ><FONT FACE="monospace" COLOR="#7193ff"><B>#</B></FONT></TD>
                    <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                        TITLE="MultiIndex([(&#x27;a&#x27;, &#x27;b&#x27;)],&#10;           names=[&#x27;l1&#x27;, &#x27;l2&#x27;])"
                        TARGET="_top"
                    ><FONT FACE="monospace" COLOR="#7193ff"><B>R</B></FONT></TD><TD>i2</TD>
                </TR>
            </TABLE>>, shape=invhouse, style=filled, tooltip="(input)\n(Series, shape: (1,), dtype: int64) l1  l2&#10;a   b     1&#10;dtype: int64"];
    <A> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BGCOLOR="wheat">
        <TR>
            <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FnOp(name=&#x27;A&#x27;, needs=[&#x27;i1&#x27;, &#x27;i2&#x27;], provides=[&#x27;o&#x27;], fn=&#x27;&lt;lambda&gt;&#x27;)" TARGET="_top"
            ><B>OP:</B> <I>A</I></TD>
            <TD BORDER="1" SIDES="b" ALIGN="right"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2" ALIGN="right">
                    <TR>
                        <TD STYLE="rounded" HEIGHT="22" WIDTH="14" FIXEDSIZE="true" VALIGN="BOTTOM" BGCOLOR="#00bbbb" TITLE="computation order" HREF="https://graphtik.readthedocs.io/en/latest/arch.html#term-steps" TARGET="_top"><FONT FACE="monospace" COLOR="white"><B>0</B></FONT></TD>
                    </TR>
                </TABLE></TD>
        </TR>
        <TR>
            <TD COLSPAN="2" ALIGN="left" TOOLTIP="&quot;a&quot;, operation(lambda a, c: np.array([1.0, 2]), &quot;A&quot;, [&quot;i1&quot;, &quot;i2&quot;], &quot;o&quot;)" TARGET="_top"
            ><B>FN:</B> ...py.&lt;locals&gt;.&lt;lambda&gt;</TD>
        </TR>
    </TABLE>>, shape=plain, tooltip=<A>];
    <o> [fillcolor=wheat, fixedsize=shape, label=<<TABLE CELLBORDER="0" CELLSPACING="0" BORDER="0">
                <TR>
                    <TD STYLE="rounded" CELLSPACING="0" CELLPADDING="0" WIDTH="8"
                        TITLE="shape: (2,), dtype: float64"
                        TARGET="_top"
                    ><FONT FACE="monospace" COLOR="#7193ff"><B>#</B></FONT></TD><TD>o</TD>
                </TR>
            </TABLE>>, shape=house, style=filled, tooltip="(output)\n(ndarray, shape: (2,), dtype: float64) [1. 2.]"];
    <i1> -> <A>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <i2> -> <A>  [arrowtail=inv, dir=back, headport=n, tailport=s];
    <A> -> <o>  [headport=n, tailport=s];
    legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled, target=_blank];
    }
    """
    assert oneliner(got) == oneliner(exp)
