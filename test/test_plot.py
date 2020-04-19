# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import abc
import pickle
import sys
from functools import partial
from operator import add

import dill
import networkx as nx
import pytest

from graphtik import base, compose, network, operation, plot
from graphtik.modifiers import optional
from graphtik.netop import NetworkOperation
from graphtik.plot import (
    Plotter,
    Style,
    active_plotter_plugged,
    get_active_plotter,
)


@pytest.fixture
def pipeline():
    return compose(
        "netop",
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
        assert ext in NetworkOperation.plot.__doc__
        assert ext in network.Network.plot.__doc__


def _striplines(s):
    if not s:
        return s
    return "/n".join(i.strip() for i in s.strip().splitlines())


def test_op_label_template_full():
    kw = dict(
        op_name="the op",
        fn_name="the fn",
        penwidth="44",
        color="red",
        fontcolor="blue",
        fillcolor="wheat",
        op_url="http://op_url.com<label>",
        op_tooltip='<op " \t tooltip>',
        op_link_target="_self",
        fn_url='http://fn_url.com/quoto"and',
        fn_tooltip="<fn\ntooltip>",
        fn_link_target="_top",
    )
    got = plot._render_template(plot.Style.op_template, **kw)
    print(got)
    exp = """
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BORDER="44" COLOR="red" BGCOLOR="wheat">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="&lt;op &quot; &#9; tooltip&gt;" HREF="http://op_url.com_label_" TARGET="_self"
                ><FONT COLOR="blue"><B>OP:</B> <I>the op</I></FONT></TD>
            </TR><TR>
                <TD ALIGN="left" TOOLTIP="&lt;fn&#10;tooltip&gt;" HREF="http://fn_url.com/quoto_and" TARGET="_top"
                ><FONT COLOR="blue"><B>FN:</B> the fn</FONT></TD>
            </TR>
        </TABLE>>
        """

    assert _striplines(got) == _striplines(exp)
    for k, v in [
        (k, v) for k, v in kw.items() if "tooltip" not in k and "url" not in k
    ]:
        assert v in got, (k, v)


def test_op_label_template_empty():
    got = plot._render_template(plot.Style.op_template)
    print(got)
    exp = """
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TARGET=""
                ></TD>
            </TR>
        </TABLE>>
        """
    assert _striplines(got) == _striplines(exp)


def test_op_label_template_fn_empty():
    got = plot._render_template(plot.Style.op_template, op_name="op", fn_name="fn")
    print(got)
    exp = """
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TARGET=""
                ><B>OP:</B> <I>op</I></TD>
            </TR><TR>
                <TD ALIGN="left" TARGET=""
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
    got = plot._render_template(plot.Style.op_template, **kw)
    print(got)
    exp = """
        <<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TARGET=""
                ></TD>
            </TR>
        </TABLE>>
        """

    assert _striplines(got) == _striplines(exp)


@pytest.mark.slow
def test_plot_formats(pipeline, tmp_path):
    ## Generate all formats  (not needing to save files)

    # run it here (and not in fixture) to ensure `last_plan` exists.
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


def test_plotters_hierarchy(pipeline: NetworkOperation, inputs, outputs):
    # Plotting original network, no plan.
    base_dot = str(pipeline.plot(inputs=inputs, outputs=outputs))
    assert base_dot
    assert f"digraph {pipeline.name} {{" in str(base_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" in str(base_dot)  # graph-label

    solution = pipeline.compute(inputs, outputs)

    # Plotting delegates to network plan.
    netop_dot = str(pipeline.plot(inputs=inputs, outputs=outputs))
    assert netop_dot
    assert netop_dot != base_dot
    assert f"digraph {pipeline.name} {{" in str(base_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" in str(base_dot)  # graph-label

    # Plotting plan alone has not label.
    plan_dot = str(pipeline.last_plan.plot(inputs=inputs, outputs=outputs))
    assert plan_dot
    assert plan_dot != base_dot
    assert plan_dot != netop_dot
    assert f"digraph {pipeline.name} {{" not in str(plan_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" not in str(plan_dot)  # graph-label

    # Plot a plan + solution, which must be different from all before.
    sol_netop_dot = str(
        pipeline.plot(inputs=inputs, outputs=outputs, solution=solution)
    )
    assert sol_netop_dot != base_dot
    assert sol_netop_dot != netop_dot
    assert sol_netop_dot != plan_dot
    assert f"digraph {pipeline.name} {{" in str(netop_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" in str(netop_dot)  # graph-label

    # Plot a solution, which must equal plan + sol.
    sol_plan_dot = str(
        pipeline.last_plan.plot(inputs=inputs, outputs=outputs, solution=solution)
    )
    head1 = "digraph plan_x9_nodes {"
    assert sol_plan_dot.startswith(head1)
    sol_dot = str(solution.plot(inputs=inputs, outputs=outputs))
    head2 = "digraph solution_x9_nodes {"
    assert sol_dot.startswith(head2)
    assert sol_plan_dot[len(head1) :] == sol_dot[len(head2) :]

    plan = pipeline.last_plan
    pipeline.last_plan = None

    # We resetted last_plan to check if it reproduces original.
    base_dot2 = str(pipeline.plot(inputs=inputs, outputs=outputs))
    assert str(base_dot2) == str(base_dot)

    # Calling plot directly on plan misses netop.name
    raw_plan_dot = str(plan.plot(inputs=inputs, outputs=outputs))
    assert f"digraph {pipeline.name} {{" not in str(raw_plan_dot)  # graph-name
    assert f"label=<{pipeline.name}>;" not in str(raw_plan_dot)  # graph-label

    # Check plan does not contain solution, unless given.
    raw_sol_plan_dot = str(plan.plot(inputs=inputs, outputs=outputs, solution=solution))
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
    assert s.target == "https://graphtik.readthedocs.io/en/latest/arch.html"
    assert repr(s) == "Ref(<class 'graphtik.plot.Style'>, 'arch_url')"

    class C:
        arch_url = "1"

    s = s.rebased(C)
    assert str(s) == "1"
    assert (
        repr(s) == "Ref(<class 'test.test_plot.test_style_Ref.<locals>.C'>, 'arch_url')"
    )

    r = plot.Ref("resched_thickness")  # int target
    str(r)  # should not scream
    assert r.target == 4
    assert repr(r) == "Ref(<class 'graphtik.plot.Style'>, 'resched_thickness')"


def test_plotter_customizations(pipeline, monkeypatch):
    ## default URL
    #
    url = Style.kw_legend["URL"]
    dot = str(pipeline.plot())
    assert url in dot

    ## New active_plotter
    #
    with active_plotter_plugged(Plotter(style=Style(kw_legend={"URL": None}))):
        dot = str(pipeline.plot())
        assert url not in dot

        ## URL --> plotter in args
        #
        url1 = "http://example.1.org"
        dot = str(pipeline.plot(plotter=Plotter(style=Style(kw_legend={"URL": url1}))))
        assert url1 in dot
        assert url not in dot
    dot = str(pipeline.plot())
    assert url in dot

    url2 = "http://example.2.org"
    with active_plotter_plugged(Plotter(style=Style(kw_legend={"URL": url2}))):
        dot = str(pipeline.plot())
        assert url2 in dot
        assert url not in dot
    dot = str(pipeline.plot())
    assert url in dot
    assert url1 not in dot

    ## URL --> plotter in args
    #
    dot = str(pipeline.plot(plotter=Plotter(style=Style(kw_legend={"URL": None}))))
    assert url not in dot

    dot = str(pipeline.plot(plotter=Plotter(style=Style(kw_legend={"URL": url2}))))
    assert url2 in dot
    assert url not in dot


def test_plotter_customizations_ignore_class(pipeline, monkeypatch):
    # Class patches ignored
    url = Style.kw_legend["URL"]
    url_ignore = "http://foo.com"
    monkeypatch.setitem(Style.kw_legend, "URL", url_ignore)
    dot = str(pipeline.plot())
    assert url in dot
    assert url_ignore not in dot


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


@pytest.mark.xfail(
    sys.version_info < (3, 7), reason="PY3.6- have different docstrings for builtins."
)
def test_node_dot_str0(dot_str_pipeline):
    dot_str = str(dot_str_pipeline.plot())
    print(dot_str)
    exp = """
        digraph graph_ {
        fontname=italic;
        label=<graph>;
        <edge> [shape=invhouse];
        <digraph&#58; strict> [shape=invhouse];
        <node> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FunctionalOperation(name=&#x27;node&#x27;, needs=[&#x27;edge&#x27;, &#x27;digraph: strict&#x27;], provides=[&#x27;&lt;graph&gt;&#x27;], fn=&#x27;add&#x27;)" TARGET=""
                ><B>OP:</B> <I>node</I></TD>
            </TR><TR>
                <TD ALIGN="left" TOOLTIP="Same as a + b." TARGET=""
                ><B>FN:</B> &lt;built-in function add&gt;</TD>
            </TR>
        </TABLE>>, shape=plain, tooltip=<node>];
        <&lt;graph&gt;> [shape=house];
        <cu&#58;sto&#58;m> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FunctionalOperation(name=&#x27;cu:sto:m&#x27;, needs=[&#x27;edge&#x27;, &#x27;digraph: strict&#x27;], provides=[&#x27;&lt;graph&gt;&#x27;], fn=&#x27;func&#x27;)" TARGET=""
                ><B>OP:</B> <I>cu:sto:m</I></TD>
            </TR><TR>
                <TD ALIGN="left" TOOLTIP="def func(a, b):&#10;    pass" TARGET=""
                ><B>FN:</B> test.test_plot.func</TD>
            </TR>
        </TABLE>>, shape=plain, tooltip=<cu:sto:m>];
        <edge> -> <node>;
        <edge> -> <cu&#58;sto&#58;m>;
        <digraph&#58; strict> -> <node>;
        <digraph&#58; strict> -> <cu&#58;sto&#58;m>;
        <node> -> <&lt;graph&gt;>;
        <cu&#58;sto&#58;m> -> <&lt;graph&gt;>;
        legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled, target=_top];
        }
        """

    assert _striplines(dot_str) == _striplines(exp)


@pytest.mark.xfail(
    sys.version_info < (3, 7), reason="PY3.6 - has different docstrings for builtins."
)
def test_node_dot_str1(dot_str_pipeline, monkeypatch):
    style = get_active_plotter().style
    monkeypatch.setattr(style, "py_item_url_format", "abc#%s")
    monkeypatch.setattr(style, "op_link_target", "_self")
    monkeypatch.setattr(style, "fn_link_target", "bad")

    ## Test node-hidding & Graph-overlaying.
    #
    overlay = nx.DiGraph()
    hidden_op = dot_str_pipeline.net.find_op_by_name("node")
    overlay.add_node(hidden_op, _no_plot=True)
    overlay.graph["splines"] = "ortho"

    sol = dot_str_pipeline.compute({"edge": 1, "digraph: strict": 2})
    dot_str = str(sol.plot(graph=overlay))

    print(dot_str)
    exp = """
        digraph solution_x5_nodes {
        fontname=italic;
        splines=ortho;
        subgraph "cluster_after pruning" {
        label="after pruning";
        <edge> [fillcolor=wheat, shape=invhouse, style=filled, tooltip="(int) 1"];
        <digraph&#58; strict> [fillcolor=wheat, shape=invhouse, style=filled, tooltip="(int) 2"];
        <&lt;graph&gt;> [fillcolor=SkyBlue, shape=house, style=filled, tooltip=None];
        <cu&#58;sto&#58;m> [label=<<TABLE CELLBORDER="0" CELLSPACING="0" STYLE="rounded" BGCOLOR="wheat">
            <TR>
                <TD BORDER="1" SIDES="b" ALIGN="left" TOOLTIP="FunctionalOperation(name=&#x27;cu:sto:m&#x27;, needs=[&#x27;edge&#x27;, &#x27;digraph: strict&#x27;], provides=[&#x27;&lt;graph&gt;&#x27;], fn=&#x27;func&#x27;)" HREF="abc#{&#x27;dot_path&#x27;: &#x27;test.test_plot.func&#x27;, &#x27;posix_path&#x27;: &#x27;test/test_plot/func&#x27;}" TARGET="bad"
                ><B>OP:</B> <I>cu:sto:m</I></TD>
            </TR><TR>
                <TD ALIGN="left" TOOLTIP="def func(a, b):&#10;    pass" HREF="abc#{&#x27;dot_path&#x27;: &#x27;test.test_plot.func&#x27;, &#x27;posix_path&#x27;: &#x27;test/test_plot/func&#x27;}" TARGET="bad"
                ><B>FN:</B> test.test_plot.func</TD>
            </TR>
        </TABLE>>, shape=plain, tooltip=<cu:sto:m>];
        }

        <edge> -> <cu&#58;sto&#58;m>;
        <digraph&#58; strict> -> <cu&#58;sto&#58;m>;
        <cu&#58;sto&#58;m> -> <&lt;graph&gt;>;
        legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled, target=_top];
        }
        """
    assert _striplines(dot_str) == _striplines(exp)
    assert "<node>" not in dot_str
