# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import abc
import sys
from functools import partial
from operator import add
from textwrap import dedent

import pytest

from graphtik import base, compose, network, operation, plot
from graphtik.modifiers import optional
from graphtik.netop import NetworkOperation
from graphtik.plot import Plotter, Style, installed_plotter, get_installed_plotter


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
    img = pipeline.plot(show=-1)
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

    img = plot.legend(show=-1)
    _check_plt_img(img)


def test_style_Ref():
    s = plot.Ref("arch_url")
    assert str(s) == "https://graphtik.readthedocs.io/en/latest/arch.html"
    assert repr(s) == "Ref(<class 'graphtik.plot.Style'>, 'arch_url')"

    class C:
        arch_url = "1"

    s = s.rebased(C)
    assert str(s) == "1"
    assert (
        repr(s) == "Ref(<class 'test.test_plot.test_style_Ref.<locals>.C'>, 'arch_url')"
    )


def test_plotter_customizations(pipeline, monkeypatch):
    ## default URL
    #
    url = Style.kw_legend["URL"]
    dot = str(pipeline.plot())
    assert url in dot

    ## New installed_plotter
    #
    with installed_plotter(Plotter(style=Style(kw_legend={"URL": None}))):
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
    with installed_plotter(Plotter(style=Style(kw_legend={"URL": url2}))):
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


@pytest.fixture()
def quoting_pipeline():
    return compose(
        "graph",
        operation(name="node", needs=["edge", "digraph: strict"], provides=["<graph>"])(
            add
        ),
    )


def test_node_quoting0(quoting_pipeline):
    dot_str = str(quoting_pipeline.plot())
    print(dot_str)
    exp = dedent(
        """
        digraph graph_ {
        fontname=italic;
        label=<graph>;
        splines=ortho;
        <edge> [shape=rect];
        <digraph&#58; strict> [shape=rect];
        <node> [fontname=italic, shape=oval, tooltip=<&lt;built-in function add&gt;>];
        <graph> [shape=rect];
        <edge> -> <node>;
        <digraph&#58; strict> -> <node>;
        <node> -> <graph>;
        legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled];
        }
        """
    ).strip()
    assert dot_str.strip() == exp


def test_node_quoting1(quoting_pipeline, monkeypatch):
    style = get_installed_plotter().style
    monkeypatch.setattr(style, "kw_op_url", {"url_format": "abc#%s", "target": "_self"})

    dot_str = str(quoting_pipeline.plot())
    print(dot_str)
    exp = dedent(
        """
        digraph graph_ {
        fontname=italic;
        label=<graph>;
        splines=ortho;
        <edge> [shape=rect];
        <digraph&#58; strict> [shape=rect];
        <node> [URL=<abc#_operator.add>, fontname=italic, shape=oval, target=_self, tooltip=<&lt;built-in function add&gt;>];
        <graph> [shape=rect];
        <edge> -> <node>;
        <digraph&#58; strict> -> <node>;
        <node> -> <graph>;
        legend [URL="https://graphtik.readthedocs.io/en/latest/_images/GraphtikLegend.svg", fillcolor=yellow, shape=component, style=filled];
        }
        """
    ).strip()
    assert dot_str.strip() == exp
