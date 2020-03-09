import os.path as osp
import re
import xml.etree
from typing import Dict
from xml.etree.ElementTree import Element

import html5lib
import pytest

from graphtik.sphinxext import _image_formats

from ..helpers import attr_check, check_xpath, flat_dict


etree_cache: Dict[str, Element] = {}


@pytest.fixture(scope="module")
def cached_etree_parse() -> xml.etree:
    def parse(fname):
        if fname in etree_cache:
            return etree_cache[fname]
        try:
            with (fname).open("rb") as fp:
                etree: xml.etree = html5lib.HTMLParser(
                    namespaceHTMLElements=False
                ).parse(fp)
                etree_cache.clear()
                etree_cache[fname] = etree
                return etree
        except Exception as ex:
            raise Exception(f"Parsing document {fname} failed due to: {ex}") from ex

    yield parse
    etree_cache.clear()


@pytest.mark.parametrize("img_format", _image_formats)
@pytest.mark.sphinx(buildername="html", testroot="graphtik-directive")
# @pytest.mark.test_params(shared_result="test_count_image_files")
def test_count_image_files(make_app, app_params, img_format):
    args, kwargs = app_params
    app = make_app(
        *args,
        confoverrides={"graphtik_default_graph_format": img_format},
        freshenv=True,
        **kwargs,
    )
    app.build(True)

    image_dir = app.outdir / "_images"
    try:

        if img_format is None:
            img_format = "svg"

        image_files = image_dir.listdir()
        n_expected = 4
        if img_format == "png":
            # x2 files for image-maps file.
            n_expected *= 2
        else:
            assert all(f.endswith(img_format) for f in image_files), image_files
        assert len(image_files) == n_expected, image_files
    finally:
        ## Clean up images for the next parametrized run
        #
        for f in image_dir.listdir():
            try:
                (image_dir / f).unlink()
            except Exception:
                pass


@pytest.mark.parametrize(
    "fname,expect",
    flat_dict(
        {
            "index.html": [
                (".//img", attr_check("src", r"\.svg$", count=5)),
                (
                    ".//img",
                    attr_check(
                        "alt",
                        "test_netop1",
                        "test_netop2",
                        "test_netop3",
                        "test_netop4",
                        "test_netop1",
                    ),
                    ".//caption",
                ),
            ]
        }
    ),
)
@pytest.mark.sphinx(buildername="html", testroot="graphtik-directive")
@pytest.mark.test_params(shared_result="test_image_nodes_default_svg")
def test_image_nodes_default_svg(app, cached_etree_parse, fname, expect):
    app.build()
    print(app.outdir / fname)

    etree = cached_etree_parse(app.outdir / fname)
    check_xpath(etree, fname, *expect)
