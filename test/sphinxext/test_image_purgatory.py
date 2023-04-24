from pathlib import Path
from typing import List

import pytest

from graphtik.sphinxext import DocFilesPurgatory, _image_formats


@pytest.fixture
def img_docs() -> List[str]:
    return [f"d{i}" for i in range(3)]


@pytest.fixture
def img_files(tmpdir) -> List[Path]:
    files = [tmpdir.join(f"f{i}") for i in range(3)]
    for f in files:
        f.ensure()
    return [Path(i) for i in files]


@pytest.fixture
def img_reg(img_docs, img_files) -> DocFilesPurgatory:
    img_reg = DocFilesPurgatory()

    img_reg.register_doc_fpath(img_docs[0], img_files[0])
    img_reg.register_doc_fpath(img_docs[0], img_files[1])
    img_reg.register_doc_fpath(img_docs[1], img_files[0])
    img_reg.register_doc_fpath(img_docs[2], img_files[2])

    return img_reg


def test_image_purgatory(img_docs, img_files, img_reg):
    for _ in range(2):
        img_reg.purge_doc(img_docs[2])
        assert list(img_reg.doc_fpaths) == img_docs[:2]
        assert img_files[0].exists()
        assert img_files[1].exists()
        assert not img_files[2].exists()

    for _ in range(2):
        img_reg.purge_doc(img_docs[1])
        assert list(img_reg.doc_fpaths) == img_docs[:1]
        assert img_files[0].exists()
        assert img_files[1].exists()
        assert not img_files[2].exists()

    img_reg.purge_doc(img_docs[0])
    assert not img_reg.doc_fpaths
    assert not img_files[0].exists()
    assert not img_files[1].exists()
    assert not img_files[2].exists()

    img_reg.purge_doc(img_docs[0])
    img_reg.purge_doc(img_docs[1])
    img_reg.purge_doc(img_docs[2])
