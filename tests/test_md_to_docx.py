"""Regression guard for the md_to_docx CLI footgun.

A multi-file invocation (`md_to_docx a.md b.md c.md`) once treated argv[2] as an OUTPUT path and
overwrote a real .md with .docx binary. This locks in the safe contract:
  - N .md inputs  -> each writes its OWN <name>.docx and NO .md is ever overwritten
  - the explicit 2-arg form (`in.md out.docx`) still works
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "md_to_docx", Path(__file__).resolve().parents[1] / "scripts" / "md_to_docx.py")
md_to_docx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(md_to_docx)

_OOXML_MAGIC = b"PK\x03\x04"


def _write_md(p: Path, title: str):
    p.write_text(f"# {title}\n\nsome **markdown** body.\n\n- one\n- two\n")


def test_multi_file_never_overwrites_an_md(tmp_path, monkeypatch):
    a, b, c = tmp_path / "a.md", tmp_path / "b.md", tmp_path / "c.md"
    for p, t in ((a, "A"), (b, "B"), (c, "C")):
        _write_md(p, t)
    monkeypatch.setattr(sys, "argv", ["md_to_docx.py", str(a), str(b), str(c)])
    assert md_to_docx.main() == 0
    # every .md is still text (NOT clobbered with OOXML), and each got its own .docx
    for p in (a, b, c):
        assert not p.read_bytes().startswith(_OOXML_MAGIC), f"{p.name} was overwritten with binary"
        assert p.with_suffix(".docx").read_bytes().startswith(_OOXML_MAGIC)


def test_explicit_two_arg_form_still_targets_docx(tmp_path, monkeypatch):
    src, out = tmp_path / "in.md", tmp_path / "out.docx"
    _write_md(src, "In")
    monkeypatch.setattr(sys, "argv", ["md_to_docx.py", str(src), str(out)])
    assert md_to_docx.main() == 0
    assert out.read_bytes().startswith(_OOXML_MAGIC)
    assert not src.read_bytes().startswith(_OOXML_MAGIC)   # source untouched
