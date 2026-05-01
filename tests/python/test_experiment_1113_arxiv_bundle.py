"""Tests for ``scripts/experiment_1113_arxiv_latex_bundle_prep.py``.

Spec: REQ-PUB-001 — arXiv LaTeX bundle preparation for Position Paper v3.

These tests cover only the helper functions added for Exp 1113 and the
end-to-end artifact-build path. They do *not* re-validate the entire
arXiv bundle on disk -- the experiment script itself does that, and
duplicating its checks here only burns CI time.

Why these specific tests:

* ``count_latex_environments`` is the cheapest pre-flight check we can
  run against ``main.tex``; if it ever returns mismatched counts on a
  pristine bundle, the bundle will not compile, so we want a unit
  test that pins its behaviour on synthetic input rather than only
  exercising it through the full end-to-end run.
* ``collect_cite_keys`` and ``collect_bib_keys`` together support the
  ``bib_citation_resolution_pct`` field; we test them independently so
  a regression in either is localised.
* ``build_artifact`` is the integration test: it should produce all
  the schema fields the .85 milestone roadmap requires, regardless of
  ``pdflatex`` availability, so we assert on the keys but not on the
  ``honest_verdict`` value (which is environment-dependent).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1113_arxiv_latex_bundle_prep.py"


def _load_module():
    """Load ``experiment_1113_arxiv_latex_bundle_prep`` as a module
    without requiring it to be installed as a package.

    The conductor exposes scripts/ via PYTHONPATH only when invoked
    through its own pipeline; tests run under pytest without that
    setup, so we hand-load the file by spec.
    """
    spec = importlib.util.spec_from_file_location("exp1113", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1113"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1113():
    return _load_module()


def test_count_latex_environments_balanced(exp1113):
    text = r"""
    \begin{document}
    \begin{abstract}foo\end{abstract}
    \end{document}
    """
    assert exp1113.count_latex_environments(text) == (2, 2)


def test_count_latex_environments_imbalanced(exp1113):
    text = r"\begin{a}\begin{b}\end{b}"
    assert exp1113.count_latex_environments(text) == (2, 1)


def test_count_latex_environments_empty(exp1113):
    assert exp1113.count_latex_environments("") == (0, 0)


def test_collect_cite_keys_simple(exp1113):
    tex = r"See \cite{foo} and \cite{bar,baz}, also \citep{qux}."
    keys = exp1113.collect_cite_keys(tex)
    assert keys == {"foo", "bar", "baz", "qux"}


def test_collect_cite_keys_handles_whitespace(exp1113):
    tex = r"\cite{ alpha , beta }"
    assert exp1113.collect_cite_keys(tex) == {"alpha", "beta"}


def test_collect_cite_keys_empty(exp1113):
    assert exp1113.collect_cite_keys("no citations here") == set()


def test_collect_bib_keys_simple(exp1113):
    bib = """
    @article{foo2020bar,
      title={Bar},
    }
    @inproceedings{baz2021qux,
      title={Qux},
    }
    """
    assert exp1113.collect_bib_keys(bib) == {"foo2020bar", "baz2021qux"}


def test_collect_bib_keys_misc_type(exp1113):
    bib = "@misc{goodfire2026,\n  title={...},\n}"
    assert exp1113.collect_bib_keys(bib) == {"goodfire2026"}


def test_count_figures_in_bundle_returns_int(exp1113):
    n = exp1113.count_figures_in_bundle()
    assert isinstance(n, int)
    assert 0 <= n <= len(exp1113.EXPECTED_FIGURE_NAMES)


def test_pdflatex_available_returns_bool(exp1113):
    assert isinstance(exp1113.pdflatex_available(), bool)


def test_index_html_has_preprint_section_returns_bool(exp1113):
    assert isinstance(exp1113.index_html_has_preprint_section(), bool)


def test_build_artifact_has_all_required_fields(exp1113):
    artifact = exp1113.build_artifact()

    required = {
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
        "figures_compiled",
        "figures_pdf_generated",
        "latex_conversion_method",
        "main_tex_path",
        "bibliography_path",
        "arxiv_bundle_path",
        "pdflatex_validation_status",
        "latex_available",
        "github_pages_updated",
        "arxiv_bundle_complete",
        "submission_ready_checklist",
        "honest_verdict",
    }
    missing = required - set(artifact.keys())
    assert not missing, f"missing required fields: {missing}"


def test_build_artifact_honest_verdict_is_known_value(exp1113):
    artifact = exp1113.build_artifact()
    assert artifact["honest_verdict"] in {
        "bundle_complete_pdf_validated",
        "bundle_complete_latex_not_installed",
        "bundle_partial",
        "failed",
    }


def test_build_artifact_checklist_is_list_of_strings(exp1113):
    artifact = exp1113.build_artifact()
    checklist = artifact["submission_ready_checklist"]
    assert isinstance(checklist, list)
    assert all(isinstance(item, str) for item in checklist)
    assert len(checklist) >= 1


def test_build_artifact_figure_count_consistent_with_bool(exp1113):
    artifact = exp1113.build_artifact()
    expected = len(exp1113.EXPECTED_FIGURE_NAMES)
    assert artifact["figures_pdf_generated"] == (artifact["figures_compiled"] == expected)
