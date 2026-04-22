"""Tests for Experiment 741 — FR-11 Formal Closure.

Coverage targets (REQ-FR11-001):
- test_spec_md_update_adds_operational: update_spec_md() adds OPERATIONAL without
  removing existing rows. (REQ-FR11-001)
- test_spec_md_update_idempotent: calling update_spec_md() twice does not produce
  duplicate sections. (REQ-FR11-001)
- test_known_issues_update_adds_closed: update_known_issues() adds FR-11 CLOSED entry
  without removing existing content. (REQ-FR11-001)
- test_known_issues_update_idempotent: calling update_known_issues() twice does not
  duplicate the entry. (REQ-FR11-001)
- test_closure_certificate_schema: write_certificate() produces a valid artifact with
  all required carnot.closure.v1 schema fields and status=OPERATIONAL. (REQ-FR11-001)
- test_verify_doc_contains_operational: verify helper returns True only when marker present.
- test_verify_certificate_valid: verify helper validates all required keys. (REQ-FR11-001)

Each test traces to REQ-FR11-001: FR-11 must be documented as operational.
The requirement is that the closure documentation is accurate, complete, and
machine-verifiable so that downstream tooling (traceability reconciler, milestone
planner) can treat FR-11 as closed without re-running the evidence experiments.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: load the module under test with repo root patched to a tmp dir
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))


def _import_module():
    """Import experiment_741 fresh each call (enables per-test path patching)."""
    import importlib
    import experiment_741_fr11_formal_closure as m
    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# REQ-FR11-001: spec.md update adds OPERATIONAL without removing existing rows
# ---------------------------------------------------------------------------


def test_spec_md_update_adds_operational():
    """update_spec_md() adds 'OPERATIONAL' to the spec without removing existing content.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    existing_content = "# Self-Learning Spec\n\n## REQ-LEARN-020: Some requirement\n\nExisting row content.\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text(existing_content, encoding="utf-8")

        with patch.object(m, "_SPEC_PATH", spec_path):
            result = m.update_spec_md()

        assert result is True
        updated = spec_path.read_text(encoding="utf-8")
        # OPERATIONAL marker must be present
        assert "OPERATIONAL" in updated
        # Existing content must not be removed
        assert "REQ-LEARN-020" in updated
        assert "Existing row content." in updated


def test_spec_md_update_idempotent():
    """update_spec_md() called twice does not produce duplicate OPERATIONAL sections.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    existing_content = "# Self-Learning Spec\n\nSome existing content.\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text(existing_content, encoding="utf-8")

        with patch.object(m, "_SPEC_PATH", spec_path):
            m.update_spec_md()
            m.update_spec_md()

        updated = spec_path.read_text(encoding="utf-8")
        # Should appear exactly once
        assert updated.count("FR-11 Formal Closure") == 1


def test_spec_md_update_missing_file():
    """update_spec_md() returns False when the spec file does not exist.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "nonexistent_spec.md"
        with patch.object(m, "_SPEC_PATH", missing):
            result = m.update_spec_md()

    assert result is False


# ---------------------------------------------------------------------------
# REQ-FR11-001: known-issues.md update adds closed entry without removing content
# ---------------------------------------------------------------------------


def test_known_issues_update_adds_closed():
    """update_known_issues() adds FR-11 CLOSED entry without removing existing issues.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    existing_content = (
        "# Known Issues\n\n"
        "| # | Issue | Severity | Workaround |\n"
        "|---|-------|----------|------------|\n"
        "| 1 | PyO3 issue | Low | Set env var |\n\n"
        "## RETRO-033 CLOSED (Exp 720)\nSome content.\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ki_path = Path(tmpdir) / "known-issues.md"
        ki_path.write_text(existing_content, encoding="utf-8")

        with patch.object(m, "_KNOWN_ISSUES_PATH", ki_path):
            result = m.update_known_issues()

        assert result is True
        updated = ki_path.read_text(encoding="utf-8")
        assert "FR-11 CLOSED" in updated
        # Existing content preserved
        assert "PyO3 issue" in updated
        assert "RETRO-033 CLOSED" in updated


def test_known_issues_update_idempotent():
    """update_known_issues() called twice does not duplicate the FR-11 CLOSED entry.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    existing_content = "# Known Issues\n\nSome open issue.\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        ki_path = Path(tmpdir) / "known-issues.md"
        ki_path.write_text(existing_content, encoding="utf-8")

        with patch.object(m, "_KNOWN_ISSUES_PATH", ki_path):
            m.update_known_issues()
            m.update_known_issues()

        updated = ki_path.read_text(encoding="utf-8")
        assert updated.count("FR-11 CLOSED") == 1


def test_known_issues_update_missing_file():
    """update_known_issues() returns False when the file does not exist.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "nonexistent.md"
        with patch.object(m, "_KNOWN_ISSUES_PATH", missing):
            result = m.update_known_issues()

    assert result is False


# ---------------------------------------------------------------------------
# REQ-FR11-001: closure certificate schema validation
# ---------------------------------------------------------------------------


def test_closure_certificate_schema():
    """write_certificate() produces a valid carnot.closure.v1 artifact with required fields.

    The required fields are: requirement, status, closed_in_milestone,
    closing_experiments, evidence, docs_updated, schema.
    status must equal OPERATIONAL.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    docs = ["openspec/capabilities/self-learning/spec.md", "ops/known-issues.md"]

    with tempfile.TemporaryDirectory() as tmpdir:
        cert_path = Path(tmpdir) / "fr11_closure_certificate.json"
        with patch.object(m, "_CERTIFICATE_PATH", cert_path):
            result = m.write_certificate(docs)

        assert result is True
        cert = json.loads(cert_path.read_text(encoding="utf-8"))

    required_fields = {
        "requirement", "status", "closed_in_milestone",
        "closing_experiments", "evidence", "docs_updated", "schema",
    }
    assert required_fields.issubset(cert.keys()), (
        f"Missing fields: {required_fields - cert.keys()}"
    )
    assert cert["requirement"] == "FR-11"
    assert cert["status"] == "OPERATIONAL"
    assert cert["schema"] == "carnot.closure.v1"
    assert 734 in cert["closing_experiments"]
    assert 738 in cert["closing_experiments"]
    assert cert["evidence"]["relay_operational"] is True
    assert cert["evidence"]["tier2_memory_functional"] is True
    assert cert["docs_updated"] == docs


# ---------------------------------------------------------------------------
# verify_doc_contains_operational helper
# ---------------------------------------------------------------------------


def test_verify_doc_contains_operational_true():
    """verify_doc_contains_operational returns True when marker present.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Implementation Status: OPERATIONAL\n")
        path = Path(f.name)

    try:
        assert m.verify_doc_contains_operational(path) is True
    finally:
        path.unlink(missing_ok=True)


def test_verify_doc_contains_operational_false():
    """verify_doc_contains_operational returns False when marker absent.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Implementation Status: Scaffolding\n")
        path = Path(f.name)

    try:
        assert m.verify_doc_contains_operational(path) is False
    finally:
        path.unlink(missing_ok=True)


def test_verify_doc_contains_operational_missing():
    """verify_doc_contains_operational returns False when file does not exist.

    Spec: REQ-FR11-001
    """
    m = _import_module()
    assert m.verify_doc_contains_operational(Path("/nonexistent/path/foo.md")) is False


# ---------------------------------------------------------------------------
# verify_certificate_valid helper
# ---------------------------------------------------------------------------


def test_verify_certificate_valid_good():
    """verify_certificate_valid returns True for a well-formed certificate.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    cert = {
        "requirement": "FR-11",
        "status": "OPERATIONAL",
        "closed_in_milestone": "2026.04.56",
        "closing_experiments": [734, 738],
        "evidence": {"relay_operational": True},
        "docs_updated": ["openspec/capabilities/self-learning/spec.md"],
        "schema": "carnot.closure.v1",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cert, f)
        path = Path(f.name)

    try:
        with patch.object(m, "_CERTIFICATE_PATH", path):
            result = m.verify_certificate_valid()
    finally:
        path.unlink(missing_ok=True)

    assert result is True


def test_verify_certificate_valid_missing_fields():
    """verify_certificate_valid returns False when required fields are absent.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    incomplete = {"requirement": "FR-11", "status": "OPERATIONAL"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(incomplete, f)
        path = Path(f.name)

    try:
        with patch.object(m, "_CERTIFICATE_PATH", path):
            result = m.verify_certificate_valid()
    finally:
        path.unlink(missing_ok=True)

    assert result is False


def test_verify_certificate_valid_wrong_status():
    """verify_certificate_valid returns False when status is not OPERATIONAL.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    cert = {
        "requirement": "FR-11",
        "status": "Scaffolding",
        "closed_in_milestone": "2026.04.56",
        "closing_experiments": [734, 738],
        "evidence": {},
        "docs_updated": [],
        "schema": "carnot.closure.v1",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cert, f)
        path = Path(f.name)

    try:
        with patch.object(m, "_CERTIFICATE_PATH", path):
            result = m.verify_certificate_valid()
    finally:
        path.unlink(missing_ok=True)

    assert result is False


def test_verify_certificate_valid_missing_file():
    """verify_certificate_valid returns False when certificate file does not exist.

    Spec: REQ-FR11-001
    """
    m = _import_module()

    with patch.object(m, "_CERTIFICATE_PATH", Path("/nonexistent/certificate.json")):
        result = m.verify_certificate_valid()

    assert result is False
