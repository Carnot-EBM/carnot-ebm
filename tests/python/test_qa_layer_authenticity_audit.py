"""Tests for scripts/qa_layer_authenticity_audit.py's pure-logic components.

This is the "who audits the auditor" tool (CLAUDE.md "QA-Layer Authenticity
Discipline") -- added 2026-07-03 after a single outer-loop session found four
real bugs in the QA/reconciliation layer in one sitting, none caught by any
existing adversarial audit because none of them were in scope.

Mirrors the untested-by-design pattern of the sibling
scripts/verifier_authenticity_audit.py for the thin LLM-CLI subprocess
wrappers (call_claude/call_gemini/call_codex -- these require live external
CLIs and are exercised by use, not unit tests). Covers the pure-logic pieces
that ARE unit-testable: function-chunk extraction, verdict parsing, the
audit-integrity (Layer 1.5) hallucination guard, and rotation-state math.

Spec refs: none (operational tooling, no OpenSpec capability).
"""

from __future__ import annotations

from pathlib import Path

import scripts.qa_layer_authenticity_audit as qla


class TestExtractRiskyFunctions:
    def test_finds_function_with_dict_get(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text(
            "def check_thing(d):\n"
            "    v = d.get('honest_verdict')\n"
            "    if v is not None:\n"
            "        return v\n"
            "    return None\n"
        )
        chunks = qla.extract_risky_functions(src)
        assert len(chunks) == 1
        assert chunks[0].label == "mod.py::check_thing"

    def test_skips_function_with_no_risky_markers(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text("def add(a, b):\n    return a + b\n")
        chunks = qla.extract_risky_functions(src)
        assert chunks == []

    def test_skips_nested_functions(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text(
            "def outer(d):\n    def inner(x):\n        return x.get('a')\n    return inner(d)\n"
        )
        chunks = qla.extract_risky_functions(src)
        # outer() itself has no risky marker at its own top level scan text
        # (the .get( is inside inner, but since chunk extraction slices by line
        # range including the nested def, outer's body text DOES contain '.get(').
        assert len(chunks) == 1
        assert chunks[0].label == "mod.py::outer"

    def test_skips_tiny_functions(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text("def f(d):\n    return d.get('x')\n")
        chunks = qla.extract_risky_functions(src)
        # body is under the 40-char floor -- excluded to avoid auditing trivial one-liners
        assert chunks == []

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        chunks = qla.extract_risky_functions(tmp_path / "does_not_exist.py")
        assert chunks == []

    def test_returns_empty_for_syntax_error(self, tmp_path: Path) -> None:
        src = tmp_path / "broken.py"
        src.write_text("def f(:\n    this is not python\n")
        chunks = qla.extract_risky_functions(src)
        assert chunks == []


class TestParseVerdict:
    def test_parses_clean_verdict(self) -> None:
        report = "## VERDICT\nCLEAN\n\n## FINDINGS\nnone found\n"
        assert qla.parse_verdict(report) == "CLEAN"

    def test_parses_real_bug_verdict(self) -> None:
        report = "## VERDICT\nREAL_BUG\n\n## FINDINGS\n1. something\n"
        assert qla.parse_verdict(report) == "REAL_BUG"

    def test_returns_unknown_when_missing(self) -> None:
        assert qla.parse_verdict("no structured output here") == "UNKNOWN"


class TestVerifyQuotedEvidence:
    def test_real_evidence_is_not_missing(self) -> None:
        body = "def _flips_gate(d):\n    return 'gate_met' in d.get('honest_verdict', '').lower()\n"
        report = "## FINDINGS\n1. Uses `d.get('honest_verdict')` without unwrapping.\n"
        high, missing = qla.verify_quoted_evidence(report, body)
        assert high
        assert missing == []

    def test_hallucinated_evidence_is_flagged_missing(self) -> None:
        body = "def _flips_gate(d):\n    return 'gate_met' in d.get('honest_verdict', '').lower()\n"
        report = "## FINDINGS\n1. Calls `np.random.randn(48)` to fabricate scores.\n"
        high, missing = qla.verify_quoted_evidence(report, body)
        assert high
        assert missing == ["np.random.randn(48)"]

    def test_low_specificity_spans_are_ignored(self) -> None:
        """Plain identifiers/short spans don't count as high-specificity evidence --
        mirrors the sibling audit's rationale (a symbol name can legitimately be
        referenced even in a wrong verdict, so it shouldn't gate the integrity check)."""
        body = "def f(d):\n    return d.get('x')\n"
        report = "## FINDINGS\n1. See `foo`.\n"
        high, missing = qla.verify_quoted_evidence(report, body)
        assert high == []
        assert missing == []


class TestRotationStateAdvances:
    """The rotation logic itself lives inline in main(); this exercises the same
    slice+wraparound math extracted to a standalone check, matching what main() does."""

    @staticmethod
    def _rotate(units: list[int], offset: int, limit: int) -> tuple[list[int], int]:
        offset = offset % len(units)
        rotated_units = units[offset:] + units[:offset]
        result = rotated_units[:limit]
        next_offset = (offset + limit) % len(rotated_units)
        return result, next_offset

    def test_successive_runs_advance_through_the_list(self) -> None:
        units = list(range(50))
        r1, off1 = self._rotate(units, 0, 20)
        assert r1 == list(range(0, 20))
        r2, off2 = self._rotate(units, off1, 20)
        assert r2 == list(range(20, 40))
        r3, off3 = self._rotate(units, off2, 20)
        # wraps around: 40..49 then 0..9
        assert r3 == list(range(40, 50)) + list(range(0, 10))

    def test_offset_beyond_list_length_wraps_via_modulo(self) -> None:
        units = list(range(10))
        r, off = self._rotate(units, 37, 3)
        assert r == [7, 8, 9]
        assert off == 0

    def test_single_run_covers_whole_short_list(self) -> None:
        units = list(range(5))
        r, off = self._rotate(units, 0, 20)
        assert r == units
        assert off == 0


class TestWholeFileAndChunkedTargetsAreDistinct:
    def test_no_overlap_between_target_sets(self) -> None:
        whole = set(qla.WHOLE_FILE_TARGETS)
        chunked = set(qla.CHUNKED_FILE_TARGETS)
        assert whole.isdisjoint(chunked)

    def test_adversarial_verify_is_chunked_not_whole(self) -> None:
        names = {p.name for p in qla.CHUNKED_FILE_TARGETS}
        assert "adversarial_verify.py" in names
        whole_names = {p.name for p in qla.WHOLE_FILE_TARGETS}
        assert "adversarial_verify.py" not in whole_names
