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

Spec refs: REQ-ARC-WMTE-6042 / SCENARIOs: the-origin-miss-is-named,
every-prompt-covers-every-class, a-new-target-is-audited-before-the-old-ones,
a-wired-guard-cannot-go-unclassified, a-constructed-missed-input-is-not-hallucination,
the-audit-never-edits.

(This file previously declared "Spec refs: none (operational tooling, no OpenSpec
capability)". That was true when it only covered chunk extraction and verdict parsing.
The 2026-07-29 extension gave it a real requirement to trace to: REQ-ARC-WMTE-6042,
the sibling of REQ-ARC-WMTE-6041 -- 6041 is about the record being rewritten, 6042 is
about the DETECTION layer's blind spots.)
"""

from __future__ import annotations

import argparse
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


class TestRotationResetsWhenTheUnitListChanges:
    """Ordering the guards first does nothing on its own -- the offset is persisted.

    Origin (2026-07-29 review): ten guard targets were added at the HEAD of the unit list
    while `ops/.qa_layer_audit_rotation.json` held `{"offset": 45}` against a 168-unit
    corpus. The conductor runs `--limit 20`, so the entire newly-covered guard surface --
    added that day because of a live incident, and including the very guard that failed to
    stop a README rewrite -- would not have been audited for roughly seven milestone-closes.
    `all_target_paths()`'s docstring asserted the opposite in plain words.
    """

    @staticmethod
    def _units(labels: list[str]) -> list[tuple[str, str, str]]:
        return [(lbl, "body", "prompt") for lbl in labels]

    def test_signature_changes_when_a_target_is_added(self) -> None:
        before = qla.units_signature(self._units(["a", "b"]))
        after = qla.units_signature(self._units(["guard", "a", "b"]))
        assert before != after

    def test_signature_is_stable_across_body_edits(self) -> None:
        """Editing a function body must NOT discard rotation progress, or an actively
        developed file would re-audit its own head slice forever and never reach the tail."""
        a = [("lbl", "one body", "p")]
        b = [("lbl", "a completely different body", "p")]
        assert qla.units_signature(a) == qla.units_signature(b)

    def test_the_count_prefix_disambiguates_a_real_label_collision(self) -> None:
        """Pins the `len(units)` prefix, which the label hash otherwise hides.

        Mutation testing caught this: dropping the count left the suite green, because for
        ordinary inputs any change to the unit list also changes the joined labels. The
        prefix earns its place on exactly one input shape -- the newline join is ambiguous,
        so ONE label containing a newline hashes identically to TWO labels split at it.
        Without the count those two lists share a signature, and a rotation offset measured
        against one would be silently reused against the other.
        """
        one_label_with_newline = [("a\nb", "body", "p")]
        two_labels = [("a", "body", "p"), ("b", "body", "p")]
        assert qla.units_signature(one_label_with_newline) != qla.units_signature(two_labels)

    def test_signature_changes_when_a_unit_is_renamed(self) -> None:
        assert qla.units_signature(self._units(["old_name"])) != qla.units_signature(
            self._units(["new_name"])
        )

    def test_stale_offset_is_discarded_when_the_list_changed(self) -> None:
        sig_now = qla.units_signature(self._units(["guard", "a", "b"]))
        prior = {"offset": 45, "units_signature": "168:deadbeefdeadbeef"}
        assert qla.resolve_rotation_offset(prior, sig_now) == 0

    def test_matching_signature_preserves_progress(self) -> None:
        sig = qla.units_signature(self._units(["a", "b", "c"]))
        assert qla.resolve_rotation_offset({"offset": 2, "units_signature": sig}, sig) == 2

    def test_legacy_state_without_a_signature_restarts(self) -> None:
        """The exact file on disk when this was found: `{"offset": 45}`, no signature.
        It must resolve to 0 so the guards are reached on the very next run."""
        sig = qla.units_signature(self._units(["guard", "a"]))
        assert qla.resolve_rotation_offset({"offset": 45}, sig) == 0

    def test_unreadable_or_malformed_state_restarts(self) -> None:
        sig = qla.units_signature(self._units(["a"]))
        assert qla.resolve_rotation_offset(None, sig) == 0
        assert qla.resolve_rotation_offset("not-a-dict", sig) == 0
        assert qla.resolve_rotation_offset({"offset": "xx", "units_signature": sig}, sig) == 0
        assert qla.resolve_rotation_offset({"offset": -5, "units_signature": sig}, sig) == 0

    def test_guards_are_at_the_head_so_a_reset_reaches_them_first(self) -> None:
        """The reset only helps because the guards are ordered first; both halves needed."""
        paths = qla.all_target_paths()
        guard_names = {p.name for p, _ in qla.GUARD_TARGETS}
        assert {p.name for p in paths[: len(guard_names)]} == guard_names


class TestOnlyRealFilesCountAsEvidenceByPath:
    """`exists_in_repo` only ever UN-voids a span, so every widening weakens the guard.

    The loose version accepted any quoted span containing a slash that resolved to anything
    on disk -- so the bare string `scripts/` was admitted as high-specificity evidence for a
    flagged verdict. A directory is a gesture, not a citation.
    """

    BODY = "def f():\n    return 1\n"

    def test_a_bare_directory_does_not_un_void_a_verdict(self, tmp_path: Path) -> None:
        (tmp_path / "scripts").mkdir()
        report = "## FINDINGS\nthe bug is in `scripts/` somewhere\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert "scripts/" in missing, (
            "a directory was accepted as evidence -- this un-voids a flagged verdict on a "
            "span that cites nothing in particular"
        )

    def test_a_real_file_still_counts(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_3946_r11l_first_solve.json").write_text("{}")
        report = (
            "## FINDINGS\nit dropped a field from `results/experiment_3946_r11l_first_solve.json`\n"
        )
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert missing == []

    def test_an_extensionless_real_file_is_not_evidence(self, tmp_path: Path) -> None:
        """Suffix allow-list, not merely is_file(): keeps the un-void path narrow."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "CNAME").write_text("carnot-ebm.org")
        report = "## FINDINGS\nsee `docs/CNAME` for the problem\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert "docs/CNAME" in missing

    def test_a_plausible_but_absent_path_is_still_flagged(self, tmp_path: Path) -> None:
        report = "## FINDINGS\nbroken in `scripts/does_not_exist.py`\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert "scripts/does_not_exist.py" in missing

    def test_a_directory_named_like_a_file_is_not_evidence(self, tmp_path: Path) -> None:
        """Pins `is_file()` specifically, which the suffix allow-list otherwise hides.

        Mutation testing caught this: reverting `is_file()` to `exists()` left the suite
        green, because the only directory the other tests try (`scripts/`) is already
        rejected by the suffix check one line earlier. The two rules overlap for every
        input in the current repo -- there is no directory here whose name ends in a
        citable suffix -- so without this case `is_file()` is a decorative rule whose test
        passes because of its neighbour. That is exactly the UNTESTED PATTERN class.

        Keeping both is deliberate rather than redundant: the suffix list answers "is this
        shaped like a citation", `is_file()` answers "is there really a file there", and a
        pipeline that mkdir'd `results/foo.json` would satisfy the first and not the second.
        """
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_9999_oops.json").mkdir()
        report = "## FINDINGS\nsee `results/experiment_9999_oops.json`\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert "results/experiment_9999_oops.json" in missing


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


# ==========================================================================================
# 2026-07-29 extension: the silent-non-firing classes, the guard prompt, the scope
# self-check, and the two integrity-guard corrections the extension needs to not eat itself.
# ==========================================================================================


class TestConstructedSectionsAreExemptFromTheIntegrityGuard:
    """The single most load-bearing test in this file.

    The extension's central question asks the reviewer to NAME AN INPUT THE CODE DOES NOT
    CONTAIN. The pre-existing integrity guard fact-checks quoted spans against the audited
    source and voids the verdict when a span is absent. Those two are in direct opposition:
    without the section exemption, every CORRECT answer to the new question is discarded as
    a hallucination, and the whole extension is decorative -- it would ask a good question
    and then throw the answer away.
    """

    OLD_LINT = "def _corrigendum_keys(d):\n    return {k for k in d if 'corrigendum' in k}\n"

    def test_missed_input_naming_an_absent_artifact_path_is_not_voided(self) -> None:
        report = (
            "## VERDICT\nSILENT_NON_FIRING\n\n"
            "## FINDINGS\n1. The key filter is narrower than the concept.\n\n"
            "## MISSED INPUT\n`results/experiment_3946_r11l_first_solve.json` lost "
            "`inference_substrate_correction_note`\n\n"
            "## RECOMMENDATION\nWIDEN_PATTERN_TO_CONCEPT\n"
        )
        _high, missing = qla.verify_quoted_evidence(report, self.OLD_LINT)
        assert missing == [], (
            "a constructed MISSED INPUT was treated as hallucinated evidence -- this is the "
            "failure mode that would make the whole 2026-07-29 extension decorative"
        )

    def test_counterexample_section_is_exempt_too(self) -> None:
        report = (
            "## VERDICT\nREAL_BUG\n\n"
            "## FINDINGS\n1. Narrow filter.\n\n"
            "## COUNTEREXAMPLE\n`results/experiment_9999_made_up.json` with "
            "`{'flagged_adversarial': true}`\n"
        )
        _high, missing = qla.verify_quoted_evidence(report, self.OLD_LINT)
        assert missing == []

    def test_hallucinated_evidence_in_findings_is_still_caught(self) -> None:
        """The exemption must not disarm the guard where it still applies."""
        report = "## FINDINGS\n1. It calls `np.random.randn(48)` to fabricate scores.\n"
        _high, missing = qla.verify_quoted_evidence(report, self.OLD_LINT)
        assert missing == ["np.random.randn(48)"]

    def test_unstructured_report_is_checked_whole(self) -> None:
        """No headings means no sections to exempt -- fall back to checking everything."""
        report = "the code calls `np.random.randn(48)` somewhere"
        _high, missing = qla.verify_quoted_evidence(report, self.OLD_LINT)
        assert missing == ["np.random.randn(48)"]


class TestRealRepoPathsCountAsEvidence:
    """A path to a file that genuinely exists is not a hallucination.

    A reviewer explaining what a guard failed to protect will naturally cite the artifact
    that lost the field. That filename cannot appear inside the guard's own source, so the
    original check called it fabricated and voided a true finding.
    """

    BODY = "def check():\n    return sorted(p for p in paths if p.endswith('.json'))\n"

    def test_existing_repo_path_is_present_when_repo_root_given(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_1.json").write_text("{}")
        report = "## FINDINGS\n1. See `results/experiment_1.json`.\n"
        high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert high == ["results/experiment_1.json"]
        assert missing == []

    def test_same_path_is_missing_without_repo_root(self, tmp_path: Path) -> None:
        """Proves the repo-existence branch is the thing doing the work, not a coincidence."""
        report = "## FINDINGS\n1. See `results/experiment_1.json`.\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY)
        assert missing == ["results/experiment_1.json"]

    def test_nonexistent_path_is_still_flagged_with_repo_root(self, tmp_path: Path) -> None:
        report = "## FINDINGS\n1. See `results/experiment_never_existed.json`.\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert missing == ["results/experiment_never_existed.json"]

    def test_path_escape_is_not_probed(self, tmp_path: Path) -> None:
        """`..` must not let the existence check wander outside the repo."""
        report = "## FINDINGS\n1. See `../../etc/passwd`.\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=tmp_path)
        assert missing == ["../../etc/passwd"]

    def test_interior_dotdot_escaping_to_a_real_file_is_not_evidence(self, tmp_path: Path) -> None:
        """The `..` rejection, on the only input shape where it is load-bearing.

        Added 2026-07-29 after mutation testing showed the check had gone decorative, and
        the reason turned out to be more interesting than the mutation: `..` is unreachable
        for a LEADING-dot path, because `candidate` is produced by `.strip("`'\\"()[],.")`
        which includes `.` -- so `../outside.json` has already become `/outside.json` by the
        time the check runs, and it is the `startswith("/")` clause that rejects it. The
        sibling case above (`../../etc/passwd`) is likewise caught by the suffix rule.

        So the `..` clause earns its place on exactly one shape: an INTERIOR traversal, which
        survives the strip, keeps a citable suffix, and resolves to a real file outside the
        given root. Only the `..` clause rejects this one.
        """
        (tmp_path / "outside.json").write_text("{}")
        repo_root = tmp_path / "repo"
        (repo_root / "sub").mkdir(parents=True)
        report = "## FINDINGS\n1. See `sub/../../outside.json`\n"
        _high, missing = qla.verify_quoted_evidence(report, self.BODY, repo_root=repo_root)
        assert missing == ["sub/../../outside.json"], (
            "an interior `..` traversal resolved to a real file outside the repo root and "
            "was accepted as evidence -- the existence check must not wander out of tree"
        )


class TestStripConstructedSections:
    def test_drops_constructed_sections_and_keeps_claims(self) -> None:
        report = (
            "## FINDINGS\nkeep-this\n\n## COUNTEREXAMPLE\ndrop-this\n\n"
            "## MISSED INPUT\ndrop-that\n\n## RATIONALE\nkeep-that\n"
        )
        scoped = qla.strip_constructed_sections(report)
        assert "keep-this" in scoped
        assert "keep-that" in scoped
        assert "drop-this" not in scoped
        assert "drop-that" not in scoped

    def test_unstructured_text_passes_through(self) -> None:
        assert qla.strip_constructed_sections("no headings at all") == "no headings at all"


class TestParseSection:
    def test_extracts_missed_input(self) -> None:
        report = "## MISSED INPUT\ninference_substrate_correction_note\n\n## RECOMMENDATION\nX\n"
        assert qla.parse_section(report, "MISSED INPUT") == "inference_substrate_correction_note"

    def test_none_found_is_treated_as_empty(self) -> None:
        report = "## MISSED INPUT\nnone found\n\n## RECOMMENDATION\nKEEP\n"
        assert qla.parse_section(report, "MISSED INPUT") == ""

    def test_absent_section_is_empty(self) -> None:
        assert qla.parse_section("## VERDICT\nCLEAN\n", "MISSED INPUT") == ""


class TestSilentNonFiringIsAFlaggedVerdict:
    def test_silent_non_firing_reaches_the_operator_action_list(self) -> None:
        assert "SILENT_NON_FIRING" in qla.FLAGGED_VERDICTS

    def test_it_is_counted_in_the_summary_table(self) -> None:
        assert "SILENT_NON_FIRING" in qla.VERDICT_ORDER

    def test_clean_is_not_flagged(self) -> None:
        assert "CLEAN" not in qla.FLAGGED_VERDICTS
        assert "MINOR_RISK" not in qla.FLAGGED_VERDICTS


class TestPromptsAskTheSilentNonFiringQuestion:
    """The prompt IS the check for an LLM audit.

    A prompt that no longer asks the question cannot get the answer, and nothing else in
    this file would notice -- the plumbing would keep working perfectly on a review that
    had stopped looking for the thing.
    """

    ALL_PROMPTS = ("PER_CHUNK_PROMPT", "PER_FILE_PROMPT", "PER_GUARD_PROMPT")

    def test_every_prompt_covers_every_bug_class(self) -> None:
        """EVERY prompt, not just the two that splice SHARED_BUG_CLASSES.

        This replaces a test that asserted the five classes were present in "both original
        prompts" -- and passed, correctly, while PER_GUARD_PROMPT contained neither class D's
        write-target framing nor class E at all. Guards are the ONLY targets that use that
        prompt, so "test tests what the author thought to test" produced a review of the
        newly-covered surface that could not raise the write-target or tracked-state-mutation
        classes even when they were staring at it: an audit fixture writing into `results/`
        on every run drew eight findings from the guard prompt and not one mentioned it.

        Driven off BUG_CLASS_MARKERS rather than a literal list so that adding a sixth class
        without wiring it into every prompt fails here instead of shipping silent.
        """
        missing: list[str] = []
        for name in self.ALL_PROMPTS:
            prompt = getattr(qla, name)
            for marker in qla.BUG_CLASS_MARKERS:
                if marker not in prompt:
                    missing.append(f"{name} is missing bug class {marker!r}")
        assert not missing, "\n".join(missing)

    def test_the_marker_list_covers_all_five_classes(self) -> None:
        """Guards the guard: the loop above is vacuous if BUG_CLASS_MARKERS is emptied."""
        assert len(qla.BUG_CLASS_MARKERS) == 5
        assert "TEST/SIDE-EFFECT MUTATION OF TRACKED STATE" in qla.BUG_CLASS_MARKERS

    def test_guard_prompt_asks_the_two_classes_it_used_to_omit(self) -> None:
        """Regression for the specific 2026-07-29 review finding, in its own right.

        Deliberately separate from the marker sweep above: that one would still pass if
        somebody 'fixed' a future omission by deleting the marker from BUG_CLASS_MARKERS.
        These two classes reaching the GUARD prompt is the finding, so it is asserted
        directly against the prompt text.
        """
        assert "HARDCODED ABSOLUTE WRITE TARGET" in qla.PER_GUARD_PROMPT
        assert "TEST/SIDE-EFFECT MUTATION OF TRACKED STATE" in qla.PER_GUARD_PROMPT
        # class D's distinguishing consequence, not merely the words "absolute path".
        # Asserted on a span that does not cross the prompt's line wrapping.
        assert "original operator checkout" in qla.PER_GUARD_PROMPT
        assert "reproducibility defect" in qla.PER_GUARD_PROMPT

    def test_guard_prompt_leads_with_silent_non_firing(self) -> None:
        assert "SILENT NON-FIRING -- the central question" in qla.PER_GUARD_PROMPT

    def test_guard_prompt_carries_the_origin_incident_verbatim(self) -> None:
        """A hostile reviewer given an abstract instruction returns abstract findings.
        The concrete precedent is what makes the answers concrete."""
        assert "inference_substrate_correction_note" in qla.PER_GUARD_PROMPT
        assert "determination_preservation_lint.py" in qla.PER_GUARD_PROMPT

    def test_guard_prompt_asks_about_fail_open(self) -> None:
        assert "fail CLOSED" in qla.PER_GUARD_PROMPT
        assert "fail OPEN" in qla.PER_GUARD_PROMPT

    def test_every_prompt_requests_the_missed_input_section(self) -> None:
        """Assert the OUTPUT SPEC, not merely a mention of the heading.

        Testing for the bare string `## MISSED INPUT` passed even when the output-format
        block had been renamed, because the closing quoting instruction mentions the heading
        too. A prompt that names a section without requesting it produces replies that never
        contain it, and `parse_section` would silently return nothing forever.
        """
        for prompt in (qla.PER_CHUNK_PROMPT, qla.PER_FILE_PROMPT, qla.PER_GUARD_PROMPT):
            assert "## MISSED INPUT\n<the concrete real-world input from class A" in prompt

    def test_every_prompt_offers_the_silent_non_firing_verdict(self) -> None:
        for prompt in (qla.PER_CHUNK_PROMPT, qla.PER_FILE_PROMPT, qla.PER_GUARD_PROMPT):
            assert "SILENT_NON_FIRING" in prompt

    def test_prompts_tell_the_reviewer_where_constructed_inputs_belong(self) -> None:
        """The section exemption only helps if the reviewer puts constructed inputs in the
        exempt sections; the prompt has to say so."""
        for prompt in (qla.PER_CHUNK_PROMPT, qla.PER_FILE_PROMPT, qla.PER_GUARD_PROMPT):
            assert "exempt from that" in prompt


class TestBuildUnitsRouting:
    def test_guard_target_gets_the_guard_prompt(self) -> None:
        target = qla.GUARD_TARGETS[0][0]
        units = qla.build_units(target)
        assert len(units) == 1
        assert units[0][2] is qla.PER_GUARD_PROMPT

    def test_chunked_target_is_split_into_functions(self) -> None:
        units = qla.build_units(qla.CHUNKED_FILE_TARGETS[0])
        assert len(units) > 1
        assert all(u[2] is qla.PER_CHUNK_PROMPT for u in units)

    def test_small_non_guard_file_is_reviewed_whole(self, tmp_path: Path) -> None:
        src = tmp_path / "small.py"
        src.write_text("def f(d):\n    return d.get('honest_verdict', '').lower()\n")
        units = qla.build_units(src)
        assert len(units) == 1
        assert units[0][2] is qla.PER_FILE_PROMPT

    def test_oversized_file_is_chunked_even_when_unlisted(self, tmp_path: Path) -> None:
        """Size routing must not depend on the file being named in CHUNKED_FILE_TARGETS --
        that list is itself a pattern list, and a huge new target would otherwise be
        'reviewed' in one skim."""
        src = tmp_path / "huge.py"
        body = "".join(
            f"def f{i}(d):\n    v = d.get('x')\n    return str(v).lower()\n\n"
            for i in range(qla.CHUNK_THRESHOLD_LINES // 4 + 20)
        )
        src.write_text(body)
        units = qla.build_units(src)
        assert len(units) > 1
        assert all(u[2] is qla.PER_CHUNK_PROMPT for u in units)

    def test_missing_file_yields_no_units_instead_of_raising(self, tmp_path: Path) -> None:
        assert qla.build_units(tmp_path / "gone.py") == []

    def test_unnormalised_path_to_a_guard_still_routes_to_the_guard_prompt(self) -> None:
        """--file is given a hand-typed path in practice. The pre-2026-07-29 code compared
        the raw Path against absolute targets, so an equivalent-but-unnormalised spelling
        never matched and the file was silently mis-routed to the wrong prompt.

        The detour through `..` is deliberate: a path that is already absolute AND normalised
        would match with or without the resolve(), so a test using one would pass even if the
        normalisation were deleted -- a test that cannot fail, which is the 'untested pattern'
        class this audit was extended to hunt.
        """
        target = qla.GUARD_TARGETS[0][0]
        detoured = qla.PROJECT_ROOT / "scripts" / ".." / "scripts" / target.name
        units = qla.build_units(detoured)
        assert units and units[0][2] is qla.PER_GUARD_PROMPT


class TestScopeSelfCheck:
    """--check-targets: the audit applying its own 'pattern list narrower than its concept'
    finding to its own target list."""

    CONFIG = (
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: some-guard\n"
        "        name: Some guard\n"
        "        entry: python3 scripts/some_guard_lint.py\n"
        "      - id: known-guard\n"
        "        entry: python3 scripts/determination_preservation_lint.py\n"
    )

    def test_parses_script_hooks_from_config(self) -> None:
        hooks = qla._precommit_script_hooks(self.CONFIG)
        assert hooks["some_guard_lint.py"] == "some-guard"
        assert hooks["determination_preservation_lint.py"] == "known-guard"

    def test_unknown_wired_guard_is_reported(self, tmp_path: Path) -> None:
        found = qla.discover_unaudited_guards(config_text=self.CONFIG, guard_dir=tmp_path)
        assert ("some_guard_lint.py", "pre-commit hook: some-guard") in found

    def test_audited_guard_is_not_reported(self, tmp_path: Path) -> None:
        found = dict(qla.discover_unaudited_guards(config_text=self.CONFIG, guard_dir=tmp_path))
        assert "determination_preservation_lint.py" not in found

    def test_acknowledged_script_is_not_reported(self, tmp_path: Path) -> None:
        config = (
            self.CONFIG + "      - id: url\n        entry: python3 scripts/canonical_url_lint.py\n"
        )
        found = dict(qla.discover_unaudited_guards(config_text=config, guard_dir=tmp_path))
        assert "canonical_url_lint.py" not in found

    def test_runtime_guard_directory_is_swept_too(self, tmp_path: Path) -> None:
        """The doc guard that landed 2026-07-29 lives under python/carnot/testing/ and is
        wired through conftest, not pre-commit. A pre-commit-only scan would never see it."""
        (tmp_path / "some_new_doc_guard.py").write_text("x = 1\n")
        (tmp_path / "helpers.py").write_text("x = 1\n")
        found = dict(qla.discover_unaudited_guards(config_text="", guard_dir=tmp_path))
        assert "some_new_doc_guard.py" in found
        assert "helpers.py" not in found, "only guard-shaped module names should be swept"

    def test_private_and_dunder_modules_are_skipped(self, tmp_path: Path) -> None:
        """The skip must be exercised by a name that WOULD otherwise match.

        A first version of this test used `__init__.py`, which the guard-shaped-name filter
        already rejects -- so deleting the underscore skip entirely left the test green. The
        filename here contains `guard` precisely so the underscore rule is the only thing
        keeping it out.
        """
        (tmp_path / "_private_guard_helpers.py").write_text("x = 1\n")
        (tmp_path / "__init__.py").write_text("")
        found = dict(qla.discover_unaudited_guards(config_text="", guard_dir=tmp_path))
        assert found == {}

    def test_live_repo_scope_is_currently_complete(self) -> None:
        """Fails the moment a guard is wired without being classified -- which is the point.

        This is the only test here that reads the real repo, and it is deliberate: it is the
        forcing function. Wiring a new guard is exactly when someone should be asked whether
        it needs reviewing, and the answer has to be written down somewhere durable rather
        than assumed.
        """
        unaudited = qla.discover_unaudited_guards()
        assert unaudited == [], (
            "A guard is wired but unclassified: "
            + ", ".join(f"{n} ({o})" for n, o in unaudited)
            + ". Add it to GUARD_TARGETS in scripts/qa_layer_authenticity_audit.py with a "
            "one-line reason it is in scope, OR to ACKNOWLEDGED_NON_QA_LAYER with the reason "
            "its failure cannot destroy or falsely admit a research determination. Leaving it "
            "unlisted is how the 2026-07-29 silent non-firing happened."
        )


class TestRunOneHoistsMissedInputsOnlyWhenTheVerdictSurvives:
    """End-to-end over the per-unit path with the LLM replaced by a fixed reply.

    Two behaviours are load-bearing and neither is visible from the pure helpers: that a
    surviving SILENT_NON_FIRING reaches the operator's missed-input list at all, and that a
    verdict VOIDED by the integrity guard does not. A voided verdict's constructed inputs
    are exactly as trustworthy as the fabricated evidence that voided it, and this list is
    the part an operator acts on first.
    """

    BODY = "def _corrigendum_keys(d):\n    return {k for k in d if 'corrigendum' in k}\n"

    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(model="stub", model_name=None)

    def _run(self, reply: str) -> tuple[list[str], list[tuple[str, str]], list, dict]:
        out: list[str] = []
        counts: dict[str, int] = {}
        flagged: list[tuple[str, str]] = []
        voids: list = []
        missed: list[tuple[str, str]] = []
        original = qla.call_model
        qla.call_model = lambda *a, **k: (True, reply)  # type: ignore[assignment]
        try:
            qla._run_one(
                "guard.py",
                self.BODY,
                qla.PER_GUARD_PROMPT,
                self._args(),
                out,
                counts,
                flagged,
                voids,
                missed,
            )
        finally:
            qla.call_model = original  # type: ignore[assignment]
        return missed, flagged, voids, counts

    def test_surviving_silent_non_firing_reaches_the_missed_input_list(self) -> None:
        reply = (
            "## VERDICT\nSILENT_NON_FIRING\n\n"
            "## FINDINGS\n1. `corrigendum` is narrower than the concept.\n\n"
            "## MISSED INPUT\ninference_substrate_correction_note\n"
        )
        missed, flagged, voids, counts = self._run(reply)
        assert missed == [("guard.py", "inference_substrate_correction_note")]
        assert flagged == [("guard.py", "SILENT_NON_FIRING")]
        assert voids == []
        assert counts["SILENT_NON_FIRING"] == 1

    def test_voided_verdict_contributes_no_missed_input(self) -> None:
        reply = (
            "## VERDICT\nSILENT_NON_FIRING\n\n"
            "## FINDINGS\n1. It calls `np.random.randn(48)` internally.\n\n"
            "## MISSED INPUT\nsomething_the_auditor_imagined\n"
        )
        missed, flagged, voids, counts = self._run(reply)
        assert missed == []
        assert flagged == []
        assert voids and voids[0][0] == "guard.py"
        assert counts["CANNOT_DETERMINE"] == 1

    def test_a_real_repo_path_cited_as_evidence_does_not_void_the_verdict(self) -> None:
        """Proves `_run_one` actually passes `repo_root` down.

        Severing that one keyword argument is invisible to every test whose reply quotes no
        repo path -- the helper stays correct and the caller stops using it. The reply here
        cites a file that genuinely exists but cannot appear inside the audited chunk, which
        is exactly what a reviewer explaining a guard's blind spot will do.
        """
        reply = (
            "## VERDICT\nSILENT_NON_FIRING\n\n"
            "## FINDINGS\n1. Nothing here covers what `scripts/adversarial_verify.py` writes.\n\n"
            "## MISSED INPUT\ninference_substrate_correction_note\n"
        )
        missed, flagged, voids, _counts = self._run(reply)
        assert voids == [], "a real repo path was treated as a hallucination"
        assert flagged == [("guard.py", "SILENT_NON_FIRING")]
        assert missed == [("guard.py", "inference_substrate_correction_note")]

    def test_clean_verdict_contributes_nothing(self) -> None:
        missed, flagged, voids, counts = self._run("## VERDICT\nCLEAN\n\n## FINDINGS\nnone found\n")
        assert (missed, flagged, voids) == ([], [], [])
        assert counts["CLEAN"] == 1


class TestGuardTargetsAreWellFormed:
    def test_every_guard_carries_a_written_reason(self) -> None:
        for path, reason in qla.GUARD_TARGETS:
            assert reason.strip(), f"{path.name} is in scope with no stated reason"
            assert len(reason) > 30, f"{path.name}'s reason is too thin to audit later"

    def test_the_two_origin_guards_are_in_scope(self) -> None:
        names = {p.name for p, _ in qla.GUARD_TARGETS}
        assert "determination_preservation_lint.py" in names
        assert "test_suite_mutation_check.py" in names

    def test_both_halves_of_the_doc_discipline_are_in_scope(self) -> None:
        """The commit-layer lint and the runtime test-time guard are different code with
        different blind spots; auditing one is not auditing the other."""
        names = {p.name for p, _ in qla.GUARD_TARGETS}
        assert "operator_curated_docs_lint.py" in names
        assert "operator_curated_doc_guard.py" in names

    def test_guards_do_not_overlap_the_other_target_lists(self) -> None:
        guards = {p for p, _ in qla.GUARD_TARGETS}
        assert guards.isdisjoint(set(qla.WHOLE_FILE_TARGETS))
        assert guards.isdisjoint(set(qla.CHUNKED_FILE_TARGETS))

    def test_all_target_paths_puts_guards_first(self) -> None:
        paths = qla.all_target_paths()
        assert paths[: len(qla.GUARD_TARGETS)] == [p for p, _ in qla.GUARD_TARGETS]
