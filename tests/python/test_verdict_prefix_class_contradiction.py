"""A success-shaped verdict string that declares itself partial.

REQ: REQ-CONDUCTOR-VERDICT-3 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-VERDICT-5 (success prefix + partial -> WARN),
-6 (blocked_ prefix + partial -> silent), -7 (success prefix + other class ->
silent), -8 (principle-wrapped verdict is read through).

Origin, 2026-08-23. Four artifacts opened honest_verdict with `complete_` while
declaring verdict_class: partial. The conductor is enum-first (REQ-CONDUCTOR-
VERDICT-2), so it believed the field, classified each task partial, and re-ran
it toward the 3-fail limit; experiment_6513's task retired permanently that way.
Anyone reading the verdict STRING saw a success.

The same incident was misdiagnosed three times as a substring false positive in
the conductor's token lists. Stripping verdict_class from those artifacts makes
the classifier return trustworthy, which proves the token lists never ran. That
misdiagnosis is why test_the_real_incident_shape_is_not_a_token_false_positive
exists below: it pins the mechanism, not just the symptom.

No test writes tracked state.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

import adversarial_verify as av  # noqa: E402

KIND = "VERDICT_PREFIX_CLASS_CONTRADICTION"


def _flags(verdict, verdict_class, **extra):
    d = {"honest_verdict": verdict, "verdict_class": verdict_class, **extra}
    out: list = []
    av.check_verdict_class_consistency(d, out)
    return out


def _kinds(flags):
    return [f.kind for f in flags]


class TestContradictionFires:
    # The four real artifacts from the origin incident, by their actual verdicts.
    @pytest.mark.parametrize(
        "verdict",
        [
            "complete_v561_evidence_corrigendum_v562_lineage_lock: ...",
            "complete_v563_independent_exact_root: immutable Exp6504 rows ...",
            "complete_v564_terminal_handoff_contract: V563 terminal failures ...",
            "complete_partial_v564_evidence_graph: row-supported structural ...",
        ],
    )
    def test_origin_incident_artifacts(self, verdict: str) -> None:
        # SCENARIO-CONDUCTOR-VERDICT-5.
        assert KIND in _kinds(_flags(verdict, "partial"))

    @pytest.mark.parametrize("prefix", list(av._TERMINAL_SUCCESS_PREFIXES))
    def test_every_terminal_prefix_is_covered(self, prefix: str) -> None:
        # A prefix present in the set but unreachable by the check would be a
        # pattern that exists on paper and does not bite.
        assert KIND in _kinds(_flags(f"{prefix}whatever happened here", "partial"))

    def test_flag_is_warn_not_critical(self) -> None:
        # Quarantining an artifact for a labelling mistake would punish a
        # possibly-sound measurement, and an overreaching gate gets bypassed.
        flag = next(f for f in _flags("complete_x: y", "partial") if f.kind == KIND)
        assert flag.severity == "warn"

    def test_detail_names_the_prefix_and_the_class(self) -> None:
        flag = next(f for f in _flags("success: y", "partial") if f.kind == KIND)
        assert "success:" in flag.detail and "partial" in flag.detail

    def test_uppercase_verdict_still_matches(self) -> None:
        assert KIND in _kinds(_flags("COMPLETE_V563_ROOT: X", "partial"))

    def test_principle_wrapped_verdict_is_read_through(self) -> None:
        # SCENARIO-CONDUCTOR-VERDICT-8. Principle-annotated fields are a
        # documented convention here, and unhandled wrapping is a known bug
        # class in this exact layer (QA-Layer Authenticity Discipline, origin
        # bug #2 -- 176 artifacts silently defeated substrate recognition).
        wrapped = {"value": "complete_v563_root: x", "principle": "why it matters"}
        assert KIND in _kinds(_flags(wrapped, "partial"))


class TestContradictionStaysSilent:
    def test_blocked_prefix_with_partial_is_honest(self) -> None:
        # SCENARIO-CONDUCTOR-VERDICT-6. experiment_6528's real shape: a blocked
        # run genuinely IS a kind of partial. Flagging it would make the check
        # cry wolf on the honest case.
        assert KIND not in _kinds(
            _flags("blocked_v565_source_model_method_contract: gates failed", "partial")
        )

    @pytest.mark.parametrize(
        "cls", ["positive", "circular_positive", "null", "blocked", "disqualified"]
    )
    def test_success_prefix_with_non_partial_class(self, cls: str) -> None:
        # SCENARIO-CONDUCTOR-VERDICT-7.
        assert KIND not in _kinds(_flags("complete_x: y", cls))

    def test_no_verdict_class_declared(self) -> None:
        out: list = []
        av.check_verdict_class_consistency({"honest_verdict": "complete_x: y"}, out)
        assert KIND not in _kinds(out)

    def test_non_string_verdict_does_not_raise(self) -> None:
        assert KIND not in _kinds(_flags(None, "partial"))
        assert KIND not in _kinds(_flags(1234, "partial"))

    def test_prefix_mid_string_is_not_a_match(self) -> None:
        # The whole reason this check is allowed to look at honest_verdict is
        # that it anchors at position 0. A mid-string "complete" must NOT fire,
        # or it becomes the drifting-substring problem REQ-CONDUCTOR-VERDICT-1
        # refuses to reintroduce.
        assert KIND not in _kinds(
            _flags("null_result: the sweep is complete_but_inconclusive", "partial")
        )


class TestMechanism:
    def test_the_real_incident_shape_is_not_a_token_false_positive(self) -> None:
        # The incident was misdiagnosed three times as the conductor's token
        # lists matching "retired"/"partial"/"blocked" inside prose. Pin the
        # actual mechanism: with verdict_class present the conductor classifies
        # from the ENUM and never reaches those lists.
        import research_conductor as rc

        artifact = {
            "honest_verdict": "complete_v563_independent_exact_root: ... retired V562 ...",
            "verdict_class": "partial",
        }
        with_class, _ = rc._verdict_is_untrustworthy(artifact)
        without_class, _ = rc._verdict_is_untrustworthy(
            {k: v for k, v in artifact.items() if k != "verdict_class"}
        )
        assert with_class is True, "enum-first should classify this partial"
        assert without_class is False, (
            "without the enum the terminal-prefix whitelist clears it -- so the "
            "token lists were never the cause"
        )
