"""Tests for the `blocked_<resource>` start-of-verdict terminal-prefix
recognition in `scripts/research_conductor.py:_verdict_is_untrustworthy`.

Origin: 2026-05-27 `.294 milestone incident where four repair-track tasks
(exp3165, exp3168, exp3169, exp3170-queued) wrote honest gated-skip
artifacts with `honest_verdict` starting `blocked_flagged_verifier:` /
`blocked_repair_gate:` after their upstream preconditions correctly
tripped. The conductor's classifier substring-matched bare `blocked` in
`_BLOCKED_TOKENS` and re-ran each task to its 3-fail retirement -
wall-time burn from a misclassification of the honest blocked-precondition
state defined in CLAUDE.md "Pre-Launch Preconditions Discipline".

This test file pins the structural fix: verdicts starting `blocked_` or
`blocked:` with a non-empty resource identifier are recognized as
TERMINAL honest-blocked states (returns False from
_verdict_is_untrustworthy), parallel to the existing `complete:` /
`success:` / `passed:` / `shipped:` terminal-prefix path.

Substring matches of "blocked" elsewhere in the verdict (the exp1473
`telemetry_claim_blocked_adversarial_audit` pattern) continue to flow
through the positive-context whitelist - those are NOT covered by this
fix and should remain trustworthy via the existing 2026-05-07 patch.

Spec coverage: CLAUDE.md "Pre-Launch Preconditions Discipline" +
"Verdict Terminal-Prefix Discipline".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_conductor():
    """Load scripts/research_conductor.py without executing the loop."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "research_conductor.py"
    spec = importlib.util.spec_from_file_location("research_conductor", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["research_conductor"] = mod
    spec.loader.exec_module(mod)
    return mod


_RC = _load_conductor()
_verdict_is_untrustworthy = _RC._verdict_is_untrustworthy


# -----------------------------------------------------------------------------
# Positive cases: `blocked_<resource>` and `blocked:<resource>` at start are
# honest terminal blocked states and must NOT be flagged untrustworthy.
# -----------------------------------------------------------------------------


class TestBlockedTerminalPrefixHonored:
    """Verdicts starting `blocked_<resource>` are honest terminal states."""

    def test_294_repair_track_blocked_flagged_verifier(self) -> None:
        """exp3168 honest_verdict pattern from the .294 incident."""
        payload = {
            "honest_verdict": (
                "blocked_flagged_verifier: gated_skip=true: exp3165 "
                "preflight_passed=false; clean rerun cannot call a model"
            )
        }
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust, "blocked_<resource> with descriptive context must be terminal"

    def test_294_repair_track_blocked_repair_gate(self) -> None:
        """exp3169 honest_verdict pattern from the .294 incident."""
        payload = {
            "honest_verdict": (
                "blocked_repair_gate: repair gate blocked: blocked_flagged_verifier; "
                "gated_skip=true: exp3168 preflight_passed=false"
            )
        }
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust, "blocked_repair_gate must be honored as terminal"

    def test_pre_launch_blocked_model_not_cached(self) -> None:
        """Pre-Launch Preconditions Discipline canonical pattern."""
        payload = {"honest_verdict": "blocked_model_not_cached_gemma_4_26B_A4B"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_pre_launch_blocked_huggingface_credentials(self) -> None:
        """Pre-Launch Preconditions Discipline canonical pattern."""
        payload = {"honest_verdict": "blocked_huggingface_credentials"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_pre_launch_blocked_cuda_unavailable(self) -> None:
        """Pre-Launch Preconditions Discipline canonical pattern."""
        payload = {"honest_verdict": "blocked_cuda_unavailable"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_pre_launch_blocked_vivado_toolchain_missing(self) -> None:
        """Pre-Launch Preconditions Discipline canonical pattern."""
        payload = {"honest_verdict": "blocked_vivado_toolchain_missing"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_pre_launch_blocked_polarfire_ssh_timeout(self) -> None:
        """Pre-Launch Preconditions Discipline canonical pattern."""
        payload = {"honest_verdict": "blocked_polarfire_ssh_timeout"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_conductor_emitted_blocked_doomed_rerun(self) -> None:
        """Conductor's own emission for doomed-rerun-blocks."""
        payload = {"honest_verdict": "blocked_doomed_rerun_no_root_cause"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_kv260_wrong_mechanism_sd_card(self) -> None:
        """KV260 SSH-Not-SD-Card Discipline emission pattern."""
        payload = {
            "honest_verdict": "blocked_kv260_wrong_mechanism_sd_card_precondition"
        }
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_blocked_colon_form_with_resource(self) -> None:
        """`blocked: <resource>` (colon-separator) is also a terminal prefix."""
        payload = {"honest_verdict": "blocked: model_not_cached_qwen35_35b"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust


# -----------------------------------------------------------------------------
# Negative cases: things that should STILL be flagged as untrustworthy.
# -----------------------------------------------------------------------------


class TestBlockedTerminalPrefixNotOverapplied:
    """The fix must not silently accept malformed or genuinely-partial verdicts."""

    def test_bare_blocked_no_resource_is_partial(self) -> None:
        """`blocked` alone (no resource) is malformed - flag as partial."""
        payload = {"honest_verdict": "blocked"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert untrust, "bare 'blocked' without resource identifier must remain partial"

    def test_bare_blocked_underscore_no_resource_is_partial(self) -> None:
        """`blocked_` (prefix with empty suffix) must NOT be honored as terminal."""
        payload = {"honest_verdict": "blocked_"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert untrust

    def test_blocked_in_middle_is_partial_unless_whitelisted(self) -> None:
        """`...blocked...` in middle of verdict is not a terminal-prefix match.

        The exp1473 positive-context pattern (`_claim_blocked` etc.) handles
        the honest-defense case via the existing whitelist. A new in-middle
        `blocked` substring without that whitelist match should still be
        flagged.
        """
        payload = {"honest_verdict": "results_were_blocked_by_corrupt_input"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert untrust

    def test_blocked_inverted_still_partial(self) -> None:
        """Verdict that starts with blocked_ but ALSO contains a strong
        partial-token in the body (e.g. 'inverted') is still terminal-blocked.

        Rationale: the blocked-prefix rule honors the precondition-fail
        convention; downstream tokens describe context but do not retract
        the honest-terminal classification. This pins that the fix is
        prefix-driven, not full-substring-driven.
        """
        payload = {
            "honest_verdict": "blocked_upstream_inverted_no_clean_rerun_present"
        }
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust


# -----------------------------------------------------------------------------
# Sanity: existing terminal-prefix paths still work.
# -----------------------------------------------------------------------------


class TestExistingTerminalPrefixesUnchanged:
    """The patch must NOT break the existing complete:/success:/passed:/shipped: paths."""

    def test_complete_colon_still_trustworthy(self) -> None:
        payload = {"honest_verdict": "complete: experiment ran cleanly"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_complete_underscore_still_trustworthy(self) -> None:
        payload = {"honest_verdict": "complete_no_improvement_honest_negative"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_success_colon_still_trustworthy(self) -> None:
        payload = {"honest_verdict": "success: all gates passed"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_passed_underscore_still_trustworthy(self) -> None:
        payload = {"honest_verdict": "passed_qwen3_logprob_telemetry_topk_available"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_partial_token_marginal_still_flagged(self) -> None:
        """Real partial verdicts (no terminal prefix, partial token in body)
        must still be flagged - we did not weaken the partial-token check."""
        payload = {"honest_verdict": "experiment_partial_marginal_below_threshold"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert untrust


# -----------------------------------------------------------------------------
# Regression case from the prior exp1473 (in-middle "blocked" as positive-
# context terminal good). Must continue to work.
# -----------------------------------------------------------------------------


class TestPriorPositiveBlockedWhitelistUnchanged:
    """The 2026-05-07 positive-context blocked-pattern fix must still apply."""

    def test_exp1473_claim_blocked_audit(self) -> None:
        """exp1473 pattern: audit successfully blocked an unsupported claim."""
        payload = {
            "honest_verdict": "telemetry_claim_blocked_adversarial_audit"
        }
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust

    def test_sakana_attack_blocked(self) -> None:
        """Sakana-defense: attack blocked = defense worked."""
        payload = {"honest_verdict": "carnot_attack_blocked_via_verifier_ensemble"}
        untrust, _ = _verdict_is_untrustworthy(payload)
        assert not untrust
