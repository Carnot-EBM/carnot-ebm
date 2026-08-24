"""Regression tests for the 2026-08-23 QA-layer audit MISSED INPUTS.

Origin: the milestone-close QA-layer audit (`ops/qa_layer_authenticity_audit_report.md`,
2026-08-23) returned six `SILENT_NON_FIRING` verdicts and one `REAL_BUG` against
`scripts/adversarial_verify.py`. Per the CLAUDE.md "QA-Layer Authenticity Discipline",
every `## MISSED INPUT` line names a concrete input that falls inside a guard's own
stated concept and gets through anyway -- so each line is both the widening and the
regression test. Each test below uses the audit's input VERBATIM, not a synthetic
happy path, so a future narrowing of the same pattern list fails here first.

Two of the seven findings are REFUTED rather than fixed; both keep a test anyway,
because a refutation that nothing guards is one refactor away from becoming true:

  * `_numeric_pairs` was reported blind to principle-wrapped numbers. It is, in
    isolation -- but the live `verify_artifact` path normalizes every top-level
    `{"principle": ..., "value": ...}` wrapper before any check runs (the 2026-07-02
    exp5161 fix). `TestNumericPairsWrappedFieldsLivePath` pins that end-to-end
    behavior so deleting the normalizer call is caught here.
  * `_declares_terminal_artifact_readiness` was reported blind to a readiness
    declaration wrapped at the PAYLOAD level. No artifact has that shape (the
    wrapper convention wraps individual fields), and REQ-INFRA-6262 deliberately
    makes principle-wrapped values gate-INELIGIBLE rather than unwrapping them.
    `TestReadinessTriggerFieldLevelWrapper` pins what a real wrapped `status`
    actually does.

Spec refs: REQ-INFRA-6262 / SCENARIO-INFRA-6262-2 (declared-artifact readiness
boundary, exercised by `TestNonCapstoneBasenameDoesNotExempt`). The remaining
tests cover operational lint helpers with no OpenSpec capability of their own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.adversarial_verify as av
from carnot.terminal_artifacts import TerminalClassification


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _kinds(report: dict[str, Any]) -> set[str]:
    return {flag["kind"] for flag in report["flags"]}


class TestIsFiniteNumberNumericTypes:
    """REAL_BUG: `_is_finite_number` rejected real finite numbers.

    Audit MISSED INPUT: `numpy.float32(0.913)` stored as an in-memory `auroc` value.
    A numpy scalar cannot reach here through `json.load`, but experiment scripts
    import this module and check their own in-memory dicts before writing them --
    that is the path where a metric silently skips every numeric check.
    """

    def test_accepts_numpy_float32_auroc(self) -> None:
        assert av._is_finite_number(np.float32(0.913)) is True

    def test_accepts_numpy_int64(self) -> None:
        assert av._is_finite_number(np.int64(7)) is True

    def test_still_rejects_numpy_bool(self) -> None:
        # bool is not a measurement; the original bool carve-out must survive
        # the widening, or True/False start entering tautology comparison.
        assert av._is_finite_number(np.bool_(True)) is False

    def test_still_rejects_python_bool(self) -> None:
        assert av._is_finite_number(True) is False
        assert av._is_finite_number(False) is False

    def test_huge_finite_int_does_not_raise(self) -> None:
        # `float(10**400)` raises OverflowError. A predicate that crashes takes
        # the whole verifier down on one oversized integer in one artifact.
        assert av._is_finite_number(10**400) is False

    def test_still_rejects_non_numbers(self) -> None:
        assert av._is_finite_number("0.5") is False
        assert av._is_finite_number(None) is False
        assert av._is_finite_number(float("inf")) is False
        assert av._is_finite_number(float("nan")) is False


class TestIsTimestampFieldMeasuredDelta:
    """SILENT_NON_FIRING: a measured delta was classified as a wall-clock instant.

    Audit MISSED INPUT: `results/experiment_5039_self_play_verifier_checkpoint.json`
    contains `"checkpoint_mtime_delta_ns": 246740982365865`. That is an elapsed
    measurement, not an instant, so exempting it removes a real metric from
    tautology detection.
    """

    def test_measured_mtime_delta_is_not_a_timestamp(self) -> None:
        assert av._is_timestamp_field("checkpoint_mtime_delta_ns") is False

    def test_elapsed_and_duration_markers_are_not_timestamps(self) -> None:
        assert av._is_timestamp_field("elapsed_timestamp_delta_s") is False
        assert av._is_timestamp_field("wall_clock_duration_ns") is False

    def test_a_runtime_is_a_span_not_an_instant(self) -> None:
        # Audit COUNTEREXAMPLE: `runtime_ms` was exempt because "time" occurs
        # inside "runtime", so `runtime_ms == solver_latency_ms` to seven
        # digits was never compared.
        assert av._is_timestamp_field("runtime_ms") is False
        assert av._is_timestamp_field("solver_latency_ms") is False

    def test_genuine_epoch_timestamps_are_still_exempt(self) -> None:
        # exp4763 origin case: two nanosecond epoch instants share leading sig
        # figs by construction. Widening must not re-open that false positive.
        assert av._is_timestamp_field("checkpoint_mtime_before_ns") is True
        assert av._is_timestamp_field("checkpoint_mtime_after_ns") is True
        assert av._is_timestamp_field("run_started_timestamp") is True
        assert av._is_timestamp_field("created_ts") is True


class TestIsCountFieldOrdinaryCountVocabulary:
    """SILENT_NON_FIRING: ordinary count nouns were not recognized as counts.

    Audit MISSED INPUT: `{"folds": 5}` -- a cross-validation fold count is exactly
    the small combinatorial integer the docstring describes, yet the classifier
    returned negative.
    """

    def test_folds_is_a_count(self) -> None:
        assert av._is_count_field("folds") is True

    @pytest.mark.parametrize(
        "name", ["trials", "replicates", "draws", "samples", "attempts", "repeats"]
    )
    def test_sibling_count_nouns_are_counts(self, name: str) -> None:
        assert av._is_count_field(name) is True

    def test_existing_count_markers_still_recognized(self) -> None:
        assert av._is_count_field("n_samples") is True
        assert av._is_count_field("completed_count") is True
        assert av._is_count_field("wins") is True

    def test_a_plain_score_is_still_not_a_count(self) -> None:
        assert av._is_count_field("auroc") is False
        assert av._is_count_field("final_loss") is False


class TestIsChanceFloorScorePermutedControl:
    """SILENT_NON_FIRING: a permutation control was not recognized as chance-floor.

    Audit MISSED INPUT: `binary_permuted_label_accuracy` is a realistic
    shuffled-label control whose expected value is 0.5, but "permuted" was absent
    from the marker tuple -- so two honest controls both sitting at the 0.5 floor
    were compared as if they were two distinct measurements.
    """

    def test_permuted_label_control_is_chance_floor(self) -> None:
        assert av._is_chance_floor_score("binary_permuted_label_accuracy") is True

    def test_permutation_control_is_chance_floor(self) -> None:
        assert av._is_chance_floor_score("label_permutation_control_score") is True

    def test_existing_markers_still_recognized(self) -> None:
        assert av._is_chance_floor_score("loo_auroc_majority_control") is True
        assert av._is_chance_floor_score("shuffled_control_auroc") is True

    def test_a_plain_accuracy_is_not_chance_floor(self) -> None:
        assert av._is_chance_floor_score("final_accuracy") is False


class TestNonCapstoneBasenameDoesNotExempt:
    """SILENT_NON_FIRING: `"capstone" in path_name` matched inside `noncapstone`.

    Audit MISSED INPUT: `complete_partial_not_attempted` as the honest verdict of a
    nonterminal partial artifact whose basename contains `noncapstone` -- the
    function returns without adding a critical flag. Same class as the 2026-07-03
    `"diffusiongemma_met" in verdict` matching inside `meta_tensor`.

    Spec: REQ-INFRA-6262 / SCENARIO-INFRA-6262-2 (nonterminal classes fail closed).
    """

    @staticmethod
    def _classification(path: str) -> TerminalClassification:
        return TerminalClassification(
            classification="partial",
            terminal=False,
            reason="partial declaration",
            path=path,
            status_raw=None,
            honest_verdict_raw="complete_partial_not_attempted",
        )

    def test_noncapstone_partial_is_still_flagged_critical(self) -> None:
        flags: list[av.Flag] = []
        av.check_terminal_artifact_readiness(
            self._classification("results/experiment_4217_noncapstone_ablation.json"), flags
        )
        assert [f.kind for f in flags] == [av.NONTERMINAL_FLAG_KIND]
        assert flags[0].severity == "critical"

    def test_a_real_capstone_partial_is_still_exempt(self) -> None:
        # The carve-out itself is deliberate: a capstone may honestly declare a
        # partial. Only the substring collision is being closed.
        flags: list[av.Flag] = []
        av.check_terminal_artifact_readiness(
            self._classification("results/experiment_4217_capstone_v420.json"), flags
        )
        assert flags == []


class TestReadinessTriggerFieldLevelWrapper:
    """REFUTED finding, pinned: the readiness trigger and the wrapper convention.

    The audit's MISSED INPUT wraps the WHOLE payload
    (`{"principle": ..., "value": {"status": "ready"}}`). No artifact has that
    shape -- the convention wraps individual fields -- and REQ-INFRA-6262 makes
    principle-wrapped values gate-ineligible on purpose. A wrapped `status` FIELD,
    which is the real shape (58 artifacts carry one), still triggers the check.
    """

    def test_field_level_wrapped_status_still_triggers(self) -> None:
        payload = {"status": {"principle": "Terminal state of the run.", "value": "complete"}}
        assert av._declares_terminal_artifact_readiness(payload) is True

    def test_bare_status_still_triggers(self) -> None:
        assert av._declares_terminal_artifact_readiness({"status": "complete"}) is True

    def test_payload_without_status_does_not_trigger(self) -> None:
        assert av._declares_terminal_artifact_readiness({"honest_verdict": "complete_x"}) is False


class TestNumericPairsWrappedFieldsLivePath:
    """REFUTED finding, pinned: wrapped numbers DO reach the tautology check.

    The audit's MISSED INPUT is a real artifact carrying
    `auroc = {"principle": "Held-out AUROC", "value": 0.997}` and
    `baseline_auroc = {"principle": "Baseline AUROC", "value": 0.512}`. Calling
    `_numeric_pairs` directly on that dict does return nothing -- but the live
    path normalizes wrappers first, so the pair is produced. 49 corpus artifacts
    carry two or more wrapped numeric top-level fields and depend on this.
    """

    def test_direct_call_on_wrapped_dict_returns_nothing(self) -> None:
        wrapped = {
            "auroc": {"principle": "Held-out AUROC", "value": 0.997},
            "baseline_auroc": {"principle": "Baseline AUROC", "value": 0.512},
        }
        assert av._numeric_pairs(wrapped) == []

    def test_normalized_dict_yields_the_pair(self) -> None:
        wrapped = {
            "auroc": {"principle": "Held-out AUROC", "value": 0.997},
            "baseline_auroc": {"principle": "Baseline AUROC", "value": 0.512},
        }
        assert av._numeric_pairs(av._normalize_principle_wrapped_fields(wrapped)) == [
            ("auroc", "baseline_auroc", 0.997, 0.512)
        ]

    def test_live_path_flags_a_wrapped_tautology_exactly_like_a_bare_one(
        self, tmp_path: Path
    ) -> None:
        common = {"experiment": 9001, "honest_verdict": "complete_probe"}
        wrapped = _write(
            tmp_path,
            "experiment_9001_wrapped.json",
            {
                **common,
                "final_train_loss": {"principle": "Training loss", "value": 0.4271883},
                "final_val_loss": {"principle": "Validation loss", "value": 0.4271883},
            },
        )
        bare = _write(
            tmp_path,
            "experiment_9002_bare.json",
            {**common, "final_train_loss": 0.4271883, "final_val_loss": 0.4271883},
        )
        assert "TAUTOLOGY" in _kinds(av.verify_artifact(wrapped))
        assert _kinds(av.verify_artifact(wrapped)) == _kinds(av.verify_artifact(bare))


class TestReportHonestVerdictReadsThroughWrapper:
    """The report's `honest_verdict` was read from the pre-normalization payload.

    155 corpus artifacts wrap `honest_verdict`. Every CHECK sees the unwrapped
    value, but the returned report read `d_raw` and fell back to `""` on a dict --
    so a caller reading the report saw an empty verdict for exactly the field the
    2026-07-02 origin incident was about.
    """

    def test_wrapped_verdict_appears_in_the_report(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "experiment_9003_wrapped_verdict.json",
            {
                "experiment": 9003,
                "honest_verdict": {
                    "principle": "Self-declared terminal state.",
                    "value": "complete_probe_ran",
                },
            },
        )
        assert av.verify_artifact(path)["honest_verdict"] == "complete_probe_ran"

    def test_bare_verdict_still_appears(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "experiment_9004_bare_verdict.json",
            {"experiment": 9004, "honest_verdict": "complete_probe_ran"},
        )
        assert av.verify_artifact(path)["honest_verdict"] == "complete_probe_ran"
