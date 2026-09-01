"""REQ-CONSTRAINT-6851 three-family isomorphic compatibility tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_6851_three_family_isomorphic_compatibility_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/constraint-verification/spec.md"
EXP6849_PATH = REPO_ROOT / "results/experiment_6849_typed_program_isomorphic_authority_audit.json"
EXP6850_PATH = REPO_ROOT / "results/experiment_6850_three_family_scoring_admission_canary.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _upstreams() -> tuple[dict[str, Any], dict[str, Any]]:
    return _read(EXP6849_PATH), _read(EXP6850_PATH)


class FakeScorer:
    """Return complete forced-score receipts without a model process."""

    def __init__(self, *, compatible_logprob: float = -0.2, violation_logprob: float = -0.6):
        self.compatible_logprob = compatible_logprob
        self.violation_logprob = violation_logprob
        self.calls: list[dict[str, Any]] = []

    def score(
        self,
        prompt_text: str,
        candidate_text: str,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(dict(identity))
        token_count = 3 if identity["exact_label"] is True else 2
        value = (
            self.compatible_logprob if identity["exact_label"] is True else self.violation_logprob
        )
        return {
            "prompt_token_ids": [1, 7, len(prompt_text)],
            "candidate_token_ids": [20 + index for index in range(token_count)],
            "token_logprobs": [value] * token_count,
            "conditional_log_likelihood": value * token_count,
            "raw_receipt": {
                "forced_sequence": True,
                "sampling": False,
                "generation": False,
                "grammar": False,
                "answer_feedback": False,
                "candidate_text_sha256": exp.sha256_text(candidate_text),
            },
        }


def _batch_receipt(model_hf_id: str, batch_index: int = 0) -> dict[str, Any]:
    canary = {
        "prompt_token_ids": [1, 2],
        "candidate_token_ids": [3, 4],
        "token_logprobs": [-0.1, -0.2],
        "conditional_log_likelihood": -0.3,
        "scientific_label": None,
        "supports_margin_claim": False,
    }
    canary["canary_hash"] = exp.canary_hash(canary)
    return {
        "model_hf_id": model_hf_id,
        "batch_index": batch_index,
        "lease_revalidation": {"owner_verified": True, "checksum": f"lease-{batch_index}"},
        "canary": canary,
    }


def _process_receipt(model: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "hf_id": model["hf_id"],
        "pid": 1000 + index,
        "start_time_ticks": 5000 + index,
        "command_hash": f"sha256:command-{index}",
        "process_group_id": 1000 + index,
        "owner_pid": 900,
        "owner_start_time_ticks": 4500,
        "ownership_token_digest": f"sha256:token-{index}",
        "port": 54000 + index,
        "gpu_uuid": "GPU-test",
        "visible_devices": "0",
        "model_hash": model["model_sha256"],
        "tokenizer_hash": model["tokenizer_receipt"]["tokenizer_sha256"],
        "owned_by_task": True,
    }


def _teardown_receipt(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hf_id": model["hf_id"],
        "ownership_verified": True,
        "process_exit_confirmed": True,
        "process_reaped": True,
        "port_release_confirmed": True,
        "leak_free": True,
        "unrelated_process_kill_count_delta": 0,
    }


def _lease_receipt(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hf_id": model["hf_id"],
        "owner": {
            "task_id": f"exp6851-{model['family']}",
            "device_uuid": "GPU-test",
            "token_opaque": True,
        },
        "phase_history": [{"phase": "terminal_complete"}],
        "release": {
            "released": True,
            "phase": "terminal_complete",
            "device_uuid": "GPU-test",
        },
        "lease_valid": True,
    }


def _complete_preconditions() -> dict[str, Any]:
    return {
        "preconditions_ready": True,
        "checks": [exp.gate_check("all_static_gates", True, True)],
        "accelerator_samples": [{"gpu_uuid": "GPU-test", "owned_by_task": True}],
    }


def test_req_constraint_6851_is_spec_anchored() -> None:
    """REQ-CONSTRAINT-6851 owns every required scenario before implementation."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text.split("## REQ-CONSTRAINT-6851:", 1)[1]
    for anchor in (
        "SCENARIO-CONSTRAINT-6851-PRECONDITIONS",
        "SCENARIO-CONSTRAINT-6851-FORCED-SCORING",
        "SCENARIO-CONSTRAINT-6851-TOKEN-ALIGNMENT",
        "SCENARIO-CONSTRAINT-6851-ISOMORPHIC-PAIRING",
        "SCENARIO-CONSTRAINT-6851-LABEL-POSITION",
        "SCENARIO-CONSTRAINT-6851-CHECKPOINT-RESTART",
        "SCENARIO-CONSTRAINT-6851-PROCESS-OWNERSHIP",
        "SCENARIO-CONSTRAINT-6851-RAW-RECEIPTS",
    ):
        assert anchor in section


def test_scenario_6851_preconditions_require_all_frozen_receipts() -> None:
    """SCENARIO-CONSTRAINT-6851-PRECONDITIONS checks gates and frozen hashes."""

    authority, admission = _upstreams()
    observed_hashes = {
        "exp6849": exp.EXPECTED_EXP6849_SHA256,
        "exp6850": exp.EXPECTED_EXP6850_SHA256,
        "exp6849_module": authority["source_artifact_hashes"]["implementation"]["module"][
            "file_sha256"
        ],
    }
    checked = exp.evaluate_preconditions(
        authority=authority,
        admission=admission,
        model_specs=admission["model_specs"],
        observed_hashes=observed_hashes,
        cuda_scoring=True,
        free_ports=True,
        lease_available=True,
    )

    assert checked["preconditions_ready"] is True
    assert checked["blocked_reasons"] == []
    assert all(row["passed"] is True for row in checked["checks"])

    changed = deepcopy(admission["model_specs"])
    changed[0]["tokenizer_receipt"]["tokenizer_sha256"] = "sha256:changed"
    failed = exp.evaluate_preconditions(
        authority=authority,
        admission=admission,
        model_specs=changed,
        observed_hashes=observed_hashes,
        cuda_scoring=True,
        free_ports=True,
        lease_available=True,
    )
    assert "unchanged_model_and_tokenizer_hashes" in failed["blocked_reasons"]


def test_scenario_6851_preconditions_emit_exact_blocked_shape() -> None:
    """SCENARIO-CONSTRAINT-6851-PRECONDITIONS keeps the failed observed value."""

    authority, admission = _upstreams()
    preconditions = {
        "preconditions_ready": False,
        "checks": [exp.gate_check("task_owned_lease", True, False)],
        "blocked_reasons": ["task_owned_lease"],
        "accelerator_samples": [],
    }
    artifact = exp.build_blocked_artifact(
        duration_s=0.5,
        preconditions=preconditions,
        model_specs=admission["model_specs"],
        source_hashes={"exp6849": exp.sha256_text(exp.canonical_json(authority))},
    )

    assert artifact["status"] == exp.BLOCKED_VERDICT
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["compatibility_stream_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "task_owned_lease"
    assert artifact["gate_check_summary"]["observed"] is False
    assert set(artifact) == set(artifact["field_principles"])


def test_scenario_6851_isomorphic_pairing_and_label_position() -> None:
    """SCENARIO-CONSTRAINT-6851-ISOMORPHIC-PAIRING joins by semantic identity."""

    authority, _ = _upstreams()
    rows = exp.materialize_score_inputs(authority, repeats=2)

    assert len(rows) == 4 * 7 * 2
    assert len({row["score_input_identity"] for row in rows}) == len(rows)
    for pair_id in {row["pair_id"] for row in rows}:
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        assert {row["transform_kind"] for row in pair_rows} == {
            "base",
            "identifier_permutation",
            "atom_rename",
            "label_swap",
            "row_reordering",
            "surface_paraphrase",
            "duplicate_removal",
        }
        assert len({row["semantic_pair_identity"] for row in pair_rows}) == 1
        assert len({row["surface_identity"] for row in pair_rows if row["repeat"] == 0}) == 7
        assert {
            row["compatible_label_position"] for row in pair_rows if row["transform_kind"] == "base"
        } == {0}
        assert {
            row["compatible_label_position"]
            for row in pair_rows
            if row["transform_kind"] == "label_swap"
        } == {1}
        assert all(
            sorted(candidate["exact_label"] for candidate in row["candidates"]) == [False, True]
            for row in pair_rows
        )

    manifest = exp.base_isomorphic_join_manifest(rows)
    assert len(manifest) == 4 * 7
    assert all(row["join_key"] == row["semantic_pair_identity"] for row in manifest)


def test_scenario_6851_forced_scoring_masks_prompt_and_reports_length_control() -> None:
    """SCENARIO-CONSTRAINT-6851-FORCED-SCORING uses candidate log-probabilities."""

    authority, admission = _upstreams()
    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]
    scorer = FakeScorer()
    row = exp.score_input_row(
        score_input=score_input,
        model_spec=admission["model_specs"][0],
        scorer=scorer,
    )

    assert row["prompt_masked"] is True
    assert row["no_sampling"] is True
    assert row["no_generation"] is True
    assert row["compatible"]["prompt_token_ids"] == row["violation"]["prompt_token_ids"]
    assert row["compatible"]["conditional_log_likelihood"] == pytest.approx(-0.6)
    assert row["violation"]["conditional_log_likelihood"] == pytest.approx(-1.2)
    assert row["sum_log_likelihood_margin"] == pytest.approx(0.6)
    assert row["mean_token_log_likelihood_margin"] == pytest.approx(0.4)
    assert row["scalar_compatibility_margin"] == row["mean_token_log_likelihood_margin"]
    assert row["compatible_candidate_length"] == 3
    assert row["violation_candidate_length"] == 2
    assert row["candidate_length_delta"] == 1
    assert row["row_hash"] == exp.row_hash(row)


@pytest.mark.parametrize("fault", ["logprob_count", "non_finite", "prompt_tokens"])
def test_scenario_6851_token_alignment_rejects_bad_receipts(fault: str) -> None:
    """SCENARIO-CONSTRAINT-6851-TOKEN-ALIGNMENT rejects incomplete raw scores."""

    authority, admission = _upstreams()
    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]

    class FaultyScorer(FakeScorer):
        def score(
            self,
            prompt_text: str,
            candidate_text: str,
            identity: Mapping[str, Any],
        ) -> dict[str, Any]:
            receipt = super().score(prompt_text, candidate_text, identity)
            if fault == "logprob_count" and identity["exact_label"] is False:
                receipt["token_logprobs"] = receipt["token_logprobs"][:-1]
            if fault == "non_finite" and identity["exact_label"] is False:
                receipt["token_logprobs"][0] = float("nan")
            if fault == "prompt_tokens" and identity["exact_label"] is False:
                receipt["prompt_token_ids"] = [99]
            return receipt

    with pytest.raises(exp.CompatibilityStreamError):
        exp.score_input_row(
            score_input=score_input,
            model_spec=admission["model_specs"][0],
            scorer=FaultyScorer(),
        )


def test_scenario_6851_checkpoint_restart_scores_only_missing_rows(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6851-CHECKPOINT-RESTART preserves verified rows."""

    authority, admission = _upstreams()
    inputs = exp.materialize_score_inputs(authority, repeats=1)[:2]
    model = admission["model_specs"][0]
    first_row = exp.score_input_row(score_input=inputs[0], model_spec=model, scorer=FakeScorer())
    checksum = exp.checkpoint_input_checksum(
        source_hashes={"fixture": "sha256:fixture"},
        model_specs=admission["model_specs"],
        score_inputs=inputs,
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    manifest = exp.build_checkpoint_manifest(
        [first_row],
        expected_row_count=2,
        input_checksum=checksum,
        batch_receipts=[_batch_receipt(str(model["hf_id"]), 0)],
    )
    exp.write_checkpoint(checkpoint_path, manifest)

    scorer = FakeScorer()
    phase = exp.score_missing_batches(
        model_spec=model,
        score_inputs=inputs,
        scorer=scorer,
        checkpoint_path=checkpoint_path,
        input_checksum=checksum,
        expected_total_row_count=2,
        batch_size=1,
        before_batch=lambda _model, batch_index: _batch_receipt(str(model["hf_id"]), batch_index),
    )

    assert phase["resumed_row_count"] == 1
    assert phase["new_row_count"] == 1
    assert len(scorer.calls) == 2
    assert len(phase["rows"]) == 2
    assert phase["checkpoint_manifest"]["complete_row_count"] == 2
    assert [row["batch_index"] for row in phase["batch_receipts"]] == [0, 1]

    tampered = deepcopy(phase["checkpoint_manifest"])
    tampered["rows"][0]["scalar_compatibility_margin"] = 999.0
    exp.write_checkpoint(checkpoint_path, tampered)
    with pytest.raises(exp.CompatibilityStreamError, match="checkpoint_row_hash_mismatch"):
        exp.verified_checkpoint_rows(checkpoint_path, input_checksum=checksum)

    exp.write_checkpoint(checkpoint_path, manifest)
    with pytest.raises(exp.CompatibilityStreamError, match="checkpoint_input_checksum_mismatch"):
        exp.verified_checkpoint_rows(checkpoint_path, input_checksum="sha256:changed")


def test_scenario_6851_checkpoint_rejects_duplicate_identity(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6851-CHECKPOINT-RESTART rejects duplicate rows."""

    authority, admission = _upstreams()
    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]
    row = exp.score_input_row(
        score_input=score_input,
        model_spec=admission["model_specs"][0],
        scorer=FakeScorer(),
    )
    manifest = exp.build_checkpoint_manifest(
        [row, row], expected_row_count=2, input_checksum="sha256:inputs"
    )
    path = tmp_path / "duplicate.json"
    exp.write_checkpoint(path, manifest)
    with pytest.raises(exp.CompatibilityStreamError, match="checkpoint_duplicate_row_identity"):
        exp.verified_checkpoint_rows(path, input_checksum="sha256:inputs")


def test_scenario_6851_raw_receipts_set_completeness_not_margin_direction() -> None:
    """SCENARIO-CONSTRAINT-6851-RAW-RECEIPTS keeps readiness effect-free."""

    authority, admission = _upstreams()
    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]
    rows = [
        exp.score_input_row(
            score_input=score_input,
            model_spec=model,
            scorer=FakeScorer(compatible_logprob=-0.9, violation_logprob=-0.1),
        )
        for model in admission["model_specs"]
    ]
    processes = [
        _process_receipt(model, index) for index, model in enumerate(admission["model_specs"])
    ]
    teardowns = [_teardown_receipt(model) for model in admission["model_specs"]]
    batches = [_batch_receipt(str(model["hf_id"])) for model in admission["model_specs"]]
    leases = [_lease_receipt(model) for model in admission["model_specs"]]
    checkpoint = exp.build_checkpoint_manifest(
        rows,
        expected_row_count=len(rows),
        input_checksum="sha256:inputs",
        batch_receipts=batches,
        process_receipts=processes,
        teardown_receipts=teardowns,
    )
    artifact = exp.build_complete_artifact(
        duration_s=90.0,
        preconditions=_complete_preconditions(),
        model_specs=admission["model_specs"],
        source_hashes={"exp6849": exp.EXPECTED_EXP6849_SHA256},
        score_inputs=[score_input],
        rows=rows,
        process_receipts=processes,
        batch_receipts=batches,
        teardown_receipts=teardowns,
        checkpoint_manifest=checkpoint,
        expected_row_count=len(rows),
        lease_receipts=leases,
    )

    assert artifact["compatibility_stream_complete_score"] == 1
    assert artifact["positive_margin_models"] == []
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["token_score_receipts"]) == 6
    assert len(artifact["per_model_margin_summary"]) == 3
    assert {row["model_hf_id"] for row in artifact["per_model_margin_summary"]} == set(
        exp.MODEL_SPECS
    )
    assert set(artifact) == set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []


def test_scenario_6851_process_ownership_and_raw_receipts_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-6851-PROCESS-OWNERSHIP requires owned clean phases."""

    authority, admission = _upstreams()
    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]
    rows = [
        exp.score_input_row(
            score_input=score_input,
            model_spec=model,
            scorer=FakeScorer(),
        )
        for model in admission["model_specs"]
    ]
    processes = [
        _process_receipt(model, index) for index, model in enumerate(admission["model_specs"])
    ]
    processes[1]["owned_by_task"] = False
    teardowns = [_teardown_receipt(model) for model in admission["model_specs"]]
    batches = [_batch_receipt(str(model["hf_id"])) for model in admission["model_specs"]]
    leases = [_lease_receipt(model) for model in admission["model_specs"]]
    checkpoint = exp.build_checkpoint_manifest(
        rows,
        expected_row_count=3,
        input_checksum="sha256:inputs",
        batch_receipts=batches,
        process_receipts=processes,
        teardown_receipts=teardowns,
    )
    artifact = exp.build_complete_artifact(
        duration_s=90.0,
        preconditions=_complete_preconditions(),
        model_specs=admission["model_specs"],
        source_hashes={"exp6849": exp.EXPECTED_EXP6849_SHA256},
        score_inputs=[score_input],
        rows=rows,
        process_receipts=processes,
        batch_receipts=batches,
        teardown_receipts=teardowns,
        checkpoint_manifest=checkpoint,
        expected_row_count=3,
        lease_receipts=leases,
    )

    assert artifact["compatibility_stream_complete_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "owned_process_receipts"
    source = Path(exp.__file__).read_text(encoding="utf-8")
    assert "OwnedLlamaCppProcess" in source
    assert "GpuLease.acquire" in source


def test_req_6851_validation_and_script_entrypoint(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6851 exposes validation and a thin non-mutating test path."""

    authority, admission = _upstreams()
    blocked = exp.build_blocked_artifact(
        duration_s=0.1,
        preconditions={
            "preconditions_ready": False,
            "checks": [exp.gate_check("cuda_scoring", True, False)],
            "blocked_reasons": ["cuda_scoring"],
            "accelerator_samples": [],
        },
        model_specs=admission["model_specs"],
        source_hashes={"authority": exp.sha256_text(exp.canonical_json(authority))},
    )
    assert exp.validate_artifact(blocked) == []

    malformed = deepcopy(blocked)
    malformed.pop("token_score_receipts")
    malformed["verdict_class"] = "unknown"
    malformed["honest_verdict"] = "blocked_without_complete_prefix"
    errors = exp.validate_artifact(malformed)
    assert "missing_field:token_score_receipts" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_complete_prefixed" in errors

    result_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    assert (
        exp.main(
            [
                "--date",
                "wrong",
                "--result-path",
                str(result_path),
                "--checkpoint-path",
                str(checkpoint_path),
            ]
        )
        == 2
    )
    assert not result_path.exists()
    wrapper = (
        REPO_ROOT
        / "scripts/experiments/experiment_6851_three_family_isomorphic_compatibility_stream.py"
    )
    assert wrapper.is_file()
    assert "package_main" in wrapper.read_text(encoding="utf-8")


def test_req_6851_materialization_and_receipt_fail_closed_branches() -> None:
    """REQ-CONSTRAINT-6851 rejects malformed fixtures and raw token receipts."""

    authority, admission = _upstreams()
    with pytest.raises(exp.CompatibilityStreamError, match="repeat_count_must_be_positive"):
        exp.materialize_score_inputs(authority, repeats=0)
    with pytest.raises(exp.CompatibilityStreamError, match="sanitized_candidate_pairs_missing"):
        exp.materialize_score_inputs({}, repeats=1)

    missing_transform = deepcopy(authority)
    missing_transform["isomorphic_transform_manifest"] = missing_transform[
        "isomorphic_transform_manifest"
    ][1:]
    with pytest.raises(exp.CompatibilityStreamError, match="qualified_transform_missing"):
        exp.materialize_score_inputs(missing_transform, repeats=1)

    invalid_labels = deepcopy(authority)
    invalid_labels["sanitized_candidate_pair_manifest"][0]["candidates"][1]["exact_label"] = True
    with pytest.raises(exp.CompatibilityStreamError, match="one_compatible_one_violation"):
        exp.materialize_score_inputs(invalid_labels, repeats=1)

    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]
    model = admission["model_specs"][0]

    class ReceiptFaultScorer(FakeScorer):
        def __init__(self, fault: str):
            super().__init__()
            self.fault = fault

        def score(
            self,
            prompt_text: str,
            candidate_text: str,
            identity: Mapping[str, Any],
        ) -> dict[str, Any]:
            receipt = super().score(prompt_text, candidate_text, identity)
            if self.fault == "prompt":
                receipt["prompt_token_ids"] = []
            elif self.fault == "candidate":
                receipt["candidate_token_ids"] = []
                receipt["token_logprobs"] = []
            elif self.fault == "raw":
                receipt["raw_receipt"] = {}
            return receipt

    for fault, message in (
        ("prompt", "prompt_tokens_missing"),
        ("candidate", "candidate_tokens_missing"),
        ("raw", "raw_token_score_receipt_missing"),
    ):
        with pytest.raises(exp.CompatibilityStreamError, match=message):
            exp.score_input_row(
                score_input=score_input,
                model_spec=model,
                scorer=ReceiptFaultScorer(fault),
            )

    duplicate_label_input = deepcopy(score_input)
    duplicate_label_input["candidates"][1]["exact_label"] = True
    with pytest.raises(exp.CompatibilityStreamError, match="scored_exact_labels_invalid"):
        exp.score_input_row(
            score_input=duplicate_label_input,
            model_spec=model,
            scorer=FakeScorer(),
        )


def test_req_6851_checkpoint_and_batch_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6851 rejects malformed checkpoints and batch canaries."""

    authority, admission = _upstreams()
    inputs = exp.materialize_score_inputs(authority, repeats=1)[:1]
    model = admission["model_specs"][0]
    missing = tmp_path / "missing.json"
    assert exp.verified_checkpoint_rows(missing, input_checksum="sha256:input") == {}

    array_path = tmp_path / "array.json"
    array_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(exp.CompatibilityStreamError, match="checkpoint_object_required"):
        exp.verified_checkpoint_rows(array_path, input_checksum="sha256:input")

    row = exp.score_input_row(score_input=inputs[0], model_spec=model, scorer=FakeScorer())
    manifest = exp.build_checkpoint_manifest(
        [row], expected_row_count=1, input_checksum="sha256:input"
    )
    manifest["checkpoint_hash"] = "sha256:changed"
    checkpoint_path = tmp_path / "changed-manifest.json"
    exp.write_checkpoint(checkpoint_path, manifest)
    with pytest.raises(exp.CompatibilityStreamError, match="checkpoint_manifest_hash_mismatch"):
        exp.verified_checkpoint_rows(checkpoint_path, input_checksum="sha256:input")

    with pytest.raises(exp.CompatibilityStreamError, match="batch_size_must_be_positive"):
        exp.score_missing_batches(
            model_spec=model,
            score_inputs=inputs,
            scorer=FakeScorer(),
            checkpoint_path=missing,
            input_checksum="sha256:input",
            expected_total_row_count=1,
            batch_size=0,
            before_batch=lambda _model, _batch: {},
        )

    bad_canary = {
        "model_hf_id": model["hf_id"],
        "batch_index": 0,
        "lease_revalidation": {},
        "canary": {
            "candidate_token_ids": [1],
            "token_logprobs": [],
            "scientific_label": "not-unlabeled",
            "supports_margin_claim": True,
            "canary_hash": "bad",
        },
    }
    assert set(exp._batch_receipt_errors(bad_canary)) == {
        "lease_revalidation",
        "canary_token_logprobs",
        "canary_token_alignment",
        "canary_scientific_label",
        "canary_margin_claim",
        "canary_hash",
    }
    missing_tokens = deepcopy(bad_canary)
    missing_tokens["canary"]["candidate_token_ids"] = []
    assert "canary_candidate_tokens" in exp._batch_receipt_errors(missing_tokens)
    with pytest.raises(exp.CompatibilityStreamError, match="batch_precondition_failed"):
        exp.score_missing_batches(
            model_spec=model,
            score_inputs=inputs,
            scorer=FakeScorer(),
            checkpoint_path=missing,
            input_checksum="sha256:input",
            expected_total_row_count=1,
            batch_size=1,
            before_batch=lambda _model, _batch: bad_canary,
        )


def test_req_6851_completeness_helpers_and_effect_verdict_branches() -> None:
    """REQ-CONSTRAINT-6851 covers positive, mixed, and malformed receipt outcomes."""

    authority, admission = _upstreams()
    score_input = exp.materialize_score_inputs(authority, repeats=1)[0]

    def build_with_signs(signs: list[bool]) -> dict[str, Any]:
        rows = [
            exp.score_input_row(
                score_input=score_input,
                model_spec=model,
                scorer=FakeScorer(
                    compatible_logprob=-0.1 if positive else -0.9,
                    violation_logprob=-0.9 if positive else -0.1,
                ),
            )
            for model, positive in zip(admission["model_specs"], signs, strict=True)
        ]
        processes = [
            _process_receipt(model, index) for index, model in enumerate(admission["model_specs"])
        ]
        teardowns = [_teardown_receipt(model) for model in admission["model_specs"]]
        batches = [_batch_receipt(str(model["hf_id"])) for model in admission["model_specs"]]
        leases = [_lease_receipt(model) for model in admission["model_specs"]]
        checkpoint = exp.build_checkpoint_manifest(
            rows,
            expected_row_count=3,
            input_checksum="sha256:inputs",
            batch_receipts=batches,
            process_receipts=processes,
            teardown_receipts=teardowns,
        )
        return exp.build_complete_artifact(
            duration_s=90.0,
            preconditions=_complete_preconditions(),
            model_specs=admission["model_specs"],
            source_hashes={"exp6849": exp.EXPECTED_EXP6849_SHA256},
            score_inputs=[score_input],
            rows=rows,
            process_receipts=processes,
            batch_receipts=batches,
            teardown_receipts=teardowns,
            checkpoint_manifest=checkpoint,
            expected_row_count=3,
            lease_receipts=leases,
        )

    positive = build_with_signs([True, True, True])
    assert positive["verdict_class"] == "positive"
    assert positive["positive_margin_models"] == list(exp.MODEL_SPECS)
    mixed = build_with_signs([True, False, False])
    assert mixed["verdict_class"] == "null"
    assert mixed["honest_verdict"] == "complete_null_three_family_isomorphic_compatibility_stream"
    assert mixed["positive_margin_models"] == [exp.MODEL_SPECS[0]]

    invalid_lease = [_lease_receipt(model) for model in admission["model_specs"]]
    invalid_lease[1]["release"]["released"] = False
    assert exp._lease_receipts_complete([]) is False
    malformed_lease = [_lease_receipt(model) for model in admission["model_specs"]]
    malformed_lease[0]["owner"] = []
    assert exp._lease_receipts_complete(malformed_lease) is False
    rows = positive["rows"]
    checkpoint = positive["checkpoint_manifest"]
    lease_failed = exp.build_complete_artifact(
        duration_s=90.0,
        preconditions=_complete_preconditions(),
        model_specs=admission["model_specs"],
        source_hashes={"exp6849": exp.EXPECTED_EXP6849_SHA256},
        score_inputs=[score_input],
        rows=rows,
        process_receipts=positive["process_receipts"],
        batch_receipts=positive["batch_canary_receipts"],
        teardown_receipts=positive["teardown_receipts"],
        checkpoint_manifest=checkpoint,
        expected_row_count=3,
        lease_receipts=invalid_lease,
    )
    assert lease_failed["compatibility_stream_complete_score"] == 0
    assert lease_failed["gate_check_summary"]["failed_check"] == "task_owned_lease_receipts"

    missing_batch = exp.build_complete_artifact(
        duration_s=90.0,
        preconditions=_complete_preconditions(),
        model_specs=admission["model_specs"],
        source_hashes={"exp6849": exp.EXPECTED_EXP6849_SHA256},
        score_inputs=[score_input],
        rows=rows,
        process_receipts=positive["process_receipts"],
        batch_receipts=positive["batch_canary_receipts"][:-1],
        teardown_receipts=positive["teardown_receipts"],
        checkpoint_manifest=checkpoint,
        expected_row_count=3,
        lease_receipts=[_lease_receipt(model) for model in admission["model_specs"]],
    )
    assert missing_batch["compatibility_stream_complete_score"] == 0
    assert missing_batch["gate_check_summary"]["failed_check"] == "fresh_batch_canaries"

    consistency_rows = [
        {
            "model_hf_id": exp.MODEL_SPECS[0],
            "semantic_pair_identity": "semantic-no-base",
            "repeat": 0,
            "transform_kind": "atom_rename",
            "scalar_compatibility_margin": 1.0,
        },
        {
            "model_hf_id": exp.MODEL_SPECS[0],
            "semantic_pair_identity": "semantic-with-base",
            "repeat": 0,
            "transform_kind": "base",
            "scalar_compatibility_margin": 1.0,
        },
        {
            "model_hf_id": exp.MODEL_SPECS[0],
            "semantic_pair_identity": "semantic-with-base",
            "repeat": 0,
            "transform_kind": "label_swap",
            "scalar_compatibility_margin": 0.0,
        },
    ]
    consistency = exp._isomorphic_consistency(consistency_rows)
    assert consistency["per_model"][0]["comparison_count"] == 1
    assert consistency["per_model"][0]["consistent_count"] == 0

    original = positive["rows"][0]
    changed_hash = deepcopy(original)
    changed_hash["scalar_compatibility_margin"] = 999.0
    assert exp._rows_have_complete_receipts([changed_hash]) is False
    missing_raw = deepcopy(original)
    missing_raw["compatible"]["raw_receipt"] = {}
    missing_raw["row_hash"] = exp.row_hash(missing_raw)
    assert exp._rows_have_complete_receipts([missing_raw]) is False
    wrong_type = deepcopy(original)
    wrong_type["compatible"]["candidate_token_ids"] = "bad"
    wrong_type["row_hash"] = exp.row_hash(wrong_type)
    assert exp._rows_have_complete_receipts([wrong_type]) is False
    empty_tokens = deepcopy(original)
    empty_tokens["compatible"]["candidate_token_ids"] = []
    empty_tokens["compatible"]["token_logprobs"] = []
    empty_tokens["row_hash"] = exp.row_hash(empty_tokens)
    assert exp._rows_have_complete_receipts([empty_tokens]) is False

    malformed = deepcopy(positive)
    malformed["inference_substrate"] = "wrong"
    malformed["verifier_is_oracle"] = True
    malformed["rows"][0]["scalar_compatibility_margin"] = 999.0
    assert {
        "invalid_inference_substrate",
        "verifier_is_oracle_must_be_false",
        "row_hash_invalid",
    }.issubset(exp.validate_artifact(malformed))
    blocked_with_row = deepcopy(positive)
    blocked_with_row["verdict_class"] = "blocked"
    assert "blocked_artifact_has_rows" in exp.validate_artifact(blocked_with_row)


def test_req_6851_main_success_path(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """REQ-CONSTRAINT-6851 reports the completed artifact from the CLI."""

    monkeypatch.setattr(
        exp,
        "run",
        lambda **_kwargs: {
            "honest_verdict": "complete_null_three_family_isomorphic_compatibility_stream",
            "compatibility_stream_complete_score": 1,
        },
    )
    assert exp.main(["--date", exp.RUN_DATE, "--runtime-dir", "/tmp/exp6851-test"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["compatibility_stream_complete_score"] == 1
