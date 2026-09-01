"""Tests for the independent compatibility shortcut audit.

Spec refs: REQ-CONSTRAINT-6852 and SCENARIO-CONSTRAINT-6852-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6852_compatibility_shortcut_authority_audit as exp
import scripts.adversarial_verify as adversarial


REPO = Path(__file__).resolve().parents[2]
PAIR_ID = "pair-one"
PAIR_SEMANTIC_ID = "sha256:pair-one"
CANDIDATES = (
    ("candidate-good", "sha256:candidate-good", True),
    ("candidate-bad", "sha256:candidate-bad", False),
)
TRANSFORMS = (
    "base",
    "identifier_permutation",
    "atom_rename",
    "label_swap",
    "row_reordering",
    "surface_paraphrase",
    "duplicate_removal",
)


def _authority() -> dict[str, Any]:
    return {
        "status": "complete",
        "authority_audit_complete_score": 1,
        "typed_program_authority_ready_score": 1,
        "isomorphic_fixture_ready_score": 1,
        "source_artifact_hashes": {
            "exp6847": {
                "path": "results/experiment_6847_v598_independent_capstone.json",
                "file_sha256": "sha256:6847",
            }
        },
        "sanitized_candidate_pair_manifest": [
            {
                "pair_id": PAIR_ID,
                "semantic_identity": PAIR_SEMANTIC_ID,
                "source_pair_id": "source-pair-one",
                "candidate_order": [row[0] for row in CANDIDATES],
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "semantic_identity": semantic_id,
                        "exact_label": exact_label,
                        "raw_sequence_inputs": {"candidate_text": candidate_id},
                    }
                    for candidate_id, semantic_id, exact_label in CANDIDATES
                ],
            }
        ],
        "isomorphic_transform_manifest": [
            {
                "pair_id": PAIR_ID,
                "transform_id": f"transform-{kind}",
                "transform_kind": kind,
                "labels_preserved": True,
            }
            for kind in TRANSFORMS
            if kind != "base"
        ],
    }


def _admission() -> dict[str, Any]:
    return {
        "status": "complete",
        "admission_canary_complete_score": 1,
        "three_family_scoring_admission_ready_score": 1,
        "models_used": list(exp.REQUIRED_MODELS),
        "model_specs": [
            {
                "hf_id": model,
                "family": f"family-{index}",
                "model_size_bytes": 30_000_000_000 + index * 1_000_000_000,
                "model_sha256": f"sha256:model-{index}",
                "tokenizer_receipt": {"tokenizer_sha256": f"sha256:tokenizer-{index}"},
            }
            for index, model in enumerate(exp.REQUIRED_MODELS)
        ],
    }


def _producer(models: tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = models or exp.REQUIRED_MODELS
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    model_margins = {model: 0.30 - index * 0.04 for index, model in enumerate(exp.REQUIRED_MODELS)}
    for model in selected:
        compatible_mean = -0.2
        violation_mean = compatible_mean - model_margins[model]
        for repeat in (0, 1):
            for transform_kind in TRANSFORMS:
                transform_id = f"transform-{transform_kind}"
                row_identity = f"{model}::{PAIR_SEMANTIC_ID}::{transform_kind}::{repeat}"
                prompt_tokens = (
                    [1, 2, 3, 4] if transform_kind == "surface_paraphrase" else [1, 2, 3]
                )
                exact_labels = [
                    {"candidate_id": candidate_id, "exact_label": exact_label}
                    for candidate_id, _, exact_label in CANDIDATES
                ]
                rows.append(
                    {
                        "row_identity": row_identity,
                        "model_hf_id": model,
                        "model_family": f"family-{exp.REQUIRED_MODELS.index(model)}",
                        "pair_id": PAIR_ID,
                        "semantic_pair_identity": PAIR_SEMANTIC_ID,
                        "source_pair_id": "source-pair-one",
                        "transform_id": transform_id,
                        "transform_kind": transform_kind,
                        "repeat": repeat,
                        "exact_labels": exact_labels,
                        "compatible_label_position": 1 if transform_kind == "label_swap" else 0,
                        "compatible_candidate_length": 14,
                        "violation_candidate_length": 14,
                        "prompt_token_count": len(prompt_tokens),
                        "sum_log_likelihood_margin": 2 * model_margins[model],
                        "mean_token_log_likelihood_margin": model_margins[model],
                        "scalar_compatibility_margin": model_margins[model],
                    }
                )
                for candidate_id, semantic_id, exact_label in CANDIDATES:
                    mean = compatible_mean if exact_label else violation_mean
                    receipts.append(
                        {
                            "row_identity": row_identity,
                            "model_hf_id": model,
                            "pair_id": PAIR_ID,
                            "semantic_pair_identity": PAIR_SEMANTIC_ID,
                            "transform_id": transform_id,
                            "transform_kind": transform_kind,
                            "repeat": repeat,
                            "candidate_id": candidate_id,
                            "candidate_semantic_identity": semantic_id,
                            "exact_label": exact_label,
                            "label": "compatible" if exact_label else "violation",
                            "prompt_token_ids": prompt_tokens,
                            "candidate_token_ids": [10, 11],
                            "token_logprobs": [mean, mean],
                            "conditional_log_likelihood": mean * 2,
                            "mean_token_log_likelihood": mean,
                        }
                    )
    return {
        "status": "complete",
        "compatibility_stream_complete_score": 1,
        "models_used": list(selected),
        "source_artifact_hashes": {
            "exp6849": {"sha256": "sha256:6849"},
            "exp6850": {"sha256": "sha256:6850"},
        },
        "rows": rows,
        "token_score_receipts": receipts,
    }


def _source_hashes(*, stale: bool = False) -> dict[str, dict[str, Any]]:
    return {
        "exp6487": {
            "path": "results/experiment_6487_representation_integrity_audit.json",
            "current_sha256": "sha256:6487",
            "recorded_sha256": "sha256:6487",
            "hash_match": True,
        },
        "exp6847": {
            "path": "results/experiment_6847_v598_independent_capstone.json",
            "current_sha256": "sha256:6847",
            "recorded_sha256": "sha256:6847",
            "hash_match": True,
        },
        "exp6849": {
            "path": "results/experiment_6849_typed_program_isomorphic_authority_audit.json",
            "current_sha256": "sha256:changed" if stale else "sha256:6849",
            "recorded_sha256": "sha256:6849",
            "hash_match": not stale,
        },
        "exp6850": {
            "path": "results/experiment_6850_three_family_scoring_admission_canary.json",
            "current_sha256": "sha256:6850",
            "recorded_sha256": "sha256:6850",
            "hash_match": True,
        },
        "exp6851": {
            "path": "results/experiment_6851_three_family_isomorphic_compatibility_stream.json",
            "current_sha256": "sha256:6851",
            "recorded_sha256": "sha256:6851",
            "hash_match": True,
        },
    }


def _states(producer_present: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "artifact_present": task_id != "exp6851" or producer_present,
            "artifact_readable": task_id != "exp6851" or producer_present,
            "state": "complete" if task_id != "exp6851" or producer_present else "missing",
            "scientific_row_count": 42 if task_id == "exp6851" and producer_present else 0,
        }
        for task_id in ("exp6849", "exp6850", "exp6851")
    ]


def _reduce(
    *,
    authority: dict[str, Any] | None = None,
    admission: dict[str, Any] | None = None,
    producer: dict[str, Any] | None = None,
    source_hashes: dict[str, dict[str, Any]] | None = None,
    states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_producer = _producer() if producer is None else producer
    return exp.reduce_evidence(
        authority=_authority() if authority is None else authority,
        admission=_admission() if admission is None else admission,
        producer=selected_producer,
        source_artifact_hashes=_source_hashes() if source_hashes is None else source_hashes,
        upstream_state_manifest=_states() if states is None else states,
        conductor_skip_manifest=[],
        run_date="20260901",
        duration_s=0.25,
    )


def test_req_constraint_6852_spec_fields_and_fresh_reducer_contract() -> None:
    """REQ-CONSTRAINT-6852 owns the schema and reducer independence."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CONSTRAINT-6852:", 1)[1]
    for ref in exp.SPEC_REFS:
        assert ref in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section

    source = (REPO / exp.MODULE_PATH).read_text(encoding="utf-8")
    for experiment in ("6849", "6850", "6851"):
        assert f"import experiment_{experiment}" not in source
        assert f"from carnot.experiment_{experiment}" not in source


def test_req_constraint_6852_recomputes_receipts_and_all_controls() -> None:
    """REQ-CONSTRAINT-6852 recomputes raw margins and reports every attack."""

    artifact = _reduce()

    assert exp.validate_artifact(artifact) == []
    assert artifact["compatibility_audit_complete_score"] == 1
    assert artifact["compatibility_claim_eligible_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert set(artifact["field_principles"]) == set(artifact)
    assert len(artifact["recomputed_margin_rows"]) == 42
    assert {row["attack_kind"] for row in artifact["shortcut_attack_results"]} == set(
        exp.ATTACK_KINDS
    )
    first = artifact["recomputed_margin_rows"][0]
    assert first["recomputed_sum_margin"] == pytest.approx(0.6)
    assert first["recomputed_mean_token_margin"] == pytest.approx(0.3)


def test_scenario_6852_missing_producer_emits_blocked_artifact() -> None:
    """SCENARIO-CONSTRAINT-6852-MISSING-PRODUCER preserves missing rows."""

    artifact = exp.reduce_evidence(
        authority=_authority(),
        admission=_admission(),
        producer=None,
        source_artifact_hashes=_source_hashes(),
        upstream_state_manifest=_states(producer_present=False),
        conductor_skip_manifest=[{"task_id": "exp6851", "status": "GATE_BLOCK"}],
        run_date="20260901",
        duration_s=0.1,
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["compatibility_claim_eligible_score"] == 0
    assert artifact["compatibility_audit_complete_score"] == 1
    assert all(row["margin"] is None for row in artifact["missing_model_manifest"])
    assert any(row["row_kind"] == "missing_scientific_unit" for row in artifact["rows"])


def test_scenario_6852_partial_models_never_average_missing_as_zero() -> None:
    """SCENARIO-CONSTRAINT-6852-PARTIAL-MODELS keeps absent margins null."""

    producer = _producer((exp.REQUIRED_MODELS[0],))
    artifact = _reduce(producer=producer)

    assert artifact["verdict_class"] == "partial"
    assert artifact["compatibility_claim_eligible_score"] == 0
    assert [row["model_hf_id"] for row in artifact["missing_model_manifest"]] == list(
        exp.REQUIRED_MODELS[1:]
    )
    assert all(row["margin"] is None for row in artifact["missing_model_manifest"])
    scored = [row for row in artifact["rows"] if row["row_kind"] == "scored_scientific_unit"]
    assert len(scored) == 1
    assert scored[0]["mean_margin"] == pytest.approx(0.3)


def test_scenario_6852_all_null_token_scores_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-6852-NULL-SCORES does not coerce null to zero."""

    producer = _producer()
    row_identity = producer["rows"][0]["row_identity"]
    for receipt in producer["token_score_receipts"]:
        if receipt["row_identity"] == row_identity:
            receipt["token_logprobs"] = [None, None]
    artifact = _reduce(producer=producer)

    assert artifact["compatibility_claim_eligible_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert any(
        row["failure_kind"] == "null_token_score" for row in artifact["authority_failure_witnesses"]
    )
    assert all(
        row["recomputed_mean_token_margin"] is not None
        for row in artifact["recomputed_margin_rows"]
    )


def test_scenario_6852_duplicate_identity_fails_closed() -> None:
    """SCENARIO-CONSTRAINT-6852-DUPLICATE-IDENTITY records both rows."""

    producer = _producer()
    producer["rows"].append(deepcopy(producer["rows"][0]))
    artifact = _reduce(producer=producer)

    witness = next(
        row
        for row in artifact["authority_failure_witnesses"]
        if row["failure_kind"] == "duplicate_row_identity"
    )
    assert witness["occurrences"] == 2
    assert artifact["verdict_class"] == "disqualified"


def test_scenario_6852_label_inversion_uses_manifest_authority() -> None:
    """SCENARIO-CONSTRAINT-6852-LABEL-INVERSION rejects producer labels."""

    producer = _producer()
    producer["token_score_receipts"][0]["exact_label"] = False
    producer["token_score_receipts"][0]["label"] = "violation"
    artifact = _reduce(producer=producer)

    witness = next(
        row
        for row in artifact["authority_failure_witnesses"]
        if row["failure_kind"] == "label_inversion"
    )
    assert witness["expected"] is True
    assert witness["observed"] is False
    assert artifact["compatibility_claim_eligible_score"] == 0


def test_scenario_6852_row_reorder_is_canonical() -> None:
    """SCENARIO-CONSTRAINT-6852-ROW-REORDER makes list order irrelevant."""

    producer = _producer()
    ordered = _reduce(producer=producer)
    reordered_producer = deepcopy(producer)
    reordered_producer["rows"].reverse()
    reordered_producer["token_score_receipts"].reverse()
    reordered = _reduce(producer=reordered_producer)

    for field in (
        "rows",
        "recomputed_margin_rows",
        "isomorphic_invariance_results",
        "shortcut_attack_results",
        "control_explanation_results",
        "gate_check_summary",
        "reproducibility_checksum",
    ):
        assert reordered[field] == ordered[field]


def test_scenario_6852_stale_hash_disqualifies_complete_rows() -> None:
    """SCENARIO-CONSTRAINT-6852-STALE-HASH reports expected and observed."""

    artifact = _reduce(source_hashes=_source_hashes(stale=True))

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["compatibility_claim_eligible_score"] == 0
    failed = next(
        row
        for row in artifact["gate_check_summary"]["failed_checks"]
        if row["check"] == "artifact_hash_authority"
    )
    assert failed["expected"] == "all recorded hashes match current bytes"
    assert failed["observed"] == ["exp6849"]


def test_scenarios_6852_shortcut_and_isomorphic_failures_are_row_supported() -> None:
    """SCENARIO-CONSTRAINT-6852-SHORTCUTS and -ISOMORPHIC fail eligibility."""

    producer = _producer()
    target_model = exp.REQUIRED_MODELS[0]
    for receipt in producer["token_score_receipts"]:
        if (
            receipt["model_hf_id"] == target_model
            and receipt["transform_kind"] == "identifier_permutation"
            and receipt["repeat"] == 0
            and receipt["exact_label"] is True
        ):
            receipt["token_logprobs"] = [-0.8, -0.8]
            receipt["conditional_log_likelihood"] = -1.6
            receipt["mean_token_log_likelihood"] = -0.8
    for row in producer["rows"]:
        if (
            row["model_hf_id"] == target_model
            and row["transform_kind"] == "identifier_permutation"
            and row["repeat"] == 0
        ):
            row["sum_log_likelihood_margin"] = -0.6
            row["mean_token_log_likelihood_margin"] = -0.3
            row["scalar_compatibility_margin"] = -0.3
    artifact = _reduce(producer=producer)

    assert artifact["verdict_class"] == "null"
    assert artifact["compatibility_claim_eligible_score"] == 0
    assert any(
        row["direction_survives"] is False for row in artifact["isomorphic_invariance_results"]
    )
    assert any(
        row["attack_kind"] == "identifier_only" and row["explains_same_or_greater"] is True
        for row in artifact["shortcut_attack_results"]
    )


def test_req_constraint_6852_conductor_records_and_real_inputs_build(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6852 inventories real task records without an LLM."""

    records = exp.parse_conductor_records(
        "\n".join(
            [
                "| 2026-09-01 15:42 UTC | Independent typed-program isomorphic authority aud | OK | done |",
                "| 2026-09-01 17:07 UTC | Three-family isomorphic fixed-sequence compatibili | FAIL | bootstrap |",
                "| 2026-09-01 17:31 UTC | Three-family isomorphic fixed-sequence compatibili | OK | done |",
            ]
        )
    )
    assert [row["status"] for row in records if row["task_id"] == "exp6851"] == [
        "FAIL",
        "OK",
    ]

    artifact = exp.build_artifact(REPO, run_date="20260901", duration_s=0.25)
    assert exp.validate_artifact(artifact) == []
    assert artifact["compatibility_audit_complete_score"] == 1
    assert artifact["compatibility_claim_eligible_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert len(artifact["upstream_state_manifest"]) == 3
    assert any(row["status"] == "FAIL" for row in artifact["conductor_skip_manifest"])

    output = tmp_path / "artifact.json"
    exp.write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_constraint_6852_validation_names_schema_drift() -> None:
    """REQ-CONSTRAINT-6852 validation reports missing principles and bad scores."""

    artifact = _reduce()
    artifact["field_principles"].pop("rows")
    artifact["compatibility_claim_eligible_score"] = 2

    assert exp.validate_artifact(artifact) == [
        "field_principles must cover every top-level field",
        "compatibility_claim_eligible_score must be 0 or 1",
    ]


@pytest.mark.parametrize(
    ("mutate", "failure_kind"),
    [
        (
            lambda producer: producer["rows"][0].__setitem__(
                "semantic_pair_identity", "sha256:wrong-pair"
            ),
            "semantic_pair_identity_mismatch",
        ),
        (
            lambda producer: producer["token_score_receipts"].pop(0),
            "candidate_receipt_count",
        ),
        (
            lambda producer: producer["token_score_receipts"][0].__setitem__(
                "prompt_token_ids", [999]
            ),
            "paired_prompt_token_mismatch",
        ),
        (
            lambda producer: producer["token_score_receipts"][0].__setitem__(
                "candidate_id", "unknown-candidate"
            ),
            "unknown_candidate_identity",
        ),
        (
            lambda producer: producer["token_score_receipts"][0].__setitem__(
                "candidate_semantic_identity", "sha256:wrong-candidate"
            ),
            "candidate_semantic_identity_mismatch",
        ),
        (
            lambda producer: producer["rows"][0].__setitem__("sum_log_likelihood_margin", 999.0),
            "producer_margin_mismatch",
        ),
    ],
)
def test_req_constraint_6852_receipt_authority_attacks(mutate: object, failure_kind: str) -> None:
    """REQ-CONSTRAINT-6852 names each malformed semantic or score receipt."""

    producer = _producer()
    mutate(producer)  # type: ignore[operator]
    artifact = _reduce(producer=producer)

    assert failure_kind in {row["failure_kind"] for row in artifact["authority_failure_witnesses"]}
    assert artifact["verdict_class"] == "disqualified"


def test_req_constraint_6852_duplicate_receipts_and_manifest_noise_fail_closed() -> None:
    """REQ-CONSTRAINT-6852 detects duplicate receipts and ignores non-record manifest noise."""

    authority = _authority()
    authority["sanitized_candidate_pair_manifest"].append("not-a-record")
    producer = _producer()
    producer["token_score_receipts"].append(deepcopy(producer["token_score_receipts"][0]))
    artifact = _reduce(authority=authority, producer=producer)

    kinds = {row["failure_kind"] for row in artifact["authority_failure_witnesses"]}
    assert "duplicate_receipt_identity" in kinds
    assert "candidate_receipt_count" in kinds


@pytest.mark.parametrize(
    ("receipt", "failure_kind"),
    [
        ({"token_logprobs": [], "candidate_token_ids": []}, "missing_token_scores"),
        ({"token_logprobs": [-0.1, None], "candidate_token_ids": [1, 2]}, "null_token_score"),
        (
            {"token_logprobs": [-0.1, float("inf")], "candidate_token_ids": [1, 2]},
            "non_finite_token_score",
        ),
        (
            {"token_logprobs": [-0.1, -0.2], "candidate_token_ids": [1]},
            "token_score_length_mismatch",
        ),
    ],
)
def test_scenario_6852_invalid_score_variants(receipt: dict[str, Any], failure_kind: str) -> None:
    """SCENARIO-CONSTRAINT-6852-NULL-SCORES rejects every incomplete score form."""

    scores, observed = exp._finite_scores(receipt)

    assert scores is None
    assert observed == failure_kind


def test_req_constraint_6852_eligible_negative_effect_is_null() -> None:
    """REQ-CONSTRAINT-6852 separates claim eligibility from effect direction."""

    producer = _producer()
    negative_margins = {
        model: -0.30 + index * 0.04 for index, model in enumerate(exp.REQUIRED_MODELS)
    }
    for receipt in producer["token_score_receipts"]:
        margin = negative_margins[receipt["model_hf_id"]]
        mean = -0.5 if receipt["exact_label"] else -0.5 - margin
        receipt["token_logprobs"] = [mean, mean]
        receipt["conditional_log_likelihood"] = mean * 2
        receipt["mean_token_log_likelihood"] = mean
    for row in producer["rows"]:
        margin = negative_margins[row["model_hf_id"]]
        row["sum_log_likelihood_margin"] = margin * 2
        row["mean_token_log_likelihood_margin"] = margin
        row["scalar_compatibility_margin"] = margin

    artifact = _reduce(producer=producer)

    assert artifact["compatibility_claim_eligible_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == (
        "complete_null_eligible_compatibility_direction_not_positive"
    )
    assert exp._direction(0.0) == "zero"
    assert exp._mean([]) is None


def test_req_constraint_6852_io_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6852 refuses unreadable sources and malformed output."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        exp._read_json(list_json)
    assert exp._load_optional(tmp_path / "missing.json") is None
    assert exp._load_optional(list_json) is None

    invalid: dict[str, Any] = {
        "field_principles": {"field_principles": "only one field"},
        "compatibility_audit_complete_score": 3,
        "compatibility_claim_eligible_score": 3,
        "inference_substrate": "wrong",
        "verifier_is_oracle": True,
        "verdict_class": "unknown",
        "honest_verdict": "not-terminal",
    }
    errors = exp.validate_artifact(invalid)
    assert errors[0].startswith("missing required fields:")
    assert "inference_substrate must name deterministic CPU independent reduction" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "verdict_class is outside the closed vocabulary" in errors
    assert "honest_verdict must be terminal" in errors
    with pytest.raises(ValueError, match="missing required fields"):
        exp.write_artifact(tmp_path / "invalid.json", invalid)


def test_req_constraint_6852_cpu_reducer_is_not_live_model_compute(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6852 keeps upstream model names from changing this substrate."""

    payload = {
        "experiment_id": "exp6852-regression",
        "status": "complete",
        "honest_verdict": "complete_null_model_compatibility_not_identifiable",
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "duration_s": 0.2,
        "random_seed": exp.RANDOM_SEED,
        "reproducibility_checksum": "sha256:test",
        "rows": [{"model_hf_id": exp.REQUIRED_MODELS[0]}],
    }
    path = tmp_path / "experiment_6852_regression.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = adversarial.verify_artifact(path)
    flag_kinds = {row["kind"] for row in report["flags"]}

    assert adversarial.duration_floor_for_artifact(payload)["reason"] == ("deterministic_verifier")
    assert "DURATION_TOO_SHORT" not in flag_kinds
    assert "METHODOLOGY_MISSING" not in flag_kinds
