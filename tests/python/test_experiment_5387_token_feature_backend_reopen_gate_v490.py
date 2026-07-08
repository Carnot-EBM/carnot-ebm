"""Tests for Exp5387 token/internal-feature backend reopen gate.

Spec refs: REQ-VERIFY-5387, SCENARIO-VERIFY-5387.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5387_token_feature_backend_reopen_gate_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _wrap(value: Any) -> dict[str, Any]:
    return {"principle": "fixture principle", "value": value}


def _value(artifact: dict[str, Any], field: str) -> Any:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _evidence_path(artifact: dict[str, Any], field: str) -> str:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return str(wrapped["evidence_path"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp5331_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5331.internal_energy_receipt_harness.v486",
        "status": _wrap("complete"),
        "honest_verdict": _wrap("complete: token_probability_receipt_ready"),
        "logits_available": False,
        "token_probability_available": True,
        "attention_available": False,
        "hidden_state_proxy_available": False,
        "token_timing_available": True,
        "internal_signal_receipt_ready": True,
        "backend_option_surface": _wrap(
            {
                "option_flags": {
                    "logit_export_option": False,
                    "attention_export_option": False,
                    "hidden_state_proxy_option": True,
                    "token_probability_option": False,
                }
            }
        ),
        "missing_backend_features": _wrap(
            [
                "logits_unavailable",
                "attention_export_unavailable",
                "hidden_state_proxy_unavailable",
            ]
        ),
        "tiny_receipt_path": _wrap(
            "/tmp/results/experiment_5331_internal_energy_tiny_receipt_v486.json"
        ),
    }


def _exp5331_schema() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5331.internal_energy_receipt_schema.v486",
        "internal_signal_receipt_ready": True,
        "availability": {
            "logits_available": False,
            "token_probability_available": True,
            "attention_available": False,
            "hidden_state_proxy_available": False,
            "token_timing_available": True,
            "raw_output_receipt_available": True,
        },
    }


def _exp5331_tiny_receipt() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5331.internal_energy_tiny_receipt.v486",
        "endpoint": "/completion",
        "completion_probabilities": [{"id": 108, "logprob": -0.22, "top_logprobs": []}],
        "logits": {"availability": "capability_absent", "top_logits": []},
        "attention": {"availability": "capability_absent", "summary": {}},
        "hidden_state_proxy": {"availability": "capability_absent", "summary": {}},
        "token_timing": {"availability": "available", "timings": {"predicted_per_token_ms": 1.0}},
        "raw_output": {"availability": "available", "tokens_evaluated": 5, "tokens_predicted": 1},
    }


def _exp5353_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5353.tokenprob_feature_audit_corrigendum.v488",
        "status": _wrap("complete"),
        "honest_verdict": _wrap("complete: tokenprob_feature_rows_ready"),
        "per_token_logprob_available": True,
        "topk_alternatives_available": True,
        "logits_available": False,
        "hidden_states_available": False,
        "attention_available": False,
        "prompt_completion_token_split_available": True,
        "token_timing_available": True,
        "tokenprob_feature_row_count": 3,
        "tokenprob_feature_rows_ready": True,
        "feature_audit": {
            "per_token_logprob_available": True,
            "topk_alternatives_available": True,
            "logits_available": False,
            "hidden_states_available": False,
            "attention_available": False,
            "prompt_completion_token_split_available": True,
            "token_timing_available": True,
            "missing_feature_names": ["logits", "attention", "hidden_states"],
        },
        "tokenprob_feature_rows": [
            {
                "prompt_id": "receipt_alpha",
                "feature_source": "backend_completion_probabilities",
                "logprob": -1.0,
                "token_index": 0,
            }
        ],
        "preconditions_checked": _wrap(
            {
                "selected_backend_kind": "llama-server",
                "selected_model_hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "retired_scope_check": {"retired_scope_reopened": False},
            }
        ),
    }


def _exp5354_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5354.arithmetic_carry_token_energy.v488",
        "status": _wrap("blocked"),
        "honest_verdict": _wrap("blocked_carry_token_features_incomplete"),
        "feature_complete_rate": 0.5625,
        "correct_vs_perturbed_margin": 0.0,
        "carry_token_energy_signal_ready": False,
        "missing_feature_names": ["no_carry_12_23:perturbed_target_logprob"],
    }


def _exp5372_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5372.token_feature_precondition_gate.v489",
        "status": "complete",
        "honest_verdict": (
            "complete: retire carry-token energy lane until logits/hidden "
            "states/attention and feature-complete controls exist"
        ),
        "future_signal_allowed": False,
        "logits_available": False,
        "hidden_states_available": False,
        "attention_available": False,
        "tokenprob_rows_available": True,
        "retire_recommendation": True,
        "forbidden_claims": ["promotion of tokenprob feature receipts into internal-energy readiness"],
    }


def _write_helper_sources(root: Path, *, with_depth_exit: bool = False) -> None:
    source_dir = root / "python/carnot"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "experiment_5331_internal_energy_receipt_harness_v486.py").write_text(
        "\n".join(
            [
                "LOGIT_EXPORT_PATTERNS = ('--logits',)",
                "ATTENTION_EXPORT_PATTERNS = ('dump-attention',)",
                "HIDDEN_PROXY_PATTERNS = ('--embedding',)",
                "def signal_availability(signal_receipt):",
                "    return {'logits_available': False, 'attention_available': False}",
            ]
        ),
        encoding="utf-8",
    )
    depth_line = "intermediate_depth_exit_receipts = True" if with_depth_exit else "depth_probe_absent = True"
    (source_dir / "experiment_5353_tokenprob_feature_audit_corrigendum_v488.py").write_text(
        "\n".join(
            [
                "def audit_backend_features(tiny_receipt, schema_artifact, internal_artifact):",
                "    return {'logits_available': False, 'hidden_states_available': False}",
                depth_line,
            ]
        ),
        encoding="utf-8",
    )


def _make_repo(root: Path, *, logits_available: bool = False, with_depth_exit: bool = False) -> Path:
    exp5331 = _exp5331_artifact()
    exp5331_schema = _exp5331_schema()
    exp5331_tiny = _exp5331_tiny_receipt()
    exp5353 = _exp5353_artifact()
    if logits_available:
        exp5331["logits_available"] = True
        exp5331_schema["availability"]["logits_available"] = True
        exp5331_tiny["logits"] = {
            "availability": "available",
            "top_logits": [{"token_id": 1, "logit": 2.0}],
        }
        exp5353["logits_available"] = True
        exp5353["feature_audit"]["logits_available"] = True
        exp5353["feature_audit"]["missing_feature_names"] = ["attention", "hidden_states"]

    _write_json(root / mod.EXP5331_RELATIVE_PATH, exp5331)
    _write_json(root / mod.EXP5331_SCHEMA_RELATIVE_PATH, exp5331_schema)
    _write_json(root / mod.EXP5331_TINY_RECEIPT_RELATIVE_PATH, exp5331_tiny)
    _write_json(root / mod.EXP5353_RELATIVE_PATH, exp5353)
    _write_json(root / mod.EXP5354_RELATIVE_PATH, _exp5354_artifact())
    _write_json(root / mod.CANONICAL_EXP5372_RELATIVE_PATH, _exp5372_artifact())
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/exclusion_manifest.yaml").write_text(
        "id: phase_d_external_text_scorer_retired_exp5163_v474\n"
        "operator_reopen_required: true\n",
        encoding="utf-8",
    )
    _write_helper_sources(root, with_depth_exit=with_depth_exit)
    return root


def _tests_run() -> list[dict[str, str]]:
    return [{"command": "unit exp5387", "outcome": "passed"}]


def test_req_verify_5387_spec_declares_backend_reopen_gate_contract() -> None:
    """REQ-VERIFY-5387: OpenSpec anchors the backend feature gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5387") : spec.index("### REQ-VERIFY-5345")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5387",
        "SCENARIO-VERIFY-5387",
        str(mod.RESULT_RELATIVE_PATH),
        "without running a new live signal benchmark",
        "scripts/research_conductor.py",
        "backend_reopen_allowed",
        "future_signal_allowed",
        "intermediate_depth_exits_available",
    ):
        assert marker in section

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5387_closes_gate_on_current_logprob_only_receipts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5387: logprob-only receipts do not reopen future signal work."""

    artifact = mod.run(
        root=_make_repo(tmp_path),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        tests_run=_tests_run(),
    )

    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "backend_reopen_allowed") is False
    assert _value(artifact, "future_signal_allowed") is False
    assert _value(artifact, "logits_available") is False
    assert _value(artifact, "hidden_states_available") is False
    assert _value(artifact, "attention_available") is False
    assert _value(artifact, "intermediate_depth_exits_available") is False
    assert _value(artifact, "clean_feature_row_provenance") is False
    assert _value(artifact, "no_live_signal_claim") is True
    assert _value(artifact, "retired_scope_reopened") is False
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _evidence_path(artifact, "logits_available") == str(mod.EXP5331_TINY_RECEIPT_RELATIVE_PATH)
    assert artifact["source_evidence"]["requested_exp5372_path_present"] is False
    assert artifact["source_evidence"]["canonical_exp5372_path_present"] is True
    assert artifact["benchmark_execution"]["new_live_signal_benchmark_run"] is False

    forbidden = set(_value(artifact, "forbidden_claims"))
    assert "text-only energy" in forbidden
    assert "incomplete token rows" in forbidden
    assert "arithmetic carry signal" in forbidden
    assert "external generated-text scoring" in forbidden
    assert "DEX-style depth claims without depth exits" in forbidden
    assert artifact["minimum_next_experiment"]["required_before_signal_claim"] == [
        "backend receipt exposing logits, hidden states, attention, or intermediate-depth exits",
        "clean row provenance from live runtime outputs",
        "feature-complete positive and negative controls",
    ]
    assert artifact["minimum_next_experiment"]["signal_quality_claim_allowed"] is False


def test_req_verify_5387_records_minimum_next_experiment_when_feature_exists(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5387: new feature receipts open only a pre-signal next experiment."""

    artifact = mod.build_artifact(
        root=_make_repo(tmp_path, logits_available=True),
        tests_run=_tests_run(),
    )

    mod.validate_artifact(artifact)
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "logits_available") is True
    assert _value(artifact, "backend_reopen_allowed") is True
    assert _value(artifact, "future_signal_allowed") is True
    assert _value(artifact, "clean_feature_row_provenance") is True
    assert _value(artifact, "retired_scope_reopened") is False
    assert artifact["minimum_next_experiment"]["name"] == "feature_receipt_positive_control"
    assert artifact["minimum_next_experiment"]["signal_quality_claim_allowed"] is False
    assert "external generated-text scoring" in _value(artifact, "forbidden_claims")


def test_req_verify_5387_blocks_when_required_semantic_sources_are_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-5387: missing source evidence writes honest_blocked and fails closed."""

    root = _make_repo(tmp_path)
    (root / mod.EXP5353_RELATIVE_PATH).unlink()

    artifact = mod.build_artifact(root=root, tests_run=_tests_run())

    mod.validate_artifact(artifact)
    assert _value(artifact, "status") == "honest_blocked"
    assert _value(artifact, "backend_reopen_allowed") is False
    assert _value(artifact, "future_signal_allowed") is False
    assert _value(artifact, "clean_feature_row_provenance") is False
    assert _value(artifact, "no_live_signal_claim") is True
    assert str(mod.EXP5353_RELATIVE_PATH) in artifact["missing_required_sources"]
    assert _value(artifact, "honest_verdict").startswith("honest_blocked")


def test_req_verify_5387_validation_rejects_unsafe_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-5387: validation rejects unsupported reopen and signal claims."""

    artifact = mod.build_artifact(root=_make_repo(tmp_path), tests_run=_tests_run())

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_signal = deepcopy(artifact)
    bad_signal["future_signal_allowed"]["value"] = True
    with pytest.raises(ValueError, match="future_signal_allowed"):
        mod.validate_artifact(bad_signal)

    bad_reopen = deepcopy(artifact)
    bad_reopen["backend_reopen_allowed"]["value"] = True
    with pytest.raises(ValueError, match="backend_reopen_allowed"):
        mod.validate_artifact(bad_reopen)

    bad_claim = deepcopy(artifact)
    bad_claim["no_live_signal_claim"]["value"] = False
    with pytest.raises(ValueError, match="no_live_signal_claim"):
        mod.validate_artifact(bad_claim)

    bad_scope = deepcopy(artifact)
    bad_scope["retired_scope_reopened"]["value"] = True
    with pytest.raises(ValueError, match="retired_scope_reopened"):
        mod.validate_artifact(bad_scope)

    bad_forbidden = deepcopy(artifact)
    bad_forbidden["forbidden_claims"]["value"] = ["text-only energy"]
    with pytest.raises(ValueError, match="forbidden_claims"):
        mod.validate_artifact(bad_forbidden)

    bad_status = deepcopy(artifact)
    bad_status["status"]["value"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(bad_tests)


def test_req_verify_5387_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5387: checked-in result is stable under deterministic replay."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(tests_run=artifact["tests_run"])

    assert artifact == replay
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "backend_reopen_allowed") is False
    assert _value(artifact, "future_signal_allowed") is False
    assert _value(artifact, "no_live_signal_claim") is True
    mod.validate_artifact(artifact)
