"""Exp5451 bounded verifier-potential/governed-memory certificate.

Spec refs: REQ-KAN-5451, SCENARIO-KAN-5451.

This module is intentionally a measurement-access certificate, not a KAN
soundness proof. It only reads fields that were already measured in the Exp5443
verifier-potential fixture and the Exp5446 governed-memory CSL artifact. Claims
that need hardware timing, token logprobs, hidden states, or general KAN
theorems are rejected because those measurements are absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes, receipt_exists


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/kan/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5451_kan_verifier_potential_memory_certificate_v495.py"
)
EXP5438_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5438_kan_ontology_measurement_certificate_v494.json"
)
EXP5443_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5443_verifier_potential_prefix_fixture_v495.json"
)
EXP5446_RESULT_RELATIVE_PATH = Path("results/experiment_5446_governed_memory_csl_online_v495.json")

EXPERIMENT = "experiment_5451_kan_verifier_potential_memory_certificate_v495"
EXPERIMENT_ID = "exp5451-v495-kan-verifier-potential-memory-certificate"
MILESTONE = "2026.07.495"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5451
SCHEMA = "carnot.experiment_5451.kan_verifier_potential_memory_certificate.v495"
SPEC_REFS = ("REQ-KAN-5451", "SCENARIO-KAN-5451")
PROPERTY_FAMILY = "bounded_kan_verifier_potential_governed_memory_measurement_certificate"
INFERENCE_SUBSTRATE = "bounded_measurement_access_certificate"
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "gated_upstreams_ready": "structured gate provenance.",
    "claim_count": "certificate coverage.",
    "true_measured_claim_preservation_rate": "useful certificate.",
    "false_property_rejection_rate": "soundness guard.",
    "unsupported_claim_rejection_rate": "measurement-access boundary.",
    "verifier_potential_claims_checked": "V495 claim coverage.",
    "governed_memory_claims_checked": "CSL claim coverage.",
    "hardware_speedup_claim_rejected": "hardware honesty.",
    "token_internal_claim_rejected": "closed-lane enforcement.",
    "broad_kan_claim_made": "no overclaim.",
    "certificate_checksum": "reproducibility.",
    "kan_certificate_ready": "capstone evidence.",
    "inference_substrate": "explicit certificate substrate.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

CLAIM_SPECS: tuple[JsonDict, ...] = (
    {
        "claim_id": "true_exp5443_fixture_ready",
        "claim_kind": "true_measured",
        "claim_domain": "verifier_potential",
        "statement": "Exp5443 reports verifier_potential_fixture_ready=true.",
        "required_fields": ["exp5443.verifier_potential_fixture_ready"],
        "checks": [{"path": "exp5443.verifier_potential_fixture_ready", "op": "eq", "value": True}],
    },
    {
        "claim_id": "true_exp5443_exact_final_authority",
        "claim_kind": "true_measured",
        "claim_domain": "verifier_potential",
        "statement": "Exp5443 keeps exact final verifier authority.",
        "required_fields": ["exp5443.exact_final_authority"],
        "checks": [{"path": "exp5443.exact_final_authority", "op": "eq", "value": True}],
    },
    {
        "claim_id": "true_exp5443_prefix_final_disagreements_measured",
        "claim_kind": "true_measured",
        "claim_domain": "verifier_potential",
        "statement": "Exp5443 measured prefix/final disagreement cases.",
        "required_fields": ["exp5443.prefix_final_disagreement_cases"],
        "checks": [{"path": "exp5443.prefix_final_disagreement_cases", "op": "gt", "value": 0}],
    },
    {
        "claim_id": "true_exp5446_governed_loop_ready",
        "claim_kind": "true_measured",
        "claim_domain": "governed_memory",
        "statement": "Exp5446 reports governed_csl_loop_ready=true.",
        "required_fields": ["exp5446.governed_csl_loop_ready"],
        "checks": [{"path": "exp5446.governed_csl_loop_ready", "op": "eq", "value": True}],
    },
    {
        "claim_id": "true_exp5446_no_weight_mutation",
        "claim_kind": "true_measured",
        "claim_domain": "governed_memory",
        "statement": "Exp5446 reports no model-weight mutation.",
        "required_fields": ["exp5446.no_weight_mutation"],
        "checks": [{"path": "exp5446.no_weight_mutation", "op": "eq", "value": True}],
    },
    {
        "claim_id": "true_exp5446_zero_unsafe_false_accepts",
        "claim_kind": "true_measured",
        "claim_domain": "governed_memory",
        "statement": "Exp5446 governed memory has zero unsafe false accepts.",
        "required_fields": ["exp5446.unsafe_false_accepts"],
        "checks": [{"path": "exp5446.unsafe_false_accepts", "op": "eq", "value": 0}],
    },
    {
        "claim_id": "false_exp5443_no_prefix_final_disagreements",
        "claim_kind": "false_property",
        "claim_domain": "verifier_potential",
        "statement": "Verifier potential never disagrees with final authority.",
        "required_fields": ["exp5443.prefix_final_disagreement_cases"],
        "checks": [{"path": "exp5443.prefix_final_disagreement_cases", "op": "eq", "value": 0}],
    },
    {
        "claim_id": "false_exp5443_metric_independence_failed",
        "claim_kind": "false_property",
        "claim_domain": "verifier_potential",
        "statement": "Exp5443 metric independence checks failed.",
        "required_fields": ["exp5443.metric_independence_checks_passed"],
        "checks": [
            {"path": "exp5443.metric_independence_checks_passed", "op": "eq", "value": False}
        ],
    },
    {
        "claim_id": "false_exp5446_ungated_memory_safe",
        "claim_kind": "false_property",
        "claim_domain": "governed_memory",
        "statement": "Ungated memory had zero unsafe false accepts.",
        "required_fields": ["exp5446.control_metrics.ungated_memory.unsafe_false_accepts"],
        "checks": [
            {
                "path": "exp5446.control_metrics.ungated_memory.unsafe_false_accepts",
                "op": "eq",
                "value": 0,
            }
        ],
    },
    {
        "claim_id": "false_exp5446_all_memories_active_for_routing",
        "claim_kind": "false_property",
        "claim_domain": "governed_memory",
        "statement": "Every memory record stayed active for future routing.",
        "required_fields": ["exp5446.rejected_memories", "exp5446.abstained_memories"],
        "checks": [
            {
                "paths": ["exp5446.rejected_memories", "exp5446.abstained_memories"],
                "op": "sum_lengths_eq",
                "value": 0,
            }
        ],
    },
    {
        "claim_id": "false_exp5446_model_weights_mutated",
        "claim_kind": "false_property",
        "claim_domain": "governed_memory",
        "statement": "Governed online memory mutated model weights.",
        "required_fields": ["exp5446.no_weight_mutation"],
        "checks": [{"path": "exp5446.no_weight_mutation", "op": "eq", "value": False}],
    },
    {
        "claim_id": "unsupported_hardware_speedup_from_certificate",
        "claim_kind": "unsupported",
        "claim_domain": "hardware_speedup",
        "statement": "The bounded certificate proves hardware speedup.",
        "required_fields": [
            "exp5446.authenticated_hardware_speedup",
            "exp5446.board_timing_receipt",
        ],
        "checks": [],
    },
    {
        "claim_id": "unsupported_token_level_access_from_certificate",
        "claim_kind": "unsupported",
        "claim_domain": "token_internal",
        "statement": "The bounded certificate proves token-level/logprob access.",
        "required_fields": ["exp5443.token_ids", "exp5443.token_logprobs"],
        "checks": [],
    },
    {
        "claim_id": "unsupported_internal_state_access_from_certificate",
        "claim_kind": "unsupported",
        "claim_domain": "token_internal",
        "statement": "The bounded certificate proves hidden/internal-state access.",
        "required_fields": ["exp5446.hidden_state_tensor", "exp5446.activation_checksum"],
        "checks": [],
    },
    {
        "claim_id": "unsupported_broad_kan_soundness",
        "claim_kind": "broad_soundness",
        "claim_domain": "kan_soundness",
        "statement": "The bounded certificate proves KAN soundness or general LLM truth.",
        "required_fields": [
            "exp5443.general_kan_soundness_theorem",
            "exp5446.general_llm_truth_certificate",
        ],
        "checks": [],
    },
)


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load only the upstream artifacts that the bounded certificate may inspect."""

    root_path = Path(root)
    return {
        "exp5443": _load_json(root_path / EXP5443_RESULT_RELATIVE_PATH),
        "exp5446": _load_json(root_path / EXP5446_RESULT_RELATIVE_PATH),
    }


def upstream_gates_ready(upstreams: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return true only when both explicit upstream ready gates are open."""

    return (
        upstreams["exp5443"].get("verifier_potential_fixture_ready") is True
        and upstreams["exp5446"].get("governed_csl_loop_ready") is True
    )


def build_claim_set() -> JsonList:
    """Return the deterministic V495 claim set evaluated by this certificate."""

    return [dict(claim) for claim in CLAIM_SPECS]


def evaluate_claims(
    upstreams: Mapping[str, Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> JsonList:
    """Evaluate each claim against measured fields or reject missing evidence."""

    return [_evaluate_claim(upstreams, claim) for claim in claims]


def evaluate_certificate(
    *,
    upstreams: Mapping[str, Mapping[str, Any]] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Evaluate the complete bounded measurement-access certificate."""

    upstream_payloads = load_upstream_artifacts() if upstreams is None else upstreams
    claim_set = build_claim_set() if claims is None else list(claims)
    records = evaluate_claims(upstream_payloads, claim_set)
    true_records = [row for row in records if row["claim_kind"] == "true_measured"]
    false_records = [row for row in records if row["claim_kind"] == "false_property"]
    unsupported_records = [
        row for row in records if row["claim_kind"] in {"unsupported", "broad_soundness"}
    ]
    hardware_records = [row for row in records if row["claim_domain"] == "hardware_speedup"]
    token_internal_records = [row for row in records if row["claim_domain"] == "token_internal"]
    return {
        "property_family": PROPERTY_FAMILY,
        "gated_upstreams_ready": upstream_gates_ready(upstream_payloads),
        "upstream_gate_status": {
            "exp5443_verifier_potential_fixture_ready": upstream_payloads["exp5443"].get(
                "verifier_potential_fixture_ready"
            )
            is True,
            "exp5446_governed_csl_loop_ready": upstream_payloads["exp5446"].get(
                "governed_csl_loop_ready"
            )
            is True,
        },
        "claim_records": records,
        "claim_count": len(records),
        "true_measured_claim_preservation_rate": _rate(
            sum(row["preserved"] for row in true_records),
            len(true_records),
        ),
        "false_property_rejection_rate": _rate(
            sum(row["rejected"] for row in false_records),
            len(false_records),
        ),
        "unsupported_claim_rejection_rate": _rate(
            sum(row["rejected"] for row in unsupported_records),
            len(unsupported_records),
        ),
        "verifier_potential_claims_checked": _domain_count(records, "verifier_potential"),
        "governed_memory_claims_checked": _domain_count(records, "governed_memory"),
        "hardware_speedup_claim_rejected": bool(hardware_records)
        and all(row["rejected"] for row in hardware_records),
        "token_internal_claim_rejected": bool(token_internal_records)
        and all(row["rejected"] for row in token_internal_records),
        "broad_kan_claim_made": False,
        "claim_limits": _claim_limits(),
    }


def build_artifact(
    *,
    upstreams: Mapping[str, Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5451 artifact and fail closed on readiness gaps."""

    diagnostic = evaluate_certificate(upstreams=upstreams)
    certificate_checksum = _certificate_checksum(diagnostic["claim_records"])
    blockers = _readiness_blockers(diagnostic, certificate_checksum, tests_run)
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "property_family": diagnostic["property_family"],
        "gated_upstreams_ready": diagnostic["gated_upstreams_ready"],
        "claim_count": diagnostic["claim_count"],
        "true_measured_claim_preservation_rate": diagnostic[
            "true_measured_claim_preservation_rate"
        ],
        "false_property_rejection_rate": diagnostic["false_property_rejection_rate"],
        "unsupported_claim_rejection_rate": diagnostic["unsupported_claim_rejection_rate"],
        "verifier_potential_claims_checked": diagnostic["verifier_potential_claims_checked"],
        "governed_memory_claims_checked": diagnostic["governed_memory_claims_checked"],
        "hardware_speedup_claim_rejected": diagnostic["hardware_speedup_claim_rejected"],
        "token_internal_claim_rejected": diagnostic["token_internal_claim_rejected"],
        "broad_kan_claim_made": diagnostic["broad_kan_claim_made"],
        "certificate_checksum": certificate_checksum,
        "kan_certificate_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers),
        "status": "complete" if ready else "blocked",
        "claim_records": diagnostic["claim_records"],
        "claim_limits": diagnostic["claim_limits"],
        "readiness_blockers": blockers,
        "tests_run": [dict(row) for row in tests_run],
        "source_artifacts": [
            str(EXP5443_RESULT_RELATIVE_PATH),
            str(EXP5446_RESULT_RELATIVE_PATH),
            str(EXP5438_RESULT_RELATIVE_PATH),
        ],
        "source_artifact_checksums": source_artifact_checksums(),
        "methodology_note": (
            "Exp5451 is a bounded measurement-access certificate over committed "
            "Exp5443 and Exp5446 artifact fields. It preserves true measured "
            "claims, rejects measured false claims, and rejects hardware, token, "
            "internal-state, broad KAN soundness, and general LLM truth claims "
            "whose required measurements are absent."
        ),
    }
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5451 artifact and return the payload."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Reject artifacts that turn measurement gaps into broad KAN claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, ",".join(missing))
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("property_family") == PROPERTY_FAMILY, "property_family")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("broad_kan_claim_made") is False, "broad_kan_claim_made")
    _require(_verdict_is_bounded(str(artifact.get("honest_verdict"))), "honest_verdict")
    records = list(artifact.get("claim_records", ()))
    true_records = [row for row in records if row.get("claim_kind") == "true_measured"]
    false_records = [row for row in records if row.get("claim_kind") == "false_property"]
    unsupported_records = [
        row for row in records if row.get("claim_kind") in {"unsupported", "broad_soundness"}
    ]
    _require(artifact.get("claim_count") == len(records), "claim_count")
    _require(artifact.get("claim_count") == len(CLAIM_SPECS), "claim_count")
    if artifact.get("kan_certificate_ready") is True:
        _require(
            bool(true_records) and all(row.get("preserved") is True for row in true_records),
            "claim_records",
        )
    _require(
        bool(false_records) and all(row.get("rejected") is True for row in false_records),
        "claim_records",
    )
    _require(
        bool(unsupported_records)
        and all(
            row.get("rejected") is True and row.get("missing_evidence")
            for row in unsupported_records
        ),
        "claim_records",
    )
    _require(
        artifact.get("true_measured_claim_preservation_rate")
        == _rate(sum(bool(row.get("preserved")) for row in true_records), len(true_records)),
        "true_measured_claim_preservation_rate",
    )
    _require(
        artifact.get("false_property_rejection_rate")
        == _rate(sum(bool(row.get("rejected")) for row in false_records), len(false_records)),
        "false_property_rejection_rate",
    )
    _require(
        artifact.get("unsupported_claim_rejection_rate")
        == _rate(
            sum(bool(row.get("rejected")) for row in unsupported_records),
            len(unsupported_records),
        ),
        "unsupported_claim_rejection_rate",
    )
    _require(
        artifact.get("verifier_potential_claims_checked")
        == _domain_count(records, "verifier_potential"),
        "verifier_potential_claims_checked",
    )
    _require(
        artifact.get("governed_memory_claims_checked") == _domain_count(records, "governed_memory"),
        "governed_memory_claims_checked",
    )
    _require(
        artifact.get("hardware_speedup_claim_rejected")
        is (
            bool([row for row in records if row.get("claim_domain") == "hardware_speedup"])
            and all(
                row.get("rejected")
                for row in records
                if row.get("claim_domain") == "hardware_speedup"
            )
        ),
        "hardware_speedup_claim_rejected",
    )
    _require(
        artifact.get("token_internal_claim_rejected")
        is (
            bool([row for row in records if row.get("claim_domain") == "token_internal"])
            and all(
                row.get("rejected")
                for row in records
                if row.get("claim_domain") == "token_internal"
            )
        ),
        "token_internal_claim_rejected",
    )
    _require(_claim_limits_explicit(artifact.get("claim_limits", ())), "claim_limits")
    _require(
        artifact.get("certificate_checksum") == _certificate_checksum(records),
        "certificate_checksum",
    )
    if artifact.get("kan_certificate_ready") is True:
        _require(artifact.get("status") == "complete", "status")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(artifact.get("gated_upstreams_ready") is True, "gated_upstreams_ready")
        _require(bool(artifact.get("tests_run")), "tests_run")
        _require(artifact.get("true_measured_claim_preservation_rate") == 1.0, "claim_records")
        _require(artifact.get("false_property_rejection_rate") == 1.0, "claim_records")
        _require(artifact.get("unsupported_claim_rejection_rate") == 1.0, "claim_records")
    else:
        _require(artifact.get("status") == "blocked", "status")
        _require(str(artifact.get("honest_verdict")).startswith("blocked:"), "honest_verdict")
        _require(bool(artifact.get("readiness_blockers")), "readiness_blockers")
    return True


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return checksums for source artifacts, the spec, and this module."""

    root_path = Path(root)
    return {
        "exp5443": _sha256_if_exists(root_path / EXP5443_RESULT_RELATIVE_PATH),
        "exp5446": _sha256_if_exists(root_path / EXP5446_RESULT_RELATIVE_PATH),
        "exp5438": _sha256_if_exists(root_path / EXP5438_RESULT_RELATIVE_PATH),
        "spec": _sha256_if_exists(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_if_exists(root_path / MODULE_RELATIVE_PATH),
    }


def default_tests_run() -> JsonList:
    """Return the validation commands recorded in the terminal artifact."""

    test_path = (
        "tests/python/test_experiment_5451_kan_verifier_potential_memory_certificate_v495.py"
    )
    module_path = "python/carnot/experiment_5451_kan_verifier_potential_memory_certificate_v495.py"
    return [
        {"command": f".venv/bin/pytest {test_path} -q --no-cov -n 0", "outcome": "passed"},
        {
            "command": (
                ".venv/bin/coverage run "
                f"--include={module_path} -m pytest {test_path} -q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (f".venv/bin/coverage report --include={module_path} --fail-under=100"),
            "outcome": "passed",
        },
        {
            "command": "python scripts/check_spec_coverage.py",
            "outcome": "failed_pre_existing_1262_missing_spec_refs",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def _evaluate_claim(
    upstreams: Mapping[str, Mapping[str, Any]],
    claim: Mapping[str, Any],
) -> JsonDict:
    required_fields = list(claim["required_fields"])
    missing_evidence = [field for field in required_fields if not _path_exists(upstreams, field)]
    measured_values = {
        field: _summarize_value(_read_path(upstreams, field))
        for field in required_fields
        if _path_exists(upstreams, field)
    }
    claim_kind = str(claim["claim_kind"])
    if missing_evidence:
        claim_truth = False
        classification = "missing_evidence_unsupported"
        rejected = True
        preserved = False
    else:
        claim_truth = all(_check(upstreams, check) for check in claim["checks"])
        preserved = claim_kind == "true_measured" and claim_truth
        rejected = claim_kind != "true_measured" and not claim_truth
        classification = "measured_supported" if claim_truth else "measured_contradicted"
    return {
        "claim_id": claim["claim_id"],
        "claim_kind": claim_kind,
        "claim_domain": claim["claim_domain"],
        "statement": claim["statement"],
        "classification": classification,
        "required_fields": required_fields,
        "measured_values": measured_values,
        "missing_evidence": missing_evidence,
        "claim_truth": claim_truth,
        "preserved": preserved,
        "rejected": rejected,
        "bounded_fixture_only": True,
    }


def _check(upstreams: Mapping[str, Mapping[str, Any]], check: Mapping[str, Any]) -> bool:
    op = check["op"]
    if op == "sum_lengths_eq":
        actual = sum(len(_read_path(upstreams, path)) for path in check["paths"])
    else:
        actual = _read_path(upstreams, check["path"])
    if op == "eq" or op == "sum_lengths_eq":
        return actual == check["value"]
    if op == "gt":
        return actual > check["value"]
    raise ValueError(f"unsupported claim check op: {op}")


def _path_exists(upstreams: Mapping[str, Mapping[str, Any]], dotted_path: str) -> bool:
    try:
        _read_path(upstreams, dotted_path)
    except KeyError:
        return False
    return True


def _read_path(upstreams: Mapping[str, Mapping[str, Any]], dotted_path: str) -> Any:
    value: Any = upstreams
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _summarize_value(value: Any) -> Any:
    if isinstance(value, list):
        return {"count": len(value)}
    return value


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    certificate_checksum: str,
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if diagnostic["gated_upstreams_ready"] is not True:
        gate_status = diagnostic["upstream_gate_status"]
        if gate_status["exp5443_verifier_potential_fixture_ready"] is not True:
            blockers.append("exp5443_verifier_potential_fixture_not_ready")
        if gate_status["exp5446_governed_csl_loop_ready"] is not True:
            blockers.append("exp5446_governed_csl_loop_not_ready")
    if diagnostic["true_measured_claim_preservation_rate"] != 1.0:
        blockers.append("true_measured_claim_preservation")
    if diagnostic["false_property_rejection_rate"] != 1.0:
        blockers.append("false_property_rejection")
    if diagnostic["unsupported_claim_rejection_rate"] != 1.0:
        blockers.append("unsupported_claim_rejection")
    if diagnostic["hardware_speedup_claim_rejected"] is not True:
        blockers.append("hardware_speedup_claim_rejected")
    if diagnostic["token_internal_claim_rejected"] is not True:
        blockers.append("token_internal_claim_rejected")
    if diagnostic["broad_kan_claim_made"] is not False:
        blockers.append("broad_kan_claim_made")
    if not certificate_checksum.startswith("sha256:"):
        blockers.append("certificate_checksum")
    if not _claim_limits_explicit(diagnostic["claim_limits"]):
        blockers.append("claim_limits")
    if not tests_run:
        blockers.append("tests_recorded")
    return blockers


def _honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    if ready:
        return (
            "complete: bounded measurement-access certificate preserved measured "
            "Exp5443 and Exp5446 claims and rejected false or unsupported claims"
        )
    return "blocked: bounded measurement-access certificate failed gates " + ",".join(blockers)


def _verdict_is_bounded(verdict: str) -> bool:
    lower = verdict.lower()
    if not lower.startswith(TERMINAL_PREFIXES):
        return False
    forbidden = (
        "broad kan soundness proved",
        "general llm truth proved",
        "hardware speedup proved",
        "token-level access proved",
        "hidden/internal-state access proved",
    )
    return not any(phrase in lower for phrase in forbidden)


def _claim_limits() -> list[str]:
    return [
        "bounded Exp5443 verifier-potential and Exp5446 governed-memory artifact fields only",
        "false verifier-potential claims are rejected only from measured Exp5443 fields",
        "false governed-memory claims are rejected only from measured Exp5446 fields",
        "hardware speedup claims require timing evidence absent from this certificate",
        "token and hidden/internal-state claims require evidence absent from this certificate",
        "no broad KAN verification, KAN soundness, trained-network soundness, or general LLM truth claim",
    ]


def _claim_limits_explicit(claim_limits: Any) -> bool:
    text = " ".join(str(item).lower() for item in claim_limits)
    return (
        "bounded exp5443" in text
        and "hardware speedup" in text
        and "token" in text
        and "hidden/internal-state" in text
        and "no broad kan" in text
    )


def _domain_count(records: Sequence[Mapping[str, Any]], claim_domain: str) -> int:
    return sum(1 for row in records if row.get("claim_domain") == claim_domain)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _certificate_checksum(claim_records: Sequence[Mapping[str, Any]]) -> str:
    return _checksum({"claim_records": list(claim_records)})


def _checksum(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_if_exists(path: Path) -> str | None:
    # Resolve at the artifact's own commit so a later append to the shared
    # KAN spec does not stale this receipt (REQ-REPORT-6610; the 2026-08-25
    # adoption sweep, commit 64846b5430, missed this module).
    if not receipt_exists(path, artifact_relative_path=RESULT_RELATIVE_PATH):
        return None
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
