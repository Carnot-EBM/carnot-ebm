#!/usr/bin/env python3
"""Build the Exp6974 claim-provenance duration-lint repair receipt.

The source Exp6967 artifact is historical evidence. This script reads and
hashes it but writes only a separate derivative receipt.

Spec ref: REQ-CONDUCTOR-6974.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct CLI import path only.
    sys.path.insert(0, str(REPO_ROOT))

import scripts.adversarial_verify as av


EXPERIMENT_ID = 6974
RANDOM_SEED = 697420260904
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP6967_PATH = Path("results/experiment_6967_certified_error_headroom_fixture.json")
OUTPUT_PATH = Path("results/experiment_6974_claim_provenance_duration_lint.json")
EXPECTED_EXP6967_SHA256 = "sha256:1685ad1bff1b82aae3a17f80d341e0593d99879809bb9afb3268060100e54fee"
INFERENCE_SUBSTRATE = "deterministic_claim_provenance_verifier"
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

REQUIRED_INPUT_PATHS = (
    EXP6967_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("tests/python/test_adversarial_verify_guards.py"),
    Path("tests/python/test_adversarial_verify_no_llm_name_rule_6593.py"),
    Path("tests/python/test_adversarial_verify_substrate_classification_5933.py"),
    Path("tests/python/test_adversarial_verify_qa_layer_missed_inputs_2026_08_23.py"),
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "classification_rule_rows",
    "mutation_fixture_rows",
    "old_new_decision_rows",
    "false_positive_rows",
    "false_negative_rows",
    "ambiguous_provenance_rows",
    "focused_test_receipts",
    "exp6967_readonly_recheck",
    "verifier_version_hash",
    "duration_lint_repair_complete_score",
    "fixture_admissibility_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required result has a stated scientific reason for inclusion.",
    "preconditions_checked": "A verifier conclusion is valid only when every frozen input is readable and identified.",
    "inference_substrate": "The receipt declares the computation it performed rather than inheriting a source model label.",
    "duration_s": "A monotonic task clock separates receipt work from copied source timing.",
    "source_artifact_hashes": "Immutable source hashes make the derivative decision repeatable without rewriting history.",
    "rows": "Per-fixture rows let readers recompute the aggregate repair decision.",
    "classification_rule_rows": "An explicit rule table makes the claim and invocation boundary falsifiable.",
    "mutation_fixture_rows": "A varied mutation corpus tests both false-positive removal and live-check retention.",
    "old_new_decision_rows": "Paired decisions isolate the behavioral change caused by the repair.",
    "false_positive_rows": "Removed false positives show that source model identifiers no longer become invocation evidence.",
    "false_negative_rows": "Missed expected criticals would reveal a weakened fabrication guard.",
    "ambiguous_provenance_rows": "Ambiguous cases stay visible because silence would look like verification.",
    "focused_test_receipts": "Terminal test receipts distinguish executed checks from planned checks.",
    "exp6967_readonly_recheck": "The incident artifact is re-evaluated while its stored stamp and bytes remain intact.",
    "verifier_version_hash": "The decision names the exact verifier implementation that produced it.",
    "duration_lint_repair_complete_score": "Completion is one only when every mutation has the expected terminal decision.",
    "fixture_admissibility_ready_score": "Readiness is one only when Exp6967 keeps its source identity and has no current live-duration critical.",
    "random_seed": "A fixed seed identifies the deterministic mutation-corpus contract.",
    "reproducibility_checksum": "A canonical checksum detects drift in decisions while excluding wall-clock noise.",
    "gate_check_summary": "Every blocked result states the failed check and its expected and observed values.",
    "verifier_is_oracle": "The verifier directly defines this infrastructure receipt, so a positive is circular.",
    "verdict_class": "A closed verdict class prevents success-shaped text from hiding a blocked or null result.",
    "honest_verdict": "A terminal prefix states the receipt outcome before any supporting metrics.",
}

CLASSIFICATION_RULE_ROWS = (
    {
        "field_family": "inference_substrate",
        "scope": "top_level_current_task",
        "rule": "Explicit live, no-LLM, aggregation, and deterministic claims lead classification.",
    },
    {
        "field_family": "invocation_boolean_or_count",
        "scope": "recursive_typed_fields",
        "rule": "Positive typed invocation fields are evidence even when nested; cited upstream fields stay external.",
    },
    {
        "field_family": "live_duration",
        "scope": "typed_current_task_fields",
        "rule": "A positive live, inference, generation, or model-load duration evidences current live work.",
    },
    {
        "field_family": "gpu_receipt",
        "scope": "typed_current_task_receipts",
        "rule": "Model-load or offload measurements in a current-task GPU receipt evidence live work.",
    },
    {
        "field_family": "methodology",
        "scope": "current_task_methodology",
        "rule": "A narrow affirmative methodology statement can evidence live inference; explicit negation wins first.",
    },
    {
        "field_family": "arbitrary_values",
        "scope": "rows_hashes_paths_diagnostics_model_labels",
        "rule": "String content alone never proves invocation; unknown model context remains ambiguous and fails closed.",
    },
)


def sha256_path(path: Path) -> str:
    """Return a prefixed hash so stored source identities cannot be mistaken for paths."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _base_payload(**overrides: Any) -> dict[str, Any]:
    """Create the shared deterministic fixture without claiming model invocation."""

    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete_circular_claim_provenance_fixture",
        "inference_substrate": "deterministic_z3_and_bounded_enumeration_reducer",
        "duration_s": 1.864951312,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "7" * 64,
        "source_rows": [
            {
                "attempt_key": "unsloth/Qwen3.6-35B-A3B-GGUF|0-3-0|direct_affine",
                "model_label": "unsloth/gemma-4-31B-it-GGUF",
            }
        ],
    }
    payload.update(overrides)
    return payload


def mutation_corpus() -> list[dict[str, Any]]:
    """Return the fixed paired-control corpus required by REQ-CONDUCTOR-6974."""

    ambiguous = _base_payload(model_specs={"name": "unsloth/gemma-4-31B-it-GGUF"})
    ambiguous.pop("inference_substrate")
    return [
        {
            "fixture_id": "genuine_live_inference",
            "category": "genuine_live",
            "expected_old_decision": "critical",
            "expected_new_decision": "critical",
            "payload": _base_payload(
                inference_substrate="live_llm_inference",
                duration_s=5.0,
                model_invoked=True,
                model_specs={"name": "unsloth/Qwen3.6-35B-A3B-GGUF"},
            ),
        },
        {
            "fixture_id": "deterministic_source_row_reducer",
            "category": "deterministic_reducer",
            "expected_old_decision": "critical",
            "expected_new_decision": "clean",
            "payload": _base_payload(),
        },
        {
            "fixture_id": "nested_input_invocation",
            "category": "nested_marker_attack",
            "expected_old_decision": "critical",
            "expected_new_decision": "critical",
            "payload": _base_payload(
                input_data={"batch": [{"invocation": {"model_invoked": True}}]},
            ),
        },
        {
            "fixture_id": "ambiguous_model_context",
            "category": "ambiguous",
            "expected_old_decision": "critical",
            "expected_new_decision": "critical",
            "payload": ambiguous,
        },
        {
            "fixture_id": "bibliographic_source_scan",
            "category": "bibliographic",
            "expected_old_decision": "clean",
            "expected_new_decision": "clean",
            "payload": _base_payload(
                inference_substrate="web_and_bibliographic_search_only",
                duration_s=0.05,
            ),
        },
        {
            "fixture_id": "live_embedding_extraction",
            "category": "embedding",
            "expected_old_decision": "critical",
            "expected_new_decision": "critical",
            "payload": _base_payload(
                inference_substrate="live_llm_embedding_extraction",
                duration_s=1.0,
                model_invoked=True,
                model_specs={"name": "unsloth/gemma-4-26B-A4B-it-GGUF"},
            ),
        },
        {
            "fixture_id": "pre_gate_block",
            "category": "pre_gate",
            "expected_old_decision": "clean",
            "expected_new_decision": "clean",
            "payload": _base_payload(
                honest_verdict="blocked_model_not_cached",
                inference_substrate="live_llm_inference",
                duration_s=0.01,
                model_invoked=False,
            ),
        },
    ]


def _legacy_duration_floor(payload: dict[str, Any]) -> float | None:
    """Reproduce the pre-repair duration choice for paired decision reporting."""

    if av._is_precondition_check_only_blocked(payload):
        return None
    classification = av._classify_inference_substrate(payload)
    if av._is_verifier_scoring_only(payload):
        return av.VERIFIER_SCORING_MIN_DURATION_S
    if av._is_aggregation_only(payload):
        return av.AGGREGATION_MIN_DURATION_S
    if av._is_deterministic_verifier(payload):
        return av.DETERMINISTIC_VERIFIER_MIN_DURATION_S
    if av._is_arc_live_agent_no_llm(payload):
        return av.ARC_LIVE_AGENT_NO_LLM_MIN_DURATION_S
    if av._is_llm_embedding_extraction(payload):
        return av.LLM_EMBEDDING_EXTRACTION_MIN_DURATION_S
    if av._is_log_analysis_local_timing(payload):
        return av.LOG_ANALYSIS_LOCAL_TIMING_MIN_DURATION_S
    if av._is_artifact_qa_lint_tests(payload):
        return av.ARTIFACT_QA_LINT_TESTS_MIN_DURATION_S
    if av._is_web_bibliographic_search_only(payload):
        return av.WEB_BIBLIOGRAPHIC_SEARCH_ONLY_MIN_DURATION_S
    if av._is_local_sota_gguf_small_n(payload):
        return av.LOCAL_SOTA_GGUF_SMALL_N_MIN_DURATION_S
    if av._is_deterministic_smt_hint_validation(payload):
        return av.DETERMINISTIC_SMT_HINT_VALIDATION_MIN_DURATION_S
    if av._is_native_gguf_backend_bisect(payload):
        return av.NATIVE_GGUF_BACKEND_BISECT_MIN_DURATION_S
    if classification["kind"] == av.SUBSTRATE_KIND_NO_LLM:
        return av.NO_LLM_DECLARED_MIN_DURATION_S
    if av._is_live_llm_inference(payload) or av._has_compute_bound_marker(payload):
        return av.COMPUTE_BOUND_MIN_DURATION_S
    return None


def _legacy_duration_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the old whole-blob marker decision without mutating the verifier."""

    floor = _legacy_duration_floor(payload)
    duration = payload.get("duration_s")
    critical = floor is not None and av._is_finite_number(duration) and float(duration) < floor
    return {
        "decision": "critical" if critical else "clean",
        "min_duration_s": floor,
        "rule": "whole_blob_compute_marker_fallback",
    }


def _current_duration_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the corrected duration decision and its raw duration-rule flags."""

    flags: list[av.Flag] = []
    av.check_duration_vs_claim(payload, flags)
    relevant = [
        flag.to_dict()
        for flag in flags
        if flag.kind in {"DURATION_TOO_SHORT", "INFERENCE_PROVENANCE_CONTRADICTION"}
    ]
    critical = any(flag["severity"] == "critical" for flag in relevant)
    return {
        "decision": "critical" if critical else "clean",
        "classification": av._classify_current_task_inference_claim(payload),
        "duration_flags": relevant,
        "duration_floor": av.duration_floor_for_artifact(payload),
    }


def evaluate_mutation_corpus() -> list[dict[str, Any]]:
    """Evaluate old and corrected rules against every required mutation fixture."""

    rows: list[dict[str, Any]] = []
    for fixture in mutation_corpus():
        old = _legacy_duration_decision(fixture["payload"])
        new = _current_duration_decision(fixture["payload"])
        passed = (
            old["decision"] == fixture["expected_old_decision"]
            and new["decision"] == fixture["expected_new_decision"]
        )
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "category": fixture["category"],
                "expected_old_decision": fixture["expected_old_decision"],
                "old_decision": old["decision"],
                "old_floor_s": old["min_duration_s"],
                "expected_new_decision": fixture["expected_new_decision"],
                "new_decision": new["decision"],
                "new_floor": new["duration_floor"],
                "claim_classification": new["classification"],
                "new_duration_flags": new["duration_flags"],
                "passed": passed,
            }
        )
    return rows


def _preconditions(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    """Check all frozen inputs before classification and preserve observed hashes."""

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str | None] = {}
    for relative in REQUIRED_INPUT_PATHS:
        path = repo_root / relative
        observed = sha256_path(path) if path.is_file() and path.stat().st_size > 0 else None
        hashes[str(relative)] = observed
        expected: Any = "readable_nonempty_file"
        passed = observed is not None
        if relative == EXP6967_PATH:
            expected = EXPECTED_EXP6967_SHA256
            passed = observed == expected
        rows.append(
            {
                "check": f"required_input:{relative}",
                "path": str(relative),
                "expected_value": expected,
                "observed_value": observed,
                "passed": passed,
            }
        )
    return rows, hashes


def _blocked_gate_rows(preconditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert failed preconditions to the repository's standard blocked shape."""

    return [
        {
            "failed_check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
        }
        for row in preconditions
        if row["passed"] is not True
    ]


def run_focused_tests(repo_root: Path) -> list[dict[str, Any]]:
    """Run the two Exp6974 test modules and preserve a terminal command receipt."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "tests/python/test_adversarial_verify_claim_provenance_6974.py",
        "tests/python/test_experiment_6974_claim_provenance_duration_lint.py",
        "-q",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180.0,
            check=False,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        outcome = "pass" if completed.returncode == 0 else "fail"
    except (OSError, subprocess.TimeoutExpired) as exc:
        exit_code = None
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        outcome = "blocked"
    return [
        {
            "name": "exp6974_focused_pytest",
            "command": " ".join(command),
            "exit_code": exit_code,
            "outcome": outcome,
            "stdout": stdout,
            "stderr": stderr,
            "duration_s": time.perf_counter() - started,
        }
    ]


def _checksum_view(artifact: dict[str, Any]) -> dict[str, Any]:
    """Remove process timing and console noise while retaining every decision."""

    view = deepcopy(artifact)
    view.pop("duration_s", None)
    view.pop("reproducibility_checksum", None)
    for receipt in view.get("focused_test_receipts", []):
        if isinstance(receipt, dict):
            receipt.pop("duration_s", None)
            receipt.pop("stdout", None)
            receipt.pop("stderr", None)
    return view


def reproducibility_checksum(artifact: dict[str, Any]) -> str:
    """Hash all stable claims and rows in canonical JSON form."""

    encoded = json.dumps(
        _checksum_view(artifact), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _base_artifact(
    *,
    date: str,
    duration_s: float,
    preconditions: list[dict[str, Any]],
    source_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Create a schema-complete neutral receipt before terminal scores are known."""

    return {
        "schema": "carnot.exp6974.claim_provenance_duration_lint.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_hashes,
        "rows": [],
        "classification_rule_rows": [dict(row) for row in CLASSIFICATION_RULE_ROWS],
        "mutation_fixture_rows": [],
        "old_new_decision_rows": [],
        "false_positive_rows": [],
        "false_negative_rows": [],
        "ambiguous_provenance_rows": [],
        "focused_test_receipts": [],
        "exp6967_readonly_recheck": {},
        "verifier_version_hash": f"sha256:{av.LOADED_GATE_VERSION}",
        "duration_lint_repair_complete_score": 0,
        "fixture_admissibility_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": [],
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_claim_provenance_duration_lint",
    }


def build_artifact(
    *,
    date: str,
    repo_root: Path,
    focused_test_receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a terminal derivative receipt without writing any source artifact."""

    started = time.perf_counter()
    preconditions, source_hashes = _preconditions(repo_root)
    artifact = _base_artifact(
        date=date,
        duration_s=0.0,
        preconditions=preconditions,
        source_hashes=source_hashes,
    )
    failed = _blocked_gate_rows(preconditions)
    if failed:
        artifact["gate_check_summary"] = failed
        artifact["exp6967_readonly_recheck"] = {
            "source_path": str(EXP6967_PATH),
            "source_hash_before": source_hashes.get(str(EXP6967_PATH)),
            "source_hash_after": source_hashes.get(str(EXP6967_PATH)),
            "source_unchanged": True,
            "stored_flagged_adversarial": None,
            "raw_findings": [],
            "live_duration_critical": None,
        }
        artifact["duration_s"] = time.perf_counter() - started
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    mutation_fixtures = mutation_corpus()
    decisions = evaluate_mutation_corpus()
    receipts = focused_test_receipts if focused_test_receipts is not None else []
    source_path = repo_root / EXP6967_PATH
    source_hash_before = sha256_path(source_path)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_report = av.verify_artifact(source_path)
    source_hash_after = sha256_path(source_path)
    raw_findings = list(source_report.get("flags", []))
    live_duration_critical = any(
        row.get("kind") == "DURATION_TOO_SHORT"
        and str(row.get("severity", "")).lower() == "critical"
        for row in raw_findings
    )

    false_positives = [
        row
        for row in decisions
        if row["old_decision"] == "critical"
        and row["expected_new_decision"] == "clean"
        and row["new_decision"] == "clean"
    ]
    false_negatives = [
        row
        for row in decisions
        if row["expected_new_decision"] == "critical" and row["new_decision"] != "critical"
    ]
    ambiguous = [
        row for row in decisions if row["claim_classification"]["state"] == av.CLAIM_STATE_AMBIGUOUS
    ]
    receipt_tests_pass = all(row.get("outcome") == "pass" for row in receipts) if receipts else True
    repair_score = int(all(row["passed"] is True for row in decisions) and receipt_tests_pass)
    fixture_score = int(
        source_hash_before == EXPECTED_EXP6967_SHA256
        and source_hash_after == source_hash_before
        and not live_duration_critical
    )

    artifact.update(
        {
            "rows": [
                {
                    "fixture_id": row["fixture_id"],
                    "category": row["category"],
                    "old_decision": row["old_decision"],
                    "new_decision": row["new_decision"],
                    "expected_new_decision": row["expected_new_decision"],
                    "passed": row["passed"],
                }
                for row in decisions
            ],
            "mutation_fixture_rows": mutation_fixtures,
            "old_new_decision_rows": decisions,
            "false_positive_rows": false_positives,
            "false_negative_rows": false_negatives,
            "ambiguous_provenance_rows": ambiguous,
            "focused_test_receipts": receipts,
            "exp6967_readonly_recheck": {
                "source_path": str(EXP6967_PATH),
                "source_hash_before": source_hash_before,
                "source_hash_after": source_hash_after,
                "source_unchanged": source_hash_after == source_hash_before,
                "stored_flagged_adversarial": source_payload.get("flagged_adversarial"),
                "stored_corrigendum_pending": source_payload.get("corrigendum_pending", []),
                "raw_findings": raw_findings,
                "live_duration_critical": live_duration_critical,
                "claim_classification": av._classify_current_task_inference_claim(source_payload),
                "gate_version": source_report.get("gate_version"),
            },
            "duration_lint_repair_complete_score": repair_score,
            "fixture_admissibility_ready_score": fixture_score,
        }
    )
    if repair_score == 1 and fixture_score == 1:
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = "complete_circular_claim_provenance_duration_lint_repair"
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_null_claim_provenance_duration_lint_not_ready"
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _expected_repair_score(artifact: dict[str, Any]) -> int:
    decisions = artifact.get("old_new_decision_rows", [])
    receipts = artifact.get("focused_test_receipts", [])
    receipt_tests_pass = all(row.get("outcome") == "pass" for row in receipts) if receipts else True
    return int(
        bool(decisions)
        and all(row.get("passed") is True for row in decisions)
        and receipt_tests_pass
    )


def _expected_fixture_score(artifact: dict[str, Any]) -> int:
    recheck = artifact.get("exp6967_readonly_recheck", {})
    return int(
        isinstance(recheck, dict)
        and recheck.get("source_hash_before") == EXPECTED_EXP6967_SHA256
        and recheck.get("source_hash_after") == recheck.get("source_hash_before")
        and recheck.get("source_unchanged") is True
        and recheck.get("live_duration_critical") is False
    )


def validate_artifact(artifact: dict[str, Any], *, repo_root: Path) -> None:
    """Recompute every terminal score and reject a forged receipt."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing_required_fields:{missing}")
    principle_missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]))
    if principle_missing:
        raise ValueError(f"field_principles_missing:{principle_missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        raise ValueError("verdict_class")
    if artifact["verifier_version_hash"] != f"sha256:{av.LOADED_GATE_VERSION}":
        raise ValueError("verifier_version_hash")

    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        if not str(artifact["honest_verdict"]).startswith("blocked_claim_provenance_duration_lint"):
            raise ValueError("honest_verdict")
        if not artifact["gate_check_summary"]:
            raise ValueError("gate_check_summary")
        if artifact["duration_lint_repair_complete_score"] != 0:
            raise ValueError("duration_lint_repair_complete_score")
        if artifact["fixture_admissibility_ready_score"] != 0:
            raise ValueError("fixture_admissibility_ready_score")
    else:
        if not all(row.get("passed") is True for row in artifact["preconditions_checked"]):
            raise ValueError("preconditions_checked")
        repair_score = _expected_repair_score(artifact)
        fixture_score = _expected_fixture_score(artifact)
        if artifact["duration_lint_repair_complete_score"] != repair_score:
            raise ValueError("duration_lint_repair_complete_score")
        if artifact["fixture_admissibility_ready_score"] != fixture_score:
            raise ValueError("fixture_admissibility_ready_score")
        expected_class = "circular_positive" if repair_score == fixture_score == 1 else "null"
        if artifact["verdict_class"] != expected_class:
            raise ValueError("verdict_class")
        expected_prefix = (
            "complete_circular_" if expected_class == "circular_positive" else "complete_null_"
        )
        if not str(artifact["honest_verdict"]).startswith(expected_prefix):
            raise ValueError("honest_verdict")
        source_path = repo_root / EXP6967_PATH
        if (
            source_path.is_file()
            and sha256_path(source_path)
            != artifact["exp6967_readonly_recheck"]["source_hash_after"]
        ):
            raise ValueError("exp6967_source_identity")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def write_json_atomic(path: Path, artifact: dict[str, Any]) -> None:
    """Replace only the requested derivative path after complete validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Run focused checks, build the receipt, validate it, and write it once."""

    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-focused-tests", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    preconditions, _ = _preconditions(repo_root)
    inputs_ready = all(row["passed"] is True for row in preconditions)
    receipts = [] if args.skip_focused_tests or not inputs_ready else run_focused_tests(repo_root)
    artifact = build_artifact(
        date=args.date,
        repo_root=repo_root,
        focused_test_receipts=receipts,
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact, repo_root=repo_root)
    output = args.output.resolve() if args.output else repo_root / OUTPUT_PATH
    write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict_class": artifact["verdict_class"],
                "duration_lint_repair_complete_score": artifact[
                    "duration_lint_repair_complete_score"
                ],
                "fixture_admissibility_ready_score": artifact["fixture_admissibility_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["verdict_class"] != "blocked" else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the required command.
    raise SystemExit(main())
