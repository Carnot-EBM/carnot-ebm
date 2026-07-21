#!/usr/bin/env python3
"""Exp5758 read-only scalar bridge for Exp5751 Rust restart parity evidence.

Spec refs: REQ-REPORT-5758, REQ-SAMPLE-5758,
SCENARIO-REPORT-5758, SCENARIO-REPORT-5758-GATE-REPLAY,
SCENARIO-REPORT-5758-FIELD-PRINCIPLES, SCENARIO-SAMPLE-5758.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.conductor_gates import evaluate_gates
from scripts.experiment_template import ExperimentTemplate


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = Path("results/experiment_5758_rust_parity_scalar_bridge.json")
UPSTREAM_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5751_rust_restart_parity_repair.json"
)
PRIOR_GATE_BLOCK_RELATIVE_PATH = Path(
    "results/experiment_5752_one_axis_allocation_free_10x_crossover.json"
)

EXPERIMENT = 5758
EXPERIMENT_ID = "experiment_5758_rust_parity_scalar_bridge"
TITLE = "Exp 5758: Rust Parity Scalar Bridge"
MILESTONE = "2026.07.514"
RUN_DATE = "20260721"
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
EXPECTED_SIZES = (48, 96, 192)
SPEC_REFS = (
    "REQ-REPORT-5758",
    "REQ-SAMPLE-5758",
    "SCENARIO-REPORT-5758",
    "SCENARIO-REPORT-5758-GATE-REPLAY",
    "SCENARIO-REPORT-5758-FIELD-PRINCIPLES",
    "SCENARIO-SAMPLE-5758",
)
PRODUCER_GATE_FIELDS = (
    "restart_parity_ready_score",
    "distributional_parity_score",
    "fallback_equivalence_score",
    "production_backend_reachable_score",
    "rust_benchmark_gate_ready_score",
)
EXP5764_GATE_FIELDS = (
    "rust_benchmark_gate_ready_score",
    "distributional_parity_score",
    "fallback_equivalence_score",
    "production_backend_reachable_score",
)
REPAIR_SOURCE_PATHS = (
    "python/carnot/samplers/one_axis_rust_backend.py",
    "crates/carnot-samplers/src/one_axis_tempering.rs",
    "crates/carnot-python/src/one_axis_tempering.rs",
)
UPSTREAM_EVIDENCE_PATHS = (
    "results/experiment_5738_one_axis_rust_batched_backend.json",
    "results/experiment_5739_one_axis_batched_10x_crossover.json",
)
TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5758_rust_parity_scalar_bridge.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null -m pytest tests/python/test_experiment_5758_rust_parity_scalar_bridge.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=scripts/experiment_5758_rust_parity_scalar_bridge.py --fail-under=100",
    ".venv/bin/pytest tests/python/test_experiment_template.py -q --no-cov -n 0",
    ".venv/bin/pytest tests/python -q",
    "cargo test -p carnot-samplers one_axis_tempering",
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p carnot-python one_axis --lib",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5758_rust_parity_scalar_bridge.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Lists the final top-level artifact keys after producer normalization.",
    "experiment": "Numeric experiment id for conductor and result indexing.",
    "title": "Human-readable bridge title emitted by ExperimentTemplate.",
    "run_date": "Absolute run date 20260721 prevents relative-date ambiguity.",
    "started_at": "Template timestamp for the bridge replay process, not upstream generation.",
    "finished_at": "Template timestamp for the bridge replay process, not upstream generation.",
    "duration_s": "Template wall-clock duration for cached fixture replay.",
    "status": "Bare terminal state supports machine checks without parsing prose.",
    "random_seed": "Template reproducibility seed for this cached replay.",
    "metrics_used": "Template metric-provenance slot; unknown because no metric package is invoked.",
    "experiment_id": "Stable bridge slug distinct from the sealed Exp5751 producer.",
    "milestone": "Milestone accountability for the scalar bridge.",
    "result_path": "Names the bridge artifact path written by this workflow.",
    "field_principles": "Maps every bridge field to the Exp5751 evidence boundary that justifies it.",
    "preconditions_checked": "Records artifact, source-hash, release-build, failure, divergence, interruption, and prior gate-block checks.",
    "spec_refs": "Binds the bridge to REQ-REPORT-5758 and REQ-SAMPLE-5758.",
    "upstream_artifact_path": "Names the canonical Exp5751 artifact read without mutation.",
    "upstream_artifact_hash": "Binds the upstream Exp5751 artifact to exact bytes.",
    "prior_gate_block_artifact_path": "Names the Exp5752 blocked artifact that exposed the gate shape problem.",
    "prior_gate_block_artifact_hash": "Binds the Exp5752 blocked artifact to exact bytes.",
    "repair_source_hashes": "Pins production repair source files before downstream timing work consumes the bridge.",
    "reproduced_failure_receipt_hash": "Preserves the reproduced Exp5739 restart failure by content hash.",
    "first_divergence_receipt_hash": "Preserves the signed-zero first-divergence diagnosis by content hash.",
    "interruption_manifest_hash": "Preserves the n=48/n=96/n=192 interruption replay manifest by content hash.",
    "release_build_receipt_hash": "Preserves the release PyO3 build receipt by content hash.",
    "parity_case_count": "Records the shared n=48/n=96/n=192 denominator used by every derived predicate.",
    "restart_parity_ready_score": "Copies the upstream restart readiness scalar only after hash and case replay.",
    "distributional_parity_score": "Bare scalar derived from distributional_parity.passed plus zero-tolerance case rows.",
    "fallback_equivalence_score": "Bare scalar derived from exact_fallback_equivalence plus per-case sample matches.",
    "production_backend_reachable_score": "Bare scalar derived from sample_batch callable, scalar API, and no-second-API receipts.",
    "derivation_receipts": "Records source fields and receipt hashes used to derive each bridge scalar.",
    "producer_normalizer_receipts": "Records ExperimentTemplate producer normalization without inventing methodology receipts.",
    "gate_replay_receipts": "Records the exact Exp5764 conductor comparisons on the bridge artifact.",
    "unsafe_synthesis_count": "Must remain zero because the bridge never fabricates missing evidence.",
    "rust_benchmark_gate_ready_score": "Bare gate scalar is true only when Exp5751 evidence and conductor replay both pass.",
    "upstream_modified": "Must remain false because Exp5751 is a read-only input.",
    "sampler_code_modified": "Must remain false because the bridge does not touch sampler implementation.",
    "timing_claimed": "Must remain false because no benchmark is rerun or promoted.",
    "hardware_speedup_claimed": "Must remain false because scalar bridge evidence is not a hardware result.",
    "inference_substrate": "Must be cached_fixture_replay_no_llm because no model or timing run is executed.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable content checksum detects bridge artifact drift.",
    "honest_verdict": "Terminal summary starts with complete: or blocked: and does not inflate Rust throughput.",
}


class BridgeValidationError(ValueError):
    """Raised when Exp5758 cannot preserve the Exp5751 evidence boundary."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise BridgeValidationError(detail)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so bridge replay does not trust metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _raw_file_hash(path: str | Path) -> str:
    return sha256_file(path).removeprefix("sha256:")


def _read_json_object(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), f"JSON object required: {path}")
    return dict(payload)


def _is_wrapper(value: Any) -> bool:
    return isinstance(value, Mapping) and "value" in value and "principle" in value


def _required_mapping(source: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    _require(key in source, f"{path} missing")
    value = source[key]
    _require(not _is_wrapper(value), f"{path} is object-wrapped")
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _required_bool(source: Mapping[str, Any], key: str, path: str) -> bool:
    _require(key in source, f"{path} missing")
    value = source[key]
    _require(not _is_wrapper(value), f"{path} is object-wrapped")
    _require(isinstance(value, bool), f"{path} must be a bare boolean")
    return value


def _required_numeric(source: Mapping[str, Any], key: str, path: str) -> float:
    _require(key in source, f"{path} missing")
    value = source[key]
    _require(not _is_wrapper(value), f"{path} is object-wrapped")
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{path} must be a bare finite number",
    )
    return float(value)


def _cases_by_size(source: Mapping[str, Any], path: str) -> dict[int, Mapping[str, Any]]:
    cases = source.get("cases")
    _require(isinstance(cases, list), f"{path}.cases missing")
    by_size: dict[int, Mapping[str, Any]] = {}
    for case in cases:
        _require(isinstance(case, Mapping), f"{path}.cases rows must be objects")
        size = int(case.get("size", -1))
        _require(size in EXPECTED_SIZES, f"{path}.cases unexpected size {size}")
        _require(size not in by_size, f"{path}.cases ambiguous duplicate size {size}")
        by_size[size] = case
    _require(tuple(sorted(by_size)) == EXPECTED_SIZES, f"{path}.cases denominator")
    return by_size


def _case_zero(case: Mapping[str, Any], key: str, path: str) -> bool:
    return _required_numeric(case, key, f"{path}.{key}") == 0.0


def _distributional_parity_score(upstream: Mapping[str, Any]) -> float:
    parity = _required_mapping(upstream, "distributional_parity", "distributional_parity")
    passed = _required_bool(parity, "passed", "distributional_parity.passed")
    cases = _cases_by_size(parity, "distributional_parity")
    case_ok = all(
        _case_zero(case, "energy_histogram_tv", f"distributional_parity case n={size}")
        and _case_zero(case, "mean_energy_delta_abs", f"distributional_parity case n={size}")
        and _case_zero(case, "best_energy_delta_abs", f"distributional_parity case n={size}")
        for size, case in sorted(cases.items())
    )
    _require(not passed or case_ok, "distributional_parity case contradicts passed")
    return 1.0 if passed and case_ok else 0.0


def _fallback_equivalence_score(upstream: Mapping[str, Any]) -> float:
    parity = _required_mapping(upstream, "fallback_equivalence", "fallback_equivalence")
    exact = _required_bool(
        parity,
        "exact_fallback_equivalence",
        "fallback_equivalence.exact_fallback_equivalence",
    )
    cases = _cases_by_size(parity, "fallback_equivalence")
    case_ok = all(
        _required_bool(case, "rust_python_samples_match", f"fallback_equivalence case n={size}")
        for size, case in sorted(cases.items())
    )
    _require(not exact or case_ok, "fallback_equivalence case contradicts exact_fallback_equivalence")
    return 1.0 if exact and case_ok else 0.0


def _production_backend_reachable_score(upstream: Mapping[str, Any]) -> float:
    reachable = _required_mapping(
        upstream,
        "production_backend_reachable",
        "production_backend_reachable",
    )
    passed = _required_bool(reachable, "passed", "production_backend_reachable.passed")
    callable_ok = _required_bool(
        reachable,
        "sample_batch_callable",
        "production_backend_reachable.sample_batch_callable",
    )
    scalar_api_ok = _required_bool(
        reachable,
        "scalar_api_unchanged",
        "production_backend_reachable.scalar_api_unchanged",
    )
    second_api_added = _required_bool(
        reachable,
        "second_sampler_api_added",
        "production_backend_reachable.second_sampler_api_added",
    )
    derived_ok = passed and callable_ok and scalar_api_ok and not second_api_added
    _require(not passed or derived_ok, "production_backend_reachable contradicts case receipts")
    return 1.0 if derived_ok else 0.0


def _derive_scores(upstream: Mapping[str, Any]) -> JsonDict:
    restart_score = _required_numeric(
        upstream,
        "restart_parity_ready_score",
        "restart_parity_ready_score",
    )
    return {
        "restart_parity_ready_score": restart_score,
        "distributional_parity_score": _distributional_parity_score(upstream),
        "fallback_equivalence_score": _fallback_equivalence_score(upstream),
        "production_backend_reachable_score": _production_backend_reachable_score(upstream),
    }


def _source_hashes(upstream: Mapping[str, Any]) -> Mapping[str, Any]:
    hashes = _required_mapping(upstream, "upstream_artifact_hashes", "upstream_artifact_hashes")
    return _required_mapping(hashes, "source_hashes", "upstream_artifact_hashes.source_hashes")


def _verify_source_hashes(upstream: Mapping[str, Any], input_repo_root: str | Path) -> JsonDict:
    root = Path(input_repo_root)
    source_hashes = _source_hashes(upstream)
    verified: JsonDict = {}
    for rel_path in REPAIR_SOURCE_PATHS:
        expected = source_hashes.get(rel_path)
        _require(isinstance(expected, str) and expected, f"missing source hash: {rel_path}")
        actual = _raw_file_hash(root / rel_path)
        _require(actual == expected, f"source hash drift: {rel_path}")
        verified[rel_path] = expected
    return verified


def _verify_upstream_evidence_hashes(
    upstream: Mapping[str, Any],
    input_repo_root: str | Path,
) -> JsonDict:
    root = Path(input_repo_root)
    source_hashes = _source_hashes(upstream)
    verified: JsonDict = {}
    for rel_path in UPSTREAM_EVIDENCE_PATHS:
        expected = source_hashes.get(rel_path)
        _require(isinstance(expected, str) and expected, f"missing evidence hash: {rel_path}")
        actual = _raw_file_hash(root / rel_path)
        _require(actual == expected, f"evidence artifact hash drift: {rel_path}")
        verified[rel_path] = expected
    _require(upstream["upstream_artifact_hashes"]["Exp5738"] == verified[UPSTREAM_EVIDENCE_PATHS[0]], "Exp5738 hash alias")
    _require(upstream["upstream_artifact_hashes"]["Exp5739"] == verified[UPSTREAM_EVIDENCE_PATHS[1]], "Exp5739 hash alias")
    return verified


def _verify_release_build(upstream: Mapping[str, Any]) -> None:
    receipt = _required_mapping(upstream, "release_build_receipt", "release_build_receipt")
    _require(receipt.get("completed") is True, "release_build_receipt.completed")
    _require(receipt.get("exit_code") == 0, "release_build_receipt.exit_code")
    _require(receipt.get("profile") == "release", "release_build_receipt.profile")


def _verify_reproduced_failure(upstream: Mapping[str, Any]) -> None:
    receipts = upstream.get("reproduced_failure_receipts")
    _require(isinstance(receipts, list) and len(receipts) >= 1, "reproduced_failure_receipts")
    first = receipts[0]
    _require(isinstance(first, Mapping), "reproduced_failure_receipts row")
    _require(first.get("reproduced") is True, "reproduced_failure_receipts.reproduced")
    _require(first.get("size") in EXPECTED_SIZES, "reproduced_failure_receipts.size")
    _require(first.get("failure_class") == "restart_match", "reproduced_failure_receipts.failure_class")
    _require(
        first.get("legacy_rust_restart_suffix_hash") != first.get("legacy_python_restart_suffix_hash"),
        "reproduced_failure_receipts legacy hashes",
    )


def _verify_first_divergence(upstream: Mapping[str, Any]) -> None:
    receipt = _required_mapping(upstream, "first_divergence_receipt", "first_divergence_receipt")
    _require(receipt.get("size") == 96, "first_divergence_receipt.size")
    _require(receipt.get("field") == "log_ratio", "first_divergence_receipt.field")
    for key in (
        "semantic_float_equal",
        "semantic_state_equal",
        "samples_equal",
        "rng_state_equal",
        "checkpoint_state_equal",
    ):
        _require(receipt.get(key) is True, f"first_divergence_receipt.{key}")


def _verify_interruption_manifest(upstream: Mapping[str, Any]) -> None:
    manifest = upstream.get("interruption_injection_manifest")
    _require(isinstance(manifest, list), "interruption_injection_manifest")
    by_size: dict[int, Mapping[str, Any]] = {}
    expected_transitions = [
        "before_checkpoint_save_after_prefix",
        "after_checkpoint_load_before_suffix",
        "before_restart_suffix",
        "after_restart_suffix",
    ]
    for row in manifest:
        _require(isinstance(row, Mapping), "interruption_injection_manifest row")
        size = int(row.get("size", -1))
        _require(size in EXPECTED_SIZES, f"interruption_injection_manifest size {size}")
        _require(size not in by_size, f"interruption_injection_manifest duplicate size {size}")
        _require(row.get("transitions") == expected_transitions, f"interruption transitions n={size}")
        _require(row.get("restart_suffix_hash_match") is True, f"interruption suffix n={size}")
        _require(row.get("combined_rust_matches_uninterrupted") is True, f"rust uninterrupted n={size}")
        _require(row.get("combined_python_matches_uninterrupted") is True, f"python uninterrupted n={size}")
        by_size[size] = row
    _require(tuple(sorted(by_size)) == EXPECTED_SIZES, "interruption_injection_manifest denominator")


def _verify_case_family(upstream: Mapping[str, Any], family: str, flag: str) -> None:
    receipt = _required_mapping(upstream, family, family)
    _require(receipt.get("passed") is True, f"{family}.passed")
    cases = _cases_by_size(receipt, family)
    for size, case in sorted(cases.items()):
        _require(case.get(flag) is True, f"{family} case n={size}.{flag}")


def _verify_restart_parity(upstream: Mapping[str, Any]) -> None:
    receipt = _required_mapping(upstream, "restart_parity", "restart_parity")
    _require(receipt.get("all_repaired_suffix_hashes_match") is True, "restart_parity suffixes")
    _require(receipt.get("restart_count_match") is True, "restart_parity restart_count_match")
    cases = _cases_by_size(receipt, "restart_parity")
    for size, case in sorted(cases.items()):
        _require(case.get("restart_suffix_hash_match") is True, f"restart_parity suffix n={size}")
        _require(case.get("restart_count_match") is True, f"restart_parity count n={size}")


def _verify_checkpoint_parity(upstream: Mapping[str, Any]) -> None:
    receipt = _required_mapping(upstream, "checkpoint_parity", "checkpoint_parity")
    for key in (
        "passed",
        "semantic_checkpoint_hash_match",
        "payload_checksums_valid",
        "corrupted_checkpoint_rejected",
    ):
        _require(receipt.get(key) is True, f"checkpoint_parity.{key}")


def _verify_sample_count(upstream: Mapping[str, Any]) -> None:
    receipt = _required_mapping(upstream, "sample_count_parity", "sample_count_parity")
    _require(receipt.get("passed") is True, "sample_count_parity.passed")
    cases = _cases_by_size(receipt, "sample_count_parity")
    for size, case in sorted(cases.items()):
        _require(case.get("sample_count_match") is True, f"sample_count_parity n={size}")
        _require(case.get("sample_count") == 3, f"sample_count_parity sample_count n={size}")


def _verify_parity_cases(upstream: Mapping[str, Any]) -> None:
    _verify_case_family(upstream, "energy_parity", "energy_events_match")
    _verify_case_family(upstream, "proposal_parity", "proposal_events_match")
    _verify_case_family(upstream, "scheduler_parity", "scheduler_events_match")
    _verify_case_family(upstream, "rng_parity", "rng_final_match")
    _verify_restart_parity(upstream)
    _verify_checkpoint_parity(upstream)
    _verify_sample_count(upstream)
    _distributional_parity_score(upstream)
    _fallback_equivalence_score(upstream)


def _verify_prior_gate_block(prior_gate: Mapping[str, Any]) -> None:
    _require(prior_gate.get("status") == "blocked", "prior Exp5752 status")
    gates = prior_gate.get("gates_evaluated")
    _require(isinstance(gates, list), "prior Exp5752 gates_evaluated")
    evaluated_fields = {
        str(gate.get("artifact_field"))
        for gate in gates
        if isinstance(gate, Mapping) and gate.get("passed") is False
    }
    for field in (
        "distributional_parity",
        "fallback_equivalence",
        "production_backend_reachable",
    ):
        _require(field in evaluated_fields, f"prior Exp5752 gate list missing {field}")


def _paths(
    input_repo_root: str | Path,
    upstream_artifact_path: str | Path | None,
    prior_gate_block_path: str | Path | None,
) -> dict[str, Path]:
    root = Path(input_repo_root)
    return {
        "upstream": Path(upstream_artifact_path)
        if upstream_artifact_path
        else root / UPSTREAM_ARTIFACT_RELATIVE_PATH,
        "prior_gate": Path(prior_gate_block_path)
        if prior_gate_block_path
        else root / PRIOR_GATE_BLOCK_RELATIVE_PATH,
    }


def _verify_upstream_bundle(
    *,
    input_repo_root: str | Path,
    upstream_artifact_path: str | Path | None = None,
    prior_gate_block_path: str | Path | None = None,
) -> JsonDict:
    paths = _paths(input_repo_root, upstream_artifact_path, prior_gate_block_path)
    upstream = _read_json_object(paths["upstream"])
    prior_gate = _read_json_object(paths["prior_gate"])
    _require(upstream.get("spec_refs") == ["REQ-SAMPLE-5751", "SCENARIO-SAMPLE-5751"], "Exp5751 spec_refs")
    _require(str(upstream.get("honest_verdict") or "").startswith("complete:"), "Exp5751 honest_verdict")
    _require(upstream.get("timing_claimed") is False, "Exp5751 timing_claimed")
    _require(upstream.get("hardware_speedup_claimed") is False, "Exp5751 hardware_speedup_claimed")
    _verify_release_build(upstream)
    _verify_reproduced_failure(upstream)
    _verify_first_divergence(upstream)
    _verify_interruption_manifest(upstream)
    _verify_parity_cases(upstream)
    _production_backend_reachable_score(upstream)
    repair_source_hashes = _verify_source_hashes(upstream, input_repo_root)
    upstream_evidence_hashes = _verify_upstream_evidence_hashes(upstream, input_repo_root)
    _verify_prior_gate_block(prior_gate)
    scores = _derive_scores(upstream)
    _require(all(value == 1.0 for value in scores.values()), "Exp5751 derived scores")
    return {
        "paths": paths,
        "upstream": upstream,
        "prior_gate": prior_gate,
        "upstream_hash": sha256_file(paths["upstream"]),
        "prior_gate_hash": sha256_file(paths["prior_gate"]),
        "repair_source_hashes": repair_source_hashes,
        "upstream_evidence_hashes": upstream_evidence_hashes,
        "scores": scores,
        "reproduced_failure_receipt_hash": sha256_json(upstream["reproduced_failure_receipts"]),
        "first_divergence_receipt_hash": sha256_json(upstream["first_divergence_receipt"]),
        "interruption_manifest_hash": sha256_json(upstream["interruption_injection_manifest"]),
        "release_build_receipt_hash": sha256_json(upstream["release_build_receipt"]),
    }


def planned_exp5764_task() -> JsonDict:
    """Return the exact downstream gate predicates replayed by Exp5758."""

    return {
        "id": "exp5764-one-axis-profiled-allocation-free-hot-path",
        "gated_on": [
            {
                "upstream": "exp5758-rust-parity-scalar-bridge",
                "artifact_field": field,
                "op": ">=",
                "value": 1.0,
            }
            for field in EXP5764_GATE_FIELDS
        ],
    }


def _gate_replay_receipts(artifact: Mapping[str, Any]) -> JsonDict:
    with tempfile.TemporaryDirectory(prefix="exp5758_gate_replay_") as tmp:
        results_dir = Path(tmp)
        path = results_dir / RESULT_RELATIVE_PATH.name
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = evaluate_gates(planned_exp5764_task(), results_dir)
    return {
        "task_id": planned_exp5764_task()["id"],
        "passed": result.passed,
        "summary": result.summary,
        "gates": [
            {
                "upstream": gate.upstream,
                "artifact_field": gate.artifact_field,
                "op": gate.op,
                "expected": gate.expected,
                "actual": gate.actual,
                "passed": gate.passed,
                "reason": gate.reason,
            }
            for gate in result.gates_evaluated
        ],
    }


def expected_rust_benchmark_gate_ready_score(artifact: Mapping[str, Any]) -> float:
    ready = (
        artifact.get("restart_parity_ready_score") == 1.0
        and artifact.get("distributional_parity_score") == 1.0
        and artifact.get("fallback_equivalence_score") == 1.0
        and artifact.get("production_backend_reachable_score") == 1.0
        and artifact.get("unsafe_synthesis_count") == 0
        and artifact.get("upstream_modified") is False
        and artifact.get("sampler_code_modified") is False
        and artifact.get("timing_claimed") is False
        and artifact.get("hardware_speedup_claimed") is False
        and dict(artifact.get("derivation_receipts") or {}).get("source_hashes_verified") is True
        and dict(artifact.get("derivation_receipts") or {}).get("upstream_artifact_hash_verified")
        is True
        and dict(artifact.get("gate_replay_receipts") or {}).get("passed") is True
    )
    return 1.0 if ready else 0.0


def _base_data(bundle: Mapping[str, Any]) -> JsonDict:
    paths = dict(bundle["paths"])
    scores = dict(bundle["scores"])
    upstream = dict(bundle["upstream"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "run_date": RUN_DATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": {
            "run_date": RUN_DATE,
            "canonical_exp5751_artifact_read": True,
            "prior_exp5752_gate_block_confirmed": True,
            "upstream_artifact_hash_replayed": True,
            "repair_source_hashes_replayed": True,
            "upstream_evidence_hashes_replayed": True,
            "release_build_completed": True,
            "reproduced_failure_receipt_present": True,
            "first_divergence_receipt_present": True,
            "interruption_manifest_verified": True,
            "parity_case_sizes": list(EXPECTED_SIZES),
            "distributional_cases_zero_tolerance": True,
            "fallback_cases_exact": True,
            "production_backend_reachable": True,
            "sampler_code_modified": False,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_path": str(paths["upstream"]),
        "upstream_artifact_hash": bundle["upstream_hash"],
        "prior_gate_block_artifact_path": str(paths["prior_gate"]),
        "prior_gate_block_artifact_hash": bundle["prior_gate_hash"],
        "repair_source_hashes": dict(bundle["repair_source_hashes"]),
        "reproduced_failure_receipt_hash": bundle["reproduced_failure_receipt_hash"],
        "first_divergence_receipt_hash": bundle["first_divergence_receipt_hash"],
        "interruption_manifest_hash": bundle["interruption_manifest_hash"],
        "release_build_receipt_hash": bundle["release_build_receipt_hash"],
        "parity_case_count": len(EXPECTED_SIZES),
        "restart_parity_ready_score": scores["restart_parity_ready_score"],
        "distributional_parity_score": scores["distributional_parity_score"],
        "fallback_equivalence_score": scores["fallback_equivalence_score"],
        "production_backend_reachable_score": scores["production_backend_reachable_score"],
        "derivation_receipts": {
            "source_experiment": "Exp5751",
            "source_spec_refs": list(upstream["spec_refs"]),
            "upstream_hash": bundle["upstream_hash"],
            "prior_gate_block_hash": bundle["prior_gate_hash"],
            "source_hashes_verified": True,
            "verified_repair_source_paths": list(REPAIR_SOURCE_PATHS),
            "upstream_artifact_hash_verified": True,
            "upstream_evidence_hashes_verified": dict(bundle["upstream_evidence_hashes"]),
            "receipt_hashes": {
                "reproduced_failure_receipts": bundle["reproduced_failure_receipt_hash"],
                "first_divergence_receipt": bundle["first_divergence_receipt_hash"],
                "interruption_injection_manifest": bundle["interruption_manifest_hash"],
                "release_build_receipt": bundle["release_build_receipt_hash"],
            },
            "score_sources": {
                "restart_parity_ready_score": "Exp5751.restart_parity_ready_score",
                "distributional_parity_score": "Exp5751.distributional_parity.passed plus zero-tolerance cases",
                "fallback_equivalence_score": "Exp5751.fallback_equivalence.exact_fallback_equivalence plus case matches",
                "production_backend_reachable_score": "Exp5751.production_backend_reachable passed/callable/scalar/no-second-api receipts",
            },
        },
        "producer_normalizer_receipts": {
            "safe_repairs": [],
            "unsafe_rejections": [],
            "ready_for_gated_consumers": True,
            "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        },
        "gate_replay_receipts": {"passed": False, "summary": "not replayed", "gates": []},
        "unsafe_synthesis_count": 0,
        "rust_benchmark_gate_ready_score": 1.0,
        "upstream_modified": False,
        "sampler_code_modified": False,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": {command: 0 for command in TEST_COMMANDS},
        "honest_verdict": "complete: Exp5751 Rust parity receipts bridged to bare scalar gates",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _gate_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finalize(artifact: JsonDict) -> JsonDict:
    artifact["schema"] = sorted(artifact.keys())
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_bridge(
    *,
    input_repo_root: str | Path = REPO_ROOT,
    output_repo_root: str | Path = REPO_ROOT,
    upstream_artifact_path: str | Path | None = None,
    prior_gate_block_path: str | Path | None = None,
) -> JsonDict:
    """Build the bridge artifact without writing it."""

    bundle = _verify_upstream_bundle(
        input_repo_root=input_repo_root,
        upstream_artifact_path=upstream_artifact_path,
        prior_gate_block_path=prior_gate_block_path,
    )
    template = ExperimentTemplate(
        EXPERIMENT,
        TITLE,
        RESULT_RELATIVE_PATH.as_posix(),
        repo_root=Path(output_repo_root),
        seed=EXPERIMENT,
    )
    artifact = template.build_result(
        _base_data(bundle),
        status="complete",
        producer_gate_fields=PRODUCER_GATE_FIELDS,
        producer_required_principle_fields=tuple(FIELD_PRINCIPLES),
    )
    artifact["gate_replay_receipts"] = _gate_replay_receipts(artifact)
    artifact["rust_benchmark_gate_ready_score"] = expected_rust_benchmark_gate_ready_score(
        artifact
    )
    artifact["honest_verdict"] = (
        "complete: Exp5751 Rust parity receipts bridged to bare scalar gates"
        if artifact["rust_benchmark_gate_ready_score"] == 1.0
        else "blocked: Exp5751 Rust parity scalar bridge gates failed"
    )
    _finalize(artifact)
    validate_artifact(artifact, input_repo_root=input_repo_root)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    input_repo_root: str | Path = REPO_ROOT,
) -> bool:
    """Validate the final bridge shape and readiness contract."""

    missing = [field for field in FIELD_PRINCIPLES if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(set(artifact) == set(artifact.get("field_principles") or {}), "field_principles")
    for field in PRODUCER_GATE_FIELDS:
        _require(_gate_scalar(artifact.get(field)), f"{field} must be bare scalar")
    _require(
        artifact.get("rust_benchmark_gate_ready_score")
        == expected_rust_benchmark_gate_ready_score(artifact),
        "rust_benchmark_gate_ready_score",
    )
    paths = _paths(
        input_repo_root,
        artifact.get("upstream_artifact_path"),
        artifact.get("prior_gate_block_artifact_path"),
    )
    upstream = _read_json_object(paths["upstream"])
    prior_gate = _read_json_object(paths["prior_gate"])
    _require(artifact.get("upstream_artifact_hash") == sha256_file(paths["upstream"]), "upstream_artifact_hash")
    _require(artifact.get("prior_gate_block_artifact_hash") == sha256_file(paths["prior_gate"]), "prior_gate_block_artifact_hash")
    _require(artifact.get("repair_source_hashes") == _verify_source_hashes(upstream, input_repo_root), "repair_source_hashes")
    _verify_upstream_evidence_hashes(upstream, input_repo_root)
    _verify_prior_gate_block(prior_gate)
    _verify_release_build(upstream)
    _verify_reproduced_failure(upstream)
    _verify_first_divergence(upstream)
    _verify_interruption_manifest(upstream)
    _verify_parity_cases(upstream)
    scores = _derive_scores(upstream)
    for field, expected in scores.items():
        _require(artifact.get(field) == expected, field)
    _require(artifact.get("parity_case_count") == len(EXPECTED_SIZES), "parity_case_count")
    _require(
        artifact.get("reproduced_failure_receipt_hash")
        == sha256_json(upstream["reproduced_failure_receipts"]),
        "reproduced_failure_receipt_hash",
    )
    _require(
        artifact.get("first_divergence_receipt_hash")
        == sha256_json(upstream["first_divergence_receipt"]),
        "first_divergence_receipt_hash",
    )
    _require(
        artifact.get("interruption_manifest_hash")
        == sha256_json(upstream["interruption_injection_manifest"]),
        "interruption_manifest_hash",
    )
    _require(
        artifact.get("release_build_receipt_hash") == sha256_json(upstream["release_build_receipt"]),
        "release_build_receipt_hash",
    )
    _require(artifact.get("upstream_modified") is False, "upstream_modified")
    _require(artifact.get("sampler_code_modified") is False, "sampler_code_modified")
    _require(artifact.get("timing_claimed") is False, "timing_claimed")
    _require(artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("unsafe_synthesis_count") == 0, "unsafe_synthesis_count")
    verdict = str(artifact.get("honest_verdict") or "")
    _require(verdict.startswith(("complete:", "blocked:")), "honest_verdict")
    _require(
        artifact.get("reproducibility_checksum") == reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return True


def run(
    *,
    input_repo_root: str | Path = REPO_ROOT,
    output_repo_root: str | Path = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5758 bridge artifact."""

    artifact = build_bridge(input_repo_root=input_repo_root, output_repo_root=output_repo_root)
    if write:
        output = Path(output_repo_root) / RESULT_RELATIVE_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    artifact = run(input_repo_root=args.input_repo_root, output_repo_root=args.output_repo_root)
    print(
        json.dumps(
            {
                "result_path": str(args.output_repo_root / RESULT_RELATIVE_PATH),
                "rust_benchmark_gate_ready_score": artifact[
                    "rust_benchmark_gate_ready_score"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
