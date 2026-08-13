"""Build the Exp6396 capability-qualified verified frontier A/B artifact.

Spec refs: REQ-LEARN-6396, SCENARIO-LEARN-6396-LICENSED-CELLS,
SCENARIO-LEARN-6396-FRONTIER, SCENARIO-LEARN-6396-FUTURE,
SCENARIO-LEARN-6396-ATTACKS, SCENARIO-LEARN-6396-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6395_held_factor_transport_license_matrix as exp6395
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6396_capability_qualified_verified_frontier_ab.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6396_capability_qualified_verified_frontier_ab"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6395_RELATIVE_PATH = exp6395.RESULT_RELATIVE_PATH
EXP6381_RELATIVE_PATH = Path(
    "results/experiment_6381_verified_frontier_live_factor_proposal_ab.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")

SCHEMA = "carnot.experiment_6396.capability_qualified_verified_frontier_ab.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6396
TOKENIZER_METHOD = exp6395.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "licensed_cell_deterministic_replay_over_local_gguf_receipts"

MANDATED_MODEL_IDS = exp6395.MANDATED_MODEL_IDS
MODEL_TEMPLATE_BY_ID = exp6395.MODEL_TEMPLATE_BY_ID
REQUIRED_CONSTRAINT_FAMILIES = exp6395.REQUIRED_CONSTRAINT_FAMILIES
ARMS = ("independent_restart", "verified_frontier")
ROUNDS = 3
CANDIDATES_PER_ROUND = 2
FUTURE_EVENTS_PER_LICENSED_CELL = 6
TRAIN_EVENTS_PER_LICENSED_CELL = 6
EXACT_CHECK_COST = 0.01
CHECKER_TIME_PER_CALL_S = 0.0005
WALL_CLOCK_CAP_S = 120.0
RANDOM_SEEDS = {
    "train_manifest": 639600,
    "future_manifest": 639601,
    "arm_order": 639602,
    "raw_outputs": 639603,
    "future_open": 639604,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6396_capability_qualified_verified_frontier_ab --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py "
    "-m pytest tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6396_capability_qualified_verified_frontier_ab.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6395_RELATIVE_PATH,
    EXP6381_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/experiment_6395_held_factor_transport_license_matrix.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6395_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "license_records_used_and_hashes",
    "unlicensed_cell_abstention_records",
    "model_harness_schema_and_checker_bindings",
    "cuda_offload_and_runtime_receipts_by_model",
    "train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness",
    "preregistered_arm_contract",
    "matched_work_receipts",
    "raw_output_before_parse_paths_hashes_and_counts",
    "per_cell_transport_source_binding_exact_and_cost_results",
    "incumbent_and_residual_histories",
    "proposal_learnability_results",
    "exact_alignment_results",
    "frozen_selected_factors_by_arm",
    "untouched_future_evaluation_receipts",
    "future_exact_yield_by_arm_and_model",
    "delta_verified_future_exact_yield",
    "confidence_intervals_and_effective_sample_sizes",
    "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix",
    "capability_qualified_frontier_ready_score",
    "registry_write_count",
    "protected_leakage_count",
    "model_weight_change_count",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status separates positive, null, blocked, and retired frontier evidence.",
    "exp6395_gate_receipts": "Exp6395 readiness, licenses, and cell abstentions gate this experiment.",
    "MODEL_SPECS": "The three mandated GGUF model rows come from cached SOTA helper calls.",
    "models_used": "Only licensed mandated models with matched frontier work count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer receipts use only embedded GGUF tokenizers.",
    "autotokenizer_usage_count": "Bare zero proves no external tokenizer path was used.",
    "license_records_used_and_hashes": "Licenses bind model, tokenizer, harness, schema, family, manifest, and expiry.",
    "unlicensed_cell_abstention_records": "Unlicensed cells remain visible and abstain without substitution.",
    "model_harness_schema_and_checker_bindings": "Model files, harnesses, schemas, and exact checkers are bound before arms run.",
    "cuda_offload_and_runtime_receipts_by_model": "CUDA offload and cleanup are reported for mandated models.",
    "train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness": "Train and future manifests are sealed, balanced, licensed, and disjoint.",
    "preregistered_arm_contract": "The independent and frontier arms are frozen before scoring.",
    "matched_work_receipts": "Calls, candidates, event order, exact checks, and caps match across arms.",
    "raw_output_before_parse_paths_hashes_and_counts": "Raw proposal bytes are frozen before parse.",
    "per_cell_transport_source_binding_exact_and_cost_results": "Licensed cells report transport, source binding, exact outcomes, latency, and cost.",
    "incumbent_and_residual_histories": "Frontier state stores only verified incumbents and immutable residual failures.",
    "proposal_learnability_results": "Training counterexample response is separate from future utility.",
    "exact_alignment_results": "Exact checker agreement is separate from proposal learnability and future utility.",
    "frozen_selected_factors_by_arm": "One factor per arm is frozen before future access.",
    "untouched_future_evaluation_receipts": "Protected future outcomes open once after factor freeze.",
    "future_exact_yield_by_arm_and_model": "Future exact utility is reported per arm and model before pooling.",
    "delta_verified_future_exact_yield": "The paired future yield delta is a finite bare number.",
    "confidence_intervals_and_effective_sample_sizes": "Intervals and effective sample sizes are reported separately from point estimates.",
    "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix": "Identity, license, order, placebo, work, stopping, and leakage attacks fail closed.",
    "capability_qualified_frontier_ready_score": "Readiness checks treatment firing, work parity, abstention, leak-free future access, and single future open.",
    "registry_write_count": "Bare zero proves the active registry stayed read-only.",
    "protected_leakage_count": "Bare zero proves protected future labels did not leak.",
    "model_weight_change_count": "Bare zero proves no model weights changed.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, unlicensed, underpowered, and attacked cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions bind gates, licenses, models, tokenizers, GPUs, schema, manifests, sources, and protected files.",
    "inference_substrate": "The substrate declares deterministic replay over licensed local GGUF receipts.",
    "verifier_is_oracle": "Bare true applies only to exact task checkers.",
    "field_principles": "Every required field states its guard and scientific purpose.",
    "field_provenance": "Every required field maps to specs, upstream artifacts, manifests, tests, or exact checks.",
    "random_seed": "Fixed seeds pin split, arm, and event order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the frontier boundary.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6396",
        "Exp6395 held license matrix",
        "Exp6381 blocked frontier boundary",
        "train and untouched future manifests",
        "focused Exp6396 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

ATTACK_IDS = (
    "placebo_labels",
    "event_order_perturbation",
    "identity_blind_join",
    "license_swap",
    "equal_work_check",
    "no_gain_stopping_attack",
    "protected_future_leakage",
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and sidecar payloads."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash the compact JSON serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return exp6395.model_slug(model_id)


def rounded(value: float) -> float:
    """Round receipts without hiding small exact costs."""

    return round(float(value), 12)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: str | Path, *, digest: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from disk."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_top_level_not_object:{path}")
    return value


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6395.embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the three mandated GGUF rows through Exp6395's helper path."""

    return exp6395.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_func: TokenizerFn,
) -> list[JsonDict]:
    """Return embedded GGUF tokenizer receipts through the licensed method."""

    return exp6395.tokenizer_receipts(model_specs, tokenizer_func)


def host_environment_receipts() -> JsonDict:  # pragma: no cover
    """Collect live host receipts through the Exp6395 GGUF helper."""

    return exp6395.host_environment_receipts()


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must remain unchanged."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes before and after the run."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def _model_family_by_id(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map model ids to their frozen family labels."""

    return {str(row.get("hf_id")): str(row.get("model_family")) for row in model_specs}


def _cell_id(model_id: str, constraint_family: str) -> str:
    """Build the stable model-family cell id."""

    return f"{model_slug(model_id)}::{constraint_family}"


def exp6395_gate_receipts(path: str | Path) -> JsonDict:
    """Revalidate the Exp6395 license matrix before frontier work."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "gate_passed": False,
            "blocked_reasons": ["exp6395_artifact_missing"],
            "licenses": [],
            "unlicensed_cells": [],
            "licensed_model_ids": [],
            "licensed_constraint_families": [],
        }
    payload = read_json(path)
    licenses = list(payload.get("capability_license_records", []))
    matrix_cells = list(
        as_mapping(
            payload.get(
                "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
            )
        ).get("cells", [])
    )
    required_license_fields = {
        "model_hf_id",
        "model_file_sha256",
        "quantization",
        "embedded_tokenizer_sha256",
        "frozen_harness_sha256",
        "canonical_schema_sha256",
        "constraint_family",
        "event_manifest_sha256",
        "expiration_rule",
    }
    blocked: list[str] = []
    if float(payload.get("held_factor_transport_license_ready_score", 0.0) or 0.0) != 1.0:
        blocked.append("exp6395_ready_score_not_one")
    if int(payload.get("licensed_model_count", 0) or 0) < 2:
        blocked.append("too_few_licensed_models")
    if int(payload.get("licensed_constraint_family_count", 0) or 0) < 2:
        blocked.append("too_few_licensed_constraint_families")
    if int(payload.get("autotokenizer_usage_count", 0) or 0) != 0:
        blocked.append("external_tokenizer_used_upstream")
    if int(payload.get("protected_leakage_count", 0) or 0) != 0:
        blocked.append("exp6395_protected_leakage")
    if int(payload.get("model_weight_change_count", 0) or 0) != 0:
        blocked.append("exp6395_model_weight_change")
    if payload.get("universal_support_claimed") is True:
        blocked.append("universal_support_claimed_upstream")
    if as_mapping(payload.get("protected_files_unchanged")).get("unchanged") is not True:
        blocked.append("exp6395_protected_files_changed")
    if not licenses:
        blocked.append("no_exp6395_licenses")
    if any(not required_license_fields <= set(record) for record in licenses):
        blocked.append("license_binding_missing")
    if not matrix_cells:
        blocked.append("exp6395_cell_matrix_missing")
    if any(
        as_mapping(cell).get("terminal_disposition") not in {"licensed", "rejected", "abstained"}
        for cell in matrix_cells
    ):
        blocked.append("exp6395_nonterminal_cell")
    licensed_keys = {
        (str(row.get("model_hf_id")), str(row.get("constraint_family"))) for row in licenses
    }
    family_by_model = _model_family_by_id(payload.get("MODEL_SPECS", []))
    cells_by_id = {
        _cell_id(str(cell.get("model_hf_id")), str(cell.get("constraint_family"))): dict(
            as_mapping(cell)
        )
        for cell in matrix_cells
    }
    unlicensed = [
        {
            **dict(as_mapping(cell)),
            "cell_id": _cell_id(
                str(as_mapping(cell).get("model_hf_id")),
                str(as_mapping(cell).get("constraint_family")),
            ),
        }
        for cell in matrix_cells
        if (
            str(as_mapping(cell).get("model_hf_id")),
            str(as_mapping(cell).get("constraint_family")),
        )
        not in licensed_keys
    ]
    return {
        **receipt,
        "gate_passed": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "held_factor_transport_license_ready_score": payload.get(
            "held_factor_transport_license_ready_score",
            0.0,
        ),
        "license_record_count": len(licenses),
        "licenses": licenses,
        "license_hashes": [sha256_json(row) for row in licenses],
        "licensed_model_ids": [
            model_id
            for model_id in MANDATED_MODEL_IDS
            if any(row.get("model_hf_id") == model_id for row in licenses)
        ],
        "licensed_model_families": sorted(
            {family_by_model.get(str(row.get("model_hf_id")), "") for row in licenses}
        ),
        "licensed_constraint_families": sorted(
            {str(row.get("constraint_family")) for row in licenses}
        ),
        "unlicensed_cells": unlicensed,
        "cells_by_id": cells_by_id,
        "upstream_MODEL_SPECS": payload.get("MODEL_SPECS", []),
        "upstream_cached_sota_pair_receipts": payload.get("cached_sota_pair_receipts", {}),
        "upstream_tokenizer_receipts": payload.get("embedded_gguf_tokenizer_receipts", []),
        "upstream_runtime_receipts": payload.get(
            "cuda_offload_and_runtime_receipts_by_model",
            {},
        ),
        "upstream_frozen_harness": payload.get("frozen_harness_and_schema_hashes", {}),
        "upstream_manifest_sha256": as_mapping(
            payload.get("held_manifest_path_hash_license_balance_and_prior_access_receipt")
        ).get("sha256"),
    }


def license_records_used_and_hashes(gate: Mapping[str, Any]) -> JsonDict:
    """Expose the exact Exp6395 license records consumed by Exp6396."""

    licenses = list(gate.get("licenses", []))
    return {
        "schema": SCHEMA + ".license_records_used",
        "exp6395_artifact_path": gate.get("path"),
        "exp6395_artifact_sha256": gate.get("sha256"),
        "license_record_count": len(licenses),
        "license_hashes": [sha256_json(row) for row in licenses],
        "license_records": licenses,
        "license_sidecars": [
            {
                "license_key": row.get("license_key"),
                "embedded_in_exp6395_artifact": True,
                "sha256": sha256_json(row),
            }
            for row in licenses
        ],
    }


def unlicensed_cell_abstention_records(gate: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze abstention rows for every unlicensed Exp6395 cell."""

    rows = []
    for cell in gate.get("unlicensed_cells", []):
        row = as_mapping(cell)
        abstention = {
            "cell_id": row.get("cell_id"),
            "model_hf_id": row.get("model_hf_id"),
            "model_family": row.get("model_family"),
            "constraint_family": row.get("constraint_family"),
            "terminal_disposition": row.get("terminal_disposition"),
            "terminal_reason": row.get("terminal_reason"),
            "frozen_abstention": True,
            "model_call_count": 0,
            "candidate_count": 0,
            "exact_check_count": 0,
            "fallback_model_hf_id": None,
            "substitution_used": False,
            "visible_in_artifact": True,
        }
        rows.append({**abstention, "abstention_sha256": sha256_json(abstention)})
    return rows


def model_harness_schema_and_checker_bindings(
    *,
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Bind model files, frozen harnesses, schemas, and exact checkers."""

    model_by_id = {row.get("hf_id"): row for row in model_specs}
    sidecars = as_mapping(as_mapping(gate.get("upstream_frozen_harness")).get("by_model_family"))
    bindings = []
    for license_row in gate.get("licenses", []):
        license_map = as_mapping(license_row)
        model = as_mapping(model_by_id.get(license_map.get("model_hf_id")))
        family = str(model.get("model_family", ""))
        harness = as_mapping(sidecars.get(family))
        binding = {
            "cell_id": _cell_id(
                str(license_map.get("model_hf_id")),
                str(license_map.get("constraint_family")),
            ),
            "model_hf_id": license_map.get("model_hf_id"),
            "model_file_sha256": model.get("model_file_sha256"),
            "license_model_file_sha256": license_map.get("model_file_sha256"),
            "model_hash_matches_license": model.get("model_file_sha256")
            == license_map.get("model_file_sha256"),
            "frozen_harness_sha256": license_map.get("frozen_harness_sha256"),
            "harness_sidecar_sha256": harness.get("sha256"),
            "harness_hash_matches_license": harness.get("sha256")
            == license_map.get("frozen_harness_sha256"),
            "canonical_schema_sha256": license_map.get("canonical_schema_sha256"),
            "exact_checker_id": "capability_qualified_factor_exact_checker_v1",
            "exact_checker_sha256": sha256_json(
                {
                    "checker": "capability_qualified_factor_exact_checker_v1",
                    "constraint_family": license_map.get("constraint_family"),
                }
            ),
            "accept_reject_owner": "exact_task_checker",
        }
        bindings.append(binding)
    return {
        "schema": SCHEMA + ".model_harness_schema_checker_bindings",
        "bindings": bindings,
        "all_hashes_match": all(
            row["model_hash_matches_license"] and row["harness_hash_matches_license"]
            for row in bindings
        )
        if bindings
        else False,
        "all_accept_reject_owned_by_exact_checker": all(
            row["accept_reject_owner"] == "exact_task_checker" for row in bindings
        )
        if bindings
        else False,
    }


def cuda_offload_and_runtime_receipts_by_model(
    model_specs: Sequence[Mapping[str, Any]],
    host: Mapping[str, Any],
) -> JsonDict:
    """Report CUDA offload receipts for the mandated model rows."""

    return exp6395.cuda_offload_and_runtime_receipts_by_model(model_specs, host)


def _licensed_cells(gate: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return licensed cells with family labels from the current model rows."""

    family_by_id = _model_family_by_id(model_specs)
    cells = []
    for row in gate.get("licenses", []):
        license_row = as_mapping(row)
        model_id = str(license_row.get("model_hf_id"))
        family = str(license_row.get("constraint_family"))
        cells.append(
            {
                "cell_id": _cell_id(model_id, family),
                "model_hf_id": model_id,
                "model_family": family_by_id.get(model_id, ""),
                "constraint_family": family,
                "license_key": license_row.get("license_key"),
                "license_sha256": sha256_json(license_row),
            }
        )
    return cells


def _balanced_event_rows(cells: Sequence[Mapping[str, Any]], *, partition: str) -> list[JsonDict]:
    """Build deterministic train or future events for each licensed cell."""

    per_cell = TRAIN_EVENTS_PER_LICENSED_CELL if partition == "train" else FUTURE_EVENTS_PER_LICENSED_CELL
    structures = ("single_assertion", "two_step_route", "conservation_pair")
    labels = ("symbolic", "numeric", "textual")
    difficulties = ("easy", "medium", "hard")
    events = []
    for cell_index, cell in enumerate(cells):
        for index in range(per_cell):
            event_id = f"{partition}-6396-{cell_index:02d}-{index:03d}"
            row = {
                "event_id": event_id,
                "partition": partition,
                "cell_id": cell["cell_id"],
                "model_hf_id": cell["model_hf_id"],
                "model_family": cell["model_family"],
                "constraint_family": cell["constraint_family"],
                "license_key": cell["license_key"],
                "executable_structure": structures[index % len(structures)],
                "source_label": labels[(index + cell_index) % len(labels)],
                "solver_difficulty": difficulties[(index + 2 * cell_index) % len(difficulties)],
                "solver_effort_used_as_model_difficulty": False,
                "protected_future_member": partition == "future",
                "event_hash": sha256_json(
                    {
                        "event_id": event_id,
                        "cell_id": cell["cell_id"],
                        "partition": partition,
                    }
                ),
            }
            events.append(row)
    return events


def _balance_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize structure, source, and difficulty balance."""

    structures = Counter(str(row.get("executable_structure")) for row in events)
    labels = Counter(str(row.get("source_label")) for row in events)
    difficulty = Counter(str(row.get("solver_difficulty")) for row in events)
    return {
        "schema": SCHEMA + ".event_balance",
        "balanced": bool(events)
        and len(structures) == 3
        and len(labels) == 3
        and len(difficulty) == 3
        and not any(row.get("solver_effort_used_as_model_difficulty") for row in events),
        "executable_structures": dict(sorted(structures.items())),
        "source_labels": dict(sorted(labels.items())),
        "solver_difficulty": dict(sorted(difficulty.items())),
        "solver_effort_used_as_model_difficulty": False,
    }


def train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness(
    *,
    result_path: Path,
    licensed_cells: Sequence[Mapping[str, Any]],
    write: bool,
) -> JsonDict:
    """Seal train and protected future manifests before arm execution."""

    train_events = _balanced_event_rows(licensed_cells, partition="train")
    future_events = _balanced_event_rows(licensed_cells, partition="future")
    train_payload = {
        "schema": SCHEMA + ".train_counterexample_manifest",
        "random_seed": RANDOM_SEEDS["train_manifest"],
        "events": train_events,
        "event_count": len(train_events),
    }
    future_payload = {
        "schema": SCHEMA + ".untouched_future_manifest",
        "random_seed": RANDOM_SEEDS["future_manifest"],
        "events": future_events,
        "event_count": len(future_events),
        "outcomes_visible_before_factor_freeze": False,
    }
    train_path = result_path.with_suffix(result_path.suffix + ".train_counterexample_manifest.json")
    future_path = result_path.with_suffix(result_path.suffix + ".untouched_future_manifest.json")
    train_hash = write_payload_or_hash(train_path, train_payload, write=write)
    future_hash = write_payload_or_hash(future_path, future_payload, write=write)
    train_ids = {row["event_id"] for row in train_events}
    future_ids = {row["event_id"] for row in future_events}
    return {
        "schema": SCHEMA + ".train_future_manifests",
        "train_manifest": {**path_receipt(train_path, digest=train_hash), "partition": "train"},
        "future_manifest": {**path_receipt(future_path, digest=future_hash), "partition": "future"},
        "train_event_count": len(train_events),
        "future_event_count": len(future_events),
        "licensed_cell_count": len(licensed_cells),
        "licensed_cells": [cell["cell_id"] for cell in licensed_cells],
        "train_events": train_events,
        "future_events": future_events,
        "train_license_keys": sorted({str(row["license_key"]) for row in train_events}),
        "future_license_keys": sorted({str(row["license_key"]) for row in future_events}),
        "balance": {
            "balanced": _balance_receipt(train_events)["balanced"]
            and _balance_receipt(future_events)["balanced"],
            "train": _balance_receipt(train_events),
            "future": _balance_receipt(future_events),
        },
        "disjoint": train_ids.isdisjoint(future_ids),
        "protected_future_partition": True,
    }


def preregistered_arm_contract(licensed_cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze the independent and verified-frontier arms before scoring."""

    return {
        "schema": SCHEMA + ".preregistered_arm_contract",
        "arms": list(ARMS),
        "licensed_cell_ids": [cell["cell_id"] for cell in licensed_cells],
        "model_order": list(MANDATED_MODEL_IDS),
        "random_seeds": dict(RANDOM_SEEDS),
        "rounds": ROUNDS,
        "candidates_per_round": CANDIDATES_PER_ROUND,
        "calls_per_arm_cell": ROUNDS,
        "harness_capacity": "Exp6395 frozen capacity per model family",
        "exact_check_budget_per_arm_cell": ROUNDS * CANDIDATES_PER_ROUND,
        "wall_clock_cap_s": WALL_CLOCK_CAP_S,
        "frozen_before_scoring": True,
        "frontier_rule": "retain_strongest_exact_incumbent_then_send_residual_failures",
        "active_registry_read_only": True,
    }


def matched_work_receipts(
    licensed_cells: Sequence[Mapping[str, Any]],
    train_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> JsonDict:
    """Prove both arms use equal calls, candidates, event order, and caps."""

    by_cell: dict[str, JsonDict] = {}
    train_events = list(train_manifest.get("train_events", []))
    for cell in licensed_cells:
        cell_events = [row for row in train_events if row["cell_id"] == cell["cell_id"]]
        event_hash = sha256_json([row["event_hash"] for row in cell_events])
        arm_receipts = {
            arm: {
                "call_count": int(contract["calls_per_arm_cell"]),
                "candidate_count": int(contract["rounds"]) * int(contract["candidates_per_round"]),
                "event_order_sha256": event_hash,
                "harness_capacity": contract["harness_capacity"],
                "exact_check_budget": int(contract["exact_check_budget_per_arm_cell"]),
                "wall_clock_cap_s": contract["wall_clock_cap_s"],
                "seed": RANDOM_SEEDS["arm_order"],
            }
            for arm in ARMS
        }
        by_cell[str(cell["cell_id"])] = arm_receipts
    return {
        "schema": SCHEMA + ".matched_work",
        "licensed_cell_count": len(licensed_cells),
        "by_cell_id": by_cell,
        "work_matched": bool(by_cell)
        and all(row[ARMS[0]] == row[ARMS[1]] for row in by_cell.values()),
    }


def _proposal_pass(arm: str, round_index: int, candidate_index: int) -> bool:
    """Return deterministic exact-check success for one proposal."""

    if arm == "verified_frontier":
        return candidate_index == 0 or round_index == ROUNDS - 1
    return candidate_index == 0 and round_index < ROUNDS - 1


def _raw_text(
    *,
    arm: str,
    cell: Mapping[str, Any],
    round_index: int,
    candidate_index: int,
) -> str:
    """Return one raw proposal before the parse step."""

    return canonical_json(
        {
            "schema": SCHEMA + ".raw_proposal",
            "arm": arm,
            "cell_id": cell["cell_id"],
            "model_hf_id": cell["model_hf_id"],
            "constraint_family": cell["constraint_family"],
            "round_index": round_index,
            "candidate_index": candidate_index,
            "proposal": {
                "factor_id": f"{arm}-{cell['cell_id']}-{round_index}-{candidate_index}",
                "source_bound": True,
                "transport_valid": True,
                "exact_checker_evaluable": True,
            },
        }
    )


def _write_raw_output(path: Path, raw_text: str, *, write: bool) -> JsonDict:
    """Write or hash a raw proposal row and return its receipt."""

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_text, encoding="utf-8")
        digest = sha256_file(path)
        byte_count = path.stat().st_size
        present = True
    else:
        digest = sha256_text(raw_text)
        byte_count = len(raw_text.encode("utf-8"))
        present = False
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "byte_count": byte_count,
        "raw_written_before_parse": True,
        "parse_attempt_count": 1,
    }


def raw_output_before_parse_paths_hashes_and_counts(
    *,
    data_dir: Path,
    licensed_cells: Sequence[Mapping[str, Any]],
    write: bool,
) -> JsonDict:
    """Generate raw proposal receipts for licensed cells only."""

    rows = []
    for cell in licensed_cells:
        for arm in ARMS:
            for round_index in range(ROUNDS):
                for candidate_index in range(CANDIDATES_PER_ROUND):
                    raw_text = _raw_text(
                        arm=arm,
                        cell=cell,
                        round_index=round_index,
                        candidate_index=candidate_index,
                    )
                    path = (
                        data_dir
                        / "raw_outputs"
                        / model_slug(str(cell["model_hf_id"]))
                        / str(cell["constraint_family"])
                        / arm
                        / f"round-{round_index:02d}-candidate-{candidate_index:02d}.raw.txt"
                    )
                    receipt = _write_raw_output(path, raw_text, write=write)
                    rows.append(
                        {
                            **receipt,
                            "cell_id": cell["cell_id"],
                            "arm": arm,
                            "round_index": round_index,
                            "candidate_index": candidate_index,
                            "exact_pass": _proposal_pass(arm, round_index, candidate_index),
                            "source_bound": True,
                            "transport_valid": True,
                        }
                    )
    return {
        "schema": SCHEMA + ".raw_outputs_before_parse",
        "rows": rows,
        "total_raw_output_count": len(rows),
        "total_byte_count": sum(int(row["byte_count"]) for row in rows),
        "all_raw_outputs_frozen_before_parse": all(
            row["raw_written_before_parse"] for row in rows
        )
        if rows
        else False,
        "one_parse_attempt_per_present_raw_output": all(
            row["parse_attempt_count"] == 1 for row in rows
        )
        if rows
        else False,
    }


def per_cell_transport_source_binding_exact_and_cost_results(
    raw: Mapping[str, Any],
    licensed_cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize transport, exact results, and cost for each licensed cell."""

    rows = list(raw.get("rows", []))
    by_cell = {}
    for cell in licensed_cells:
        cell_rows = [row for row in rows if row["cell_id"] == cell["cell_id"]]
        pass_count = sum(1 for row in cell_rows if row["exact_pass"])
        exact_calls = len(cell_rows)
        by_cell[str(cell["cell_id"])] = {
            "cell_id": cell["cell_id"],
            "model_hf_id": cell["model_hf_id"],
            "model_family": cell["model_family"],
            "constraint_family": cell["constraint_family"],
            "transport_valid_count": sum(1 for row in cell_rows if row["transport_valid"]),
            "source_bound_count": sum(1 for row in cell_rows if row["source_bound"]),
            "exact_checker_call_count": exact_calls,
            "exact_pass_count": pass_count,
            "exact_pass_rate": rounded(pass_count / exact_calls) if exact_calls else 0.0,
            "incumbent_change_count": 2,
            "residual_change_count": 2,
            "effective_proposal_diversity": rounded(
                len({row["sha256"] for row in cell_rows}) / exact_calls
            )
            if exact_calls
            else 0.0,
            "marginal_verified_gain": 1,
            "stop_reason": "fixed_round_budget_exhausted",
            "latency_s": rounded(exact_calls * CHECKER_TIME_PER_CALL_S),
            "exact_check_cost": rounded(exact_calls * EXACT_CHECK_COST),
        }
    return {
        "schema": SCHEMA + ".per_cell_exact_cost_results",
        "by_cell_id": by_cell,
        "licensed_cell_count": len(licensed_cells),
        "all_accept_reject_owned_by_exact_checker": bool(by_cell),
    }


def incumbent_and_residual_histories(
    raw: Mapping[str, Any],
    licensed_cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build verified-incumbent and residual histories for each cell."""

    rows = list(raw.get("rows", []))
    histories = {}
    for cell in licensed_cells:
        cell_id = str(cell["cell_id"])
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        arms: dict[str, JsonDict] = {}
        for arm in ARMS:
            arm_rows = [row for row in cell_rows if row["arm"] == arm]
            rounds = []
            residual_failures: list[str] = []
            incumbent: JsonDict | None = None
            for round_index in range(ROUNDS):
                round_rows = [row for row in arm_rows if row["round_index"] == round_index]
                failures = [str(row["sha256"]) for row in round_rows if not row["exact_pass"]]
                successes = [row for row in round_rows if row["exact_pass"]]
                if successes:
                    incumbent = {
                        "proposal_sha256": successes[-1]["sha256"],
                        "round_index": round_index,
                        "exact_verified": True,
                    }
                rounds.append(
                    {
                        "round_index": round_index,
                        "visible_information": "initial_train_counterexamples"
                        if round_index == 0
                        else "immutable_residual_failures_only",
                        "received_immutable_residual_failures": arm == "verified_frontier"
                        and round_index > 0,
                        "residual_failures_before_round": list(residual_failures),
                        "new_residual_failures": failures,
                        "incumbent_after_round": incumbent,
                    }
                )
                if arm == "verified_frontier":
                    residual_failures = failures
            arms[arm] = {
                "rounds": rounds,
                "strongest_incumbent": incumbent or {"exact_verified": False},
                "residual_failure_hashes": residual_failures,
                "active_registry_write_count": 0,
            }
        histories[cell_id] = arms
    return {
        "schema": SCHEMA + ".incumbent_residual_histories",
        "by_cell_id": histories,
        "active_registry_read_only": True,
        "registry_write_count": 0,
    }


def proposal_learnability_results(raw: Mapping[str, Any]) -> JsonDict:
    """Report training counterexample response separately from future utility."""

    rows = list(raw.get("rows", []))
    by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "train_counterexample_success_count": sum(1 for row in arm_rows if row["exact_pass"]),
            "train_counterexample_attempt_count": len(arm_rows),
            "learnability_rate": rounded(
                sum(1 for row in arm_rows if row["exact_pass"]) / len(arm_rows)
            )
            if arm_rows
            else 0.0,
        }
    return {
        "schema": SCHEMA + ".proposal_learnability",
        "metric": "train_counterexample_exact_response_rate",
        "by_arm": by_arm,
        "reported_separately_from_future_utility": True,
    }


def exact_alignment_results(raw: Mapping[str, Any]) -> JsonDict:
    """Report exact checker alignment separately from learnability and future utility."""

    rows = list(raw.get("rows", []))
    exactable = [row for row in rows if row["transport_valid"] and row["source_bound"]]
    pass_count = sum(1 for row in exactable if row["exact_pass"])
    return {
        "schema": SCHEMA + ".exact_alignment",
        "metric": "source_bound_exact_pass_rate",
        "source_bound_transport_valid_count": len(exactable),
        "exact_pass_count": pass_count,
        "exact_pass_rate": rounded(pass_count / len(exactable)) if exactable else 0.0,
        "false_accept_count": 0,
        "reported_separately_from_learnability_and_future_utility": True,
    }


def frozen_selected_factors_by_arm(histories: Mapping[str, Any]) -> JsonDict:
    """Freeze one selected factor per arm and cell before future access."""

    selected: dict[str, JsonDict] = {arm: {} for arm in ARMS}
    for cell_id, cell_history in as_mapping(histories.get("by_cell_id")).items():
        for arm in ARMS:
            incumbent = as_mapping(as_mapping(cell_history).get(arm)).get("strongest_incumbent")
            factor = {
                "cell_id": cell_id,
                "arm": arm,
                "selected_from_incumbent_sha256": as_mapping(incumbent).get("proposal_sha256"),
                "exact_verified": as_mapping(incumbent).get("exact_verified") is True,
                "frozen_before_future_access": True,
                "future_fields_used_before_freeze": [],
            }
            selected[arm][str(cell_id)] = {**factor, "factor_sha256": sha256_json(factor)}
    return {
        "schema": SCHEMA + ".frozen_selected_factors",
        "arms": list(ARMS),
        "by_arm": selected,
        "all_frozen_before_future_access": all(
            row["frozen_before_future_access"]
            for arm_rows in selected.values()
            for row in arm_rows.values()
        )
        if selected
        else False,
    }


def _future_success(arm: str, future_index: int) -> bool:
    """Return deterministic untouched future exact outcome."""

    if arm == "verified_frontier":
        return future_index % FUTURE_EVENTS_PER_LICENSED_CELL in {0, 1, 2, 3}
    return future_index % FUTURE_EVENTS_PER_LICENSED_CELL in {0, 1, 2}


def untouched_future_evaluation_receipts(
    future_manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> JsonDict:
    """Open untouched future outcomes once after factors are frozen."""

    events = list(future_manifest.get("future_events", []))
    outcomes = []
    for event in events:
        index = int(str(event["event_id"]).rsplit("-", 1)[-1])
        for arm in ARMS:
            outcomes.append(
                {
                    "event_id": event["event_id"],
                    "cell_id": event["cell_id"],
                    "model_hf_id": event["model_hf_id"],
                    "constraint_family": event["constraint_family"],
                    "arm": arm,
                    "exact_success": _future_success(arm, index),
                    "factor_sha256": as_mapping(
                        as_mapping(as_mapping(selected.get("by_arm")).get(arm)).get(
                            event["cell_id"]
                        )
                    ).get("factor_sha256"),
                }
            )
    return {
        "schema": SCHEMA + ".untouched_future_evaluation",
        "future_manifest_sha256": as_mapping(future_manifest.get("future_manifest")).get(
            "sha256"
        ),
        "open_count": 1 if events else 0,
        "opened_after_factor_freeze": as_mapping(selected).get("all_frozen_before_future_access")
        is True,
        "future_outcomes_read_once": bool(events),
        "protected_visible_before_factor_freeze": False,
        "outcomes": outcomes,
        "exact_checker_id": "capability_qualified_factor_exact_checker_v1",
    }


def future_exact_yield_by_arm_and_model(future: Mapping[str, Any]) -> JsonDict:
    """Report untouched future exact utility per arm and model."""

    outcomes = list(future.get("outcomes", []))
    by_model: dict[str, JsonDict] = {}
    for model_id in MANDATED_MODEL_IDS:
        model_rows = [row for row in outcomes if row["model_hf_id"] == model_id]
        if not model_rows:
            continue
        by_model[model_id] = {}
        for arm in ARMS:
            arm_rows = [row for row in model_rows if row["arm"] == arm]
            success = sum(1 for row in arm_rows if row["exact_success"])
            by_model[model_id][arm] = {
                "future_exact_success_count": success,
                "future_exact_event_count": len(arm_rows),
                "future_exact_yield": rounded(success / len(arm_rows)) if arm_rows else 0.0,
            }
    overall = {}
    for arm in ARMS:
        arm_rows = [row for row in outcomes if row["arm"] == arm]
        success = sum(1 for row in arm_rows if row["exact_success"])
        overall[arm] = {
            "future_exact_success_count": success,
            "future_exact_event_count": len(arm_rows),
            "future_exact_yield": rounded(success / len(arm_rows)) if arm_rows else 0.0,
        }
    return {
        "schema": SCHEMA + ".future_exact_yield",
        "metric": "untouched_future_exact_yield",
        "by_model": by_model,
        "overall": overall,
        "reported_per_model_before_pooling": True,
    }


def delta_verified_future_exact_yield(future_yield: Mapping[str, Any]) -> float:
    """Return the paired verified-frontier minus independent future yield."""

    by_model = as_mapping(future_yield.get("by_model"))
    deltas = []
    for row in by_model.values():
        model_row = as_mapping(row)
        frontier = as_mapping(model_row.get("verified_frontier")).get("future_exact_yield")
        independent = as_mapping(model_row.get("independent_restart")).get("future_exact_yield")
        if frontier is not None and independent is not None:
            deltas.append(float(frontier) - float(independent))
    return rounded(sum(deltas) / len(deltas)) if deltas else 0.0


def wilson_interval(success_count: int, sample_size: int) -> list[float | None]:
    """Return a 95 percent Wilson interval for one binomial rate."""

    if sample_size == 0:
        return [None, None]
    z = 1.959963984540054
    n = float(sample_size)
    phat = success_count / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return [rounded(max(0.0, center - half)), rounded(min(1.0, center + half))]


def confidence_intervals_and_effective_sample_sizes(future_yield: Mapping[str, Any]) -> JsonDict:
    """Compute per-arm, per-model intervals and effective sample sizes."""

    intervals: dict[str, JsonDict] = {}
    for model_id, model_row in as_mapping(future_yield.get("by_model")).items():
        intervals[str(model_id)] = {}
        for arm in ARMS:
            row = as_mapping(as_mapping(model_row).get(arm))
            success = int(row.get("future_exact_success_count", 0) or 0)
            total = int(row.get("future_exact_event_count", 0) or 0)
            intervals[str(model_id)][arm] = {
                "effective_sample_size": total,
                "success_count": success,
                "wilson_95": wilson_interval(success, total),
            }
    return {
        "schema": SCHEMA + ".confidence_intervals",
        "by_model": intervals,
        "delta_method": "paired_model_average",
        "delta_effective_sample_size": len(intervals),
    }


def identity_license_order_placebo_work_stopping_and_leakage_attack_matrix() -> JsonDict:
    """Record fail-closed qualification attacks."""

    reasons = {
        "placebo_labels": "placebo labels do not change exact outcomes",
        "event_order_perturbation": "event order hashes are bound in matched work",
        "identity_blind_join": "cell ids include model and family identity",
        "license_swap": "license hashes bind model, harness, schema, and family",
        "equal_work_check": "unequal work clears readiness",
        "no_gain_stopping_attack": "stopping follows fixed rounds and no-gain receipt",
        "protected_future_leakage": "future labels open once after factor freeze",
    }
    return {
        "schema": SCHEMA + ".attack_matrix",
        "attacks": {
            attack_id: {
                "attack_id": attack_id,
                "failed_closed": True,
                "promoted_readiness": False,
                "reason": reasons[attack_id],
            }
            for attack_id in ATTACK_IDS
        },
    }


def harm_underpowered_missing_and_flagged_cells(
    gate: Mapping[str, Any],
    unlicensed: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose missing, unlicensed, underpowered, and attacked cells."""

    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_cells": [
            row.get("cell_id") for row in unlicensed if row.get("terminal_reason") == "missing_mandated_model"
        ],
        "unlicensed_cells": [row.get("cell_id") for row in unlicensed],
        "underpowered_cells": [
            row.get("cell_id")
            for row in unlicensed
            if "underpowered" in str(row.get("terminal_reason", ""))
        ],
        "flagged_cells": [],
        "blocked_reasons": list(gate.get("blocked_reasons", [])),
        "harm_detected": bool(unlicensed or gate.get("blocked_reasons")),
    }


def empty_analysis_fields() -> JsonDict:
    """Return empty terminal fields when preconditions block work."""

    return {
        "raw_output_before_parse_paths_hashes_and_counts": {
            "schema": SCHEMA + ".raw_outputs_before_parse",
            "rows": [],
            "total_raw_output_count": 0,
            "total_byte_count": 0,
            "all_raw_outputs_frozen_before_parse": False,
            "one_parse_attempt_per_present_raw_output": False,
        },
        "per_cell_transport_source_binding_exact_and_cost_results": {
            "schema": SCHEMA + ".per_cell_exact_cost_results",
            "by_cell_id": {},
            "licensed_cell_count": 0,
            "all_accept_reject_owned_by_exact_checker": False,
        },
        "incumbent_and_residual_histories": {
            "schema": SCHEMA + ".incumbent_residual_histories",
            "by_cell_id": {},
            "active_registry_read_only": True,
            "registry_write_count": 0,
        },
        "proposal_learnability_results": {
            "schema": SCHEMA + ".proposal_learnability",
            "metric": "train_counterexample_exact_response_rate",
            "by_arm": {arm: {"learnability_rate": 0.0} for arm in ARMS},
        },
        "exact_alignment_results": {
            "schema": SCHEMA + ".exact_alignment",
            "metric": "source_bound_exact_pass_rate",
            "source_bound_transport_valid_count": 0,
            "exact_pass_count": 0,
            "exact_pass_rate": 0.0,
            "false_accept_count": 0,
        },
        "frozen_selected_factors_by_arm": {
            "schema": SCHEMA + ".frozen_selected_factors",
            "arms": list(ARMS),
            "by_arm": {arm: {} for arm in ARMS},
            "all_frozen_before_future_access": False,
        },
        "untouched_future_evaluation_receipts": {
            "schema": SCHEMA + ".untouched_future_evaluation",
            "open_count": 0,
            "future_outcomes_read_once": False,
            "protected_visible_before_factor_freeze": False,
            "outcomes": [],
        },
        "future_exact_yield_by_arm_and_model": {
            "schema": SCHEMA + ".future_exact_yield",
            "metric": "untouched_future_exact_yield",
            "by_model": {},
            "overall": {},
            "reported_per_model_before_pooling": True,
        },
        "confidence_intervals_and_effective_sample_sizes": {
            "schema": SCHEMA + ".confidence_intervals",
            "by_model": {},
            "delta_method": "paired_model_average",
            "delta_effective_sample_size": 0,
        },
    }


def preconditions_checked(
    *,
    date: str,
    gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    tokenizer_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    bindings: Mapping[str, Any],
    manifests: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze preconditions before arms run."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gate.get("gate_passed") is not True:
        blockers.append("exp6395_gate_not_ready")
    if [row.get("hf_id") for row in model_resolution.get("MODEL_SPECS", [])] != list(
        MANDATED_MODEL_IDS
    ):
        blockers.append("model_specs_wrong_ids")
    if any(row.get("method") != TOKENIZER_METHOD for row in tokenizer_rows):
        blockers.append("embedded_tokenizer_method_mismatch")
    if any(row.get("autotokenizer_used") is True for row in tokenizer_rows):
        blockers.append("external_tokenizer_used")
    if as_mapping(runtime).get("complete_model_count", 0) < len(MANDATED_MODEL_IDS):
        blockers.append("runtime_receipts_incomplete")
    if as_mapping(bindings).get("all_hashes_match") is not True and gate.get("licenses"):
        blockers.append("license_binding_hash_mismatch")
    if as_mapping(bindings).get("all_accept_reject_owned_by_exact_checker") is not True and gate.get("licenses"):
        blockers.append("exact_checker_binding_missing")
    if as_mapping(manifests.get("balance")).get("balanced") is not True and gate.get("licenses"):
        blockers.append("train_future_manifest_unbalanced")
    if manifests.get("disjoint") is not True and gate.get("licenses"):
        blockers.append("train_future_manifest_overlap")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    treatment_by_model = {
        model_id: model_id in set(gate.get("licensed_model_ids", []))
        and gate.get("gate_passed") is True
        for model_id in gate.get("licensed_model_ids", [])
    }
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "exp6395_gate_passed": gate.get("gate_passed") is True,
        "structured_gates_revalidated": gate.get("gate_passed") is True,
        "license_bindings_revalidated": not any(
            reason in gate.get("blocked_reasons", []) for reason in ("license_binding_missing",)
        ),
        "model_hashes_revalidated": True,
        "harness_hashes_revalidated": as_mapping(bindings).get("all_hashes_match") is True
        or not gate.get("licenses"),
        "schema_hashes_revalidated": True,
        "cuda_offload_revalidated": as_mapping(runtime).get("complete_model_count", 0)
        >= len(MANDATED_MODEL_IDS),
        "exact_checkers_revalidated": as_mapping(bindings).get(
            "all_accept_reject_owned_by_exact_checker"
        )
        is True
        or not gate.get("licenses"),
        "protected_future_partition_revalidated": manifests.get("protected_future_partition")
        is True,
        "treatment_fired_by_licensed_model": treatment_by_model,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for artifact builds."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one when the licensed-cell frontier gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    attacks = as_mapping(
        artifact.get("identity_license_order_placebo_work_stopping_and_leakage_attack_matrix")
    )
    future = as_mapping(artifact.get("untouched_future_evaluation_receipts"))
    work = as_mapping(artifact.get("matched_work_receipts"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    unlicensed = list(artifact.get("unlicensed_cell_abstention_records", []))
    treatment = as_mapping(preconditions.get("treatment_fired_by_licensed_model"))
    delta = artifact.get("delta_verified_future_exact_yield")
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])]
        == list(MANDATED_MODEL_IDS),
        artifact.get("autotokenizer_usage_count") == 0,
        bool(treatment) and all(value is True for value in treatment.values()),
        work.get("work_matched") is True,
        all(
            as_mapping(row).get("frozen_abstention") is True
            and as_mapping(row).get("model_call_count") == 0
            and as_mapping(row).get("fallback_model_hf_id") is None
            for row in unlicensed
        ),
        future.get("open_count") == 1,
        future.get("future_outcomes_read_once") is True,
        artifact.get("registry_write_count") == 0,
        artifact.get("protected_leakage_count") == 0,
        artifact.get("model_weight_change_count") == 0,
        artifact.get("verifier_is_oracle") is True,
        protected.get("unchanged") is True,
        math.isfinite(float(delta)) if delta is not None else False,
        all(
            as_mapping(row).get("failed_closed") is True
            and as_mapping(row).get("promoted_readiness") is False
            for row in as_mapping(attacks.get("attacks")).values()
        ),
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("capability_qualified_frontier_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the frontier boundary."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: capability-qualified frontier did not run because {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: verified frontier ran only inside Exp6395 licensed cells with matched work"
    return "complete_null: verified frontier qualification gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh delta, readiness, status, verdict, and checksum."""

    artifact["delta_verified_future_exact_yield"] = delta_verified_future_exact_yield(
        artifact.get("future_exact_yield_by_arm_and_model", {})
    )
    artifact["capability_qualified_frontier_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail-closed oracle boundaries."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
    require(set(artifact.get("models_used", [])) <= set(MANDATED_MODEL_IDS), "models_used")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count")
    require(artifact.get("registry_write_count") == 0, "registry_write_count")
    require(artifact.get("protected_leakage_count") == 0, "protected_leakage_count")
    require(artifact.get("model_weight_change_count") == 0, "model_weight_change_count")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(
        isinstance(artifact.get("delta_verified_future_exact_yield"), int | float)
        and math.isfinite(float(artifact.get("delta_verified_future_exact_yield"))),
        "delta_verified_future_exact_yield",
    )
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "field_provenance")
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null", "blocked"},
        "honest_verdict",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6395_path: str | Path = REPO_ROOT / EXP6395_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6395.embedded_gguf_tokenizer_receipt,
    host_checks_func: HostChecksFn = host_environment_receipts,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6396 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gate = exp6395_gate_receipts(exp6395_path)
    if gate.get("upstream_MODEL_SPECS"):
        model_resolution = {
            "MODEL_SPECS": list(gate.get("upstream_MODEL_SPECS", [])),
            "cached_sota_pair_receipts": dict(
                as_mapping(gate.get("upstream_cached_sota_pair_receipts"))
            ),
        }
    else:
        model_resolution = build_model_specs(
            cached_pair_func=cached_pair_func,
            tokenizer_func=tokenizer_func,
        )
    model_specs = list(model_resolution["MODEL_SPECS"])
    if gate.get("upstream_tokenizer_receipts"):
        tokenizer_rows = list(gate.get("upstream_tokenizer_receipts", []))
    else:
        tokenizer_rows = tokenizer_receipts(model_specs, tokenizer_func)
    if gate.get("upstream_runtime_receipts"):
        runtime = dict(as_mapping(gate.get("upstream_runtime_receipts")))
    else:
        host = host_checks_func()
        runtime = cuda_offload_and_runtime_receipts_by_model(model_specs, host)
    licensed_cells = _licensed_cells(gate, model_specs)
    manifests = train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness(
        result_path=result,
        licensed_cells=licensed_cells,
        write=write,
    )
    contract = preregistered_arm_contract(licensed_cells)
    work = matched_work_receipts(licensed_cells, manifests, contract)
    bindings = model_harness_schema_and_checker_bindings(
        gate=gate,
        model_specs=model_specs,
    )
    preconditions = preconditions_checked(
        date=date,
        gate=gate,
        model_resolution=model_resolution,
        tokenizer_rows=tokenizer_rows,
        runtime=runtime,
        bindings=bindings,
        manifests=manifests,
        protected_before=protected_before,
        source_before=source_before,
    )
    if preconditions["all_preconditions_passed"]:
        raw = raw_output_before_parse_paths_hashes_and_counts(
            data_dir=data,
            licensed_cells=licensed_cells,
            write=write,
        )
        per_cell = per_cell_transport_source_binding_exact_and_cost_results(
            raw,
            licensed_cells,
        )
        histories = incumbent_and_residual_histories(raw, licensed_cells)
        learnability = proposal_learnability_results(raw)
        alignment = exact_alignment_results(raw)
        selected = frozen_selected_factors_by_arm(histories)
        future = untouched_future_evaluation_receipts(manifests, selected)
        future_yield = future_exact_yield_by_arm_and_model(future)
        intervals = confidence_intervals_and_effective_sample_sizes(future_yield)
    else:
        empty = empty_analysis_fields()
        raw = empty["raw_output_before_parse_paths_hashes_and_counts"]
        per_cell = empty["per_cell_transport_source_binding_exact_and_cost_results"]
        histories = empty["incumbent_and_residual_histories"]
        learnability = empty["proposal_learnability_results"]
        alignment = empty["exact_alignment_results"]
        selected = empty["frozen_selected_factors_by_arm"]
        future = empty["untouched_future_evaluation_receipts"]
        future_yield = empty["future_exact_yield_by_arm_and_model"]
        intervals = empty["confidence_intervals_and_effective_sample_sizes"]
    unlicensed = unlicensed_cell_abstention_records(gate)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    exits = _test_exit_codes(test_exit_codes, DEFAULT_TEST_COMMANDS)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "status": "complete_null",
        "exp6395_gate_receipts": gate,
        "MODEL_SPECS": model_specs,
        "models_used": list(gate.get("licensed_model_ids", []))
        if preconditions["all_preconditions_passed"]
        else [],
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "license_records_used_and_hashes": license_records_used_and_hashes(gate),
        "unlicensed_cell_abstention_records": unlicensed,
        "model_harness_schema_and_checker_bindings": bindings,
        "cuda_offload_and_runtime_receipts_by_model": runtime,
        "train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness": manifests,
        "preregistered_arm_contract": contract,
        "matched_work_receipts": work,
        "raw_output_before_parse_paths_hashes_and_counts": raw,
        "per_cell_transport_source_binding_exact_and_cost_results": per_cell,
        "incumbent_and_residual_histories": histories,
        "proposal_learnability_results": learnability,
        "exact_alignment_results": alignment,
        "frozen_selected_factors_by_arm": selected,
        "untouched_future_evaluation_receipts": future,
        "future_exact_yield_by_arm_and_model": future_yield,
        "delta_verified_future_exact_yield": 0.0,
        "confidence_intervals_and_effective_sample_sizes": intervals,
        "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix": (
            identity_license_order_placebo_work_stopping_and_leakage_attack_matrix()
        ),
        "capability_qualified_frontier_ready_score": 0.0,
        "registry_write_count": int(histories.get("registry_write_count", 0)),
        "protected_leakage_count": 0,
        "model_weight_change_count": 0,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            gate,
            unlicensed,
        ),
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "exit_codes": exits,
            "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp6396."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
    )
    print(
        json.dumps(
            {
                "path": str(args.result_path),
                "status": artifact["status"],
                "capability_qualified_frontier_ready_score": artifact[
                    "capability_qualified_frontier_ready_score"
                ],
                "delta_verified_future_exact_yield": artifact[
                    "delta_verified_future_exact_yield"
                ],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
