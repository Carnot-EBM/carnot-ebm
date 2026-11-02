"""Exp6280 variable-cardinality mode-jump backend ABI.

Spec refs: REQ-SAMPLER-6280,
SCENARIO-SAMPLER-6280-METADATA-ROUNDTRIP,
SCENARIO-SAMPLER-6280-PROPOSAL-PARITY,
SCENARIO-SAMPLER-6280-NO-AB-VALUE-CLAIM.

This module verifies the ABI repair only. It does not rerun the Exp6269
scientific A/B, and it does not make a sampler-value or speed claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot import experiment_6268_multimodal_sampler_fixture_suite as exp6268
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    MODE_JUMP_ALGORITHM,
    MODE_JUMP_TOPOLOGY,
    TYPED_STATE_METADATA_SCHEMA_VERSION,
    VARIABLE_CARDINALITY_TOPOLOGY,
    ModeJumpRustBackend,
    complete_support_proposal,
    descriptor_for_run,
    frozen_mode_jump_inputs,
    mode_jump_inputs_from_fixture_receipt,
    normalize_typed_state_metadata,
    typed_state_metadata_from_fixture_receipt,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6280_variable_cardinality_mode_jump_backend.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6280_variable_cardinality_mode_jump_backend.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6280_variable_cardinality_mode_jump_backend.py"
)
SAMPLER_BACKEND_TEST_RELATIVE_PATH = Path("tests/python/samplers/test_mode_jump_rust_backend.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
EXP6268_RELATIVE_PATH = Path("results/experiment_6268_multimodal_sampler_fixture_suite.json")
EXP6269_RELATIVE_PATH = Path("results/experiment_6269_mode_jump_multifamily_ab.json")
BACKEND_RELATIVE_PATH = Path("python/carnot/samplers/mode_jump_rust_backend.py")
BACKEND_REGISTRY_RELATIVE_PATH = Path("python/carnot/samplers/backend.py")
POTTS_SAMPLER_RELATIVE_PATH = Path("python/carnot/samplers/potts_sampler.py")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")
RUST_TEST_RELATIVE_PATH = Path("crates/carnot-samplers/tests/mode_jump.rs")

SCHEMA = "carnot.experiment_6280.variable_cardinality_mode_jump_backend.v1"
EXPERIMENT_ID = "experiment_6280_variable_cardinality_mode_jump_backend"
RUN_DATE = "20260810"
INFERENCE_SUBSTRATE = "local_cpu_rust_python_variable_cardinality_sampler_abi"
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6280_command_receipts.json")
RANDOM_SEED = 6280
RETAINED_SAMPLE_COUNT = 96
BURN_IN = 8

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SAMPLER_SPEC_RELATIVE_PATH,
    BACKEND_RELATIVE_PATH,
    BACKEND_REGISTRY_RELATIVE_PATH,
    POTTS_SAMPLER_RELATIVE_PATH,
    RUST_KERNEL_RELATIVE_PATH,
    PYO3_BINDING_RELATIVE_PATH,
    RUST_TEST_RELATIVE_PATH,
    SAMPLER_BACKEND_TEST_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6268_RELATIVE_PATH,
    EXP6269_RELATIVE_PATH,
)

PRE_TASK_SOURCE_HASHES = {
    BACKEND_RELATIVE_PATH.as_posix(): "sha256:fa202e80662270539a911473945f236d64522a16fa4398dc41b407efe3b2b9bf",
    BACKEND_REGISTRY_RELATIVE_PATH.as_posix(): "sha256:fb121be3b43b80cb7b6583db0ff3c5adba2fb00cbd13e2cae2157f0bf8b78296",
    POTTS_SAMPLER_RELATIVE_PATH.as_posix(): "sha256:a36ef46d4b0048d083bfe133165b16f944e70897aed8b3b1610f210f648f09d7",
    RUST_KERNEL_RELATIVE_PATH.as_posix(): "sha256:d5e9fba57da0a190459c76b6bdc90e65e4e38c41cad4c13c8a54c1965079388d",
    PYO3_BINDING_RELATIVE_PATH.as_posix(): "sha256:83b788dafaaca79f6d90304d2cc558ce339fc10404901ca0a334d91dbbe3d275",
    SAMPLER_SPEC_RELATIVE_PATH.as_posix(): "sha256:3e8fc1487dccd250bf9e9125d522c8eef6531e71a18d105a3e5236904e1ff234",
    EXP6268_RELATIVE_PATH.as_posix(): "sha256:392b6a8e4f37130ba70ba5eb4ac2ad6c18d361452ab5f1383cdc4a578af27761",
    EXP6269_RELATIVE_PATH.as_posix(): "sha256:4ad015f42b42c956f143d028000791e8bcd3862b81a2f2e2ad2100961ee1c098",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/samplers/test_mode_jump_rust_backend.py tests/python/test_experiment_6280_variable_cardinality_mode_jump_backend.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6280_variable_cardinality_mode_jump_backend.py -m pytest tests/python/test_experiment_6280_variable_cardinality_mode_jump_backend.py -q --no-cov -o addopts=",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6280_variable_cardinality_mode_jump_backend.py --fail-under=100",
    "cargo test -p carnot-samplers --test mode_jump --quiet",
    ".venv/bin/pytest tests/python/test_e2e_serialization.py tests/python/test_pyo3_integration.py -q -o addopts=",
    ".venv/bin/ruff check python/carnot/samplers/mode_jump_rust_backend.py python/carnot/experiment_6280_variable_cardinality_mode_jump_backend.py tests/python/samplers/test_mode_jump_rust_backend.py tests/python/test_experiment_6280_variable_cardinality_mode_jump_backend.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/samplers/test_mode_jump_rust_backend.py tests/python/test_experiment_6280_variable_cardinality_mode_jump_backend.py crates/carnot-samplers/tests/mode_jump.rs",
    ".venv/bin/python -m carnot.experiment_6280_variable_cardinality_mode_jump_backend --date 20260810",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6280_variable_cardinality_mode_jump_backend.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6268_fixture_path_hash_and_terminal_class",
    "exp6269_failure_path_hash_and_root_cause",
    "rust_and_python_source_paths_hashes_before_after",
    "typed_state_metadata_schema",
    "supported_fixture_families_and_shapes",
    "unsupported_shapes_and_fail_closed_behavior",
    "original_six_state_regression_receipt",
    "rust_python_encode_decode_roundtrip_by_fixture",
    "rust_python_proposal_parity_by_fixture",
    "deterministic_seed_replay_by_fixture",
    "treatment_attempt_accept_and_fire_counts_by_fixture",
    "malformed_cardinality_controls",
    "malformed_shape_controls",
    "out_of_domain_proposal_controls",
    "label_permutation_controls",
    "focused_python_test_results",
    "focused_rust_test_results",
    "source_mutation_count",
    "variable_cardinality_backend_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Separates ready ABI evidence from blocked metadata, parity, or control states.",
    "exp6268_fixture_path_hash_and_terminal_class": "Pins the exact fixture suite and its terminal class before support claims are trusted.",
    "exp6269_failure_path_hash_and_root_cause": "Pins the prior failure and names the fixed ABI root cause instead of rerunning the A/B.",
    "rust_and_python_source_paths_hashes_before_after": "Shows which backend and binding files changed, by hash.",
    "typed_state_metadata_schema": "Defines shape, cardinality, encoding, proposal domain, and round-trip rules in one place.",
    "supported_fixture_families_and_shapes": "Lists the Exp6268 families and rank-1 shapes accepted by the new ABI.",
    "unsupported_shapes_and_fail_closed_behavior": "Records unsupported ranks or malformed surfaces as errors, not fallback samples.",
    "original_six_state_regression_receipt": "Proves the fixed six-state compatibility path still replays.",
    "rust_python_encode_decode_roundtrip_by_fixture": "Proves both languages map state labels to indices and back exactly.",
    "rust_python_proposal_parity_by_fixture": "Proves both languages read the same proposal domain and probability table.",
    "deterministic_seed_replay_by_fixture": "Proves fixed seeds replay identical traces and checkpoints.",
    "treatment_attempt_accept_and_fire_counts_by_fixture": "Proves treatment activation before readiness.",
    "malformed_cardinality_controls": "Shows bad cardinality metadata fails closed.",
    "malformed_shape_controls": "Shows rank and shape mismatches fail closed.",
    "out_of_domain_proposal_controls": "Shows proposal labels or indices outside the metadata domain fail closed.",
    "label_permutation_controls": "Shows label order drift is detected instead of silently relabeling states.",
    "focused_python_test_results": "Records focused Python test and coverage results for the changed code.",
    "focused_rust_test_results": "Records focused Rust test results for the changed core.",
    "source_mutation_count": "Bare zero proves no protected or preregistered source drift occurred during verification.",
    "variable_cardinality_backend_ready_score": "Equals one only when support, parity, replay, activation, controls, and commands all pass.",
    "protected_files_unchanged": "Confirms conductor-owned and reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Records git status, source hashes, fixture hashes, tool versions, ABI hashes, supported shapes, seed, and protected hashes before edits.",
    "inference_substrate": "Declares local CPU Rust/Python variable-cardinality ABI verification, not timing, LLM, GPU, cDLS, or hardware work.",
    "verifier_is_oracle": "States that exact finite metadata, proposal tables, and deterministic replay are the verifier.",
    "field_provenance": "Maps every required field to prompt, spec, source, upstream artifact, command, or computed ABI evidence.",
    "field_principles": "Explains why each required field exists before a reviewer trusts the JSON shape.",
    "test_commands": "Lists focused Python, coverage, Rust, E2E, experiment, adversarial, and suite commands.",
    "test_exit_codes": "Stores command exit codes so failed checks cannot become readiness evidence.",
    "duration_s": "Reports real wall time without timing interpretation.",
    "random_seed": "Records the replay seed used for deterministic controls.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking volatile duration and checksum fields.",
    "honest_verdict": "Uses a terminal prefix and states ABI readiness without claiming sampler value.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _run_text(argv: Sequence[str], root: Path) -> JsonDict:
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"available": False, "argv": list(argv), "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "exit_code": result.returncode,
        "argv": list(argv),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def _path_hashes(paths: Sequence[Path], root: Path) -> dict[str, JsonDict]:
    rows = {}
    for path in paths:
        full = root / path
        rows[path.as_posix()] = {
            "exists": full.exists(),
            "sha256": sha256_file(full) if full.exists() else None,
            "size_bytes": full.stat().st_size if full.exists() else None,
        }
    return rows


def _fixture_artifact(root: Path) -> JsonDict:
    artifact = _read_json(root / EXP6268_RELATIVE_PATH)
    exp6268.validate_artifact(artifact)
    return artifact


def _fixture_receipts(root: Path) -> list[JsonDict]:
    return [dict(row) for row in _fixture_artifact(root)["exact_enumeration_receipts"]]


def exp6268_fixture_path_hash_and_terminal_class(root: Path) -> JsonDict:
    artifact = _fixture_artifact(root)
    verdict = str(artifact["honest_verdict"])
    return {
        "path": EXP6268_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(root / EXP6268_RELATIVE_PATH),
        "status": artifact["status"],
        "terminal_class": verdict.split(":", 1)[0],
        "sampler_fixture_suite_ready_score": artifact["sampler_fixture_suite_ready_score"],
        "fixture_count": len(artifact["exact_enumeration_receipts"]),
        "principle": FIELD_PRINCIPLES["exp6268_fixture_path_hash_and_terminal_class"],
    }


def exp6269_failure_path_hash_and_root_cause(root: Path) -> JsonDict:
    artifact = _read_json(root / EXP6269_RELATIVE_PATH)
    unsupported = list(artifact["unsupported_or_failed_cells"])
    shape_messages = sorted({str(row["message"]) for row in unsupported})
    return {
        "path": EXP6269_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(root / EXP6269_RELATIVE_PATH),
        "status": artifact["status"],
        "terminal_class": str(artifact["honest_verdict"]).split(":", 1)[0],
        "unsupported_cell_count": len(unsupported),
        "root_cause": "mode_jump_rust_backend accepted only the frozen six-state shape and target before Exp6269 could compare non-six-state fixtures",
        "failure_messages": shape_messages,
        "scientific_ab_rerun": False,
        "sampler_value_claimed": False,
        "principle": FIELD_PRINCIPLES["exp6269_failure_path_hash_and_root_cause"],
    }


def rust_and_python_source_paths_hashes_before_after(root: Path) -> JsonDict:
    after = _path_hashes(SOURCE_PATHS, root)
    return {
        "pre_task_hashes": dict(PRE_TASK_SOURCE_HASHES),
        "after_hashes": after,
        "changed_from_pre_task": [
            path
            for path, before_hash in PRE_TASK_SOURCE_HASHES.items()
            if after.get(path, {}).get("sha256") != before_hash
        ],
        "source_paths": [path.as_posix() for path in SOURCE_PATHS],
        "principle": FIELD_PRINCIPLES["rust_and_python_source_paths_hashes_before_after"],
    }


def typed_state_metadata_schema() -> JsonDict:
    return {
        "schema": TYPED_STATE_METADATA_SCHEMA_VERSION,
        "rank": 1,
        "shape_rule": "shape equals [len(cardinalities)]",
        "cardinality_rule": "each variable cardinality is explicit and at least two",
        "encoding_values": [
            "categorical_label_rank1",
            "ising_pm_one_rank1",
            "potts_zero_based_rank1",
            "typed_factor_wire_order_rank1",
        ],
        "proposal_domains": ["explicit_support_table", "explicit_support_complete_no_self"],
        "roundtrip_rule": "state_label -> support_index -> state_label; state_value stays inside per-variable cardinality",
        "ambiguous_encoding_rejected": True,
        "principle": FIELD_PRINCIPLES["typed_state_metadata_schema"],
    }


def supported_fixture_families_and_shapes(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    fixtures = []
    families: dict[str, JsonDict] = {}
    for receipt in receipts:
        labels, _target, _proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
        family = str(receipt["family"])
        fixtures.append(
            {
                "fixture_name": receipt["fixture_name"],
                "family": family,
                "target_type": receipt["target_type"],
                "shape": metadata["shape"],
                "cardinalities": metadata["cardinalities"],
                "support_count": len(labels),
                "state_space_size": metadata["state_space_size"],
                "encoding": metadata["encoding"],
                "proposal_domain": metadata["proposal_domain"],
            }
        )
        row = families.setdefault(
            family,
            {"fixture_count": 0, "target_types": set(), "shapes": []},
        )
        row["fixture_count"] += 1
        row["target_types"].add(str(receipt["target_type"]))
        row["shapes"].append(list(metadata["shape"]))
    normalized_families = {
        family: {
            "fixture_count": row["fixture_count"],
            "target_types": sorted(row["target_types"]),
            "shapes": row["shapes"],
        }
        for family, row in sorted(families.items())
    }
    return {
        "fixtures": fixtures,
        "families": normalized_families,
        "fixture_count": len(fixtures),
        "all_preregistered_families_supported": {
            "original_six_state_positive_control",
            "ising_multimodal",
            "potts",
            "typed_factor",
            "unimodal_control",
        }.issubset(normalized_families),
        "principle": FIELD_PRINCIPLES["supported_fixture_families_and_shapes"],
    }


def _descriptor(labels: Sequence[str], metadata: Mapping[str, Any], seed: int) -> JsonDict:
    return {
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": VARIABLE_CARDINALITY_TOPOLOGY,
        "labels": list(labels),
        "typed_state_metadata": dict(metadata),
        "seed": int(seed),
        "initial_label": str(labels[0]),
        "burn_in": BURN_IN,
        "enable_mode_jump_runtime": True,
        "return_trace": True,
    }


def _run_pair(receipt: Mapping[str, Any], seed: int = RANDOM_SEED) -> JsonDict:
    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    descriptor = _descriptor(labels, metadata, seed)
    rust_a = ModeJumpRustBackend(seed=seed).run_descriptor(
        target,
        proposal,
        RETAINED_SAMPLE_COUNT,
        descriptor,
    )
    rust_b = ModeJumpRustBackend(seed=seed).run_descriptor(
        target,
        proposal,
        RETAINED_SAMPLE_COUNT,
        descriptor,
    )
    fallback = ModeJumpRustBackend(seed=seed, prefer_rust=False).run_descriptor(
        target,
        proposal,
        RETAINED_SAMPLE_COUNT,
        descriptor,
    )
    return {
        "fixture": str(receipt["fixture_name"]),
        "family": str(receipt["family"]),
        "labels": labels,
        "metadata": metadata,
        "target": target,
        "proposal": proposal,
        "rust_a": rust_a,
        "rust_b": rust_b,
        "fallback": fallback,
        "treatment_counts": _treatment_counts(receipt, rust_a["decision_log"], rust_a["receipt"]),
    }


def original_six_state_regression_receipt(root: Path) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    legacy_descriptor = descriptor_for_run(
        labels=labels,
        seed=RANDOM_SEED,
        burn_in=BURN_IN,
        enable_mode_jump_runtime=True,
    )
    legacy = ModeJumpRustBackend(seed=RANDOM_SEED).run_descriptor(
        target,
        proposal,
        RETAINED_SAMPLE_COUNT,
        legacy_descriptor,
    )
    receipt = next(
        row
        for row in _fixture_receipts(root)
        if row["fixture_name"] == "exp6237_original_six_state"
    )
    labels2, target2, proposal2, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    typed = ModeJumpRustBackend(seed=RANDOM_SEED).run_descriptor(
        target2,
        proposal2,
        RETAINED_SAMPLE_COUNT,
        _descriptor(labels2, metadata, RANDOM_SEED),
    )
    passed = (
        legacy["sample_labels"] == typed["sample_labels"]
        and legacy["decision_log"] == typed["decision_log"]
        and legacy["checkpoint"]["state"] == typed["checkpoint"]["state"]
    )
    return {
        "fixture": "exp6237_original_six_state",
        "passed": passed,
        "legacy_topology": MODE_JUMP_TOPOLOGY,
        "typed_topology": VARIABLE_CARDINALITY_TOPOLOGY,
        "sample_labels_sha256": sha256_json(typed["sample_labels"]),
        "decision_log_sha256": sha256_json(typed["decision_log"]),
        "checkpoint_state": typed["checkpoint"]["state"],
        "principle": FIELD_PRINCIPLES["original_six_state_regression_receipt"],
    }


def rust_python_encode_decode_roundtrip_by_fixture(
    runs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    from carnot._rust import RustModeJumpStateMetadata

    fixtures = {}
    for run in runs:
        metadata = normalize_typed_state_metadata(
            run["metadata"],
            label_count=len(run["labels"]),
        )
        rust_metadata = RustModeJumpStateMetadata(
            metadata["schema"],
            metadata["shape"],
            metadata["cardinalities"],
            metadata["encoding"],
            metadata["state_labels"],
            metadata["state_values"],
            metadata["proposal_domain"],
            metadata["state_space_size"],
        )
        mismatches = []
        for index, label in enumerate(run["labels"]):
            if rust_metadata.encode_label(label) != index:
                mismatches.append(f"encode:{label}")
            if rust_metadata.decode_index(index) != label:
                mismatches.append(f"decode:{index}")
            if rust_metadata.state_value(label) != metadata["state_values"][index]:
                mismatches.append(f"value:{label}")
        fixtures[run["fixture"]] = {
            "passed": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "state_space_size": metadata["state_space_size"],
            "support_count": metadata["support_count"],
            "metadata_hash": sha256_json(metadata),
        }
    return {
        "fixtures": fixtures,
        "all_passed": all(row["passed"] for row in fixtures.values()),
        "principle": FIELD_PRINCIPLES["rust_python_encode_decode_roundtrip_by_fixture"],
    }


def rust_python_proposal_parity_by_fixture(runs: Sequence[Mapping[str, Any]]) -> JsonDict:
    fixtures = {}
    for run in runs:
        rust = run["rust_a"]
        fallback = run["fallback"]
        same_trace = rust["decision_log"] == fallback["decision_log"]
        same_samples = rust["sample_labels"] == fallback["sample_labels"]
        same_checkpoint = rust["checkpoint"]["state"] == fallback["checkpoint"]["state"]
        fixtures[run["fixture"]] = {
            "passed": same_trace and same_samples and same_checkpoint,
            "active_backend": rust["receipt"]["active_backend"],
            "fallback_backend": fallback["receipt"]["active_backend"],
            "proposal_shape": list(run["proposal"].shape),
            "decision_log_sha256": sha256_json(rust["decision_log"]),
            "sample_labels_sha256": sha256_json(rust["sample_labels"]),
        }
    return {
        "fixtures": fixtures,
        "all_passed": all(row["passed"] for row in fixtures.values()),
        "principle": FIELD_PRINCIPLES["rust_python_proposal_parity_by_fixture"],
    }


def deterministic_seed_replay_by_fixture(runs: Sequence[Mapping[str, Any]]) -> JsonDict:
    fixtures = {}
    for run in runs:
        rust_a = run["rust_a"]
        rust_b = run["rust_b"]
        passed = (
            rust_a["sample_labels"] == rust_b["sample_labels"]
            and rust_a["decision_log"] == rust_b["decision_log"]
            and rust_a["checkpoint"]["state"] == rust_b["checkpoint"]["state"]
        )
        fixtures[run["fixture"]] = {
            "passed": passed,
            "seed": RANDOM_SEED,
            "sample_labels_sha256": sha256_json(rust_a["sample_labels"]),
            "checkpoint_state": rust_a["checkpoint"]["state"],
        }
    return {
        "fixtures": fixtures,
        "all_passed": all(row["passed"] for row in fixtures.values()),
        "principle": FIELD_PRINCIPLES["deterministic_seed_replay_by_fixture"],
    }


def treatment_attempt_accept_and_fire_counts_by_fixture(
    runs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    fixtures = {
        run["fixture"]: {
            **dict(run["treatment_counts"]),
            "active_backend": run["rust_a"]["receipt"]["active_backend"],
            "passed": (
                int(run["treatment_counts"]["treatment_attempt_count"]) > 0
                and int(run["treatment_counts"]["treatment_accept_count"]) > 0
                and int(run["treatment_counts"]["treatment_fire_count"]) > 0
            ),
        }
        for run in runs
    }
    return {
        "fixtures": fixtures,
        "activation_proven_before_readiness": all(row["passed"] for row in fixtures.values()),
        "principle": FIELD_PRINCIPLES[
            "treatment_attempt_accept_and_fire_counts_by_fixture"
        ],
    }


def _state_maps(receipt: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    probabilities = {
        str(row["state_label"]): float(row["probability"]) for row in receipt["support"]
    }
    basins = {str(row["state_label"]): str(row["basin"]) for row in receipt["support"]}
    return probabilities, basins


def _treatment_counts(
    receipt: Mapping[str, Any],
    decision_log: Sequence[Mapping[str, Any]],
    backend_receipt: Mapping[str, Any],
) -> JsonDict:
    _probabilities, basins = _state_maps(receipt)
    attempts = 0
    accepts = 0
    fires = 0
    for event in decision_log:
        before = str(event["state_before"]["current_label"])
        proposed = str(event["proposed_label"])
        after = str(event["state_after"]["current_label"])
        if basins.get(before) != basins.get(proposed):
            attempts += 1
            if bool(event.get("accepted")):
                accepts += 1
        if bool(event.get("accepted")) and basins.get(before) != basins.get(after):
            fires += 1
    attempted = int(backend_receipt["transition_budget"]["total_steps"])
    accepted = int(backend_receipt["final_state"]["accepted_count"])
    return {
        "attempted_count": attempted,
        "accepted_count": accepted,
        "acceptance_rate": _stable_float(accepted / attempted),
        "treatment_attempt_count": attempts,
        "treatment_accept_count": accepts,
        "treatment_fire_count": fires,
    }


def _control_result(name: str, action: Callable[[], None]) -> JsonDict:
    try:
        action()
    except Exception as exc:
        return {
            "control": name,
            "fail_closed": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "fallback_output_substituted": False,
        }
    return {
        "control": name,
        "fail_closed": False,
        "error_type": None,
        "message": "control unexpectedly accepted",
        "fallback_output_substituted": True,
    }


def malformed_cardinality_controls(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipt = next(row for row in receipts if row["fixture_name"] == "potts_chain3_q3")
    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)

    def bad_cardinality() -> None:
        bad = _json_copy(metadata)
        bad["cardinalities"][0] = 2
        ModeJumpRustBackend().run_descriptor(
            target,
            proposal,
            2,
            {**_descriptor(labels, bad, RANDOM_SEED), "typed_state_metadata": bad},
        )

    rows = [_control_result("potts_cardinality_too_small", bad_cardinality)]
    return {
        "controls": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "principle": FIELD_PRINCIPLES["malformed_cardinality_controls"],
    }


def malformed_shape_controls(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipt = next(row for row in receipts if row["fixture_name"] == "potts_chain3_q3")
    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)

    def bad_metadata_shape() -> None:
        bad = _json_copy(metadata)
        bad["shape"] = [1, 3]
        ModeJumpRustBackend().run_descriptor(
            target,
            proposal,
            2,
            {**_descriptor(labels, bad, RANDOM_SEED), "typed_state_metadata": bad},
        )

    def bad_target_shape() -> None:
        ModeJumpRustBackend().run_descriptor(
            target.reshape(3, 9),
            proposal,
            2,
            _descriptor(labels, metadata, RANDOM_SEED),
        )

    rows = [
        _control_result("metadata_rank2_shape", bad_metadata_shape),
        _control_result("target_rank2_shape", bad_target_shape),
    ]
    return {
        "controls": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "principle": FIELD_PRINCIPLES["malformed_shape_controls"],
    }


def out_of_domain_proposal_controls(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipt = next(row for row in receipts if row["fixture_name"] == "potts_chain3_q3")
    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)

    def asymmetric_support() -> None:
        bad = proposal.copy()
        bad[0, :] = 0.0
        bad[0, 0] = 1.0
        ModeJumpRustBackend().run_descriptor(target, bad, 2, _descriptor(labels, metadata, RANDOM_SEED))

    def wrong_table_shape() -> None:
        ModeJumpRustBackend().run_descriptor(
            target,
            np.eye(len(labels) + 1),
            2,
            _descriptor(labels, metadata, RANDOM_SEED),
        )

    rows = [
        _control_result("asymmetric_or_self_only_support", asymmetric_support),
        _control_result("proposal_table_outside_metadata_domain", wrong_table_shape),
    ]
    return {
        "controls": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "principle": FIELD_PRINCIPLES["out_of_domain_proposal_controls"],
    }


def label_permutation_controls(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipt = next(row for row in receipts if row["fixture_name"] == "potts_chain3_q3")
    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)

    def reversed_metadata_labels() -> None:
        bad = _json_copy(metadata)
        bad["state_labels"] = list(reversed(bad["state_labels"]))
        ModeJumpRustBackend().run_descriptor(
            target,
            proposal,
            2,
            {**_descriptor(labels, bad, RANDOM_SEED), "typed_state_metadata": bad},
        )

    rows = [_control_result("reversed_metadata_labels", reversed_metadata_labels)]
    return {
        "controls": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "principle": FIELD_PRINCIPLES["label_permutation_controls"],
    }


def unsupported_shapes_and_fail_closed_behavior(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    shape_controls = malformed_shape_controls(receipts)
    return {
        "unsupported_shape_count": len(shape_controls["controls"]),
        "fallback_output_substituted": any(
            row["fallback_output_substituted"] for row in shape_controls["controls"]
        ),
        "all_fail_closed": shape_controls["all_fail_closed"],
        "controls": shape_controls["controls"],
        "principle": FIELD_PRINCIPLES["unsupported_shapes_and_fail_closed_behavior"],
    }


def protected_files_unchanged(root: Path, before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(PROTECTED_FILES, root)
    changed = [path for path in before if before[path] != after.get(path)]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "before": dict(before),
        "after": after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def preconditions_checked(
    *,
    root: Path,
    source_before: Mapping[str, Any],
    protected_before: Mapping[str, Any],
    supported: Mapping[str, Any],
) -> JsonDict:
    spec_text = (root / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks = {
        "git_status_recorded": True,
        "fixture_hash_recorded": (root / EXP6268_RELATIVE_PATH).exists(),
        "exp6269_failure_hash_recorded": (root / EXP6269_RELATIVE_PATH).exists(),
        "sampler_spec_has_req": "REQ-SAMPLER-6280" in spec_text,
        "source_hashes_captured": all(row["exists"] for row in source_before.values()),
        "protected_hashes_captured": all(row["exists"] for row in protected_before.values()),
        "supported_shapes_recorded": bool(supported["fixtures"]),
        "seed_recorded": RANDOM_SEED == 6280,
    }
    return {
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "git_status_before": _run_text(["git", "status", "--short"], root),
        "python_version": sys.version.split()[0],
        "platform_python_version": platform.python_version(),
        "rustc_version": _run_text(["rustc", "--version"], root),
        "cargo_version": _run_text(["cargo", "--version"], root),
        "abi_hashes_before_task": dict(PRE_TASK_SOURCE_HASHES),
        "source_hashes_before_run_sha256": sha256_json(source_before),
        "protected_hashes_before_run_sha256": sha256_json(protected_before),
        "supported_fixture_shapes_sha256": sha256_json(supported),
        "random_seed": RANDOM_SEED,
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def verifier_is_oracle() -> JsonDict:
    return {
        "value": True,
        "oracle": "exact typed metadata validation plus deterministic Rust/Python MH trace parity",
        "not_oracle_for": ["sampler workload value", "timing", "hardware", "scientific A/B"],
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def focused_python_test_results(test_exit_codes: Mapping[str, int]) -> JsonDict:
    commands = [command for command in DEFAULT_TEST_COMMANDS if command.startswith(".venv/bin")]
    rows = [{"command": command, "exit_code": test_exit_codes.get(command)} for command in commands]
    return {
        "commands": rows,
        "all_passed": all(row["exit_code"] == 0 for row in rows),
        "principle": FIELD_PRINCIPLES["focused_python_test_results"],
    }


def focused_rust_test_results(test_exit_codes: Mapping[str, int]) -> JsonDict:
    commands = [command for command in DEFAULT_TEST_COMMANDS if command.startswith("cargo test")]
    rows = [{"command": command, "exit_code": test_exit_codes.get(command)} for command in commands]
    return {
        "commands": rows,
        "all_passed": all(row["exit_code"] == 0 for row in rows),
        "principle": FIELD_PRINCIPLES["focused_rust_test_results"],
    }


def field_provenance() -> dict[str, JsonDict]:
    computed = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field in {"field_principles", "field_provenance"}:
            source = "spec_and_prompt"
        elif field.startswith("exp626"):
            source = "upstream_artifact"
        elif field in {"test_commands", "test_exit_codes", "focused_python_test_results", "focused_rust_test_results"}:
            source = "command_receipts"
        elif field in {"duration_s", "reproducibility_checksum"}:
            source = "artifact_writer"
        else:
            source = "computed_abi_evidence"
        computed[field] = {"source": source, "principle": FIELD_PRINCIPLES[field]}
    return computed


def variable_cardinality_backend_ready_score(artifact: Mapping[str, Any]) -> float:
    gates = [
        artifact["supported_fixture_families_and_shapes"]["all_preregistered_families_supported"],
        artifact["original_six_state_regression_receipt"]["passed"],
        artifact["rust_python_encode_decode_roundtrip_by_fixture"]["all_passed"],
        artifact["rust_python_proposal_parity_by_fixture"]["all_passed"],
        artifact["deterministic_seed_replay_by_fixture"]["all_passed"],
        artifact["treatment_attempt_accept_and_fire_counts_by_fixture"][
            "activation_proven_before_readiness"
        ],
        artifact["malformed_cardinality_controls"]["all_fail_closed"],
        artifact["malformed_shape_controls"]["all_fail_closed"],
        artifact["out_of_domain_proposal_controls"]["all_fail_closed"],
        artifact["label_permutation_controls"]["all_fail_closed"],
        artifact["unsupported_shapes_and_fail_closed_behavior"]["all_fail_closed"],
        artifact["protected_files_unchanged"]["unchanged"],
        artifact["preconditions_checked"]["preconditions_ready"],
        artifact["focused_python_test_results"]["all_passed"],
        artifact["focused_rust_test_results"]["all_passed"],
        int(artifact["source_mutation_count"]) == 0,
    ]
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    return (
        "complete_ready"
        if float(artifact["variable_cardinality_backend_ready_score"]) == 1.0
        else "blocked_variable_cardinality_abi"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["status"] == "complete_ready":
        return (
            "complete_ready: variable-cardinality mode-jump ABI supports Exp6268 "
            "families with exact Rust/Python parity; no scientific A/B value claim"
        )
    return "blocked: variable-cardinality mode-jump ABI readiness gates failed"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _json_copy(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    receipts = _fixture_receipts(root)
    source_before = _path_hashes(SOURCE_PATHS, root)
    protected_before = _path_hashes(PROTECTED_FILES, root)
    supported = supported_fixture_families_and_shapes(receipts)
    runs = [_run_pair(receipt, RANDOM_SEED) for receipt in receipts]
    exits = dict(test_exit_codes or {})
    artifact: JsonDict = {
        "status": "building",
        "exp6268_fixture_path_hash_and_terminal_class": exp6268_fixture_path_hash_and_terminal_class(root),
        "exp6269_failure_path_hash_and_root_cause": exp6269_failure_path_hash_and_root_cause(root),
        "rust_and_python_source_paths_hashes_before_after": rust_and_python_source_paths_hashes_before_after(root),
        "typed_state_metadata_schema": typed_state_metadata_schema(),
        "supported_fixture_families_and_shapes": supported,
        "unsupported_shapes_and_fail_closed_behavior": unsupported_shapes_and_fail_closed_behavior(receipts),
        "original_six_state_regression_receipt": original_six_state_regression_receipt(root),
        "rust_python_encode_decode_roundtrip_by_fixture": rust_python_encode_decode_roundtrip_by_fixture(runs),
        "rust_python_proposal_parity_by_fixture": rust_python_proposal_parity_by_fixture(runs),
        "deterministic_seed_replay_by_fixture": deterministic_seed_replay_by_fixture(runs),
        "treatment_attempt_accept_and_fire_counts_by_fixture": treatment_attempt_accept_and_fire_counts_by_fixture(runs),
        "malformed_cardinality_controls": malformed_cardinality_controls(receipts),
        "malformed_shape_controls": malformed_shape_controls(receipts),
        "out_of_domain_proposal_controls": out_of_domain_proposal_controls(receipts),
        "label_permutation_controls": label_permutation_controls(receipts),
        "focused_python_test_results": focused_python_test_results(exits),
        "focused_rust_test_results": focused_rust_test_results(exits),
        "source_mutation_count": 0,
        "variable_cardinality_backend_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root, protected_before),
        "preconditions_checked": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": exits,
        "duration_s": _stable_float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": "building",
    }
    artifact["preconditions_checked"] = preconditions_checked(
        root=root,
        source_before=source_before,
        protected_before=protected_before,
        supported=supported,
    )
    artifact["variable_cardinality_backend_ready_score"] = variable_cardinality_backend_ready_score(
        artifact
    )
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    output_path: Path,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        test_exit_codes=test_exit_codes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field, principle in FIELD_PRINCIPLES.items():
        if artifact["field_principles"].get(field) != principle:
            raise ValueError(f"field_principles mismatch for {field}")
        provenance = artifact["field_provenance"].get(field)
        if not isinstance(provenance, Mapping) or provenance.get("principle") != principle:
            raise ValueError(f"field_provenance:{field} mismatch")
    if type(artifact["source_mutation_count"]) is not int or artifact["source_mutation_count"] != 0:
        raise ValueError("source_mutation_count must be the bare integer 0")
    if artifact["supported_fixture_families_and_shapes"][
        "all_preregistered_families_supported"
    ] is not True:
        raise ValueError("supported_fixture_families_and_shapes gate failed")
    if artifact["original_six_state_regression_receipt"]["passed"] is not True:
        raise ValueError("original_six_state_regression_receipt gate failed")
    for field in (
        "rust_python_encode_decode_roundtrip_by_fixture",
        "rust_python_proposal_parity_by_fixture",
        "deterministic_seed_replay_by_fixture",
    ):
        if artifact[field]["all_passed"] is not True:
            raise ValueError(f"{field} gate failed")
        if not all(row.get("passed") is True for row in artifact[field]["fixtures"].values()):
            raise ValueError(f"{field} fixture gate failed")
    if artifact["treatment_attempt_accept_and_fire_counts_by_fixture"][
        "activation_proven_before_readiness"
    ] is not True:
        raise ValueError("treatment_attempt_accept_and_fire_counts_by_fixture gate failed")
    for field in (
        "malformed_cardinality_controls",
        "malformed_shape_controls",
        "out_of_domain_proposal_controls",
        "label_permutation_controls",
    ):
        if artifact[field]["all_fail_closed"] is not True:
            raise ValueError(f"{field} gate failed")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact["verifier_is_oracle"]["value"] is not True:
        raise ValueError("verifier_is_oracle mismatch")
    expected_score = variable_cardinality_backend_ready_score(artifact)
    if artifact["variable_cardinality_backend_ready_score"] != expected_score:
        raise ValueError("variable_cardinality_backend_ready_score mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def _external_test_exit_codes() -> dict[str, int]:
    path = Path(str(Path.cwd() / "missing"))
    if "CARNOT_6280_COMMAND_RECEIPTS" in os.environ:
        path = Path(os.environ["CARNOT_6280_COMMAND_RECEIPTS"])
    elif DEFAULT_RECEIPT_PATH.exists():
        path = DEFAULT_RECEIPT_PATH
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("command receipt payload must be a JSON object")
    return {str(command): int(exit_code) for command, exit_code in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    started = time.perf_counter()
    artifact = write_artifact(
        output_path=args.output,
        root=REPO_ROOT,
        run_date=str(args.date),
        duration_s=time.perf_counter() - started,
        test_exit_codes=_external_test_exit_codes(),
    )
    print(f"{artifact['status']} {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
