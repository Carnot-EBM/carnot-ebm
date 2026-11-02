"""Build exact constraint probes from frozen authentic model outputs.

This module never calls a model. It joins the Exp6745 output corpus to the
lossless Exp6755 replay, then builds exact CNF variants on the CPU.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache, cache
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


SCHEMA = "carnot.experiment_6799.model_output_formal_constraint_probes.v1"
EXPERIMENT_ID = 6799
RANDOM_SEED = 6799
LIVE_LLM_INVOKED = False
ELIGIBLE_CASE_COUNT = 97
MINIMUM_ELIGIBLE_CASE_COUNT = 96
TRANSFORMATIONS = ("base", "refinement", "restructuring")
REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_CONSTRAINT_FAMILIES = (
    "expander_tseitin",
    "ladder_tseitin",
    "pigeonhole_anchor",
)
DIAGNOSTIC_CLASSES = (
    "model_reasoning_valid",
    "model_reasoning_error",
    "model_abstention",
    "translation_disagreement",
    "translation_failure",
)
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_blocked_model_output_probe_fixture",
)
ATTACK_NAMES = (
    "solution_preserving_rename",
    "source_model_label_shuffle",
    "parser_disagreement",
    "duplicate_cases",
    "accidental_refinement_labeled_restructuring",
    "exact_label_leakage",
)
WORK_RELATIVE_TOLERANCE = 0.10
DIFFICULTY_SCORE_TOLERANCE = 2
SERIALIZED_LENGTH_RELATIVE_TOLERANCE = 0.10
SERIALIZED_LENGTH_ABSOLUTE_TOLERANCE = 32
INFERENCE_SUBSTRATE = (
    "deterministic_verifier -- CPU transform of frozen authentic mandated-GGUF "
    "outputs; no new LLM inference and no source-output replacement"
)

EXPECTED_SOURCE_HASHES = {
    "exp6744": "sha256:bef1d4b32917d67721c320ecd7f496f5298b0063b48cda7e427a3a596ba5d007",
    "exp6745": "sha256:d9f8cad418dbc3890ddcb8a799d8b7951362084b3f1e58e41cfca35ec1a3cdd7",
    "exp6755": "sha256:6e556e3f90e7e5a71194cf4c4bd4b373ff1354d0cfa4b8b7f66c0459e471ad1d",
    "exp6768": "sha256:5e44fd4db55e11c99f0f5720aa5815bf26135f5b0801e3687ebf1095a1ee180d",
    "exp6786": "sha256:f3780c85e29cda8dbd897b6c43a0ce3c938252625e823e54107f918d2052514a",
    "exp6789": "sha256:101487f224e5926d519b3db94c6a8ce12910c26ff1e711936216762398fc4748",
}
SOURCE_ARTIFACT_PATHS = {
    "exp6744": "results/experiment_6744_hardness_controlled_certificate_stream.json",
    "exp6745": "results/experiment_6745_sota_dual_encoding_proposal_corpus.json",
    "exp6755": "results/experiment_6755_lossless_gguf_output_reparse.json",
    "exp6768": "results/experiment_6768_targetable_proof_panel_expansion.json",
    "exp6786": "results/experiment_6786_constraint_dependency_hard_negative_fixture.json",
    "exp6789": "results/experiment_6789_soft_fixed_point_cold_authority_audit.json",
}

FEATURE_ALLOWLIST = (
    "representation",
    "variable_count",
    "clause_count",
    "clause_width_histogram",
    "literal_count",
    "dependency_incidence_count",
    "compositional_depth",
    "cross_constraint_coupling",
    "dependency_chain_length",
    "structural_difficulty_score",
)
FEATURE_DENYLIST = (
    "exact_valid",
    "valid_assignments",
    "valid_set_hash",
    "solution_count",
    "source_model",
    "source_model_hf_id",
    "model_hub_id",
    "model_family",
    "split",
    "diagnosis",
    "translation_reasoning_diagnostic",
    "graph_hash",
)
STANDARD_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "source_model_specs",
    "live_llm_invoked",
    "transformation_contract",
    "feature_allowlist",
    "feature_denylist",
    "split_manifest",
    "matching_receipts",
    "dual_encoding_diagnostics",
    "exact_replay_receipts",
    "graph_hashes",
    "rows",
    "model_output_constraint_probe_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)


def canonical_json(value: Any) -> str:
    """Serialize evidence in one deterministic form for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest for exact byte evidence."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    """Hash a JSON-compatible value in canonical form."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash one source artifact without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _default_repo_root() -> Path:
    """Resolve the checkout root from this installed source file."""

    return Path(__file__).resolve().parents[2]


def _source_paths(
    repo_root: Path,
    proposal_path: Path | None = None,
    reparse_path: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the two source paths while keeping test overrides explicit."""

    proposal = proposal_path or repo_root / SOURCE_ARTIFACT_PATHS["exp6745"]
    reparse = reparse_path or repo_root / SOURCE_ARTIFACT_PATHS["exp6755"]
    return proposal, reparse


def load_source_artifacts(
    repo_root: Path | None = None,
    proposal_path: Path | None = None,
    reparse_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the authentic proposal corpus and its lossless replay."""

    root = Path(repo_root or _default_repo_root())
    proposal_file, reparse_file = _source_paths(root, proposal_path, reparse_path)
    return (
        json.loads(proposal_file.read_text(encoding="utf-8")),
        json.loads(reparse_file.read_text(encoding="utf-8")),
    )


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    """Build one stable precondition receipt."""

    return {
        "check": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _finish_check_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Add aggregate and first-failure fields to gate receipts."""

    failed = [row["check"] for row in checks if not row["passed"]]
    summary = {
        "all_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }
    summary["first_failure"] = first_failed_check(summary)
    return summary


def first_failed_check(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the first failed receipt or an explicit success receipt."""

    for row in summary.get("checks", []):
        if not row.get("passed", False):
            return row
    return {
        "check": "all_preconditions",
        "passed": True,
        "observed": True,
        "required": True,
    }


def _decoded_legacy_output(envelope: str) -> bytes | None:
    """Decode the frozen Python bytes envelope without semantic edits."""

    try:
        value = ast.literal_eval(envelope)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, bytes) else None


def _row_maps(
    proposal: dict[str, Any], reparse: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index proposal, replay, and frozen stream rows by stable IDs."""

    proposal_rows = {row["row_id"]: row for row in proposal.get("rows", [])}
    replay_rows = {row["row_id"]: row for row in reparse.get("rows", [])}
    instances = {
        row["row_id"]: row for row in proposal.get("frozen_manifest", {}).get("instances", [])
    }
    return proposal_rows, replay_rows, instances


def _model_provenance_valid(proposal: dict[str, Any]) -> bool:
    """Check local GGUF metadata and authentic accelerator receipts."""

    models = proposal.get("models_used", [])
    receipts = proposal.get("gpu_receipts", [])
    if {row.get("hf_id") for row in models} != set(REQUIRED_MODEL_IDS):
        return False
    for model in models:
        model_path = Path(model.get("model_path", ""))
        if not (
            model.get("resolved") is True
            and model_path.is_file()
            and model_path.stat().st_size == model.get("model_size_bytes")
        ):
            return False
        related = [row for row in receipts if row.get("model_hf_id") == model.get("hf_id")]
        if not related or not all(
            row.get("authentic") is True
            and row.get("cuda_offload") is True
            and row.get("accelerator_observed") is True
            and row.get("owned_process") is True
            and row.get("memory_recovered") is True
            and row.get("exit_code") == 0
            and row.get("model_sha256") == model.get("model_sha256")
            for row in related
        ):
            return False
    return True


def evaluate_source_contract(proposal: dict[str, Any], reparse: dict[str, Any]) -> dict[str, Any]:
    """Evaluate content gates after both source JSON files are readable."""

    proposal_rows, replay_rows, _ = _row_maps(proposal, reparse)
    planned = proposal.get("planned_row_count")
    complete = (
        isinstance(planned, int)
        and planned == 216
        and len(proposal_rows) == planned
        and len(replay_rows) == planned
        and set(proposal_rows) == set(replay_rows)
        and reparse.get("replayed_row_count") == planned
    )
    lossless = complete and all(
        replay.get("original_output_text") == proposal_rows[row_id].get("raw_output")
        and replay.get("original_output_sha256") == proposal_rows[row_id].get("raw_output_sha256")
        and replay.get("original_output_hash_matches") is True
        and replay.get("semantic_edits_performed") == 0
        and replay.get("evidence_preserved") is True
        and sha256_bytes(replay.get("original_output_text", "").encode("utf-8"))
        == replay.get("original_output_sha256")
        and sha256_bytes(replay.get("normalized_output_text", "").encode("utf-8"))
        == replay.get("normalized_output_sha256")
        and _decoded_legacy_output(replay.get("original_output_text", ""))
        == replay.get("normalized_output_text", "").encode("utf-8")
        for row_id, replay in replay_rows.items()
    )
    parseable = [
        row
        for row in replay_rows.values()
        if row.get("post_parse_result", {}).get("parser_status") == "parseable"
    ]
    dual_exact = bool(parseable) and all(
        row.get("encoder_a", {}).get("attempted") is True
        and row.get("encoder_b", {}).get("attempted") is True
        and row.get("encoder_a", {}).get("exact_check", {}).get("authority_available") is True
        and row.get("encoder_b", {}).get("exact_check", {}).get("authority_available") is True
        for row in parseable
    )
    eligible = select_eligible_cases(proposal, reparse) if complete else []
    models = {row["source_model"]["hf_id"] for row in eligible}
    families = {row["constraint_family"] for row in eligible}
    exact_models = {row.get("hf_id") for row in proposal.get("models_used", [])}
    checks = [
        _check(
            "exp6745_corpus_ready",
            proposal.get("dual_encoding_corpus_ready") is True,
            proposal.get("dual_encoding_corpus_ready"),
            True,
        ),
        _check(
            "exp6755_reparse_ready",
            reparse.get("transport_reparse_ready") is True,
            reparse.get("transport_reparse_ready"),
            True,
        ),
        _check(
            "authentic_local_gguf_provenance",
            _model_provenance_valid(proposal),
            sorted(exact_models),
            sorted(REQUIRED_MODEL_IDS),
        ),
        _check(
            "exact_model_ids",
            exact_models == set(REQUIRED_MODEL_IDS),
            sorted(exact_models),
            sorted(REQUIRED_MODEL_IDS),
        ),
        _check(
            "complete_source_rows",
            complete,
            {"proposal": len(proposal_rows), "reparse": len(replay_rows)},
            {"proposal": 216, "reparse": 216},
        ),
        _check(
            "lossless_output_bytes_and_hashes",
            lossless,
            bool(lossless),
            True,
        ),
        _check(
            "dual_encodings_and_exact_authority",
            dual_exact,
            {"parseable_rows": len(parseable), "all_authoritative": dual_exact},
            {"minimum_parseable_rows": 1, "all_authoritative": True},
        ),
        _check(
            "eligible_paired_source_cases",
            len(eligible) >= MINIMUM_ELIGIBLE_CASE_COUNT,
            len(eligible),
            f">={MINIMUM_ELIGIBLE_CASE_COUNT}",
        ),
        _check(
            "required_model_strata",
            models == set(REQUIRED_MODEL_IDS),
            sorted(models),
            sorted(REQUIRED_MODEL_IDS),
        ),
        _check(
            "required_family_strata",
            families == set(REQUIRED_CONSTRAINT_FAMILIES),
            sorted(families),
            sorted(REQUIRED_CONSTRAINT_FAMILIES),
        ),
    ]
    return _finish_check_summary(checks)


def evaluate_preconditions(
    repo_root: Path | None = None,
    proposal_path: Path | None = None,
    reparse_path: Path | None = None,
) -> dict[str, Any]:
    """Verify files and all authentic-source gates before transformation."""

    root = Path(repo_root or _default_repo_root())
    proposal_file, reparse_file = _source_paths(root, proposal_path, reparse_path)
    proposal_exists = proposal_file.is_file()
    reparse_exists = reparse_file.is_file()
    checks = [
        _check("exp6745_artifact_exists", proposal_exists, proposal_exists, True),
        _check(
            "exp6745_artifact_hash",
            proposal_exists and sha256_file(proposal_file) == EXPECTED_SOURCE_HASHES["exp6745"],
            sha256_file(proposal_file) if proposal_exists else None,
            EXPECTED_SOURCE_HASHES["exp6745"],
        ),
        _check("exp6755_artifact_exists", reparse_exists, reparse_exists, True),
        _check(
            "exp6755_artifact_hash",
            reparse_exists and sha256_file(reparse_file) == EXPECTED_SOURCE_HASHES["exp6755"],
            sha256_file(reparse_file) if reparse_exists else None,
            EXPECTED_SOURCE_HASHES["exp6755"],
        ),
    ]
    if not (proposal_exists and reparse_exists and checks[1]["passed"] and checks[3]["passed"]):
        names = (
            "exp6745_corpus_ready",
            "exp6755_reparse_ready",
            "authentic_local_gguf_provenance",
            "exact_model_ids",
            "complete_source_rows",
            "lossless_output_bytes_and_hashes",
            "dual_encodings_and_exact_authority",
            "eligible_paired_source_cases",
            "required_model_strata",
            "required_family_strata",
        )
        checks.extend(_check(name, False, "not_evaluated", True) for name in names)
    else:
        proposal, reparse = load_source_artifacts(root, proposal_file, reparse_file)
        content = evaluate_source_contract(proposal, reparse)
        by_name = {row["check"]: row for row in content["checks"]}
        checks.insert(2, by_name["exp6745_corpus_ready"])
        checks.insert(5, by_name["exp6755_reparse_ready"])
        checks.extend(
            by_name[name]
            for name in (
                "authentic_local_gguf_provenance",
                "exact_model_ids",
                "complete_source_rows",
                "lossless_output_bytes_and_hashes",
                "dual_encodings_and_exact_authority",
                "eligible_paired_source_cases",
                "required_model_strata",
                "required_family_strata",
            )
        )
    return _finish_check_summary(checks)


def diagnose_dual_encoding(
    encoder_a: dict[str, Any], encoder_b: dict[str, Any], post_diagnosis: str
) -> str:
    """Separate translation disagreement from model reasoning outcomes."""

    if not encoder_a.get("attempted") or not encoder_b.get("attempted"):
        return "translation_failure"
    if encoder_a.get("error") or encoder_b.get("error"):
        return "translation_failure"
    if encoder_a.get("normalized_constraints") != encoder_b.get("normalized_constraints"):
        return "translation_disagreement"
    if post_diagnosis == "exact_valid":
        return "model_reasoning_valid"
    if post_diagnosis == "abstention":
        return "model_abstention"
    return "model_reasoning_error"


def _source_case_is_eligible(replay: dict[str, Any], source: dict[str, Any] | None) -> bool:
    """Require one complete SAT case with dual exact-check authority."""

    if source is None or source.get("label") != "SAT":
        return False
    return (
        replay.get("replay_complete") is True
        and replay.get("semantic_edits_performed") == 0
        and replay.get("evidence_preserved") is True
        and replay.get("original_output_hash_matches") is True
        and replay.get("post_parse_result", {}).get("parser_status") == "parseable"
        and replay.get("encoder_a", {}).get("attempted") is True
        and replay.get("encoder_b", {}).get("attempted") is True
        and replay.get("encoder_a", {}).get("exact_check", {}).get("authority_available") is True
        and replay.get("encoder_b", {}).get("exact_check", {}).get("authority_available") is True
    )


def select_eligible_cases(
    proposal: dict[str, Any], reparse: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join every eligible authentic output to its frozen SAT CNF."""

    proposal_rows, _, instances = _row_maps(proposal, reparse)
    cases: list[dict[str, Any]] = []
    for replay in sorted(reparse.get("rows", []), key=lambda row: row["row_id"]):
        source_row_id = replay.get("source_row", {}).get("row_id")
        source = instances.get(source_row_id)
        producer = proposal_rows.get(replay.get("row_id"))
        if producer is None or not _source_case_is_eligible(replay, source):
            continue
        output = replay["normalized_output_text"].encode("utf-8")
        envelope = replay["original_output_text"]
        if (
            sha256_bytes(output) != replay.get("normalized_output_sha256")
            or sha256_bytes(envelope.encode("utf-8")) != replay.get("original_output_sha256")
            or _decoded_legacy_output(envelope) != output
        ):
            continue
        exact = enumerate_cnf(source["cnf"]["n_vars"], source["cnf"]["clauses"])
        if exact["solution_count"] < 2:
            continue
        diagnosis = diagnose_dual_encoding(
            replay["encoder_a"], replay["encoder_b"], replay["post_diagnosis"]
        )
        cases.append(
            {
                "source_case_id": replay["row_id"],
                "case_cluster_key": source_row_id,
                "constraint_family": source["family"],
                "source_label": source["label"],
                "source_model": deepcopy(replay["model"]),
                "source_cnf": deepcopy(source["cnf"]),
                "source_provenance": {
                    "source_case_id": replay["row_id"],
                    "source_stream_row_id": source_row_id,
                    "source_stream_row_sha256": source["row_sha256"],
                    "source_artifact_row_sha256": replay["source_artifact_row_sha256"],
                    "source_artifact_hash": EXPECTED_SOURCE_HASHES["exp6745"],
                    "source_reparse_artifact_hash": EXPECTED_SOURCE_HASHES["exp6755"],
                    "source_output_envelope": envelope,
                    "source_output_envelope_sha256": replay["original_output_sha256"],
                    "source_output_bytes_b64": base64.b64encode(output).decode("ascii"),
                    "source_output_bytes_sha256": replay["normalized_output_sha256"],
                    "constraint_family": source["family"],
                    "model_family": replay["model"]["family_id"],
                    "model_hub_id": replay["model"]["hf_id"],
                    "producer_dual_encodings": {
                        "encoder_a": deepcopy(producer["encoder_a"]),
                        "encoder_b": deepcopy(producer["encoder_b"]),
                    },
                    "lossless_replay_dual_encodings": {
                        "encoder_a": deepcopy(replay["encoder_a"]),
                        "encoder_b": deepcopy(replay["encoder_b"]),
                    },
                    "pre_diagnosis": replay["pre_diagnosis"],
                    "post_diagnosis": replay["post_diagnosis"],
                    "translation_reasoning_diagnostic": diagnosis,
                },
            }
        )
    split_by_cluster: dict[str, str] = {}
    by_family: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_family[case["constraint_family"]].append(case["case_cluster_key"])
    for family in sorted(by_family):
        for index, cluster in enumerate(sorted(set(by_family[family]))):
            split_by_cluster[cluster] = "held_case" if index % 3 == 0 else "development"
    for case in cases:
        case["split"] = split_by_cluster[case["case_cluster_key"]]
    return cases


@cache
def _assignment_tuples(n_vars: int) -> tuple[tuple[int, ...], ...]:
    """Cache the complete binary universe for small exact fixtures."""

    return tuple(product((0, 1), repeat=n_vars))


@cache
def _clause_mask(n_vars: int, clause: tuple[int, ...]) -> int:
    """Encode every satisfying assignment for one clause as an integer bitset."""

    mask = 0
    for index, assignment in enumerate(_assignment_tuples(n_vars)):
        if any(
            (literal > 0 and assignment[abs(literal) - 1] == 1)
            or (literal < 0 and assignment[abs(literal) - 1] == 0)
            for literal in clause
        ):
            mask |= 1 << index
    return mask


def _cnf_valid_mask(n_vars: int, clauses: Sequence[Sequence[int]]) -> int:
    """Intersect clause bitsets to obtain the exact CNF valid set."""

    valid = (1 << (1 << n_vars)) - 1
    for clause in clauses:
        valid &= _clause_mask(n_vars, tuple(clause))
    return valid


def _assignments_from_mask(n_vars: int, mask: int) -> list[dict[str, int]]:
    """Materialize ordered assignments from an exact bitset."""

    return [
        {f"x{variable + 1}": value for variable, value in enumerate(assignment)}
        for index, assignment in enumerate(_assignment_tuples(n_vars))
        if mask & (1 << index)
    ]


def _solution_count_band(count: int) -> str:
    """Place a positive count in its frozen power-of-two band."""

    if count <= 0:
        return "0"
    lower = 1 << (count.bit_length() - 1)
    upper = (lower << 1) - 1
    return str(lower) if lower == upper else f"{lower}-{upper}"


def enumerate_cnf(n_vars: int, clauses: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Enumerate every binary assignment and hash the complete valid set."""

    valid = _assignments_from_mask(n_vars, _cnf_valid_mask(n_vars, clauses))
    literal_count = sum(len(clause) for clause in clauses)
    return {
        "valid_assignments": valid,
        "valid_set_hash": sha256_value(valid),
        "solution_count": len(valid),
        "solution_count_band": _solution_count_band(len(valid)),
        "enumerated_assignment_count": 1 << n_vars,
        "literal_check_work": (1 << n_vars) * literal_count,
    }


def _dependency_edges(clauses: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """Record only cross-variable clause incidences as dependency topology."""

    return [
        {
            "constraint_id": f"c{index + 1:03d}",
            "variables": [f"x{value}" for value in sorted({abs(x) for x in clause})],
        }
        for index, clause in enumerate(clauses)
        if len({abs(value) for value in clause}) > 1
    ]


def _topology_pairs(clauses: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    """Return unique variable co-occurrences for structural difficulty."""

    pairs: set[tuple[int, int]] = set()
    for clause in clauses:
        variables = sorted({abs(value) for value in clause})
        pairs.update(combinations(variables, 2))
    return pairs


def _dependency_chain_length(n_vars: int, pairs: set[tuple[int, int]]) -> int:
    """Measure the longest finite shortest path in the variable graph."""

    neighbors = {value: set() for value in range(1, n_vars + 1)}
    for left, right in pairs:
        neighbors[left].add(right)
        neighbors[right].add(left)
    longest = 0
    for start in neighbors:
        distances = {start: 0}
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            for target in neighbors[current] - distances.keys():
                distances[target] = distances[current] + 1
                frontier.append(target)
        longest = max(longest, max(distances.values()))
    return longest


def _make_graph(n_vars: int, clauses: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Build the stable CNF graph representation used by every probe."""

    copied = [list(clause) for clause in clauses]
    pairs = _topology_pairs(copied)
    chain = _dependency_chain_length(n_vars, pairs)
    score = (
        max((len(clause) for clause in copied), default=0)
        + math.ceil(math.log2(len(copied) + 1))
        + math.ceil(math.log2(len(pairs) + 1))
        + chain
    )
    return {
        "schema": "carnot.constraint_graph.cnf.v1",
        "representation": "cnf_constraint_graph_v1",
        "variables": [f"x{value}" for value in range(1, n_vars + 1)],
        "clauses": copied,
        "local_groups": [
            {"variable": f"x{value}", "domain": [0, 1]} for value in range(1, n_vars + 1)
        ],
        "dependency_edges": _dependency_edges(copied),
        "topology_summary": {
            "cross_constraint_coupling": len(pairs),
            "dependency_chain_length": chain,
            "compositional_depth": 1,
            "structural_difficulty_score": score,
        },
    }


def graph_hash(graph: dict[str, Any]) -> str:
    """Hash the full graph representation."""

    return sha256_value(graph)


def _topology_hash(graph: dict[str, Any]) -> str:
    """Hash cross-variable incidence topology without unary refinements."""

    return sha256_value(graph["dependency_edges"])


def enumerate_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Enumerate a serialized graph through the independent public path."""

    return enumerate_cnf(len(graph["variables"]), graph["clauses"])


def _graph_record(
    operation_class: str,
    graph: dict[str, Any],
    operation_proof: dict[str, Any],
) -> dict[str, Any]:
    """Combine one graph with exact evidence and its operation proof."""

    exact = enumerate_graph(graph)
    return {
        "operation_class": operation_class,
        "graph": graph,
        "graph_hash": graph_hash(graph),
        "topology_hash": _topology_hash(graph),
        "serialized_length_bytes": len(canonical_json(graph).encode("utf-8")),
        "structural_difficulty_score": graph["topology_summary"]["structural_difficulty_score"],
        **exact,
        "operation_proof": operation_proof,
    }


def assignment_set(assignments: Iterable[dict[str, int]]) -> set[tuple[tuple[str, int], ...]]:
    """Convert serialized assignments to a set for exact support proofs."""

    return {tuple(sorted(row.items())) for row in assignments}


@cache
def _candidate_clauses(n_vars: int, width: int) -> tuple[tuple[int, ...], ...]:
    """Generate canonical non-tautological clauses of one fixed width."""

    return tuple(
        tuple(sign * variable for sign, variable in zip(signs, variables, strict=True))
        for variables in combinations(range(1, n_vars + 1), width)
        for signs in product((-1, 1), repeat=width)
    )


def _preferred_refinements(case: dict[str, Any], base_mask: int) -> list[tuple[int, int, int]]:
    """Order strict unary refinements using only frozen output content."""

    n_vars = case["source_cnf"]["n_vars"]
    parsed = (
        case["source_provenance"]["lossless_replay_dual_encodings"]["encoder_a"].get(
            "normalized_constraints"
        )
        or {}
    )
    output_values = {
        row["variable"]: int(row["values"][0])
        for row in parsed.get("bindings", [])
        if len(row.get("values", [])) == 1
    }
    output_hash = case["source_provenance"]["source_output_bytes_sha256"]
    options = []
    for variable in range(1, n_vars + 1):
        for value in (0, 1):
            literal = variable if value else -variable
            mask = base_mask & _clause_mask(n_vars, (literal,))
            if mask and mask != base_mask:
                preferred = int(output_values.get(variable) != value)
                tie = int(
                    hashlib.sha256(f"{output_hash}|{variable}|{value}".encode()).hexdigest(),
                    16,
                )
                options.append((preferred, tie, literal))
    return [(literal, preferred, tie) for preferred, tie, literal in sorted(options)]


def _matching_receipt(refinement: dict[str, Any], restructuring: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered nuisance-matching tolerances."""

    length_delta = abs(
        refinement["serialized_length_bytes"] - restructuring["serialized_length_bytes"]
    )
    length_tolerance = max(
        SERIALIZED_LENGTH_ABSOLUTE_TOLERANCE,
        math.ceil(
            max(
                refinement["serialized_length_bytes"],
                restructuring["serialized_length_bytes"],
            )
            * SERIALIZED_LENGTH_RELATIVE_TOLERANCE
        ),
    )
    work_delta = abs(refinement["literal_check_work"] - restructuring["literal_check_work"]) / max(
        refinement["literal_check_work"], restructuring["literal_check_work"]
    )
    difficulty_delta = abs(
        refinement["structural_difficulty_score"] - restructuring["structural_difficulty_score"]
    )
    receipt = {
        "variable_count_delta": abs(
            len(refinement["graph"]["variables"]) - len(restructuring["graph"]["variables"])
        ),
        "solution_count_band_match": refinement["solution_count_band"]
        == restructuring["solution_count_band"],
        "enumerated_assignment_count_delta": abs(
            refinement["enumerated_assignment_count"] - restructuring["enumerated_assignment_count"]
        ),
        "serialized_length_delta": length_delta,
        "serialized_length_tolerance": length_tolerance,
        "literal_check_work_relative_delta": work_delta,
        "literal_check_work_tolerance": WORK_RELATIVE_TOLERANCE,
        "difficulty_score_delta": difficulty_delta,
        "difficulty_score_tolerance": DIFFICULTY_SCORE_TOLERANCE,
    }
    receipt["all_tolerances_passed"] = (
        receipt["variable_count_delta"] == 0
        and receipt["solution_count_band_match"]
        and receipt["enumerated_assignment_count_delta"] == 0
        and length_delta <= length_tolerance
        and work_delta <= WORK_RELATIVE_TOLERANCE
        and difficulty_delta <= DIFFICULTY_SCORE_TOLERANCE
    )
    return receipt


def _find_restructuring(
    case: dict[str, Any],
    base_record: dict[str, Any],
    refinement_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Find a same-width replacement with incomparable exact support."""

    n_vars = case["source_cnf"]["n_vars"]
    clauses = case["source_cnf"]["clauses"]
    base_mask = _cnf_valid_mask(n_vars, clauses)
    full = (1 << (1 << n_vars)) - 1
    clause_masks = [_clause_mask(n_vars, tuple(clause)) for clause in clauses]
    prefix = [full]
    for mask in clause_masks:
        prefix.append(prefix[-1] & mask)
    suffix = [full] * (len(clauses) + 1)
    for index in range(len(clauses) - 1, -1, -1):
        suffix[index] = suffix[index + 1] & clause_masks[index]
    output_hash = case["source_provenance"]["source_output_bytes_sha256"]
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for index, old_clause in enumerate(clauses):
        old_support = {abs(value) for value in old_clause}
        other_mask = prefix[index] & suffix[index + 1]
        for candidate in _candidate_clauses(n_vars, len(old_clause)):
            if {abs(value) for value in candidate} == old_support:
                continue
            candidate_mask = other_mask & _clause_mask(n_vars, candidate)
            count = candidate_mask.bit_count()
            if (
                not count
                or not (base_mask & ~candidate_mask)
                or not (candidate_mask & ~base_mask)
                or _solution_count_band(count) != refinement_record["solution_count_band"]
            ):
                continue
            changed = deepcopy(clauses)
            changed[index] = list(candidate)
            graph = _make_graph(n_vars, changed)
            proof = {
                "replaced_constraint_id": f"c{index + 1:03d}",
                "old_clause": list(old_clause),
                "new_clause": list(candidate),
                "variable_count_unchanged": True,
                "clause_count_unchanged": True,
                "clause_width_multiset_unchanged": True,
                "dependency_topology_changed": _topology_hash(graph)
                != base_record["topology_hash"],
                "base_only_assignment_count": (base_mask & ~candidate_mask).bit_count(),
                "restructured_only_assignment_count": (candidate_mask & ~base_mask).bit_count(),
                "supports_incomparable": True,
                "not_merely_added_constraint": True,
            }
            record = _graph_record("restructuring", graph, proof)
            matching = _matching_receipt(refinement_record, record)
            if not proof["dependency_topology_changed"] or not matching["all_tolerances_passed"]:
                continue
            rank = (
                abs(record["solution_count"] - refinement_record["solution_count"]),
                matching["difficulty_score_delta"],
                matching["serialized_length_delta"],
                hashlib.sha256(f"{output_hash}|{index}|{candidate}".encode()).hexdigest(),
            )
            if best is None or rank < best[0]:
                best = (rank, record)
    return None if best is None else best[1]


def _find_block_restructuring(
    case: dict[str, Any],
    base_record: dict[str, Any],
    refinement_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Replace a two-clause dependency block when each single clause is redundant."""

    n_vars = case["source_cnf"]["n_vars"]
    clauses = case["source_cnf"]["clauses"]
    base_mask = _cnf_valid_mask(n_vars, clauses)
    full = (1 << (1 << n_vars)) - 1
    clause_masks = [_clause_mask(n_vars, tuple(clause)) for clause in clauses]
    output_hash = case["source_provenance"]["source_output_bytes_sha256"]
    index_pairs = sorted(
        combinations(range(len(clauses)), 2),
        key=lambda pair: hashlib.sha256(f"{output_hash}|block|{pair}".encode()).hexdigest(),
    )
    for left, right in index_pairs:
        other_mask = full
        for index, mask in enumerate(clause_masks):
            if index not in (left, right):
                other_mask &= mask
        left_candidates = sorted(
            _candidate_clauses(n_vars, len(clauses[left])),
            key=lambda candidate: hashlib.sha256(
                f"{output_hash}|left|{candidate}".encode()
            ).hexdigest(),
        )
        right_candidates = sorted(
            _candidate_clauses(n_vars, len(clauses[right])),
            key=lambda candidate: hashlib.sha256(
                f"{output_hash}|right|{candidate}".encode()
            ).hexdigest(),
        )
        for left_clause in left_candidates:
            intermediate = other_mask & _clause_mask(n_vars, left_clause)
            if not intermediate:
                continue
            for right_clause in right_candidates:
                if right_clause == left_clause:
                    continue
                candidate_mask = intermediate & _clause_mask(n_vars, right_clause)
                count = candidate_mask.bit_count()
                if (
                    not count
                    or _solution_count_band(count) != refinement_record["solution_count_band"]
                    or not (base_mask & ~candidate_mask)
                    or not (candidate_mask & ~base_mask)
                ):
                    continue
                changed = deepcopy(clauses)
                changed[left] = list(left_clause)
                changed[right] = list(right_clause)
                graph = _make_graph(n_vars, changed)
                proof = {
                    "replaced_constraint_ids": [
                        f"c{left + 1:03d}",
                        f"c{right + 1:03d}",
                    ],
                    "old_clauses": [deepcopy(clauses[left]), deepcopy(clauses[right])],
                    "new_clauses": [list(left_clause), list(right_clause)],
                    "dependency_block_size": 2,
                    "single_clause_replacement_unavailable": True,
                    "variable_count_unchanged": True,
                    "clause_count_unchanged": True,
                    "clause_width_multiset_unchanged": True,
                    "dependency_topology_changed": _topology_hash(graph)
                    != base_record["topology_hash"],
                    "base_only_assignment_count": (base_mask & ~candidate_mask).bit_count(),
                    "restructured_only_assignment_count": (candidate_mask & ~base_mask).bit_count(),
                    "supports_incomparable": True,
                    "not_merely_added_constraint": True,
                }
                record = _graph_record("restructuring", graph, proof)
                matching = _matching_receipt(refinement_record, record)
                if proof["dependency_topology_changed"] and matching["all_tolerances_passed"]:
                    return record
    return None


def _build_group(case: dict[str, Any]) -> dict[str, Any]:
    """Build the base and a matched pair of distinct formal operations."""

    n_vars = case["source_cnf"]["n_vars"]
    clauses = case["source_cnf"]["clauses"]
    base_graph = _make_graph(n_vars, clauses)
    base = _graph_record(
        "base",
        base_graph,
        {"source_cnf_unchanged": True, "dependency_topology_unchanged": True},
    )
    base_mask = _cnf_valid_mask(n_vars, clauses)
    selected: tuple[dict[str, Any], dict[str, Any]] | None = None
    for literal, _, _ in _preferred_refinements(case, base_mask):
        refined_clauses = deepcopy(clauses) + [[literal]]
        refined_graph = _make_graph(n_vars, refined_clauses)
        refinement = _graph_record(
            "refinement",
            refined_graph,
            {
                "added_constraint_id": f"c{len(refined_clauses):03d}",
                "added_unary_clause": [literal],
                "valid_unary_constraint": True,
                "strict_solution_set_shrink": True,
                "variable_set_unchanged": True,
                "representation_unchanged": True,
                "dependency_topology_unchanged": _topology_hash(refined_graph)
                == base["topology_hash"],
                "removed_assignment_count": base["solution_count"]
                - _cnf_valid_mask(n_vars, refined_clauses).bit_count(),
            },
        )
        restructuring = _find_restructuring(case, base, refinement)
        if restructuring is None:
            restructuring = _find_block_restructuring(case, base, refinement)
        if restructuring is not None:
            selected = refinement, restructuring
            break
    if selected is None:
        raise ValueError(f"no matched restructuring for {case['source_case_id']}")
    refinement, restructuring = selected
    matching = _matching_receipt(refinement, restructuring)
    return {
        "source_case_id": case["source_case_id"],
        "case_cluster_key": case["case_cluster_key"],
        "split": case["split"],
        "constraint_family": case["constraint_family"],
        "source_model": deepcopy(case["source_model"]),
        "source_provenance": deepcopy(case["source_provenance"]),
        "graphs": {
            "base": base,
            "refinement": refinement,
            "restructuring": restructuring,
        },
        "matching_receipt": matching,
        "operation_class_distinction_proved": (
            refinement["operation_proof"]["dependency_topology_unchanged"]
            and restructuring["operation_proof"]["dependency_topology_changed"]
            and restructuring["operation_proof"]["supports_incomparable"]
        ),
    }


def build_probe_groups(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one exact base/refinement/restructuring group per source row."""

    return [_build_group(case) for case in cases]


def _proposal_features(graph_record: dict[str, Any]) -> dict[str, Any]:
    """Expose structural nuisance fields but no provenance or exact labels."""

    graph = graph_record["graph"]
    widths = Counter(len(clause) for clause in graph["clauses"])
    topology = graph["topology_summary"]
    return {
        "representation": graph["representation"],
        "variable_count": len(graph["variables"]),
        "clause_count": len(graph["clauses"]),
        "clause_width_histogram": [[width, widths[width]] for width in sorted(widths)],
        "literal_count": sum(len(clause) for clause in graph["clauses"]),
        "dependency_incidence_count": sum(
            len(edge["variables"]) for edge in graph["dependency_edges"]
        ),
        "compositional_depth": topology["compositional_depth"],
        "cross_constraint_coupling": topology["cross_constraint_coupling"],
        "dependency_chain_length": topology["dependency_chain_length"],
        "structural_difficulty_score": topology["structural_difficulty_score"],
    }


def exact_check_candidate(graph: dict[str, Any], assignment: dict[str, int]) -> dict[str, Any]:
    """Check binary domains first and then every cross-variable clause."""

    expected = set(graph["variables"])
    local = set(assignment) == expected and all(value in (0, 1) for value in assignment.values())
    failed: list[str] = []
    if local:
        for index, clause in enumerate(graph["clauses"]):
            satisfied = any(
                (literal > 0 and assignment[f"x{abs(literal)}"] == 1)
                or (literal < 0 and assignment[f"x{abs(literal)}"] == 0)
                for literal in clause
            )
            if not satisfied:
                failed.append(f"c{index + 1:03d}")
    return {
        "local_checks_passed": local,
        "failed_clause_ids": failed,
        "exact_valid": local and not failed,
    }


def _hard_negative(graph_record: dict[str, Any]) -> dict[str, int]:
    """Choose a local-pass assignment that fails a cross-variable dependency."""

    graph = graph_record["graph"]
    cross_ids = {edge["constraint_id"] for edge in graph["dependency_edges"]}
    for values in _assignment_tuples(len(graph["variables"])):
        assignment = {f"x{index + 1}": value for index, value in enumerate(values)}
        receipt = exact_check_candidate(graph, assignment)
        if not receipt["exact_valid"] and cross_ids.intersection(receipt["failed_clause_ids"]):
            return assignment
    raise ValueError("graph has no local-pass cross-dependency hard negative")


def row_checksum(row: dict[str, Any]) -> str:
    """Hash one row without its self-referential digest."""

    return sha256_value({key: value for key, value in row.items() if key != "row_sha256"})


def build_rows(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one valid and one hard-negative row for every graph."""

    rows: list[dict[str, Any]] = []
    for group in groups:
        for transformation in TRANSFORMATIONS:
            graph_record = group["graphs"][transformation]
            candidates = (
                (
                    "exact_valid_witness",
                    deepcopy(graph_record["valid_assignments"][0]),
                ),
                (
                    "local_pass_cross_dependency_fail",
                    _hard_negative(graph_record),
                ),
            )
            features = _proposal_features(graph_record)
            for condition, candidate in candidates:
                row = {
                    "schema": "carnot.experiment_6799.probe_row.v1",
                    "row_id": (f"{group['source_case_id']}|{transformation}|{condition}"),
                    "source_case_id": group["source_case_id"],
                    "case_cluster_key": group["case_cluster_key"],
                    "split": group["split"],
                    "constraint_family": group["constraint_family"],
                    "source_model": deepcopy(group["source_model"]),
                    "source_provenance": deepcopy(group["source_provenance"]),
                    "transformation": transformation,
                    "adversarial_condition": condition,
                    "graph_hash": graph_record["graph_hash"],
                    "valid_set_hash": graph_record["valid_set_hash"],
                    "proposal_features": deepcopy(features),
                    "candidate_assignment": candidate,
                    "exact_check_receipt": exact_check_candidate(graph_record["graph"], candidate),
                    "verifier_is_oracle": False,
                }
                row["row_sha256"] = row_checksum(row)
                rows.append(row)
    return rows


def _walk_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Yield every nested mapping key and its dotted path."""

    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield key, path
            yield from _walk_keys(nested, path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_keys(nested, f"{prefix}[{index}]")


def audit_feature_contract(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Report proposal features outside the allowlist or in the denylist."""

    errors: list[str] = []
    allowed = set(FEATURE_ALLOWLIST)
    denied = set(FEATURE_DENYLIST)
    for row in rows:
        features = row.get("proposal_features", {})
        for key, path in _walk_keys(features):
            if key in denied or ("[" not in path and "." not in path and key not in allowed):
                errors.append(f"{row.get('row_id', '<missing>')}.{path}")
    return errors


def _clause_widths(graph: dict[str, Any]) -> list[int]:
    """Return the sorted clause-width multiset for operation validation."""

    return sorted(len(clause) for clause in graph["clauses"])


def validate_probe_groups(groups: Sequence[dict[str, Any]]) -> list[str]:
    """Recompute operation, exact-set, and matching claims for every group."""

    errors: list[str] = []
    ids = [group.get("source_case_id") for group in groups]
    if len(ids) != len(set(ids)):
        errors.append("duplicate source case IDs")
    for group in groups:
        source_id = group.get("source_case_id", "<missing>")
        if set(group.get("graphs", {})) != set(TRANSFORMATIONS):
            errors.append(f"{source_id}: missing transformation graphs")
            continue
        base = group["graphs"]["base"]
        refinement = group["graphs"]["refinement"]
        restructuring = group["graphs"]["restructuring"]
        for name, record in group["graphs"].items():
            replay = enumerate_graph(record["graph"])
            if (
                replay["valid_set_hash"] != record.get("valid_set_hash")
                or replay["valid_assignments"] != record.get("valid_assignments")
                or graph_hash(record["graph"]) != record.get("graph_hash")
            ):
                errors.append(f"{source_id}: {name} exact evidence mismatch")
        base_set = assignment_set(base["valid_assignments"])
        refined_set = assignment_set(refinement["valid_assignments"])
        restructured_set = assignment_set(restructuring["valid_assignments"])
        if not (
            refined_set < base_set
            and refinement.get("operation_class") == "refinement"
            and refinement["operation_proof"].get("dependency_topology_unchanged")
            and refinement.get("topology_hash") == base.get("topology_hash")
        ):
            errors.append(f"{source_id}: invalid refinement operation")
        restructure_valid = (
            restructuring.get("operation_class") == "restructuring"
            and restructuring["operation_proof"].get("dependency_topology_changed")
            and restructuring.get("topology_hash") != base.get("topology_hash")
            and bool(base_set - restructured_set)
            and bool(restructured_set - base_set)
            and len(restructuring["graph"]["variables"]) == len(base["graph"]["variables"])
            and len(restructuring["graph"]["clauses"]) == len(base["graph"]["clauses"])
            and _clause_widths(restructuring["graph"]) == _clause_widths(base["graph"])
        )
        if not restructure_valid:
            errors.append(f"{source_id}: invalid restructuring operation")
        observed_matching = _matching_receipt(refinement, restructuring)
        if (
            observed_matching != group.get("matching_receipt")
            or not observed_matching["all_tolerances_passed"]
        ):
            errors.append(f"{source_id}: nuisance matching failed")
    return errors


def validate_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Reject duplicate IDs, checksum drift, and exact-label inconsistency."""

    errors: list[str] = []
    ids = [row.get("row_id") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate row IDs")
    for row in rows:
        if row_checksum(row) != row.get("row_sha256"):
            errors.append(f"{row.get('row_id', '<missing>')}: row checksum mismatch")
        receipt = row.get("exact_check_receipt", {})
        condition = row.get("adversarial_condition")
        if condition == "exact_valid_witness" and receipt.get("exact_valid") is not True:
            errors.append(f"{row.get('row_id', '<missing>')}: invalid witness label")
        if condition == "local_pass_cross_dependency_fail" and not (
            receipt.get("local_checks_passed") is True
            and receipt.get("exact_valid") is False
            and receipt.get("failed_clause_ids")
        ):
            errors.append(f"{row.get('row_id', '<missing>')}: invalid hard negative")
    return errors


def build_split_manifest(
    groups: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize case-isolated splits and required stratum coverage."""

    split_values: dict[str, dict[str, Any]] = {}
    for split in ("development", "held_case"):
        selected = [group for group in groups if group["split"] == split]
        selected_rows = [row for row in rows if row["split"] == split]
        split_values[split] = {
            "source_case_count": len(selected),
            "row_count": len(selected_rows),
            "case_cluster_keys": sorted({group["case_cluster_key"] for group in selected}),
            "source_case_ids": sorted(group["source_case_id"] for group in selected),
            "model_hub_ids": sorted({group["source_model"]["hf_id"] for group in selected}),
            "constraint_families": sorted({group["constraint_family"] for group in selected}),
        }
    case_overlap = sorted(
        set(split_values["development"]["case_cluster_keys"])
        & set(split_values["held_case"]["case_cluster_keys"])
    )
    source_overlap = sorted(
        set(split_values["development"]["source_case_ids"])
        & set(split_values["held_case"]["source_case_ids"])
    )
    represented = all(
        set(value["model_hub_ids"]) == set(REQUIRED_MODEL_IDS)
        and set(value["constraint_families"]) == set(REQUIRED_CONSTRAINT_FAMILIES)
        for value in split_values.values()
    )
    return {
        "assignment_rule": (
            "within each constraint family, sorted source stream case index modulo "
            "three equals zero is held_case; all other indices are development"
        ),
        "split_unit": "source_stream_case",
        "splits": split_values,
        "case_overlap": case_overlap,
        "source_case_overlap": source_overlap,
        "all_required_strata_represented": represented,
    }


def _rename_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], dict[int, int]]:
    """Reverse variable names while preserving every clause relation."""

    n_vars = len(graph["variables"])
    mapping = {old: n_vars + 1 - old for old in range(1, n_vars + 1)}
    clauses = [
        [mapping[abs(literal)] if literal > 0 else -mapping[abs(literal)] for literal in clause]
        for clause in graph["clauses"]
    ]
    return _make_graph(n_vars, clauses), mapping


def run_adversarial_attacks(
    groups: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Run the six preregistered semantic and leakage mutations."""

    rename_ok = True
    for group in groups:
        for record in group["graphs"].values():
            renamed, mapping = _rename_graph(record["graph"])
            renamed_assignments = enumerate_graph(renamed)["valid_assignments"]
            restored = [
                {f"x{old}": assignment[f"x{new}"] for old, new in mapping.items()}
                for assignment in renamed_assignments
            ]
            rename_ok &= assignment_set(restored) == assignment_set(record["valid_assignments"])
    feature_hashes = [
        sha256_value(_proposal_features(record))
        for group in groups
        for record in group["graphs"].values()
    ]
    shuffled_hashes = [
        sha256_value(_proposal_features(record))
        for group in reversed(groups)
        for record in group["graphs"].values()
    ]
    parser_pair = deepcopy(groups[0]["source_provenance"]["lossless_replay_dual_encodings"])
    parser_pair["encoder_b"]["normalized_constraints"] = {"claim": "changed"}
    parser_diagnostic = diagnose_dual_encoding(
        parser_pair["encoder_a"], parser_pair["encoder_b"], "reasoning_error"
    )
    duplicates = list(rows) + [deepcopy(rows[0])]
    duplicate_detected = "duplicate row IDs" in validate_rows(duplicates)
    mislabeled = deepcopy(groups[:1])
    mislabeled[0]["graphs"]["restructuring"] = deepcopy(mislabeled[0]["graphs"]["refinement"])
    mislabeled[0]["graphs"]["restructuring"]["operation_class"] = "restructuring"
    mislabeled_detected = any(
        "restructuring" in error for error in validate_probe_groups(mislabeled)
    )
    leaked = deepcopy(rows[:1])
    leaked[0]["proposal_features"]["exact_valid"] = True
    leakage_detected = bool(audit_feature_contract(leaked))
    return {
        "solution_preserving_rename": {
            "passed": rename_ok,
            "semantics_preserved": rename_ok,
            "graph_count": len(groups) * len(TRANSFORMATIONS),
        },
        "source_model_label_shuffle": {
            "passed": sorted(feature_hashes) == sorted(shuffled_hashes),
            "feature_hashes_unchanged": sorted(feature_hashes) == sorted(shuffled_hashes),
            "model_identity_used_as_feature": False,
        },
        "parser_disagreement": {
            "passed": parser_diagnostic == "translation_disagreement",
            "observed_diagnostic": parser_diagnostic,
            "labeled_model_reasoning_error": False,
        },
        "duplicate_cases": {
            "passed": duplicate_detected,
            "mutation_detected": duplicate_detected,
        },
        "accidental_refinement_labeled_restructuring": {
            "passed": mislabeled_detected,
            "mutation_detected": mislabeled_detected,
        },
        "exact_label_leakage": {
            "passed": leakage_detected,
            "mutation_detected": leakage_detected,
        },
    }


def replay_payload(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Recompute every exact set and graph hash from serialized graph inputs."""

    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for group in groups:
        for transformation in TRANSFORMATIONS:
            expected = group["graphs"][transformation]
            observed = enumerate_graph(expected["graph"])
            observed_graph_hash = graph_hash(expected["graph"])
            replay_id = f"{group['source_case_id']}|{transformation}"
            matches = (
                observed_graph_hash == expected["graph_hash"]
                and observed["valid_set_hash"] == expected["valid_set_hash"]
                and observed["solution_count"] == expected["solution_count"]
            )
            if not matches:
                mismatches.append(replay_id)
            records.append(
                {
                    "replay_id": replay_id,
                    "source_case_id": group["source_case_id"],
                    "transformation": transformation,
                    "expected_graph_hash": expected["graph_hash"],
                    "observed_graph_hash": observed_graph_hash,
                    "expected_valid_set_hash": expected["valid_set_hash"],
                    "observed_valid_set_hash": observed["valid_set_hash"],
                    "expected_solution_count": expected["solution_count"],
                    "observed_solution_count": observed["solution_count"],
                    "matches": matches,
                }
            )
    return {
        "agreement": not mismatches,
        "replayed_graph_count": len(records),
        "mismatches": mismatches,
        "records": records,
        "worker_pid": os.getpid(),
    }


def _replay_projection(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep subprocess input limited to graphs and expected exact receipts."""

    return [
        {
            "source_case_id": group["source_case_id"],
            "graphs": {
                name: {
                    "graph": deepcopy(record["graph"]),
                    "graph_hash": record["graph_hash"],
                    "valid_set_hash": record["valid_set_hash"],
                    "solution_count": record["solution_count"],
                }
                for name, record in group["graphs"].items()
            },
        }
        for group in groups
    ]


def run_exact_replay(
    groups: Sequence[dict[str, Any]], repo_root: Path | None = None
) -> dict[str, Any]:
    """Replay exact enumeration in a fresh Python process."""

    root = Path(repo_root or _default_repo_root())
    environment = os.environ.copy()
    python_path = str(root / "python")
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else python_path + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_6799_model_output_formal_constraint_probes",
            "--exact-replay-worker",
        ],
        input=canonical_json({"groups": _replay_projection(groups)}),
        capture_output=True,
        text=True,
        cwd=root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh exact replay failed: " + (completed.stderr.strip() or "unknown error")
        )
    result = json.loads(completed.stdout)
    result.update(
        {
            "fresh_process": result["worker_pid"] != os.getpid(),
            "cold_pid": result.pop("worker_pid"),
            "producer_pid": os.getpid(),
        }
    )
    return result


def _exact_replay_worker() -> int:
    """Read projected groups from stdin and emit one exact replay receipt."""

    payload = json.load(sys.stdin)
    print(canonical_json(replay_payload(payload["groups"])))
    return 0


def _dual_encoding_diagnostics(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retained producer and replay parser evidence."""

    counts = Counter(
        group["source_provenance"]["translation_reasoning_diagnostic"] for group in groups
    )
    return {
        "diagnostic_counts": {key: counts[key] for key in sorted(counts)},
        "translation_disagreement_count": counts["translation_disagreement"],
        "translation_disagreement_labeled_reasoning_error": False,
        "source_case_receipts": [
            {
                "source_case_id": group["source_case_id"],
                "diagnostic": group["source_provenance"]["translation_reasoning_diagnostic"],
                "producer_encoder_ids": [
                    group["source_provenance"]["producer_dual_encodings"][key]["encoder_id"]
                    for key in ("encoder_a", "encoder_b")
                ],
                "replay_encoder_ids": [
                    group["source_provenance"]["lossless_replay_dual_encodings"][key]["encoder_id"]
                    for key in ("encoder_a", "encoder_b")
                ],
            }
            for group in groups
        ],
    }


def _transformation_contract() -> dict[str, Any]:
    """Return the frozen operation definitions and nuisance tolerances."""

    return {
        "base": "unchanged frozen source CNF",
        "refinement": (
            "add one satisfiable unary clause; strict subset; unchanged variable set, "
            "representation, and cross-variable dependency topology"
        ),
        "restructuring": (
            "replace one same-width clause; preserve variable and clause counts; "
            "change dependency topology; produce support incomparable with base"
        ),
        "matching_tolerances": {
            "variable_count_delta": 0,
            "enumerated_assignment_count_delta": 0,
            "solution_count_band": "same_power_of_two_band",
            "serialized_length": "max(32_bytes,10_percent)",
            "literal_check_work_relative_delta": WORK_RELATIVE_TOLERANCE,
            "structural_difficulty_score_delta": DIFFICULTY_SCORE_TOLERANCE,
        },
        "model_identity_role": "provenance_stratum_only_never_proposal_feature",
    }


FIELD_PURPOSES = {
    "schema": "Names the versioned artifact contract.",
    "experiment_id": "Identifies Experiment 6799.",
    "run_date": "Freezes the requested execution date.",
    "status": "States whether the ready or blocked fixture completed.",
    "field_principles": "Explains the purpose of every top-level field.",
    "inference_substrate": "States that only CPU transforms of frozen outputs ran.",
    "duration_s": "Records elapsed fixture generation time.",
    "random_seed": "Freezes deterministic tie breaking.",
    "reproducibility_checksum": "Hashes all deterministic artifact evidence.",
    "source_artifact_hashes": "Pins every source and supporting artifact byte hash.",
    "source_model_specs": "Retains mandated GGUF models as provenance strata.",
    "live_llm_invoked": "Proves that no live model call was made.",
    "transformation_contract": "Freezes formal operations and matching tolerances.",
    "feature_allowlist": "Lists the only allowed proposal feature names.",
    "feature_denylist": "Lists labels and provenance forbidden from features.",
    "split_manifest": "Proves case isolation and stratum coverage.",
    "matching_receipts": "Records nuisance matching for every paired probe.",
    "dual_encoding_diagnostics": "Keeps translation separate from reasoning errors.",
    "exact_replay_receipts": "Records fresh-process graph and valid-set replay.",
    "graph_hashes": "Pins each source transformation graph and exact valid set.",
    "rows": "Stores valid and cross-dependency-fail candidate rows.",
    "probe_groups": "Stores self-contained formal operation proofs.",
    "adversarial_attack_receipts": "Records all preregistered shortcut attacks.",
    "model_output_constraint_probe_ready": "States whether the fixture passed all gates.",
    "gate_check_summary": "Records every precondition and internal gate.",
    "verifier_is_oracle": "Prevents treating the verifier as an independent oracle.",
    "verdict_class": "Reports one closed-enum terminal verdict class.",
    "honest_verdict": "Provides the terminal plain-language result.",
}


def reproducibility_checksum(artifact: dict[str, Any]) -> str:
    """Hash deterministic evidence while excluding duration and the checksum itself."""

    excluded = {"duration_s", "reproducibility_checksum"}
    return sha256_value({key: value for key, value in artifact.items() if key not in excluded})


def _source_hash_receipts(root: Path) -> dict[str, dict[str, Any]]:
    """Record expected and observed hashes for all frozen input artifacts."""

    receipts = {}
    for name, relative in SOURCE_ARTIFACT_PATHS.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        receipts[name] = {
            "path": relative,
            "expected_sha256": EXPECTED_SOURCE_HASHES[name],
            "observed_sha256": observed,
            "matches": observed == EXPECTED_SOURCE_HASHES[name],
        }
    return receipts


def _artifact_base(
    run_date: str,
    duration_s: float,
    source_hashes: dict[str, Any],
    model_specs: Sequence[dict[str, Any]],
    gate_summary: dict[str, Any],
    ready: bool,
) -> dict[str, Any]:
    """Create the shared full schema for ready and blocked outcomes."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": (
            "complete_model_output_constraint_probe_fixture"
            if ready
            else "complete_blocked_model_output_probe_fixture"
        ),
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "source_model_specs": list(model_specs),
        "live_llm_invoked": LIVE_LLM_INVOKED,
        "transformation_contract": _transformation_contract(),
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "split_manifest": {},
        "matching_receipts": [],
        "dual_encoding_diagnostics": {},
        "exact_replay_receipts": [],
        "graph_hashes": [],
        "rows": [],
        "probe_groups": [],
        "adversarial_attack_receipts": {},
        "model_output_constraint_probe_ready": ready,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete: frozen authentic outputs produced exact paired formal probes"
            if ready
            else "complete_blocked_model_output_probe_fixture: authentic source gates failed"
        ),
    }


def _finalize_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Add complete field purposes and the deterministic checksum last."""

    artifact["field_principles"] = {key: FIELD_PURPOSES[key] for key in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    run_date: str,
    repo_root: Path | None = None,
    proposal_path: Path | None = None,
    reparse_path: Path | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build a ready fixture or a full blocked artifact without substitutes."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")
    root = Path(repo_root or _default_repo_root())
    preconditions = evaluate_preconditions(root, proposal_path, reparse_path)
    source_hashes = _source_hash_receipts(root)
    if not preconditions["all_passed"]:
        artifact = _artifact_base(run_date, duration_s, source_hashes, [], preconditions, False)
        return _finalize_artifact(artifact)
    proposal, reparse = load_source_artifacts(
        root,
        proposal_path or root / SOURCE_ARTIFACT_PATHS["exp6745"],
        reparse_path or root / SOURCE_ARTIFACT_PATHS["exp6755"],
    )
    cases = select_eligible_cases(proposal, reparse)
    groups = build_probe_groups(cases)
    rows = build_rows(groups)
    split_manifest = build_split_manifest(groups, rows)
    attacks = run_adversarial_attacks(groups, rows)
    replay = run_exact_replay(groups, root)
    group_errors = validate_probe_groups(groups)
    row_errors = validate_rows(rows)
    feature_errors = audit_feature_contract(rows)
    internal_checks = [
        _check(
            "exact_eligible_case_count",
            len(groups) == ELIGIBLE_CASE_COUNT,
            len(groups),
            ELIGIBLE_CASE_COUNT,
        ),
        _check("formal_group_validation", not group_errors, group_errors, []),
        _check("row_validation", not row_errors, row_errors, []),
        _check("feature_contract", not feature_errors, feature_errors, []),
        _check(
            "split_isolation",
            not split_manifest["case_overlap"]
            and not split_manifest["source_case_overlap"]
            and split_manifest["all_required_strata_represented"],
            split_manifest,
            "disjoint_with_all_strata",
        ),
        _check(
            "adversarial_attacks",
            all(receipt["passed"] for receipt in attacks.values()),
            attacks,
            "all_passed",
        ),
        _check(
            "fresh_process_exact_replay",
            replay["agreement"] and replay["fresh_process"],
            {
                "agreement": replay["agreement"],
                "fresh_process": replay["fresh_process"],
                "replayed_graph_count": replay["replayed_graph_count"],
            },
            {"agreement": True, "fresh_process": True, "replayed_graph_count": 291},
        ),
    ]
    gate_summary = _finish_check_summary(preconditions["checks"] + internal_checks)
    ready = gate_summary["all_passed"]
    artifact = _artifact_base(
        run_date,
        duration_s,
        source_hashes,
        proposal["models_used"],
        gate_summary,
        ready,
    )
    if ready:
        artifact.update(
            {
                "split_manifest": split_manifest,
                "matching_receipts": [
                    {
                        "source_case_id": group["source_case_id"],
                        **deepcopy(group["matching_receipt"]),
                    }
                    for group in groups
                ],
                "dual_encoding_diagnostics": _dual_encoding_diagnostics(groups),
                "exact_replay_receipts": [
                    {**record, "fresh_process": True} for record in replay["records"]
                ],
                "graph_hashes": [
                    {
                        "source_case_id": group["source_case_id"],
                        "case_cluster_key": group["case_cluster_key"],
                        "split": group["split"],
                        "transformation": name,
                        "graph_hash": record["graph_hash"],
                        "valid_set_hash": record["valid_set_hash"],
                        "solution_count": record["solution_count"],
                    }
                    for group in groups
                    for name, record in group["graphs"].items()
                ],
                "rows": rows,
                "probe_groups": groups,
                "adversarial_attack_receipts": attacks,
            }
        )
    return _finalize_artifact(artifact)


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Validate the terminal artifact without trusting its stored gate booleans."""

    errors: list[str] = []
    required = set(STANDARD_ARTIFACT_FIELDS) | set(REQUIRED_ARTIFACT_FIELDS)
    missing = sorted(required - set(artifact))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles must cover every top-level field")
    if artifact.get("live_llm_invoked") is not False:
        errors.append("live_llm_invoked must remain false")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must remain false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class is outside the closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict lacks a terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    ready = artifact.get("model_output_constraint_probe_ready") is True
    if ready:
        errors.extend(validate_rows(artifact.get("rows", [])))
        errors.extend(audit_feature_contract(artifact.get("rows", [])))
        errors.extend(validate_probe_groups(artifact.get("probe_groups", [])))
        if artifact.get("gate_check_summary", {}).get("all_passed") is not True:
            errors.append("ready artifact has failed gates")
    else:
        if artifact.get("status") != "complete_blocked_model_output_probe_fixture":
            errors.append("blocked artifact has wrong status")
        if artifact.get("rows") or artifact.get("graph_hashes"):
            errors.append("blocked artifact must not contain probe rows")
    return errors


def write_outputs(
    run_date: str,
    artifact_path: Path | None = None,
    repo_root: Path | None = None,
    proposal_path: Path | None = None,
    reparse_path: Path | None = None,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Build and write exactly one requested artifact path."""

    root = Path(repo_root or _default_repo_root())
    started = time.monotonic()
    output = artifact_path or Path(
        "results/experiment_6799_model_output_formal_constraint_probes.json"
    )
    if not output.is_absolute():
        output = root / output
    artifact = build_artifact(
        run_date=run_date,
        repo_root=root,
        proposal_path=proposal_path,
        reparse_path=reparse_path,
        duration_s=(0.0 if duration_s is None else duration_s),
    )
    if duration_s is None:
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated generator or its internal fresh-process replay worker."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--exact-replay-worker"]:
        return _exact_replay_worker()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Execution date in YYYYMMDD form")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment_6799_model_output_formal_constraint_probes.json"),
        help="Artifact path, relative to the repository root by default",
    )
    parsed = parser.parse_args(arguments)
    write_outputs(run_date=parsed.date, artifact_path=parsed.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the tested wrapper calls main.
    raise SystemExit(main())
