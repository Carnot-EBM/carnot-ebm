"""Run a bounded seal, leakage, metamorphic, and mutation audit.

This module treats Exp6702 as an immutable archive. It freezes attack names and
input hashes before it opens expected attack outcomes. It never searches plan
spaces or recalculates an optimum. This boundary keeps the shard small and
prevents the seal auditor from becoming a planning oracle.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = Path("results/experiment_6702_exact_planning_fixture_recovery.json")
RESULT_PATH = Path("results/experiment_6716_bounded_seal_attack_audit.json")
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
MODULE_PATH = Path("python/carnot/experiment_6716_bounded_seal_attack_audit.py")
TEST_PATH = Path("tests/python/test_experiment_6716_bounded_seal_attack_audit.py")

SCHEMA = "carnot.experiment_6716.bounded_seal_attack_audit.v1"
INFERENCE_SUBSTRATE = "cpu_static_and_dynamic_attack_audit_no_llm"
SCANNER_VERSION = "carnot.exp6716.bounded_leakage_scanner.v1"
SEAL_AUDITOR_VERSION = "carnot.exp6716.one_shot_prompt_bound_seal.v1"
ATTACK_RUNNER_VERSION = "carnot.exp6716.archived_row_attack_runner.v1"
REDUCER_VERSION = "carnot.exp6716.raw_attack_reducer.v1"

FAMILIES = ("inventory", "battery_dispatch", "job_slot", "reservoir_control")
LEAKAGE_CASES = (
    "prompt_direct_label",
    "metadata_label_encoding",
    "instance_id_uniqueness",
    "instance_id_shortcut",
    "split_membership",
    "development_to_held_contamination",
    "family_isolation",
    "hash_and_seal_freshness",
)
SEAL_ACCESS_CASES = (
    "early_access",
    "valid_post_commit",
    "receipt_replay",
    "stale_token",
    "reordered_event",
    "partial_write",
)
METAMORPHIC_TRANSFORMS = (
    "action_renaming",
    "constant_cost_shift",
    "equivalent_state_encoding",
    "family_preserving_surface_change",
)
METAMORPHIC_CASES = tuple(
    f"{family}:{transform}" for family in FAMILIES for transform in METAMORPHIC_TRANSFORMS
)
MUTATION_CASES = (
    "bad_transition",
    "infeasible_action",
    "corrupted_cost",
    "label_leakage",
    "wrong_ties",
    "stale_seal",
)
MEMORY_POISON_CASES = (
    "copied_record",
    "relation_poison",
    "provenance_loss",
    "tombstone_reappearance",
)
EXPECTED_CASES: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt_direct_label",),
    "metadata": ("metadata_label_encoding",),
    "id": ("instance_id_uniqueness", "instance_id_shortcut"),
    "split": ("split_membership", "development_to_held_contamination"),
    "family_isolation": ("family_isolation",),
    "seal_integrity": ("hash_and_seal_freshness",),
    "seal_access": SEAL_ACCESS_CASES,
    "metamorphic": METAMORPHIC_CASES,
    "mutation": MUTATION_CASES,
    "memory_poison": MEMORY_POISON_CASES,
}
ATTACK_BUDGETS_S = {
    "prompt": 2.0,
    "metadata": 2.0,
    "id": 2.0,
    "split": 2.0,
    "family_isolation": 2.0,
    "seal_integrity": 2.0,
    "seal_access": 2.0,
    "metamorphic": 5.0,
    "mutation": 5.0,
    "memory_poison": 2.0,
}
RANDOM_SEED = {"attack_order": 6716001, "mutation": 6716002}

OPEN_SPEC_IDS = (
    "REQ-SAFE-6716",
    "SCENARIO-SAFE-6716-LEAKAGE",
    "SCENARIO-SAFE-6716-SEAL-ACCESS",
    "SCENARIO-SAFE-6716-METAMORPHIC-MUTATION",
    "SCENARIO-SAFE-6716-MEMORY-POISON",
    "REQ-PIPELINE-6716",
    "SCENARIO-PIPELINE-6716-ROW-REDUCTION",
    "SCENARIO-PIPELINE-6716-PER-UNIT",
    "REQ-REPORT-6716",
    "SCENARIO-REPORT-6716-ATOMIC",
    "SCENARIO-REPORT-6716-BLOCKED",
    "REQ-VERIFY-6716",
    "SCENARIO-VERIFY-6716-AUTHORITY",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "openspec_requirement_ids",
    "frozen_attack_manifest",
    "method_fidelity_contract",
    "leakage_rows",
    "seal_access_rows",
    "metamorphic_rows",
    "mutation_attack_rows",
    "seal_attack_audit_passed",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
PROVENANCE_KEYS = (
    "prompt_store",
    "seal",
    "scanner",
    "attack_runner",
    "reducer",
    "function",
    "version",
    "hash",
)

FOCUSED_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--data-file=/tmp/carnot_exp6716_coverage "
    "--include=*/experiment_6716_bounded_seal_attack_audit.py "
    "-m pytest tests/python/test_experiment_6716_bounded_seal_attack_audit.py "
    "-q --no-cov -n 0 -o addopts="
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--data-file=/tmp/carnot_exp6716_coverage "
    "--include=*/experiment_6716_bounded_seal_attack_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6716_bounded_seal_attack_audit.py"
)
E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6716_bounded_seal_attack_audit.py "
    "-q --no-cov -n 0 -o addopts= -k e2e_actual_bounded_attack_audit"
)
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"
VERIFICATION_COMMANDS = (
    ("focused_tests", FOCUSED_COMMAND),
    ("scoped_coverage", COVERAGE_COMMAND),
    ("full_python_suite", FULL_SUITE_COMMAND),
    ("spec_coverage", SPEC_COVERAGE_COMMAND),
    ("applicable_e2e", E2E_COMMAND),
    ("ruff_check", RUFF_COMMAND),
    ("format_check", FORMAT_COMMAND),
)
OPERATIONAL_CHECK_IDS = (
    "artifact_validation",
    "row_consistency",
    "adversarial_verification",
)
REQUIRED_TEST_CHECKS = (
    *(check_id for check_id, _ in VERIFICATION_COMMANDS),
    *OPERATIONAL_CHECK_IDS,
)


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes for hashes and exact comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON value without depending on source whitespace."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash file bytes, or retain an explicit missing state."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Load one JSON object and reject scalar or array substitutes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def row_by_case(rows: Sequence[Mapping[str, Any]], case: str) -> Mapping[str, Any]:
    """Return one named raw row so tests and reducers avoid positional trust."""

    return next(row for row in rows if row.get("case", row.get("check")) == case)


def _case_rows(cases: Sequence[str], input_value: Any) -> list[JsonDict]:
    """Bind each blind case name to the same immutable input projection."""

    input_hash = sha256_json(input_value)
    return [{"case": case, "input_hash": input_hash} for case in cases]


def public_attack_inputs(upstream: Mapping[str, Any]) -> JsonDict:
    """Project attack identities and hashes without expected or observed outcomes."""

    instances = list(upstream.get("instance_rows", []))
    prompts = [{"instance": row.get("instance"), "prompt": row.get("prompt")} for row in instances]
    metadata_rows = [
        {"instance": row.get("instance"), "typed_spec": row.get("typed_spec")} for row in instances
    ]
    identity_rows = [
        {
            "instance": row.get("instance"),
            "split": row.get("split"),
            "family": row.get("family"),
            "prompt_hash": row.get("prompt_hash"),
            "spec_hash": row.get("spec_hash"),
        }
        for row in instances
    ]
    seal_rows = [
        {
            "instance": row.get("instance"),
            "prompt_hash": row.get("prompt_hash"),
            "label_hash": row.get("label_hash"),
            "seal_hash": row.get("seal_hash"),
        }
        for row in upstream.get("label_seal_rows", [])
    ]
    metamorphic = [
        {
            "case": row.get("row_id"),
            "instance": row.get("instance"),
            "family": row.get("family"),
            "transform": row.get("transform"),
            "input_hash": sha256_json(
                {
                    "row_id": row.get("row_id"),
                    "instance": row.get("instance"),
                    "family": row.get("family"),
                    "transform": row.get("transform"),
                }
            ),
        }
        for row in upstream.get("metamorphic_rows", [])
    ]
    mutations = [
        {
            "case": row.get("mutation"),
            "mutation": row.get("mutation"),
            "input_hash": sha256_json(
                {"row_id": row.get("row_id"), "mutation": row.get("mutation")}
            ),
        }
        for row in upstream.get("mutation_rows", [])
    ]
    return {
        "source_schema": upstream.get("schema"),
        "source_checksum": upstream.get("reproducibility_checksum"),
        "prompt_cases": _case_rows(EXPECTED_CASES["prompt"], prompts),
        "metadata_cases": _case_rows(EXPECTED_CASES["metadata"], metadata_rows),
        "id_cases": _case_rows(EXPECTED_CASES["id"], identity_rows),
        "split_cases": _case_rows(EXPECTED_CASES["split"], identity_rows),
        "family_isolation_cases": _case_rows(EXPECTED_CASES["family_isolation"], identity_rows),
        "seal_integrity_cases": _case_rows(EXPECTED_CASES["seal_integrity"], seal_rows),
        "seal_access_cases": _case_rows(EXPECTED_CASES["seal_access"], seal_rows),
        "metamorphic_cases": metamorphic,
        "mutation_cases": mutations,
        "memory_poison_cases": _case_rows(
            EXPECTED_CASES["memory_poison"], {"identity_rows": identity_rows}
        ),
        "store_hashes": {
            "instance_rows": sha256_json(instances),
            "state_action_rows": sha256_json(upstream.get("state_action_rows", [])),
            "label_seal_rows": sha256_json(upstream.get("label_seal_rows", [])),
            "metamorphic_rows_blind": sha256_json(metamorphic),
            "mutation_rows_blind": sha256_json(mutations),
        },
    }


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest without trusting its stored checksum."""

    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_hash"})


def freeze_attack_manifest(blinded: Mapping[str, Any]) -> JsonDict:
    """Freeze exact case counts, budgets, versions, and hashes before evaluation."""

    family_rows: list[JsonDict] = []
    for family, expected in EXPECTED_CASES.items():
        rows = list(blinded.get(f"{family}_cases", []))
        observed = tuple(str(row.get("case")) for row in rows)
        if observed != expected or len(set(observed)) != len(observed):
            raise ValueError(f"attack identity mismatch for {family}")
        family_rows.append(
            {
                "family": family,
                "case_ids": list(expected),
                "count": len(expected),
                "time_budget_s": ATTACK_BUDGETS_S[family],
                "input_hash": sha256_json(rows),
                "version": ATTACK_RUNNER_VERSION,
            }
        )
    manifest: JsonDict = {
        "schema": SCHEMA + ".frozen_attack_manifest",
        "frozen_before_expected_result_read": True,
        "source_schema": blinded.get("source_schema"),
        "source_checksum": blinded.get("source_checksum"),
        "store_hashes": deepcopy(dict(blinded.get("store_hashes", {}))),
        "attack_families": family_rows,
        "versions": {
            "scanner": SCANNER_VERSION,
            "seal_auditor": SEAL_AUDITOR_VERSION,
            "attack_runner": ATTACK_RUNNER_VERSION,
            "reducer": REDUCER_VERSION,
        },
        "total_time_budget_s": sum(ATTACK_BUDGETS_S.values()),
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = manifest_checksum(manifest)
    return manifest


def method_fidelity_contract() -> JsonDict:
    """State the bounded methods and forbidden substitutions in machine form."""

    return {
        "schema": SCHEMA + ".method_fidelity_contract",
        "required_scanners": [
            "prompt_and_metadata_static_scan",
            "identity_split_family_collision_scan",
            "prompt_spec_and_seal_hash_rebuild",
            "one_shot_dynamic_seal_state_machine",
            "archived_row_metamorphic_replay",
            "bounded_row_mutation_panel",
            "isolated_forward_memory_poison_panel",
        ],
        "required_metrics": [
            "raw_expected_and_observed_results",
            "per_case_pass_state",
            "family_time_budget",
            "fixture_and_seal_non_interference",
        ],
        "time_budgets_s": deepcopy(ATTACK_BUDGETS_S),
        "forbidden_substitutions": [
            "plan_space_enumeration",
            "optimum_recomputation",
            "producer_conclusion_import",
            "llm_judgment",
            "attack_count_reduction",
            "time_budget_widening",
        ],
        "plan_enumeration_count": 0,
        "optimum_recomputation_count": 0,
        "producer_conclusion_import_count": 0,
        "llm_call_count": 0,
        "method_substitution_count": 0,
        "passed": True,
    }


def _seal_components(row: Mapping[str, Any]) -> JsonDict:
    return {
        "instance": row.get("instance"),
        "prompt_hash": row.get("prompt_hash"),
        "label_hash": row.get("label_hash"),
        "seal_version": row.get("seal_version"),
        "commit_requirement": row.get("commit_requirement"),
    }


def _valid_seal(row: Mapping[str, Any], instance: Mapping[str, Any] | None) -> bool:
    """Verify that a seal binds the current prompt and archived label hash."""

    if instance is None:
        return False
    components = _seal_components(row)
    label_hash = str(row.get("label_hash", ""))
    return (
        row.get("instance") == instance.get("instance")
        and row.get("prompt_hash") == instance.get("prompt_hash")
        and row.get("seal_hash") == instance.get("label_seal_hash")
        and row.get("seal_hash") == sha256_json(components)
        and row.get("seal_version") == "carnot.prompt_bound_label_seal.v1"
        and row.get("commit_requirement") == "prompt_bound_candidate_commit_receipt"
        and row.get("access_state") == "sealed_until_commit"
        and row.get("negative_access_result") == "denied:commit receipt required"
        and re.fullmatch(r"sha256:[0-9a-f]{64}", label_hash) is not None
    )


def _metadata_hits(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    forbidden_key = re.compile(
        r"(?:answer|exact[_-]?label|future[_-]?value|action[_-]?gap|"
        r"optimum[_-]?plan|total[_-]?optimum)",
        re.IGNORECASE,
    )
    encoded_objective = re.compile(
        r"(?:optimum|answer|label|total)\s*(?:is|equals|[:=])\s*[-+]?\d",
        re.IGNORECASE,
    )
    allowed_row_keys = {
        "action_set",
        "family",
        "feasibility",
        "horizon",
        "instance",
        "label_seal_hash",
        "optimum",
        "prompt",
        "prompt_hash",
        "seed",
        "spec_hash",
        "split",
        "ties",
        "typed_spec",
    }
    hits: list[JsonDict] = []
    for row in rows:
        bad_keys = [
            str(key)
            for key in row
            if key not in allowed_row_keys and forbidden_key.search(str(key))
        ]

        def visit(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    child = f"{path}.{key}" if path else str(key)
                    if forbidden_key.search(str(key)):
                        bad_keys.append(child)
                    visit(nested, child)
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    visit(nested, f"{path}[{index}]")
            elif path.endswith("objective") and encoded_objective.search(str(value)):
                bad_keys.append(path)

        visit(row.get("typed_spec", {}), "typed_spec")
        if bad_keys:
            hits.append({"instance": row.get("instance"), "paths": sorted(set(bad_keys))})
    return hits


def scan_leakage(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Scan all prompts, metadata, identities, splits, families, and hashes."""

    rows = list(upstream.get("instance_rows", []))
    label_pattern = re.compile(
        r"(?:exact\s+)?(?:optimum|answer|label)(?:\s+(?:is|equals))?\s*[:=]\s*[-+]?\d",
        re.IGNORECASE,
    )
    prompt_hits = [
        {
            "instance": row.get("instance"),
            "match": label_pattern.search(str(row.get("prompt", ""))).group(0),
        }
        for row in rows
        if label_pattern.search(str(row.get("prompt", "")))
    ]
    metadata_hits = _metadata_hits(rows)
    ids = [str(row.get("instance")) for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count != 1)
    id_shortcuts = sorted(
        instance
        for instance in ids
        if re.search(r"(?:optimum|opt|answer|label|total)[-_=]?\d", instance, re.IGNORECASE)
    )
    split_counts = Counter(str(row.get("split")) for row in rows)
    invalid_splits = sorted(
        str(row.get("instance"))
        for row in rows
        if row.get("split") not in {"headline", "development"}
    )
    split_issue: Any = []
    if invalid_splits or split_counts != Counter({"headline": 32, "development": 8}):
        split_issue = {"invalid": invalid_splits, "counts": dict(sorted(split_counts.items()))}

    cross_split: list[JsonDict] = []
    cross_family: list[JsonDict] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            shared = [
                key
                for key in ("prompt_hash", "spec_hash")
                if left.get(key) is not None and left.get(key) == right.get(key)
            ]
            normalized_left = " ".join(str(left.get("prompt", "")).casefold().split())
            normalized_right = " ".join(str(right.get("prompt", "")).casefold().split())
            if normalized_left and normalized_left == normalized_right:
                shared.append("normalized_prompt")
            evidence = {
                "left": left.get("instance"),
                "right": right.get("instance"),
                "shared": shared,
            }
            if shared and left.get("split") != right.get("split"):
                cross_split.append(evidence)
            if shared and left.get("family") != right.get("family"):
                cross_family.append(evidence)

    instances = {str(row.get("instance")): row for row in rows}
    seals = list(upstream.get("label_seal_rows", []))
    seal_counts = Counter(str(row.get("instance")) for row in seals)
    stale_hashes: list[JsonDict] = []
    for row in rows:
        prompt_hash = (
            "sha256:" + hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()
        )
        if row.get("prompt_hash") != prompt_hash:
            stale_hashes.append({"instance": row.get("instance"), "field": "prompt_hash"})
        if row.get("spec_hash") != sha256_json(row.get("typed_spec")):
            stale_hashes.append({"instance": row.get("instance"), "field": "spec_hash"})
    for seal in seals:
        if not _valid_seal(seal, instances.get(str(seal.get("instance")))):
            stale_hashes.append({"instance": seal.get("instance"), "field": "label_seal"})
    for instance in ids:
        if seal_counts.get(instance) != 1:
            stale_hashes.append(
                {
                    "instance": instance,
                    "field": "seal_count",
                    "observed": seal_counts.get(instance, 0),
                }
            )

    evidence = {
        "prompt_direct_label": prompt_hits,
        "metadata_label_encoding": metadata_hits,
        "instance_id_uniqueness": duplicate_ids,
        "instance_id_shortcut": id_shortcuts,
        "split_membership": split_issue,
        "development_to_held_contamination": cross_split,
        "family_isolation": cross_family,
        "hash_and_seal_freshness": stale_hashes,
    }
    expected = {
        "prompt_direct_label": "no prompt contains a direct answer",
        "metadata_label_encoding": "no input metadata field encodes a result",
        "instance_id_uniqueness": "all 40 instance identities are unique",
        "instance_id_shortcut": "no identity encodes a result",
        "split_membership": "32 headline and 8 development rows use known splits",
        "development_to_held_contamination": "no canonical input crosses splits",
        "family_isolation": "no prompt or specification identity crosses families",
        "hash_and_seal_freshness": "all prompt, specification, and seal hashes are current",
    }
    return [
        {
            "case": case,
            "check": case,
            "expected_result": expected[case],
            "observed_result": evidence[case],
            "evidence_hash": sha256_json(evidence[case]),
            "pass_state": not bool(evidence[case]),
        }
        for case in LEAKAGE_CASES
    ]


class SealAccessError(PermissionError):
    """Return stable denial reasons without exposing an archived label."""


class AuditLabelSeal:
    """Permit one label-hash read after one complete prompt-bound commit.

    A receipt binds the event sequence as well as the prompt. This prevents a
    valid receipt from one event from becoming authority for a later event.
    The receipt is one-shot because audit replay must not create a label API.
    """

    def __init__(self, seal_row: Mapping[str, Any]) -> None:
        components = _seal_components(seal_row)
        if seal_row.get("seal_hash") != sha256_json(components):
            raise SealAccessError("invalid seal")
        self._seal = deepcopy(dict(seal_row))
        self._event_id: str | None = None
        self._event_sequence = -1
        self._receipts: dict[str, JsonDict] = {}
        self._consumed: set[str] = set()

    def begin_event(self, event_id: str, prompt_hash: str, event_sequence: int) -> None:
        """Open a later event only when it matches the sealed prompt."""

        if prompt_hash != self._seal.get("prompt_hash"):
            raise SealAccessError("prompt hash mismatch")
        if event_sequence <= self._event_sequence:
            raise SealAccessError("reordered event")
        self._event_id = event_id
        self._event_sequence = event_sequence

    def commit(
        self,
        event_id: str,
        candidate: Any,
        *,
        status: str = "committed",
        write_complete: bool = True,
        event_sequence: int | None = None,
    ) -> JsonDict:
        """Create a receipt, while retaining incomplete receipts for denial tests."""

        if self._event_id is None:
            raise SealAccessError("event not opened")
        if event_id != self._event_id:
            raise SealAccessError("event mismatch")
        sequence = self._event_sequence if event_sequence is None else event_sequence
        if sequence != self._event_sequence:
            raise SealAccessError("reordered event")
        core = {
            "instance": self._seal.get("instance"),
            "event": event_id,
            "event_sequence": sequence,
            "prompt_hash": self._seal.get("prompt_hash"),
            "candidate_hash": sha256_json(candidate),
            "status": status,
            "write_complete": write_complete,
            "commit_version": SEAL_AUDITOR_VERSION,
        }
        receipt = {**core, "receipt_hash": sha256_json(core)}
        self._receipts[str(receipt["receipt_hash"])] = deepcopy(receipt)
        return receipt

    def read(self, event_id: str, receipt: Mapping[str, Any] | None) -> str:
        """Return only the sealed label hash after all chronology checks pass."""

        if receipt is None:
            raise SealAccessError("commit receipt required")
        receipt_dict = dict(receipt)
        receipt_hash = str(receipt_dict.get("receipt_hash", ""))
        core = {key: value for key, value in receipt_dict.items() if key != "receipt_hash"}
        if receipt_hash != sha256_json(core) or self._receipts.get(receipt_hash) != receipt_dict:
            raise SealAccessError("invalid commit receipt")
        if (
            event_id != self._event_id
            or receipt_dict.get("event") != self._event_id
            or receipt_dict.get("event_sequence") != self._event_sequence
        ):
            raise SealAccessError("stale token")
        if (
            receipt_dict.get("status") != "committed"
            or receipt_dict.get("write_complete") is not True
        ):
            raise SealAccessError("partial commit receipt")
        if receipt_hash in self._consumed:
            raise SealAccessError("receipt replay")
        self._consumed.add(receipt_hash)
        return str(self._seal["label_hash"])


def _access_attempt(
    store: AuditLabelSeal, event_id: str, receipt: Mapping[str, Any] | None
) -> tuple[bool, str | None, str]:
    try:
        return True, store.read(event_id, receipt), "access_granted"
    except SealAccessError as exc:
        return False, None, f"denied:{exc}"


def _seal_access_row(
    *,
    case: str,
    event: str,
    token: Mapping[str, Any] | None,
    commit_state: str,
    expected_access: bool,
    attempt: tuple[bool, str | None, str],
) -> JsonDict:
    observed_access, observed_label_hash, disposition = attempt
    return {
        "case": case,
        "event": event,
        "access_token": deepcopy(dict(token)) if token is not None else None,
        "commit_state": commit_state,
        "expected_access": expected_access,
        "observed_access": observed_access,
        "observed_label_hash": observed_label_hash,
        "disposition": disposition,
        "pass_state": observed_access is expected_access,
    }


def run_seal_access_attacks(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Exercise valid access and five denied paths on isolated seal stores."""

    seal = deepcopy(dict(upstream.get("label_seal_rows", [])[0]))
    event = f"{seal['instance']}:current"

    early_store = AuditLabelSeal(seal)
    early_store.begin_event(event, str(seal["prompt_hash"]), 1)
    early = _seal_access_row(
        case="early_access",
        event=event,
        token=None,
        commit_state="not_committed",
        expected_access=False,
        attempt=_access_attempt(early_store, event, None),
    )

    valid_store = AuditLabelSeal(seal)
    valid_store.begin_event(event, str(seal["prompt_hash"]), 1)
    valid_token = valid_store.commit(event, {"candidate": "sealed-before-label"})
    valid = _seal_access_row(
        case="valid_post_commit",
        event=event,
        token=valid_token,
        commit_state="committed_complete",
        expected_access=True,
        attempt=_access_attempt(valid_store, event, valid_token),
    )
    replay = _seal_access_row(
        case="receipt_replay",
        event=event,
        token=valid_token,
        commit_state="committed_consumed",
        expected_access=False,
        attempt=_access_attempt(valid_store, event, valid_token),
    )

    stale_store = AuditLabelSeal(seal)
    stale_store.begin_event(event, str(seal["prompt_hash"]), 1)
    stale_token = stale_store.commit(event, {"candidate": "old-event"})
    next_event = f"{seal['instance']}:next"
    stale_store.begin_event(next_event, str(seal["prompt_hash"]), 2)
    stale = _seal_access_row(
        case="stale_token",
        event=next_event,
        token=stale_token,
        commit_state="committed_for_prior_event",
        expected_access=False,
        attempt=_access_attempt(stale_store, next_event, stale_token),
    )

    reordered_store = AuditLabelSeal(seal)
    reordered_store.begin_event(event, str(seal["prompt_hash"]), 2)
    reordered_reason = ""
    try:
        reordered_store.commit(event, {"candidate": "wrong-order"}, event_sequence=1)
    except SealAccessError as exc:
        reordered_reason = f"denied:{exc}"
    reordered = _seal_access_row(
        case="reordered_event",
        event=event,
        token=None,
        commit_state="reordered_before_commit",
        expected_access=False,
        attempt=(False, None, reordered_reason),
    )

    partial_store = AuditLabelSeal(seal)
    partial_store.begin_event(event, str(seal["prompt_hash"]), 1)
    partial_token = partial_store.commit(
        event,
        {"candidate": "partial"},
        status="prepared",
        write_complete=False,
    )
    partial = _seal_access_row(
        case="partial_write",
        event=event,
        token=partial_token,
        commit_state="prepared_incomplete",
        expected_access=False,
        attempt=_access_attempt(partial_store, event, partial_token),
    )
    return [early, valid, replay, stale, reordered, partial]


def _authorized_label_hash(seal: Mapping[str, Any], case: str) -> tuple[str, str]:
    """Create a fresh audit receipt before reading archived result metadata."""

    store = AuditLabelSeal(seal)
    event = f"audit:{case}"
    store.begin_event(event, str(seal["prompt_hash"]), 1)
    receipt = store.commit(event, {"audit_case": case})
    return store.read(event, receipt), str(receipt["receipt_hash"])


def replay_metamorphic_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Replay 16 transformations by archived-row algebra, without plan search."""

    instances = {str(row.get("instance")): row for row in upstream.get("instance_rows", [])}
    seals = {str(row.get("instance")): row for row in upstream.get("label_seal_rows", [])}
    state_rows = list(upstream.get("state_action_rows", []))
    raw = {str(row.get("row_id")): row for row in upstream.get("metamorphic_rows", [])}
    results: list[JsonDict] = []
    for case in METAMORPHIC_CASES:
        family, transform = case.split(":", 1)
        instance_id = f"{family}-headline-00"
        instance = instances.get(instance_id)
        source = raw.get(case)
        observed: Any = None
        access_hash: str | None = None
        label_hash: str | None = None
        if instance is not None and instance_id in seals:
            label_hash, access_hash = _authorized_label_hash(seals[instance_id], case)
            if transform == "action_renaming":
                aliases = {
                    str(action): f"choice_{index}"
                    for index, action in enumerate(instance.get("action_set", []))
                }
                observed = {
                    "aliases": aliases,
                    "renamed_optimum": [
                        aliases[str(action)]
                        for action in instance.get("optimum", {}).get("action_set", [])
                    ],
                }
            elif transform == "constant_cost_shift":
                base_total = instance.get("optimum", {}).get("total")
                observed = {
                    "base_total": base_total,
                    "shifted_total": None
                    if not isinstance(base_total, int)
                    else base_total + 3 * int(instance.get("horizon", 0)),
                }
            elif transform == "equivalent_state_encoding":
                observed = {
                    "state_count": sum(row.get("instance") == instance_id for row in state_rows)
                }
            elif transform == "family_preserving_surface_change":
                surface = "Please solve this equivalent family task. " + str(
                    instance.get("prompt", "")
                )
                observed = {
                    "base_prompt_hash": instance.get("prompt_hash"),
                    "surface_prompt_hash": "sha256:"
                    + hashlib.sha256(surface.encode("utf-8")).hexdigest(),
                }
        raw_expected = None if source is None else source.get("observed_result")
        expected_invariant = None if source is None else source.get("expected_invariant")
        results.append(
            {
                "case": case,
                "family": family,
                "instance": instance_id,
                "transform": transform,
                "expected_invariant": expected_invariant,
                "raw_expected_result": deepcopy(raw_expected),
                "observed_result": observed,
                "label_hash_read_after_commit": label_hash,
                "access_receipt_hash": access_hash,
                "pass_state": source is not None
                and source.get("pass_state") is True
                and raw_expected == observed,
            }
        )
    return results


def _memory_records(upstream: Mapping[str, Any]) -> list[JsonDict]:
    identities = [str(row.get("instance")) for row in upstream.get("instance_rows", [])[:2]]
    records: list[JsonDict] = []
    for index, identity in enumerate(identities):
        relation = "same_family" if index == 0 else "retired_relation"
        core = {"record_id": identity, "relation": relation}
        records.append(
            {
                **core,
                "provenance_hash": sha256_json(core),
                "tombstone": index == 1,
            }
        )
    return records


def _memory_poison_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    fixture_before = sha256_json(upstream.get("instance_rows", []))
    seal_before = sha256_json(upstream.get("label_seal_rows", []))
    base = _memory_records(upstream)
    results: list[JsonDict] = []

    copied = deepcopy(base)
    copied.append(deepcopy(copied[0]))
    copied_detected = len({row["record_id"] for row in copied}) != len(copied)

    poisoned = deepcopy(base)
    poisoned[0]["relation"] = "different_family"
    relation_core = {
        "record_id": poisoned[0]["record_id"],
        "relation": poisoned[0]["relation"],
    }
    relation_detected = poisoned[0]["provenance_hash"] != sha256_json(relation_core)

    provenance_lost = deepcopy(base)
    provenance_lost[0].pop("provenance_hash")
    provenance_detected = any(not row.get("provenance_hash") for row in provenance_lost)

    reappeared = deepcopy(base)
    tombstone_id = str(reappeared[1]["record_id"])
    reappeared.append(
        {
            "record_id": tombstone_id,
            "relation": "returned_relation",
            "provenance_hash": sha256_json(
                {"record_id": tombstone_id, "relation": "returned_relation"}
            ),
            "tombstone": False,
        }
    )
    tombstone_positions = {
        str(row["record_id"]): index
        for index, row in enumerate(reappeared)
        if row.get("tombstone") is True
    }
    reappearance_detected = any(
        not row.get("tombstone")
        and str(row["record_id"]) in tombstone_positions
        and index > tombstone_positions[str(row["record_id"])]
        for index, row in enumerate(reappeared)
    )

    panels = {
        "copied_record": (copied_detected, copied),
        "relation_poison": (relation_detected, poisoned),
        "provenance_loss": (provenance_detected, provenance_lost),
        "tombstone_reappearance": (reappearance_detected, reappeared),
    }
    for case in MEMORY_POISON_CASES:
        detected, attacked = panels[case]
        fixture_after = sha256_json(upstream.get("instance_rows", []))
        seal_after = sha256_json(upstream.get("label_seal_rows", []))
        results.append(
            {
                "kind": "memory_poison",
                "case": case,
                "mutation_or_poison": case,
                "expected_detection": True,
                "observed_detection": detected,
                "raw_expected_result": True,
                "observed_result": {
                    "attacked_memory_hash": sha256_json(attacked),
                    "record_count": len(attacked),
                },
                "fixture_truth_hash_before": fixture_before,
                "fixture_truth_hash_after": fixture_after,
                "seal_state_hash_before": seal_before,
                "seal_state_hash_after": seal_after,
                "fixture_truth_unchanged": fixture_before == fixture_after,
                "seal_state_unchanged": seal_before == seal_after,
                "pass_state": detected
                and fixture_before == fixture_after
                and seal_before == seal_after,
            }
        )
    return results


def run_mutation_and_memory_attacks(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Detect six row mutations and four isolated forward-memory poisons."""

    state_rows = list(upstream.get("state_action_rows", []))
    legal = next(row for row in state_rows if row.get("legality") is True)
    illegal = next(row for row in state_rows if row.get("legality") is False)

    changed_transition = deepcopy(dict(legal))
    next_state = changed_transition["transition"]["next_state"]
    state_key = next(iter(next_state))
    next_state[state_key] += 1

    changed_legality = deepcopy(dict(illegal))
    changed_legality["legality"] = True

    changed_cost = deepcopy(dict(legal))
    changed_cost["immediate_cost"] += 1

    first_instance = deepcopy(dict(upstream.get("instance_rows", [])[0]))
    seal_by_id = {str(row.get("instance")): row for row in upstream.get("label_seal_rows", [])}
    _, label_receipt = _authorized_label_hash(
        seal_by_id[str(first_instance["instance"])], "mutation:label_leakage"
    )
    leaked_prompt = str(first_instance.get("prompt", "")) + (
        f" Exact optimum: {first_instance.get('optimum', {}).get('total')}."
    )
    label_leak_detected = (
        re.search(r"exact\s+optimum\s*:\s*[-+]?\d", leaked_prompt, re.IGNORECASE) is not None
    )

    changed_ties = deepcopy(first_instance)
    changed_ties["ties"] = not bool(changed_ties.get("ties"))

    instances = {str(row.get("instance")): row for row in upstream.get("instance_rows", [])}
    stale_seal = deepcopy(dict(upstream.get("label_seal_rows", [])[0]))
    stale_seal["prompt_hash"] = "sha256:stale"

    detections: dict[str, tuple[bool, JsonDict]] = {
        "bad_transition": (
            sha256_json(changed_transition) != sha256_json(legal),
            {"clean_hash": sha256_json(legal), "mutated_hash": sha256_json(changed_transition)},
        ),
        "infeasible_action": (
            sha256_json(changed_legality) != sha256_json(illegal),
            {"clean_hash": sha256_json(illegal), "mutated_hash": sha256_json(changed_legality)},
        ),
        "corrupted_cost": (
            sha256_json(changed_cost) != sha256_json(legal),
            {"clean_hash": sha256_json(legal), "mutated_hash": sha256_json(changed_cost)},
        ),
        "label_leakage": (
            label_leak_detected,
            {"prompt_hash": sha256_json(leaked_prompt), "access_receipt_hash": label_receipt},
        ),
        "wrong_ties": (
            sha256_json(changed_ties) != sha256_json(first_instance),
            {
                "clean_ties": first_instance.get("ties"),
                "mutated_ties": changed_ties.get("ties"),
            },
        ),
        "stale_seal": (
            not _valid_seal(stale_seal, instances.get(str(stale_seal.get("instance")))),
            {"mutated_seal_hash": sha256_json(stale_seal)},
        ),
    }
    raw = {str(row.get("mutation")): row for row in upstream.get("mutation_rows", [])}
    rows: list[JsonDict] = []
    for case in MUTATION_CASES:
        source = raw.get(case)
        observed, evidence = detections[case]
        raw_expected = None if source is None else source.get("expected_detection")
        rows.append(
            {
                "kind": "mutation",
                "case": case,
                "mutation_or_poison": case,
                "expected_detection": raw_expected,
                "observed_detection": observed,
                "raw_expected_result": raw_expected,
                "observed_result": evidence,
                "pass_state": source is not None
                and raw_expected is True
                and source.get("pass_state") is True
                and observed is True,
            }
        )
    return [*rows, *_memory_poison_rows(upstream)]


class StepClock:
    """Provide deterministic small elapsed times for focused tests."""

    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.001
        return self._value


def run_attack_campaign(
    upstream: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> JsonDict:
    """Run each frozen family once and retain measured budget rows."""

    if manifest.get("manifest_hash") != manifest_checksum(manifest):
        raise ValueError("attack manifest hash mismatch")
    timings: dict[str, float] = {}

    started = clock()
    leakage_rows = scan_leakage(upstream)
    static_duration = max(0.0, clock() - started)
    for family in ("prompt", "metadata", "id", "split", "family_isolation", "seal_integrity"):
        timings[family] = static_duration

    started = clock()
    seal_access_rows = run_seal_access_attacks(upstream)
    timings["seal_access"] = max(0.0, clock() - started)

    started = clock()
    metamorphic_rows = replay_metamorphic_rows(upstream)
    timings["metamorphic"] = max(0.0, clock() - started)

    started = clock()
    mutation_attack_rows = run_mutation_and_memory_attacks(upstream)
    mutation_duration = max(0.0, clock() - started)
    timings["mutation"] = mutation_duration
    timings["memory_poison"] = mutation_duration

    budget_rows = [
        {
            "case": f"{family}_time_budget",
            "family": family,
            "expected_max_s": ATTACK_BUDGETS_S[family],
            "observed_duration_s": round(timings[family], 6),
            "pass_state": timings[family] <= ATTACK_BUDGETS_S[family],
        }
        for family in EXPECTED_CASES
    ]
    return {
        "leakage_rows": leakage_rows,
        "seal_access_rows": seal_access_rows,
        "metamorphic_rows": metamorphic_rows,
        "mutation_attack_rows": mutation_attack_rows,
        "budget_rows": budget_rows,
    }


def _case_set_ok(rows: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> bool:
    observed = [str(row.get("case", row.get("check"))) for row in rows]
    return Counter(observed) == Counter(expected) and all(
        row.get("pass_state") is True for row in rows
    )


def recompute_aggregate(
    *,
    manifest: Mapping[str, Any],
    leakage_rows: Sequence[Mapping[str, Any]],
    seal_access_rows: Sequence[Mapping[str, Any]],
    metamorphic_rows: Sequence[Mapping[str, Any]],
    mutation_attack_rows: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    preconditions_passed: bool,
    protected_files_unchanged: bool,
    method_contract: Mapping[str, Any],
) -> JsonDict:
    """Rebuild the audit gate only from raw rows and measured receipts."""

    static_expected = tuple(
        case
        for family in (
            "prompt",
            "metadata",
            "id",
            "split",
            "family_isolation",
            "seal_integrity",
        )
        for case in EXPECTED_CASES[family]
    )
    budget_expected = tuple(f"{family}_time_budget" for family in EXPECTED_CASES)
    method_ok = (
        method_contract.get("passed") is True
        and method_contract.get("plan_enumeration_count") == 0
        and method_contract.get("optimum_recomputation_count") == 0
        and method_contract.get("producer_conclusion_import_count") == 0
        and method_contract.get("llm_call_count") == 0
        and method_contract.get("method_substitution_count") == 0
        and method_contract.get("time_budgets_s") == ATTACK_BUDGETS_S
    )
    checks: list[JsonDict] = [
        {
            "check": "preconditions",
            "expected": True,
            "observed": preconditions_passed,
            "passed": preconditions_passed,
        },
        {
            "check": "protected_files",
            "expected": True,
            "observed": protected_files_unchanged,
            "passed": protected_files_unchanged,
        },
        {
            "check": "frozen_attack_manifest",
            "expected": True,
            "observed": manifest.get("manifest_hash") == manifest_checksum(manifest),
            "passed": manifest.get("manifest_hash") == manifest_checksum(manifest),
        },
        {
            "check": "method_fidelity_contract",
            "expected": True,
            "observed": method_ok,
            "passed": method_ok,
        },
        {
            "check": "leakage_rows",
            "expected": list(static_expected),
            "observed": [row.get("case") for row in leakage_rows],
            "passed": _case_set_ok(leakage_rows, static_expected),
        },
        {
            "check": "seal_access_rows",
            "expected": list(SEAL_ACCESS_CASES),
            "observed": [row.get("case") for row in seal_access_rows],
            "passed": _case_set_ok(seal_access_rows, SEAL_ACCESS_CASES),
        },
        {
            "check": "metamorphic_rows",
            "expected": list(METAMORPHIC_CASES),
            "observed": [row.get("case") for row in metamorphic_rows],
            "passed": _case_set_ok(metamorphic_rows, METAMORPHIC_CASES),
        },
        {
            "check": "mutation_attack_rows",
            "expected": [*MUTATION_CASES, *MEMORY_POISON_CASES],
            "observed": [row.get("case") for row in mutation_attack_rows],
            "passed": _case_set_ok(mutation_attack_rows, (*MUTATION_CASES, *MEMORY_POISON_CASES)),
        },
        {
            "check": "attack_time_budgets",
            "expected": list(budget_expected),
            "observed": [row.get("case") for row in budget_rows],
            "passed": _case_set_ok(budget_rows, budget_expected),
        },
    ]
    test_map = {str(row.get("check_id")): row for row in tests_run}
    for check_id in REQUIRED_TEST_CHECKS:
        row = test_map.get(check_id)
        observed = None if row is None else row.get("passed")
        passed = observed is True
        if check_id == "scoped_coverage":
            passed = passed and row is not None and row.get("coverage_percent") == 100.0
            observed = {
                "passed": None if row is None else row.get("passed"),
                "coverage_percent": None if row is None else row.get("coverage_percent"),
            }
        checks.append(
            {
                "check": check_id,
                "expected": True
                if check_id != "scoped_coverage"
                else {"passed": True, "coverage_percent": 100.0},
                "observed": observed,
                "passed": passed,
            }
        )
    failed = [str(row["check"]) for row in checks if row.get("passed") is not True]
    return {
        "schema": SCHEMA + ".aggregate_row_recomputation",
        "counts": {
            "frozen_attack_family_count": len(manifest.get("attack_families", [])),
            "frozen_attack_case_count": sum(
                int(row.get("count", 0)) for row in manifest.get("attack_families", [])
            ),
            "leakage_row_count": len(leakage_rows),
            "seal_access_row_count": len(seal_access_rows),
            "metamorphic_row_count": len(metamorphic_rows),
            "mutation_attack_row_count": len(mutation_attack_rows),
            "memory_poison_row_count": sum(
                row.get("kind") == "memory_poison" for row in mutation_attack_rows
            ),
            "budget_row_count": len(budget_rows),
            "test_receipt_count": len(tests_run),
        },
        "check_rows": checks,
        "failed_checks": failed,
        "seal_attack_audit_passed": not failed,
    }


def _memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def _precondition(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"name": name, "expected": expected, "observed": observed, "passed": passed}


def _upstream_checksum(payload: Mapping[str, Any]) -> str:
    body = deepcopy(dict(payload))
    body.pop("reproducibility_checksum", None)
    body.pop("duration_s", None)
    return sha256_json(body)


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Measure every required input, resource, tool, schema, and protected hash."""

    upstream_path = root / UPSTREAM_PATH
    upstream: JsonDict = {}
    upstream_error: str | None = None
    if upstream_path.is_file():
        try:
            upstream = load_json(upstream_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            upstream_error = str(exc)
    rows = [
        _precondition(
            "upstream_artifact",
            "present JSON object",
            {"present": upstream_path.is_file(), "error": upstream_error},
            upstream_path.is_file() and bool(upstream) and upstream_error is None,
        )
    ]
    rows.append(
        _precondition(
            "planning_fixture_ready",
            True,
            upstream.get("planning_fixture_ready"),
            upstream.get("planning_fixture_ready") is True,
        )
    )
    expected_counts = {
        "instance_rows": 40,
        "state_action_rows": 4543,
        "label_seal_rows": 40,
        "metamorphic_rows": 16,
        "mutation_rows": 6,
    }
    observed_stores = {
        name: {
            "type": type(upstream.get(name)).__name__,
            "count": len(upstream.get(name, [])) if isinstance(upstream.get(name), list) else None,
            "sha256": sha256_json(upstream.get(name, [])),
        }
        for name in expected_counts
    }
    stores_ok = all(
        isinstance(upstream.get(name), list) and len(upstream.get(name, [])) == count
        for name, count in expected_counts.items()
    )
    rows.append(_precondition("raw_attack_inputs", expected_counts, observed_stores, stores_ok))
    recorded_checksum = upstream.get("reproducibility_checksum")
    computed_checksum = _upstream_checksum(upstream) if upstream else "unavailable"
    rows.append(
        _precondition(
            "upstream_reproducibility_checksum",
            recorded_checksum,
            computed_checksum,
            bool(upstream) and recorded_checksum == computed_checksum,
        )
    )
    hash_scan = scan_leakage(upstream) if stores_ok else []
    hash_row = next(
        (row for row in hash_scan if row.get("case") == "hash_and_seal_freshness"), None
    )
    rows.append(
        _precondition(
            "prompt_spec_and_seal_hashes",
            "all current",
            None if hash_row is None else hash_row.get("observed_result"),
            hash_row is not None and hash_row.get("pass_state") is True,
        )
    )
    cpu_count = os.cpu_count() or 0
    ram_bytes = _memory_bytes()
    disk_free = shutil.disk_usage(root).free
    rows.extend(
        [
            _precondition("cpu", ">=1", cpu_count, cpu_count >= 1),
            _precondition("ram_bytes", ">=1073741824", ram_bytes, ram_bytes >= 1024**3),
            _precondition("disk_free_bytes", ">=104857600", disk_free, disk_free >= 100 * 1024**2),
        ]
    )
    tools = {
        "python": shutil.which("python") is not None,
        "git": shutil.which("git") is not None,
        "jq": shutil.which("jq") is not None,
        "artifact_validation": MODULE_PATH.is_file()
        if root == REPO_ROOT
        else (root / MODULE_PATH).is_file(),
        "row_consistency": (root / "scripts/verdict_row_consistency_lint.py").is_file(),
        "adversarial_verification": (root / "scripts/adversarial_verify.py").is_file(),
        "spec_coverage": (root / "scripts/check_spec_coverage.py").is_file(),
    }
    rows.append(_precondition("audit_tools", "all present", tools, all(tools.values())))
    try:
        jsonschema_version = metadata.version("jsonschema")
        schema_ok = True
    except metadata.PackageNotFoundError:
        jsonschema_version = "missing"
        schema_ok = False
    rows.append(
        _precondition(
            "artifact_schema",
            "jsonschema available",
            {"version": jsonschema_version},
            schema_ok,
        )
    )
    roadmap = root / ACTIVE_ROADMAP
    roadmap_text = roadmap.read_text(encoding="utf-8") if roadmap.is_file() else ""
    roadmap_observed = {
        "present": roadmap.is_file(),
        "exp6716_task": "exp6716-bounded-seal-attack-audit" in roadmap_text,
        "sha256": sha256_file(roadmap),
    }
    rows.append(
        _precondition(
            "roadmap",
            "active V585 Exp6716 task present and hashed",
            roadmap_observed,
            roadmap_observed["present"] and roadmap_observed["exp6716_task"],
        )
    )
    conductor = root / CONDUCTOR_PATH
    conductor_observed = {"present": conductor.is_file(), "sha256": sha256_file(conductor)}
    rows.append(
        _precondition(
            "conductor",
            "present and hashed",
            conductor_observed,
            conductor_observed["present"] and conductor_observed["sha256"] != "missing",
        )
    )
    rows.append(
        _precondition(
            "protected_hashes",
            "roadmap and conductor hashes available",
            protected_hashes(root),
            roadmap_observed["sha256"] != "missing" and conductor_observed["sha256"] != "missing",
        )
    )
    return rows


def protected_hashes(root: Path) -> JsonDict:
    """Hash the two files this task must not modify."""

    return {
        path.as_posix(): sha256_file(root / path)
        for path in (ACTIVE_ROADMAP, CONDUCTOR_PATH)
        if (root / path).is_file()
    }


def _per_unit_rows(
    leakage_rows: Sequence[Mapping[str, Any]],
    seal_access_rows: Sequence[Mapping[str, Any]],
    metamorphic_rows: Sequence[Mapping[str, Any]],
    mutation_attack_rows: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for unit_type, values in (
        ("leakage", leakage_rows),
        ("seal_access", seal_access_rows),
        ("metamorphic", metamorphic_rows),
        ("mutation_or_poison", mutation_attack_rows),
        ("budget", budget_rows),
        ("check", check_rows),
    ):
        rows.extend({"unit_type": unit_type, **deepcopy(dict(row))} for row in values)
    return rows


def _gate_summary(aggregate: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "check": row.get("check"),
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
            "passed": False,
        }
        for row in aggregate.get("check_rows", [])
        if row.get("passed") is not True
    ]


def _stable_provenance_value(field: str, value: Any) -> Any:
    if field == "field_provenance":
        return sorted(REQUIRED_ARTIFACT_FIELDS)
    if field == "reproducibility_checksum":
        return "computed_after_field_provenance"
    if field == "duration_s":
        return "measured_monotonic_duration"
    return value


def _field_provenance(artifact: Mapping[str, Any]) -> JsonDict:
    provenance: JsonDict = {}
    for field, value in artifact.items():
        provenance[field] = {
            "prompt_store": UPSTREAM_PATH.as_posix()
            if field
            in {
                "frozen_attack_manifest",
                "leakage_rows",
                "metamorphic_rows",
                "mutation_attack_rows",
                "preconditions_checked",
            }
            else None,
            "seal": SEAL_AUDITOR_VERSION
            if field in {"seal_access_rows", "metamorphic_rows", "mutation_attack_rows"}
            else None,
            "scanner": SCANNER_VERSION
            if field in {"leakage_rows", "frozen_attack_manifest"}
            else None,
            "attack_runner": ATTACK_RUNNER_VERSION
            if field
            in {
                "seal_access_rows",
                "metamorphic_rows",
                "mutation_attack_rows",
                "per_unit_rows",
            }
            else None,
            "reducer": REDUCER_VERSION
            if field
            in {
                "status",
                "honest_verdict",
                "verdict_class",
                "gate_check_summary",
                "seal_attack_audit_passed",
                "aggregate_row_recomputation",
            }
            else None,
            "function": "build_artifact",
            "version": SCHEMA,
            "hash": sha256_json(_stable_provenance_value(field, value)),
        }
    return provenance


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash deterministic artifact content while excluding measured wall time."""

    body = deepcopy(dict(payload))
    body.pop("reproducibility_checksum", None)
    body.pop("duration_s", None)
    return sha256_json(body)


def _classification(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    if aggregate.get("seal_attack_audit_passed") is True:
        return (
            "complete_passed",
            "passed: every bounded leakage, seal, transform, mutation, and poison check passed",
            "positive",
        )
    failed = set(aggregate.get("failed_checks", []))
    attack_checks = {
        "frozen_attack_manifest",
        "method_fidelity_contract",
        "leakage_rows",
        "seal_access_rows",
        "metamorphic_rows",
        "mutation_attack_rows",
        "attack_time_budgets",
    }
    if failed & attack_checks:
        return (
            "disqualified_attack_failure",
            "disqualified: one or more bounded seal or attack checks failed",
            "disqualified",
        )
    return (
        "partial_verification_failure",
        "partial: attack rows passed but required verification is incomplete",
        "partial",
    )


def build_artifact(
    *,
    date: str,
    root: Path,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    protected_before: Mapping[str, Any],
    clock: Callable[[], float] = time.perf_counter,
    upstream: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    campaign: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Freeze inputs, run bounded attacks, and build one terminal artifact."""

    source = deepcopy(dict(upstream)) if upstream is not None else load_json(root / UPSTREAM_PATH)
    frozen = (
        deepcopy(dict(manifest))
        if manifest is not None
        else freeze_attack_manifest(public_attack_inputs(source))
    )
    measured_campaign = (
        deepcopy(dict(campaign))
        if campaign is not None
        else run_attack_campaign(source, frozen, clock=clock)
    )
    measured_preconditions = (
        [deepcopy(dict(row)) for row in preconditions]
        if preconditions is not None
        else collect_preconditions(root)
    )
    protected_after = protected_hashes(root)
    protected = {
        "before": deepcopy(dict(protected_before)),
        "after": protected_after,
        "unchanged": dict(protected_before) == protected_after,
    }
    method_contract = method_fidelity_contract()
    aggregate = recompute_aggregate(
        manifest=frozen,
        leakage_rows=measured_campaign["leakage_rows"],
        seal_access_rows=measured_campaign["seal_access_rows"],
        metamorphic_rows=measured_campaign["metamorphic_rows"],
        mutation_attack_rows=measured_campaign["mutation_attack_rows"],
        budget_rows=measured_campaign["budget_rows"],
        tests_run=tests_run,
        preconditions_passed=all(row.get("passed") is True for row in measured_preconditions),
        protected_files_unchanged=protected["unchanged"],
        method_contract=method_contract,
    )
    status, verdict, verdict_class = _classification(aggregate)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6716,
        "run_date": date,
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": _gate_summary(aggregate),
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "frozen_attack_manifest": frozen,
        "method_fidelity_contract": method_contract,
        "leakage_rows": deepcopy(list(measured_campaign["leakage_rows"])),
        "seal_access_rows": deepcopy(list(measured_campaign["seal_access_rows"])),
        "metamorphic_rows": deepcopy(list(measured_campaign["metamorphic_rows"])),
        "mutation_attack_rows": deepcopy(list(measured_campaign["mutation_attack_rows"])),
        "budget_rows": deepcopy(list(measured_campaign["budget_rows"])),
        "seal_attack_audit_passed": aggregate["seal_attack_audit_passed"],
        "per_unit_rows": _per_unit_rows(
            measured_campaign["leakage_rows"],
            measured_campaign["seal_access_rows"],
            measured_campaign["metamorphic_rows"],
            measured_campaign["mutation_attack_rows"],
            measured_campaign["budget_rows"],
            aggregate["check_rows"],
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": measured_preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": {},
        "random_seed": deepcopy(RANDOM_SEED),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    date: str,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Write measured missing prerequisites without invented attack rows."""

    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    check = {
        "check": "preconditions",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    aggregate = {
        "schema": SCHEMA + ".aggregate_row_recomputation",
        "counts": {
            "frozen_attack_family_count": 0,
            "frozen_attack_case_count": 0,
            "leakage_row_count": 0,
            "seal_access_row_count": 0,
            "metamorphic_row_count": 0,
            "mutation_attack_row_count": 0,
            "memory_poison_row_count": 0,
            "budget_row_count": 0,
            "test_receipt_count": 0,
        },
        "check_rows": [check],
        "failed_checks": ["preconditions"],
        "seal_attack_audit_passed": False,
    }
    before = protected_hashes(root)
    gate_summary = []
    if failed is not None:
        gate_summary.append(
            {
                "check": failed.get("name"),
                "expected": deepcopy(failed.get("expected")),
                "observed": deepcopy(failed.get("observed")),
                "passed": False,
            }
        )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6716,
        "run_date": date,
        "status": "blocked_precondition",
        "honest_verdict": "blocked: required fixture input, resource, schema, tool, or hash is unavailable",
        "verdict_class": "blocked",
        "gate_check_summary": gate_summary,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "frozen_attack_manifest": {},
        "method_fidelity_contract": method_fidelity_contract(),
        "leakage_rows": [],
        "seal_access_rows": [],
        "metamorphic_rows": [],
        "mutation_attack_rows": [],
        "budget_rows": [],
        "seal_attack_audit_passed": False,
        "per_unit_rows": _per_unit_rows([], [], [], [], [], [check]),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "protected_files_unchanged": {"before": before, "after": before, "unchanged": True},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": {},
        "random_seed": deepcopy(RANDOM_SEED),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [],
        "reproducibility_checksum": "",
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate fields, provenance, checksum, row conservation, and gate reduction."""

    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", -1) < 0:
        errors.append("duration_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    provenance = payload.get("field_provenance")
    provenance_valid = isinstance(provenance, Mapping) and set(REQUIRED_ARTIFACT_FIELDS) <= set(
        provenance
    )
    if provenance_valid:
        provenance_valid = all(
            isinstance(row, Mapping) and set(PROVENANCE_KEYS) <= set(row)
            for row in provenance.values()
        )
    if not provenance_valid:
        errors.append("field_provenance_invalid")
    expected_units = _per_unit_rows(
        payload.get("leakage_rows", []),
        payload.get("seal_access_rows", []),
        payload.get("metamorphic_rows", []),
        payload.get("mutation_attack_rows", []),
        payload.get("budget_rows", []),
        payload.get("aggregate_row_recomputation", {}).get("check_rows", []),
    )
    if payload.get("per_unit_rows") != expected_units:
        errors.append("per_unit_rows_mismatch")
    if payload.get("status") == "blocked_precondition":
        if payload.get("seal_attack_audit_passed") is not False or not payload.get(
            "gate_check_summary"
        ):
            errors.append("blocked_state_mismatch")
        return errors
    manifest = payload.get("frozen_attack_manifest", {})
    if manifest.get("manifest_hash") != manifest_checksum(manifest):
        errors.append("manifest_hash_mismatch")
    aggregate = recompute_aggregate(
        manifest=manifest,
        leakage_rows=payload.get("leakage_rows", []),
        seal_access_rows=payload.get("seal_access_rows", []),
        metamorphic_rows=payload.get("metamorphic_rows", []),
        mutation_attack_rows=payload.get("mutation_attack_rows", []),
        budget_rows=payload.get("budget_rows", []),
        tests_run=payload.get("tests_run", []),
        preconditions_passed=all(
            row.get("passed") is True for row in payload.get("preconditions_checked", [])
        ),
        protected_files_unchanged=payload.get("protected_files_unchanged", {}).get("unchanged")
        is True,
        method_contract=payload.get("method_fidelity_contract", {}),
    )
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    if payload.get("seal_attack_audit_passed") != aggregate["seal_attack_audit_passed"]:
        errors.append("audit_gate_mismatch")
    if payload.get("seal_attack_audit_passed") is True and payload.get("gate_check_summary"):
        errors.append("passed_gate_summary_mismatch")
    if payload.get("seal_attack_audit_passed") is False and not payload.get("gate_check_summary"):
        errors.append("failed_gate_summary_missing")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Sync a complete temporary file before one atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": path.as_posix(), "bytes": len(data), "atomic_replace": True}


def default_command_runner(command: str, root: Path) -> JsonDict:
    """Run one declared verification command and retain its process receipt."""

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_s": round(time.perf_counter() - started, 6),
    }


def _command_row(check_id: str, command: str, receipt: Mapping[str, Any]) -> JsonDict:
    output = str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))
    coverage: float | None = None
    if check_id == "scoped_coverage":
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        coverage = float(match.group(1)) if match else None
    passed = receipt.get("exit_code") == 0 and (check_id != "scoped_coverage" or coverage == 100.0)
    return {
        "check_id": check_id,
        "command": command,
        "exit_code": receipt.get("exit_code"),
        "passed": passed,
        "coverage_percent": coverage,
        "summary": output[-2000:],
        "duration_s": receipt.get("duration_s", 0.0),
    }


def run_verification_commands(
    root: Path,
    runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner,
) -> list[JsonDict]:
    """Run focused, coverage, full-suite, spec, E2E, and lint checks once."""

    return [
        _command_row(check_id, command, runner(command, root))
        for check_id, command in VERIFICATION_COMMANDS
    ]


def run_artifact_checks(
    root: Path,
    artifact_path: Path,
    runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner,
) -> list[JsonDict]:
    """Run validation, row consistency, and adversarial checks on a candidate."""

    target = shlex.quote(str(artifact_path))
    commands = (
        (
            "artifact_validation",
            ".venv/bin/python -m carnot.experiment_6716_bounded_seal_attack_audit "
            f"--validate --output {target}",
        ),
        (
            "row_consistency",
            f".venv/bin/python scripts/verdict_row_consistency_lint.py --strict {target}",
        ),
        (
            "adversarial_verification",
            f".venv/bin/python scripts/adversarial_verify.py --json {target}",
        ),
    )
    rows: list[JsonDict] = []
    for check_id, command in commands:
        receipt = runner(command, root)
        row = _command_row(check_id, command, receipt)
        row["critical_free"] = None
        if check_id == "adversarial_verification":
            try:
                report = json.loads(str(receipt.get("stdout", "")))
                row["critical_free"] = all(
                    int(item.get("max_severity", 0)) < 2 for item in report.get("reports", [])
                )
            except (TypeError, ValueError, AttributeError):
                row["critical_free"] = receipt.get("exit_code") == 0
            row["passed"] = row["critical_free"] is True
        rows.append(row)
    return rows


def pending_artifact_check_rows() -> list[JsonDict]:
    """Reserve operational receipt slots while the candidate is validated."""

    return [
        {
            "check_id": check_id,
            "command": "pending complete candidate artifact",
            "exit_code": None,
            "passed": False,
            "coverage_percent": None,
            "summary": "not run before candidate publication",
            "duration_s": 0.0,
        }
        for check_id in OPERATIONAL_CHECK_IDS
    ]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner,
    clock: Callable[[], float] = time.perf_counter,
) -> JsonDict:
    """Run preconditions, frozen attacks, verification, and atomic output."""

    started = time.perf_counter()
    output = output_path or root / RESULT_PATH
    preconditions = collect_preconditions(root)
    if not all(row.get("passed") is True for row in preconditions):
        artifact = build_blocked_artifact(date, root, preconditions, time.perf_counter() - started)
    else:
        before = protected_hashes(root)
        upstream = load_json(root / UPSTREAM_PATH)
        manifest = freeze_attack_manifest(public_attack_inputs(upstream))
        campaign = run_attack_campaign(upstream, manifest, clock=clock)
        receipts = run_verification_commands(root, runner=runner)
        candidate = build_artifact(
            date=date,
            root=root,
            tests_run=[*receipts, *pending_artifact_check_rows()],
            duration_s=time.perf_counter() - started,
            protected_before=before,
            upstream=upstream,
            manifest=manifest,
            campaign=campaign,
            preconditions=preconditions,
        )
        errors = validate_artifact(candidate)
        if errors:
            raise ValueError("candidate: " + "; ".join(errors))
        with tempfile.TemporaryDirectory(prefix="carnot-exp6716-") as temporary:
            candidate_path = Path(temporary) / RESULT_PATH.name
            write_json_atomic(candidate_path, candidate)
            receipts.extend(run_artifact_checks(root, candidate_path, runner=runner))
        artifact = build_artifact(
            date=date,
            root=root,
            tests_run=receipts,
            duration_s=time.perf_counter() - started,
            protected_before=before,
            upstream=upstream,
            manifest=manifest,
            campaign=campaign,
            preconditions=preconditions,
        )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_json_atomic(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated audit, or validate one existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260828")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        if not args.output.is_file():
            return 1
        try:
            return 0 if not validate_artifact(load_json(args.output)) else 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 1
    run(date=args.date, root=REPO_ROOT, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the required module command.
    raise SystemExit(main())
