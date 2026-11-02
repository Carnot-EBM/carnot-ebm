"""Build the independent V619 evidence capstone.

The reducer reads stored contracts and task receipts. It does not rerun any
science branch. Missing task receipts remain visible and do not stop the final
capstone, because an honest absence is evidence about milestone completion.

Spec refs: REQ-REPORT-7075 and SCENARIO-REPORT-7075-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import yaml

from carnot.experiment_7063_v619_contract_preflight import parse_markdown_contract


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.619"
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7075_v619_capstone.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 707520260906
PROMPT_TAIL = "Do NOT push. Do NOT modify scripts/research_conductor.py."

EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp7063-v619-contract-preflight",
        "title": "V619 Markdown and YAML task-contract preflight",
        "deliverable": "results/experiment_7063_v619_contract_preflight.json",
    },
    {
        "id": "exp7064-exact-entrance-constraint-fixture",
        "title": "Exact source-grouped entrance constraint fixture",
        "deliverable": "results/experiment_7064_v619_exact_entrance_fixture.json",
    },
    {
        "id": "exp7065-three-family-entrance-proposal-bank",
        "title": "Three-family SOTA entrance proposal bank",
        "deliverable": "results/experiment_7065_v619_three_family_entrance_bank.json",
    },
    {
        "id": "exp7066-entrance-bank-independent-audit",
        "title": "Cold recomputation of entrance-bank support",
        "deliverable": "results/experiment_7066_v619_entrance_bank_audit.json",
    },
    {
        "id": "exp7067-hopfield-entrance-energy-selection",
        "title": "Hopfield-style entrance energy selection comparison",
        "deliverable": "results/experiment_7067_v619_entrance_energy_selection.json",
    },
    {
        "id": "exp7068-hierarchical-branch-fidelity-control",
        "title": "Lossless categorical mass-rebalancing evaluation",
        "deliverable": "results/experiment_7068_v619_hierarchical_branch_control.json",
    },
    {
        "id": "exp7069-context-bound-experience-contract",
        "title": "BCIT use-validate-reject state machine",
        "deliverable": "results/experiment_7069_v619_context_authorization_contract.json",
    },
    {
        "id": "exp7070-bcit-prospective-self-learning",
        "title": "Prospective context-bound continuous self-learning comparison",
        "deliverable": "results/experiment_7070_v619_bcit_self_learning.json",
    },
    {
        "id": "exp7071-bcit-drift-rollback-audit",
        "title": "Fresh-process self-learning drift and rollback audit",
        "deliverable": "results/experiment_7071_v619_bcit_drift_audit.json",
    },
    {
        "id": "exp7072-live-arc-compaction-ab",
        "title": "Claim-grade live ARC compaction generalization A/B",
        "deliverable": "results/experiment_7072_v619_live_arc_compaction_ab.json",
    },
    {
        "id": "exp7073-entrance-energy-ising-parity",
        "title": "QUBO translation and finite-distribution equivalence",
        "deliverable": "results/experiment_7073_v619_entrance_ising_parity.json",
    },
    {
        "id": "exp7074-degree16-placement-sampler-audit",
        "title": "Degree-16 placement and finite-sampler audit",
        "deliverable": "results/experiment_7074_v619_degree16_sampler_audit.json",
    },
    {
        "id": "exp7075-v619-capstone",
        "title": "V619 evidence matrix",
        "deliverable": "results/experiment_7075_v619_capstone.json",
    },
)
EXPECTED_TASK_COUNT = len(EXPECTED_TASKS)
EXPECTED_ID_ORDER = tuple(row["id"] for row in EXPECTED_TASKS)

EXPECTED_GATES: dict[str, tuple[tuple[str, str, int], ...]] = {
    EXPECTED_ID_ORDER[2]: ((EXPECTED_ID_ORDER[1], "entrance_fixture_ready_score", 1),),
    EXPECTED_ID_ORDER[3]: ((EXPECTED_ID_ORDER[2], "entrance_proposal_bank_complete_score", 1),),
    EXPECTED_ID_ORDER[4]: (
        (EXPECTED_ID_ORDER[3], "entrance_support_audit_ready_score", 1),
        (EXPECTED_ID_ORDER[3], "entrance_selector_headroom_ready_score", 1),
    ),
    EXPECTED_ID_ORDER[5]: ((EXPECTED_ID_ORDER[3], "entrance_support_audit_ready_score", 1),),
    EXPECTED_ID_ORDER[7]: (
        (EXPECTED_ID_ORDER[6], "context_authorization_contract_ready_score", 1),
    ),
    EXPECTED_ID_ORDER[8]: ((EXPECTED_ID_ORDER[7], "bcit_comparison_complete_score", 1),),
    EXPECTED_ID_ORDER[10]: (
        (EXPECTED_ID_ORDER[4], "entrance_energy_comparison_complete_score", 1),
    ),
    EXPECTED_ID_ORDER[11]: ((EXPECTED_ID_ORDER[10], "ising_parity_ready_score", 1),),
}
PRODUCER_FIELDS: dict[str, tuple[str, ...]] = {
    EXPECTED_ID_ORDER[1]: ("entrance_fixture_ready_score",),
    EXPECTED_ID_ORDER[2]: ("entrance_proposal_bank_complete_score",),
    EXPECTED_ID_ORDER[3]: (
        "entrance_support_audit_ready_score",
        "entrance_selector_headroom_ready_score",
    ),
    EXPECTED_ID_ORDER[4]: ("entrance_energy_comparison_complete_score",),
    EXPECTED_ID_ORDER[6]: ("context_authorization_contract_ready_score",),
    EXPECTED_ID_ORDER[7]: ("bcit_comparison_complete_score",),
    EXPECTED_ID_ORDER[10]: ("ising_parity_ready_score",),
}

EXPECTED_PRIORS: dict[str, tuple[JsonDict, ...]] = {
    EXPECTED_ID_ORDER[0]: (
        {
            "experiment_id": "exp7050-v618-active-contract-preflight",
            "verdict": "complete_disqualified_v618_markdown_yaml_contract_mismatch",
            "addressed_by": "V619 writes one exact 13-row Markdown and YAML contract before activation and parses the two files independently.",
            "retire_if_same_verdict": True,
        },
    ),
    EXPECTED_ID_ORDER[1]: (
        {
            "experiment_id": "exp5708-sota-exact-constraint-canary",
            "verdict": "blocked: parse_failures",
            "addressed_by": "Exp7064 builds deterministic exhaustive entrance labels and replayed witnesses before any model output, instead of depending on a live model parser.",
            "retire_if_same_verdict": True,
        },
    ),
    EXPECTED_ID_ORDER[2]: (
        {
            "experiment_id": "exp6200-three-family-raw-code-transport-canary",
            "verdict": "complete_partial: Exp6200 coverage=18/18 cells; ready_families=[]",
            "addressed_by": "Exp7065 uses short structured entrance proposals, all cached mandated GGUFs, matched llama.cpp transport, raw-first checkpoints, and exact post-generation parsing.",
            "retire_if_same_verdict": True,
        },
    ),
    EXPECTED_ID_ORDER[4]: (
        {
            "experiment_id": "exp1006-energy-selection-ssd",
            "verdict": "blocked_gate_check_failed",
            "addressed_by": "Exp7067 first requires an independently audited fixed proposal bank with measured held headroom, then compares a new Hopfield-style entrance energy against MRV, logits, frequency, uniform, and shuffled controls.",
            "retire_if_same_verdict": True,
        },
    ),
    EXPECTED_ID_ORDER[7]: (
        {
            "experiment_id": "exp6978-transactional-constraint-self-learning",
            "verdict": "complete_null_transactional_constraint_self_learning",
            "addressed_by": "Exp7070 adds context-bound authorization, a validate state, immutable source groups, and equal-budget controls before each transactional commit.",
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "exp7021-prospective-belief-utility",
            "verdict": "complete_null_prospective_belief_utility_not_demonstrated",
            "addressed_by": "Exp7070 replaces context-free belief reuse with BCIT records tied to policy, source, schema, support, conflicts, and bounded current-state validation.",
            "retire_if_same_verdict": True,
        },
    ),
    EXPECTED_ID_ORDER[8]: (
        {
            "experiment_id": "exp6979-self-learning-cold-audit",
            "verdict": "complete_null_self_learning_cold_audit",
            "addressed_by": "Exp7071 audits a deterministic context-bound event contract in a fresh subprocess with realistic duration, explicit drift mutations, and row-level recomputation.",
            "retire_if_same_verdict": True,
        },
    ),
    EXPECTED_ID_ORDER[9]: (
        {
            "experiment_id": "exp6473-tool-loop-compaction-pilot-ab",
            "verdict": "complete: compaction pilot A/B measured on 13 paired live cells; one or more gates FAIL (G-Q is a pilot signal, never a claim)",
            "addressed_by": "Exp7072 uses at least 30 new paired Qwen cells, the current typed identity contract, an explicit compaction-fire gate, a mandated Gemma replication, and claim-grade intervals.",
            "retire_if_same_verdict": True,
        },
    ),
}

COMMON_UPSTREAM_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
BRANCH_DECISIONS = frozenset(
    {
        "release",
        "retain_default_off",
        "retire_null",
        "retire_disqualified",
        "blocked_resource",
        "needs_independent_replication",
    }
)
COMPARISON_TASKS = frozenset(
    {
        EXPECTED_ID_ORDER[4],
        EXPECTED_ID_ORDER[5],
        EXPECTED_ID_ORDER[7],
        EXPECTED_ID_ORDER[9],
        EXPECTED_ID_ORDER[11],
    }
)
BRANCH_TASKS: dict[str, tuple[str, ...]] = {
    "entrance": EXPECTED_ID_ORDER[1:6],
    "self_learning": EXPECTED_ID_ORDER[6:9],
    "arc_compaction": (EXPECTED_ID_ORDER[9],),
    "ising": EXPECTED_ID_ORDER[10:12],
}
BRANCH_TARGETS: dict[str, tuple[str, ...]] = {
    "entrance": (EXPECTED_ID_ORDER[4],),
    "self_learning": (EXPECTED_ID_ORDER[7], EXPECTED_ID_ORDER[8]),
    "arc_compaction": (EXPECTED_ID_ORDER[9],),
    "ising": (EXPECTED_ID_ORDER[10], EXPECTED_ID_ORDER[11]),
}

REQUIRED_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
        "duration_s",
        "source_artifact_hashes",
        "cited_upstream_artifacts",
        "rows",
        "task_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "contract_recomputation_rows",
        "gate_recomputation_rows",
        "artifact_discovery_rows",
        "artifact_validation_rows",
        "checksum_validation_rows",
        "source_hash_rows",
        "model_identity_rows",
        "row_headline_recomputation_rows",
        "science_result_rows",
        "artifact_schema_result_rows",
        "task_contract_result_rows",
        "verdict_class_rows",
        "circularity_rows",
        "branch_decision_rows",
        "retirement_rows",
        "default_off_rows",
        "hardware_claim_boundary_rows",
        "entrance_branch_decision",
        "self_learning_branch_decision",
        "arc_compaction_branch_decision",
        "ising_branch_decision",
        "milestone_release_ready_score",
        "v620_handoff_rows",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)


def canonical_bytes(value: Any, *, ensure_ascii: bool = True, newline: bool = False) -> bytes:
    """Serialize evidence with a stable order so byte hashes are reproducible."""

    text = json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":"))
    return (text + ("\n" if newline else "")).encode("utf-8")


def sha256_file(path: Path) -> str | None:
    """Hash one readable file without loading a large artifact twice."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic capstone content and exclude measured elapsed time."""

    stable = deepcopy(dict(artifact))
    stable.pop("reproducibility_checksum", None)
    stable.pop("duration_s", None)
    return "sha256:" + hashlib.sha256(canonical_bytes(stable)).hexdigest()


def _remove_timing(value: Any) -> Any:
    """Remove nested timing fields used by upstream timing-free conventions."""

    if isinstance(value, Mapping):
        return {
            key: _remove_timing(child)
            for key, child in value.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_remove_timing(child) for child in value]
    return value


def _checksum_candidates(artifact: Mapping[str, Any]) -> set[str]:
    """Recompute the documented checksum conventions used by V619 producers."""

    base = deepcopy(dict(artifact))
    projections: list[Any] = []
    dropped = deepcopy(base)
    dropped.pop("reproducibility_checksum", None)
    projections.append(dropped)
    blanked = deepcopy(base)
    blanked["reproducibility_checksum"] = ""
    projections.append(blanked)
    no_duration = deepcopy(dropped)
    no_duration.pop("duration_s", None)
    projections.append(no_duration)
    zero_duration = deepcopy(blanked)
    zero_duration["duration_s"] = 0.0
    projections.append(zero_duration)
    projections.append(_remove_timing(base))
    return {
        "sha256:"
        + hashlib.sha256(
            canonical_bytes(item, ensure_ascii=ascii_mode, newline=with_newline)
        ).hexdigest()
        for item in projections
        for ascii_mode in (True, False)
        for with_newline in (False, True)
    }


def _infer_verdict_class(verdict: object) -> str | None:
    """Read the terminal meaning without trusting the declared class field."""

    text = str(verdict).lower()
    if "disqualified" in text:
        return "disqualified"
    if "circular_positive" in text or "circular-positive" in text:
        return "circular_positive"
    if "blocked" in text:
        return "blocked"
    if "partial" in text:
        return "partial"
    if "null" in text or "no_improvement" in text:
        return "null"
    if text.startswith(("complete", "success", "passed", "shipped")):
        return "positive"
    return None


def classify_artifact(artifact: Mapping[str, Any] | None) -> JsonDict:
    """Keep missing and every closed verdict class separate for later axes."""

    if artifact is None:
        return {
            "declared_verdict_class": None,
            "inferred_verdict_class": None,
            "effective_verdict_class": "missing",
            "class_matches": None,
        }
    declared = artifact.get("verdict_class")
    inferred = _infer_verdict_class(artifact.get("honest_verdict"))
    matches = declared in VERDICT_CLASSES and declared == inferred
    effective = str(declared) if matches else "disqualified"
    if effective == "positive" and artifact.get("verifier_is_oracle") is True:
        effective = "circular_positive"
    return {
        "declared_verdict_class": declared,
        "inferred_verdict_class": inferred,
        "effective_verdict_class": effective,
        "class_matches": matches,
    }


def _readable(path: Path) -> tuple[bool, str]:
    """Report exact precondition state without repairing a missing source."""

    if not path.is_file():
        return False, "missing"
    try:
        return (
            path.stat().st_size > 0,
            "readable_nonempty_file" if path.stat().st_size > 0 else "empty",
        )
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _preconditions(root: Path, output_path: Path) -> list[JsonDict]:
    """Check only prerequisites whose absence prevents capstone execution."""

    rows = []
    for check, relative in (
        ("active_roadmap_readable", ROADMAP_PATH),
        ("v619_design_readable", DESIGN_PATH),
        ("conductor_log_readable", CONDUCTOR_LOG_PATH),
        ("exclusion_manifest_readable", EXCLUSION_PATH),
    ):
        passed, observed = _readable(root / relative)
        rows.append(
            {
                "check": check,
                "passed": passed,
                "expected_value": "readable_nonempty_file",
                "observed_value": observed,
            }
        )
    target = output_path if output_path.is_absolute() else root / output_path
    parent = target.parent
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    rows.append(
        {
            "check": "capstone_path_writable",
            "passed": writable,
            "expected_value": "writable_parent",
            "observed_value": "writable_parent" if writable else "missing_or_unwritable_parent",
        }
    )
    return rows


def _normalized_gates(task: Mapping[str, Any]) -> tuple[tuple[Any, Any, Any], ...]:
    """Normalize structured gates while preserving source order."""

    gates = task.get("gated_on", task.get("gates", []))
    if not isinstance(gates, list):
        return ()
    return tuple(
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("value"))
        for gate in gates
        if isinstance(gate, Mapping) and gate.get("op", "==") == "=="
    )


def _parse_contracts(root: Path) -> tuple[JsonDict, list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Parse YAML and Markdown from separate bytes, then compare fixed rows."""

    yaml_document = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    yaml_tasks = yaml_document.get("tasks", []) if isinstance(yaml_document, Mapping) else []
    markdown_document = parse_markdown_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
    markdown_tasks = markdown_document.get("tasks", [])
    yaml_rows = [dict(task) if isinstance(task, Mapping) else {} for task in yaml_tasks]
    markdown_rows = [dict(task) if isinstance(task, Mapping) else {} for task in markdown_tasks]
    comparisons = []
    width = max(EXPECTED_TASK_COUNT, len(yaml_rows), len(markdown_rows))
    for index in range(width):
        expected = EXPECTED_TASKS[index] if index < EXPECTED_TASK_COUNT else {}
        yaml_row = yaml_rows[index] if index < len(yaml_rows) else {}
        markdown_row = markdown_rows[index] if index < len(markdown_rows) else {}
        task_id = expected.get("id")
        expected_gates = EXPECTED_GATES.get(str(task_id), ())
        yaml_gates = _normalized_gates(yaml_row)
        markdown_gates = _normalized_gates(markdown_row)
        prompt = str(yaml_row.get("prompt", ""))
        producer_fields = PRODUCER_FIELDS.get(str(task_id), ())
        producer_fields_match = all(field in prompt for field in producer_fields)
        expected_priors = list(EXPECTED_PRIORS.get(str(task_id), ()))
        observed_priors = yaml_row.get("prior_failures", [])
        priors_match = observed_priors == expected_priors
        base_fields = ("id", "title", "deliverable")
        matches = (
            all(
                yaml_row.get(field) == expected.get(field) == markdown_row.get(field)
                for field in base_fields
            )
            and yaml_gates == expected_gates == markdown_gates
            and producer_fields_match
            and prompt.rstrip().endswith(PROMPT_TAIL)
            and priors_match
            and (str(task_id) != EXPECTED_ID_ORDER[-1] or not yaml_gates)
        )
        comparisons.append(
            {
                "order": index + 1,
                "task_id": task_id,
                "expected": {field: expected.get(field) for field in base_fields},
                "yaml": {field: yaml_row.get(field) for field in base_fields},
                "markdown": {field: markdown_row.get(field) for field in base_fields},
                "expected_gates": [list(row) for row in expected_gates],
                "yaml_gates": [list(row) for row in yaml_gates],
                "markdown_gates": [list(row) for row in markdown_gates],
                "producer_fields": list(producer_fields),
                "producer_fields_match": producer_fields_match,
                "prompt_tail_match": prompt.rstrip().endswith(PROMPT_TAIL),
                "expected_prior_failures": expected_priors,
                "observed_prior_failures": observed_priors,
                "prior_failures_match": priors_match,
                "ungated_capstone": str(task_id) != EXPECTED_ID_ORDER[-1] or not yaml_gates,
                "matches": matches,
            }
        )
    metadata = {
        "yaml_milestone": yaml_document.get("milestone")
        if isinstance(yaml_document, Mapping)
        else None,
        "markdown_milestone": markdown_document.get("milestone"),
        "milestone_matches": yaml_document.get("milestone")
        == MILESTONE
        == markdown_document.get("milestone")
        if isinstance(yaml_document, Mapping)
        else False,
    }
    return metadata, yaml_rows, markdown_rows, comparisons


def _load_artifact(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    """Load one declared deliverable and preserve parse errors as row evidence."""

    if not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, Mapping):
        return None, "artifact_root_not_mapping"
    return value, None


def _source_hash_evidence(root: Path, task_id: str, artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute each named source hash without assuming an absent file exists."""

    source = artifact.get("source_artifact_hashes", {})
    items: list[tuple[object, object]] = []
    if isinstance(source, Mapping) and isinstance(source.get("files"), list):
        items.extend(
            (row.get("path"), row.get("sha256"))
            for row in source["files"]
            if isinstance(row, Mapping)
        )
    elif isinstance(source, Mapping):
        items.extend((key, value) for key, value in source.items() if isinstance(value, str))
    rows = []
    for raw_path, expected in items:
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved)
        rows.append(
            {
                "task_id": task_id,
                "source_path": str(raw_path),
                "expected_hash": expected,
                "observed_hash": observed,
                "matches": observed == expected,
            }
        )
    return rows


def _model_identity_row(task_id: str, artifact: Mapping[str, Any] | None) -> JsonDict:
    """Require live identity rows only when the task reached live inference."""

    if artifact is None:
        return {
            "task_id": task_id,
            "applicable": False,
            "result": "missing_artifact",
            "valid": None,
        }
    substrate = str(artifact.get("inference_substrate", ""))
    if substrate != "live_llm_inference":
        return {
            "task_id": task_id,
            "applicable": False,
            "result": "non_live_substrate",
            "valid": True,
        }
    rows = artifact.get("model_identity_rows")
    if artifact.get("verdict_class") == "blocked" and isinstance(rows, list) and not rows:
        return {
            "task_id": task_id,
            "applicable": True,
            "result": "blocked_before_invocation",
            "valid": True,
        }
    valid = isinstance(rows, list) and bool(rows) and all(isinstance(row, Mapping) for row in rows)
    return {
        "task_id": task_id,
        "applicable": True,
        "result": "identity_rows_present" if valid else "identity_rows_missing",
        "valid": valid,
    }


def _headline_row(task_id: str, artifact: Mapping[str, Any] | None) -> JsonDict:
    """Recompute a declared two-arm headline from comparable unit rows."""

    empty = {
        "task_id": task_id,
        "status": "not_available",
        "claimed": None,
        "recomputed": None,
        "matches_claimed": True,
    }
    if artifact is None:
        return empty
    spec = artifact.get("comparative_headline")
    rows = artifact.get("rows")
    if not isinstance(spec, Mapping):
        status = "not_declared" if task_id in COMPARISON_TASKS else "not_comparative"
        return {**empty, "status": status}
    if not isinstance(rows, list):
        return {**empty, "status": "rows_missing", "matches_claimed": False}
    treatment_field = str(spec.get("treatment_field", "treatment"))
    control_field = str(spec.get("control_field", "control"))
    higher = spec.get("higher_is_better", True) is not False
    wins = losses = ties = comparable = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        treatment = row.get(treatment_field)
        control = row.get(control_field)
        if (
            not isinstance(treatment, (int, float))
            or isinstance(treatment, bool)
            or not isinstance(control, (int, float))
            or isinstance(control, bool)
        ):
            continue
        comparable += 1
        delta = float(treatment) - float(control)
        if delta == 0:
            ties += 1
        elif (delta > 0) == higher:
            wins += 1
        else:
            losses += 1
    recomputed = {"wins": wins, "losses": losses, "ties": ties, "comparable_units": comparable}
    claimed = {key: spec.get(key) for key in recomputed}
    return {
        "task_id": task_id,
        "status": "recomputed",
        "claimed": claimed,
        "recomputed": recomputed,
        "matches_claimed": claimed == recomputed,
    }


def _upstream_schema_errors(
    artifact: Mapping[str, Any], classification: Mapping[str, Any]
) -> list[str]:
    """Validate common evidence fields without letting science determine schema."""

    errors = [f"missing_field:{field}" for field in COMMON_UPSTREAM_FIELDS if field not in artifact]
    if classification.get("class_matches") is not True:
        errors.append("verdict_class_mismatch")
    if artifact.get("reproducibility_checksum") not in _checksum_candidates(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        if (
            not isinstance(summary, Mapping)
            or summary.get("failed_check") in {None, ""}
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_gate_summary_incomplete")
    if artifact.get("verdict_class") == "positive" and artifact.get("verifier_is_oracle") is True:
        errors.append("positive_depends_on_oracle")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles_not_mapping")
    return sorted(set(errors))


def _gate_evidence_matches(
    artifact: Mapping[str, Any], field: str, observed: Any, expected: Any
) -> bool:
    """Compare a consumer's stored gate receipt with fresh producer evidence."""

    rows = artifact.get("upstream_gate_rows")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        check = row.get("check", row.get("artifact_field"))
        if check == field:
            return (
                row.get("observed_value") == observed
                and row.get("expected_value") == expected
                and row.get("passed") == (observed == expected)
            )
    return False


def _branch_decision(
    branch: str,
    artifacts: Mapping[str, Mapping[str, Any] | None],
    classifications: Mapping[str, Mapping[str, Any]],
    valid_by_task: Mapping[str, bool],
) -> JsonDict:
    """Choose one closed terminal disposition from target and dependency evidence."""

    task_ids = BRANCH_TASKS[branch]
    target_ids = BRANCH_TARGETS[branch]
    classes = {task_id: classifications[task_id]["effective_verdict_class"] for task_id in task_ids}
    target_classes = [classes[task_id] for task_id in target_ids]
    if "disqualified" in target_classes:
        decision, reason = "retire_disqualified", "A target receipt is disqualified."
    elif any(value in {"missing", "blocked"} for value in classes.values()):
        decision, reason = "blocked_resource", "A required task is missing or resource-blocked."
    elif any(value == "partial" for value in target_classes):
        decision, reason = "needs_independent_replication", "A target receipt is partial."
    elif any(value == "circular_positive" for value in target_classes):
        decision, reason = (
            "needs_independent_replication",
            "Target value depends on circular or oracle evidence.",
        )
    elif any(not valid_by_task.get(task_id, False) for task_id in task_ids):
        decision, reason = (
            "needs_independent_replication",
            "Required branch evidence failed artifact validation.",
        )
    elif any(value == "null" for value in target_classes):
        if branch == "self_learning" and not all(value == "null" for value in target_classes):
            decision, reason = (
                "needs_independent_replication",
                "Self-learning has one null without its matching audit.",
            )
        else:
            decision, reason = "retire_null", "The complete target evidence is null."
    elif all(value == "positive" for value in target_classes):
        decision, reason = (
            "release",
            "All target receipts are valid and oracle-distinct positive evidence.",
        )
    else:
        decision, reason = (
            "needs_independent_replication",
            "The target evidence does not justify release or retirement.",
        )
    return {
        "branch": branch,
        "task_ids": list(task_ids),
        "target_task_ids": list(target_ids),
        "evidence_classes": classes,
        "decision": decision,
        "reason": reason,
    }


def _hardware_rows() -> list[JsonDict]:
    """State the strongest hardware claim that each branch can support."""

    return [
        {
            "branch": "entrance",
            "boundary": "RTX proposal generation and deterministic CPU labels only; no hardware speed claim.",
        },
        {
            "branch": "self_learning",
            "boundary": "Deterministic CPU authorization and replay only; no accelerated learning claim.",
        },
        {
            "branch": "arc_compaction",
            "boundary": "Only owned RTX live-agent runs can earn credit; offline adapters cannot.",
        },
        {
            "branch": "ising",
            "boundary": "Software Ising parity only; no Z1, FPGA, power, latency, or speed claim.",
        },
    ]


def _retirement_rule_rows(classifications: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute each V619 retire-if-same rule from current and prior classes."""

    rows = []
    for task_id in (EXPECTED_ID_ORDER[7], EXPECTED_ID_ORDER[8], EXPECTED_ID_ORDER[9]):
        current_class = classifications[task_id]["effective_verdict_class"]
        for prior in EXPECTED_PRIORS[task_id]:
            prior_class = _infer_verdict_class(prior["verdict"])
            if task_id == EXPECTED_ID_ORDER[9] and "gates fail" in prior["verdict"].lower():
                prior_class = "null"
            same_verdict = current_class == prior_class
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior["experiment_id"],
                    "prior_verdict": prior["verdict"],
                    "prior_verdict_class": prior_class,
                    "current_verdict_class": current_class,
                    "retire_if_same_verdict": prior["retire_if_same_verdict"],
                    "same_verdict": same_verdict,
                    "retirement_triggered": (
                        prior["retire_if_same_verdict"] is True
                        and same_verdict
                        and current_class == "null"
                    ),
                }
            )
    return rows


def _handoff_rows(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Turn terminal V619 decisions into bounded V620 evidence requests."""

    questions = {
        "entrance": (
            "Can an oracle-distinct entrance energy beat MRV and logit controls?",
            "A valid three-model bank, cold support audit, and held per-unit selector rows.",
        ),
        "self_learning": (
            "Does context-bound reuse improve held chronological outcomes without harmful transfer?",
            "At least 120 frozen events across 12 source groups plus a fresh-process drift audit.",
        ),
        "arc_compaction": (
            "Does activated compaction improve claim-grade live hidden-game outcomes?",
            "At least 30 eligible paired Qwen cells, Gemma replication, and observed treatment firing.",
        ),
        "ising": (
            "Does the selected entrance energy preserve exact QUBO, Ising, and finite-sampler semantics?",
            "A completed entrance-energy artifact followed by exact parity and degree-16 software audits.",
        ),
    }
    hardware = {row["branch"]: row["boundary"] for row in _hardware_rows()}
    rows = []
    for decision in decisions:
        branch = str(decision["branch"])
        disposition = str(decision["decision"])
        question, evidence = questions[branch]
        rows.append(
            {
                "branch": branch,
                "highest_value_unresolved_question": None
                if disposition.startswith("retire_")
                else question,
                "required_evidence": None if disposition.startswith("retire_") else evidence,
                "retired_scopes": [branch] if disposition.startswith("retire_") else [],
                "default_off_flags": [branch]
                if disposition
                in {"blocked_resource", "needs_independent_replication", "retain_default_off"}
                else [],
                "hardware_boundary": hardware[branch],
                "continuous_learning_follow_up": (
                    "Build one larger immutable BCIT stream before rerunning Exp7070 and Exp7071."
                    if branch == "self_learning" and not disposition.startswith("retire_")
                    else None
                ),
            }
        )
    return rows


def _field_principles() -> dict[str, str]:
    """Give each required field one falsifiable scientific purpose."""

    return {
        field: f"The {field} field preserves evidence needed to reproduce or falsify the V619 capstone."
        for field in sorted(REQUIRED_FIELDS)
    }


def _base_artifact(
    run_date: str, preconditions: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Create a schema-complete shell for both blocked and completed outcomes."""

    empty_rows = {
        "contract_recomputation_rows",
        "gate_recomputation_rows",
        "artifact_discovery_rows",
        "artifact_validation_rows",
        "checksum_validation_rows",
        "source_hash_rows",
        "model_identity_rows",
        "row_headline_recomputation_rows",
        "science_result_rows",
        "artifact_schema_result_rows",
        "task_contract_result_rows",
        "verdict_class_rows",
        "circularity_rows",
        "branch_decision_rows",
        "retirement_rows",
        "default_off_rows",
        "hardware_claim_boundary_rows",
        "v620_handoff_rows",
        "rows",
        "task_rows",
    }
    artifact: JsonDict = {field: [] for field in empty_rows}
    artifact.update(
        {
            "field_principles": _field_principles(),
            "preconditions_checked": deepcopy(list(preconditions)),
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": round(duration_s, 6),
            "source_artifact_hashes": {},
            "cited_upstream_artifacts": [],
            "expected_task_count": EXPECTED_TASK_COUNT,
            "observed_task_count": 0,
            "expected_id_order": list(EXPECTED_ID_ORDER),
            "observed_id_order": [],
            "entrance_branch_decision": "blocked_resource",
            "self_learning_branch_decision": "blocked_resource",
            "arc_compaction_branch_decision": "blocked_resource",
            "ising_branch_decision": "blocked_resource",
            "milestone_release_ready_score": 0,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "gate_check_summary": {},
            "verifier_is_oracle": False,
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_v619_capstone_prerequisite_missing",
            "run_date": run_date,
            "schema": "v619_capstone_v1",
        }
    )
    return artifact


def build_capstone(
    root: Path = REPO_ROOT, run_date: str = "20260906", output_path: Path = DEFAULT_OUTPUT_PATH
) -> JsonDict:
    """Recompute the V619 contract and every available evidence receipt."""

    started = time.monotonic()
    preconditions = _preconditions(root, output_path)
    artifact = _base_artifact(run_date, preconditions, time.monotonic() - started)
    failed = next((row for row in preconditions if row["passed"] is not True), None)
    if failed is not None:
        artifact["gate_check_summary"] = (
            {
                key: failed[key]
                for key in ("passed", "failed_check", "expected_value", "observed_value")
            }
            if "failed_check" in failed
            else {
                "passed": False,
                "failed_check": failed["check"],
                "expected_value": failed["expected_value"],
                "observed_value": failed["observed_value"],
            }
        )
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    metadata, yaml_rows, _markdown_rows, contract_rows = _parse_contracts(root)
    artifact["contract_recomputation_rows"] = contract_rows
    artifact["observed_task_count"] = len(yaml_rows)
    artifact["observed_id_order"] = [row.get("id") for row in yaml_rows]
    contract_ok = (
        metadata["milestone_matches"]
        and len(yaml_rows) == EXPECTED_TASK_COUNT
        and all(row["matches"] for row in contract_rows)
    )
    artifact["task_contract_result_rows"] = [
        {
            "task_id": row["task_id"],
            "result": "pass" if row["matches"] else "fail",
            "matches": row["matches"],
        }
        for row in contract_rows
    ]

    artifacts: dict[str, Mapping[str, Any] | None] = {}
    parse_errors: dict[str, str | None] = {}
    paths: dict[str, Path] = {}
    for task in EXPECTED_TASKS[:-1]:
        task_id = task["id"]
        path = root / task["deliverable"]
        value, error = _load_artifact(path)
        artifacts[task_id] = value
        parse_errors[task_id] = error
        paths[task_id] = path
        artifact["artifact_discovery_rows"].append(
            {
                "task_id": task_id,
                "declared_deliverable": task["deliverable"],
                "discovery_result": "unreadable"
                if error
                else "available"
                if value is not None
                else "missing",
                "error": error,
            }
        )

    classifications = {task_id: classify_artifact(value) for task_id, value in artifacts.items()}
    gate_errors: dict[str, list[str]] = {task_id: [] for task_id in artifacts}
    for consumer_id, gates in EXPECTED_GATES.items():
        consumer = artifacts.get(consumer_id)
        for producer_id, field, expected in gates:
            producer = artifacts.get(producer_id)
            observed = producer.get(field) if producer is not None else None
            passed = observed == expected
            evidence_match = (
                _gate_evidence_matches(consumer, field, observed, expected)
                if consumer is not None
                else None
            )
            artifact["gate_recomputation_rows"].append(
                {
                    "consumer_task_id": consumer_id,
                    "producer_task_id": producer_id,
                    "artifact_field": field,
                    "operator": "==",
                    "expected_value": expected,
                    "observed_value": observed,
                    "passed": passed,
                    "consumer_gate_evidence_matches": evidence_match,
                }
            )
            if consumer is not None and evidence_match is not True:
                gate_errors[consumer_id].append("gate_evidence_mismatch")

    valid_by_task: dict[str, bool] = {}
    for task in EXPECTED_TASKS[:-1]:
        task_id = task["id"]
        value = artifacts[task_id]
        classification = classifications[task_id]
        artifact["verdict_class_rows"].append({"task_id": task_id, **classification})
        artifact["science_result_rows"].append(
            {"task_id": task_id, "science_result": classification["effective_verdict_class"]}
        )
        artifact["circularity_rows"].append(
            {
                "task_id": task_id,
                "verifier_is_oracle": value.get("verifier_is_oracle") if value else None,
                "circular": classification["effective_verdict_class"] == "circular_positive",
            }
        )
        model_row = _model_identity_row(task_id, value)
        artifact["model_identity_rows"].append(model_row)
        headline_row = _headline_row(task_id, value)
        artifact["row_headline_recomputation_rows"].append(headline_row)
        if value is None:
            errors = ["artifact_unreadable"] if parse_errors[task_id] else ["artifact_missing"]
            checksum_match = None
            source_rows: list[JsonDict] = []
        else:
            errors = _upstream_schema_errors(value, classification)
            errors.extend(gate_errors[task_id])
            if model_row["valid"] is False:
                errors.append("model_identity_invalid")
            if headline_row["matches_claimed"] is False:
                errors.append("row_headline_mismatch")
            checksum_match = value.get("reproducibility_checksum") in _checksum_candidates(value)
            source_rows = _source_hash_evidence(root, task_id, value)
            if any(row["matches"] is not True for row in source_rows):
                errors.append("source_hash_mismatch")
        artifact["source_hash_rows"].extend(source_rows)
        errors = sorted(set(errors))
        schema_result = (
            "missing"
            if value is None and not parse_errors[task_id]
            else "fail"
            if errors
            else "pass"
        )
        artifact["artifact_schema_result_rows"].append(
            {"task_id": task_id, "result": schema_result, "errors": errors}
        )
        artifact["checksum_validation_rows"].append(
            {"task_id": task_id, "checksum_matches": checksum_match}
        )
        source_valid = all(row["matches"] is True for row in source_rows)
        valid = value is not None and not errors and checksum_match is True and source_valid
        valid_by_task[task_id] = valid
        artifact["artifact_validation_rows"].append(
            {
                "task_id": task_id,
                "valid": valid,
                "schema_result": schema_result,
                "checksum_matches": checksum_match,
                "source_hashes_match": source_valid,
                "errors": errors,
            }
        )

    decisions = [
        _branch_decision(branch, artifacts, classifications, valid_by_task)
        for branch in BRANCH_TASKS
    ]
    artifact["branch_decision_rows"] = decisions
    decision_map = {row["branch"]: row["decision"] for row in decisions}
    artifact["entrance_branch_decision"] = decision_map["entrance"]
    artifact["self_learning_branch_decision"] = decision_map["self_learning"]
    artifact["arc_compaction_branch_decision"] = decision_map["arc_compaction"]
    artifact["ising_branch_decision"] = decision_map["ising"]
    artifact["retirement_rows"] = _retirement_rule_rows(classifications)
    artifact["default_off_rows"] = [
        {
            "branch": branch,
            "default_off": decision
            in {"blocked_resource", "needs_independent_replication", "retain_default_off"},
        }
        for branch, decision in decision_map.items()
    ]
    artifact["hardware_claim_boundary_rows"] = _hardware_rows()
    artifact["v620_handoff_rows"] = _handoff_rows(decisions)

    headlines_ok = all(
        row["matches_claimed"] is not False for row in artifact["row_headline_recomputation_rows"]
    )
    no_oracle_positive = all(
        row["effective_verdict_class"] != "circular_positive" or task_id == EXPECTED_ID_ORDER[1]
        for task_id, row in classifications.items()
    )
    target_evidence_valid = all(
        decision["decision"] != "release"
        or all(valid_by_task.get(task_id, False) for task_id in decision["target_task_ids"])
        for decision in decisions
    )
    decisions_terminal = all(row["decision"] in BRANCH_DECISIONS for row in decisions)
    no_unresolved_replication = all(
        row["decision"] != "needs_independent_replication" for row in decisions
    )
    release_ready = (
        contract_ok
        and headlines_ok
        and no_oracle_positive
        and target_evidence_valid
        and decisions_terminal
        and no_unresolved_replication
    )
    artifact["milestone_release_ready_score"] = int(release_ready)

    if not contract_ok or any(row["decision"] == "retire_disqualified" for row in decisions):
        verdict_class = "disqualified"
        verdict = "complete_disqualified_v619_contract_or_target_evidence"
    elif any(
        row["decision"] == "needs_independent_replication" and "circular" in row["reason"].lower()
        for row in decisions
    ):
        verdict_class = "circular_positive"
        verdict = "complete_circular_positive_v619_requires_independent_replication"
    elif release_ready and all(row["decision"] == "release" for row in decisions):
        verdict_class = "positive"
        verdict = "complete_positive_v619_branches_release"
    else:
        verdict_class = "null"
        verdict = "complete_null_v619_terminal_branch_dispositions"
    artifact["verdict_class"] = verdict_class
    artifact["honest_verdict"] = verdict

    artifact["artifact_schema_result_rows"].append(
        {"task_id": EXPECTED_ID_ORDER[-1], "result": "pass", "errors": []}
    )
    artifact["science_result_rows"].append(
        {"task_id": EXPECTED_ID_ORDER[-1], "science_result": verdict_class}
    )
    artifact["verdict_class_rows"].append(
        {
            "task_id": EXPECTED_ID_ORDER[-1],
            "declared_verdict_class": verdict_class,
            "inferred_verdict_class": verdict_class,
            "effective_verdict_class": verdict_class,
            "class_matches": True,
        }
    )
    artifact["circularity_rows"].append(
        {
            "task_id": EXPECTED_ID_ORDER[-1],
            "verifier_is_oracle": False,
            "circular": verdict_class == "circular_positive",
        }
    )
    contract_result_by_task = {
        row["task_id"]: row["result"] for row in artifact["task_contract_result_rows"]
    }
    artifact["task_rows"] = [
        {
            "task_id": task_id,
            "science_result": next(
                row["science_result"]
                for row in artifact["science_result_rows"]
                if row["task_id"] == task_id
            ),
            "artifact_schema_result": next(
                row["result"]
                for row in artifact["artifact_schema_result_rows"]
                if row["task_id"] == task_id
            ),
            "task_contract_result": contract_result_by_task.get(task_id, "fail"),
        }
        for task_id in EXPECTED_ID_ORDER
    ]
    artifact["rows"] = deepcopy(artifact["task_rows"])
    capstone_sources = [ROADMAP_PATH, DESIGN_PATH, CONDUCTOR_LOG_PATH, EXCLUSION_PATH]
    capstone_sources.extend(
        Path(task["deliverable"])
        for task in EXPECTED_TASKS[:-1]
        if artifacts[task["id"]] is not None
    )
    artifact["source_artifact_hashes"] = {
        str(path): sha256_file(root / path) for path in capstone_sources
    }
    artifact["cited_upstream_artifacts"] = [
        {
            "task_id": task["id"],
            "path": task["deliverable"],
            "sha256": sha256_file(paths[task["id"]]),
        }
        for task in EXPECTED_TASKS[:-1]
        if artifacts[task["id"]] is not None
    ]
    artifact["gate_check_summary"] = {
        "passed": release_ready,
        "failed_check": None if release_ready else "milestone_release_ready_score",
        "expected_value": 1,
        "observed_value": int(release_ready),
    }
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the capstone's closed schema and terminal checksum rules."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in artifact]
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not str(principles.get(field, "")).strip() for field in REQUIRED_FIELDS
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("expected_task_count") != EXPECTED_TASK_COUNT:
        errors.append("expected_task_count_mismatch")
    observed_ids = artifact.get("observed_id_order")
    if not isinstance(observed_ids, list) or artifact.get("observed_task_count") != len(
        observed_ids
    ):
        errors.append("observed_task_count_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    classification = classify_artifact(artifact)
    if classification["class_matches"] is not True:
        errors.append("verdict_class_mismatch")
    for field in (
        "entrance_branch_decision",
        "self_learning_branch_decision",
        "arc_compaction_branch_decision",
        "ising_branch_decision",
    ):
        if artifact.get(field) not in BRANCH_DECISIONS:
            errors.append(f"invalid_branch_decision:{field}")
    if artifact.get("milestone_release_ready_score") not in {0, 1}:
        errors.append("milestone_release_ready_score_invalid")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        if (
            not isinstance(summary, Mapping)
            or summary.get("failed_check") in {None, ""}
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_gate_summary_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """Write one validated capstone to the requested local output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260906")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.root / args.output
    artifact = build_capstone(args.root, args.date, args.output)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid capstone: " + ", ".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
