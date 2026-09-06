"""Build the V617 release, hold, repair, and retirement disposition.

The reducer reads stored task artifacts and independently parsed contracts. It
does not run a model or infer a downstream claim from an upstream ready flag.
Missing upstream work is useful terminal evidence, so it becomes blocked data.

Spec refs: REQ-CAP-7049 and SCENARIO-CAP-7049-*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml

from carnot.experiment_7038_v617_active_contract_preflight import parse_markdown_contract


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.617"
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/capstone/spec.md")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7049_v617_capstone_disposition.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 704920260906
UNIFORM_SCOPE_KEY = "v617_uniform_belief_influence_repeated_exact_outcome_null"

EXPECTED_TASK_IDS = (
    "exp7038-v617-active-contract-preflight",
    "exp7039-live-model-report-channel-forensics",
    "exp7040-typed-model-identity-report-bridge",
    "exp7041-identity-report-channel-cold-audit",
    "exp7042-typed-identity-belief-shadow-trace",
    "exp7043-belief-shadow-trace-cold-audit",
    "exp7044-uniform-belief-two-model-live-ab",
    "exp7045-uniform-belief-value-cold-audit",
    "exp7046-frontier-stratified-exact-outcome-curriculum",
    "exp7047-selective-belief-exact-advantage-csl",
    "exp7048-selective-belief-two-model-live-ab",
    "exp7049-v617-evidence-disposition-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_TASK_IDS)

TASK_PATHS = {
    EXPECTED_TASK_IDS[0]: "results/experiment_7038_v617_active_contract_preflight.json",
    EXPECTED_TASK_IDS[1]: "results/experiment_7039_v617_model_report_forensics.json",
    EXPECTED_TASK_IDS[2]: "results/experiment_7040_v617_typed_identity_bridge.json",
    EXPECTED_TASK_IDS[3]: "results/experiment_7041_v617_identity_attack_audit.json",
    EXPECTED_TASK_IDS[4]: "results/experiment_7042_v617_belief_shadow_trace.json",
    EXPECTED_TASK_IDS[5]: "results/experiment_7043_v617_belief_shadow_cold_audit.json",
    EXPECTED_TASK_IDS[6]: "results/experiment_7044_v617_uniform_belief_live_ab.json",
    EXPECTED_TASK_IDS[7]: "results/experiment_7045_v617_uniform_belief_cold_audit.json",
    EXPECTED_TASK_IDS[8]: "results/experiment_7046_v617_frontier_stratified_curriculum.json",
    EXPECTED_TASK_IDS[9]: "results/experiment_7047_v617_selective_belief_csl.json",
    EXPECTED_TASK_IDS[10]: "results/experiment_7048_v617_selective_belief_live_ab.json",
    EXPECTED_TASK_IDS[11]: str(DEFAULT_OUTPUT_PATH),
}

EXPECTED_GATES = {
    EXPECTED_TASK_IDS[2]: ((EXPECTED_TASK_IDS[1], "arc_report_channel_forensics_ready_score"),),
    EXPECTED_TASK_IDS[3]: ((EXPECTED_TASK_IDS[2], "arc_typed_identity_bridge_ready_score"),),
    EXPECTED_TASK_IDS[4]: (
        (EXPECTED_TASK_IDS[2], "arc_typed_identity_bridge_ready_score"),
        (EXPECTED_TASK_IDS[3], "arc_identity_report_attack_audit_ready_score"),
    ),
    EXPECTED_TASK_IDS[5]: ((EXPECTED_TASK_IDS[4], "belief_shadow_transport_ready_score"),),
    EXPECTED_TASK_IDS[6]: (
        (EXPECTED_TASK_IDS[4], "belief_shadow_transport_ready_score"),
        (EXPECTED_TASK_IDS[5], "belief_shadow_trace_audit_ready_score"),
    ),
    EXPECTED_TASK_IDS[7]: ((EXPECTED_TASK_IDS[6], "uniform_belief_live_ab_complete_score"),),
    EXPECTED_TASK_IDS[8]: (
        (EXPECTED_TASK_IDS[7], "uniform_belief_value_audit_complete_score"),
    ),
    EXPECTED_TASK_IDS[9]: ((EXPECTED_TASK_IDS[8], "belief_frontier_curriculum_ready_score"),),
    EXPECTED_TASK_IDS[10]: (
        (EXPECTED_TASK_IDS[9], "selective_belief_policy_safety_score"),
        (EXPECTED_TASK_IDS[9], "selective_belief_policy_nontrivial_score"),
    ),
}

OWN_GATE_FIELDS = {
    EXPECTED_TASK_IDS[0]: ("v617_task_contract_conforms_score",),
    EXPECTED_TASK_IDS[1]: ("arc_report_channel_forensics_ready_score",),
    EXPECTED_TASK_IDS[2]: ("arc_typed_identity_bridge_ready_score",),
    EXPECTED_TASK_IDS[3]: ("arc_identity_report_attack_audit_ready_score",),
    EXPECTED_TASK_IDS[4]: ("belief_shadow_transport_ready_score",),
    EXPECTED_TASK_IDS[5]: ("belief_shadow_trace_audit_ready_score",),
    EXPECTED_TASK_IDS[6]: ("uniform_belief_live_ab_complete_score",),
    EXPECTED_TASK_IDS[7]: ("uniform_belief_value_audit_complete_score",),
    EXPECTED_TASK_IDS[8]: ("belief_frontier_curriculum_ready_score",),
    EXPECTED_TASK_IDS[9]: (
        "selective_belief_policy_safety_score",
        "selective_belief_policy_nontrivial_score",
    ),
    EXPECTED_TASK_IDS[10]: ("selective_belief_live_ab_complete_score",),
}

VALUE_POSITIVE_FIELDS = {
    EXPECTED_TASK_IDS[6]: "uniform_belief_value_positive_score",
    EXPECTED_TASK_IDS[7]: "uniform_belief_value_audit_positive_score",
    EXPECTED_TASK_IDS[10]: "selective_belief_live_value_positive_score",
}

CLOSED_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_FIELDS = {
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "roadmap_contract_rows",
    "task_artifact_rows",
    "artifact_hash_rows",
    "verdict_consistency_rows",
    "gate_replay_rows",
    "row_consistency_rows",
    "identity_claim_rows",
    "uniform_value_claim_rows",
    "continuous_learning_claim_rows",
    "selective_transfer_claim_rows",
    "solve_claim_rows",
    "hardware_claim_rows",
    "blocked_input_rows",
    "null_input_rows",
    "disqualified_input_rows",
    "release_hold_repair_retire_rows",
    "exclusion_manifest_rows",
    "documentation_reconciliation_rows",
    "production_default_unchanged",
    "next_handoff",
    "expected_task_count",
    "observed_task_count",
    "expected_id_order",
    "observed_id_order",
    "v617_capstone_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}

_PRINCIPLE_OVERRIDES = {
    "inference_substrate": "This value separates stored-evidence reduction from new model inference.",
    "source_artifact_hashes": "Content hashes expose source drift before a claim is reused.",
    "rows": "Claim rows keep unlike identity, value, learning, solve, and hardware evidence separate.",
    "production_default_unchanged": "A new belief policy must prove value before it can affect normal production.",
    "v617_capstone_complete_score": "Completion measures terminal classification, not whether every input was positive.",
    "reproducibility_checksum": "A canonical digest makes silent result changes detectable.",
    "verifier_is_oracle": "The evidence reducer audits claims but does not create an oracle-distinct result.",
    "honest_verdict": "A terminal summary prevents an external block from becoming repeated partial work.",
}


def task_number(task_id: object) -> int | None:
    """Return the experiment number so Markdown gates can use short IDs."""

    match = re.match(r"exp(\d+)(?:-|$)", str(task_id), re.I)
    return int(match.group(1)) if match else None


def canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically for evidence checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str | None:
    """Hash one file without changing its timestamps or contents."""

    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_checksum(artifact: Mapping[str, Any]) -> str:
    """Recompute the checksum convention used by V617 task artifacts."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the final capstone fields with the same canonical convention."""

    return upstream_checksum(artifact)


def _field_principles() -> dict[str, str]:
    """Give every required field a short scientific reason."""

    return {
        field: _PRINCIPLE_OVERRIDES.get(
            field,
            f"The {field} field preserves evidence needed to reproduce or falsify the V617 disposition.",
        )
        for field in sorted(REQUIRED_FIELDS)
    }


def _terminal_prefix(verdict: object) -> bool:
    """Require the project terminal marker before descriptive verdict text."""

    return str(verdict).lower().startswith(TERMINAL_PREFIXES)


def _class_from_verdict(verdict: object) -> str | None:
    """Derive the closed class without trusting the declared class field."""

    text = str(verdict).lower()
    if not _terminal_prefix(text):
        return None
    if "disqualified" in text:
        return "disqualified"
    if "circular_positive" in text or "circular-positive" in text:
        return "circular_positive"
    if "blocked" in text:
        return "blocked"
    if "partial" in text:
        return "partial"
    if "null" in text:
        return "null"
    return "positive"


def _canonical_gates(task: Mapping[str, Any]) -> list[tuple[Any, Any, Any, Any]]:
    """Normalize structured gates without losing order or scalar values."""

    gates = task.get("gated_on", task.get("gates", []))
    if not isinstance(gates, list):
        return []
    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
        if isinstance(gate, Mapping)
    ]


def _load_yaml_contract(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Parse only the activated YAML source into its own contract rows."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {}, [], f"{type(exc).__name__}: {exc}"
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return {}, [], "active roadmap is not a task mapping"
    rows = []
    for order, raw in enumerate(document["tasks"], 1):
        if not isinstance(raw, Mapping):
            rows.append({"order": order, "id": None, "title": None, "deliverable": None, "gates": []})
            continue
        rows.append(
            {
                "order": order,
                "id": raw.get("id"),
                "title": raw.get("title"),
                "deliverable": raw.get("deliverable"),
                "gates": _canonical_gates(raw),
            }
        )
    return dict(document), rows, None


def _load_markdown_contract(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Parse Markdown through the independent V617 table parser."""

    try:
        parsed = parse_markdown_contract(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return {}, [], f"{type(exc).__name__}: {exc}"
    rows = [
        {
            "order": row.get("order"),
            "id": row.get("id"),
            "title": row.get("title"),
            "deliverable": row.get("deliverable"),
            "gates": _canonical_gates(row),
        }
        for row in parsed.get("tasks", [])
        if isinstance(row, Mapping)
    ]
    return parsed, rows, None


def _readable_file(path: Path) -> tuple[bool, str]:
    """Report file readability without creating a replacement."""

    if not path.is_file():
        return False, "missing"
    try:
        return (bool(path.read_bytes()), "readable_nonempty" if path.stat().st_size else "empty")
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _writable_file(path: Path) -> tuple[bool, str]:
    """Check an existing document or output parent without writing a probe."""

    target = path if path.exists() else path.parent
    available = target.exists() and os.access(target, os.W_OK)
    return available, "writable" if available else "not_writable"


def _preconditions(root: Path, output_path: Path) -> list[dict[str, Any]]:
    """Keep capstone-input checks separate from missing upstream evidence."""

    checks: list[tuple[str, bool, str, str]] = []
    for name, relative in (
        ("v617_active_yaml_readable", ROADMAP_PATH),
        ("v617_markdown_readable", DESIGN_PATH),
        ("exclusion_manifest_readable", EXCLUSION_PATH),
        ("capstone_spec_readable", SPEC_PATH),
    ):
        available, observed = _readable_file(root / relative)
        checks.append((name, available, "readable_nonempty_file", observed))
    results_dir = root / "results"
    checks.append(
        (
            "results_directory_readable",
            results_dir.is_dir() and os.access(results_dir, os.R_OK),
            "readable_directory",
            "readable_directory" if results_dir.is_dir() and os.access(results_dir, os.R_OK) else "missing_or_unreadable",
        )
    )
    target = output_path if output_path.is_absolute() else root / output_path
    writable, observed = _writable_file(target)
    checks.append(("artifact_path_writable", writable, "writable", observed))
    for relative in (Path("ops/status.md"), Path("ops/changelog.md"), Path("_bmad/traceability.md")):
        writable, observed = _writable_file(root / relative)
        checks.append((f"documentation_path_writable:{relative}", writable, "writable", observed))
    return [
        {"check": name, "available": available, "expected_value": expected, "observed_value": observed}
        for name, available, expected, observed in checks
    ]


def _contract_rows(
    yaml_rows: Sequence[Mapping[str, Any]], markdown_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Compare two separately built contract views against the fixed ID order."""

    rows = []
    for index in range(max(EXPECTED_TASK_COUNT, len(yaml_rows), len(markdown_rows))):
        expected = EXPECTED_TASK_IDS[index] if index < EXPECTED_TASK_COUNT else None
        yaml_row = yaml_rows[index] if index < len(yaml_rows) else {}
        markdown_row = markdown_rows[index] if index < len(markdown_rows) else {}
        matches = (
            yaml_row.get("order") == index + 1
            and markdown_row.get("order") == index + 1
            and yaml_row.get("id") == expected
            and markdown_row.get("id") == expected
            and yaml_row.get("title") == markdown_row.get("title")
            and yaml_row.get("deliverable") == markdown_row.get("deliverable")
            and yaml_row.get("gates") == markdown_row.get("gates")
        )
        rows.append(
            {
                "order": index + 1,
                "expected_id": expected,
                "yaml_id": yaml_row.get("id"),
                "markdown_id": markdown_row.get("id"),
                "title_matches": yaml_row.get("title") == markdown_row.get("title"),
                "deliverable_matches": yaml_row.get("deliverable") == markdown_row.get("deliverable"),
                "gates_match": yaml_row.get("gates") == markdown_row.get("gates"),
                "matches": matches,
            }
        )
    return rows


def _source_hash_rows(root: Path, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recompute every declared upstream source hash from current bytes."""

    declared = artifact.get("source_artifact_hashes")
    if not isinstance(declared, Mapping):
        return [{"path": None, "declared_hash": None, "observed_hash": None, "matches": False}]
    rows = []
    for raw_path, expected in sorted(declared.items(), key=lambda item: str(item[0])):
        relative = Path(str(raw_path))
        path = relative if relative.is_absolute() else root / relative
        observed = sha256_file(path)
        rows.append(
            {
                "path": str(raw_path),
                "declared_hash": expected,
                "observed_hash": observed,
                "matches": expected == observed,
            }
        )
    return rows


def _headline_consistent(task_id: str, artifact: Mapping[str, Any], declared: str) -> bool:
    """Check ready and value headlines against the artifact's declared class."""

    fields = OWN_GATE_FIELDS.get(task_id, ())
    values = [artifact.get(field) for field in fields]
    if any(value not in (0, 1) for value in values):
        return False
    if declared in {"positive", "circular_positive", "null"} and any(value != 1 for value in values):
        if not (task_id == EXPECTED_TASK_IDS[9] and declared == "null" and values == [1, 0]):
            return False
    if declared == "blocked" and any(value != 0 for value in values):
        return False
    positive_field = VALUE_POSITIVE_FIELDS.get(task_id)
    if positive_field is not None:
        positive_value = artifact.get(positive_field)
        if positive_value not in (0, 1):
            return False
        if (declared == "positive") != (positive_value == 1):
            return False
    return True


def _task_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, JsonDict]]:
    """Load exact task paths and turn integrity failures into disqualification."""

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS[:-1]:
        relative = TASK_PATHS[task_id]
        path = root / relative
        file_hash = sha256_file(path)
        if file_hash is None:
            rows.append(
                {
                    "task_id": task_id,
                    "artifact_path": relative,
                    "artifact_state": "missing",
                    "artifact_hash": None,
                    "declared_verdict_class": None,
                    "effective_verdict_class": "blocked",
                    "honest_verdict": "complete_blocked_upstream_artifact_missing",
                    "integrity_errors": ["artifact_missing"],
                    "source_hash_rows": [],
                }
            )
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "artifact_path": relative,
                    "artifact_state": "unreadable",
                    "artifact_hash": file_hash,
                    "declared_verdict_class": None,
                    "effective_verdict_class": "blocked",
                    "honest_verdict": "complete_blocked_upstream_artifact_unreadable",
                    "integrity_errors": [f"artifact_unreadable:{type(exc).__name__}"],
                    "source_hash_rows": [],
                }
            )
            continue
        if not isinstance(loaded, dict):
            rows.append(
                {
                    "task_id": task_id,
                    "artifact_path": relative,
                    "artifact_state": "unreadable",
                    "artifact_hash": file_hash,
                    "declared_verdict_class": None,
                    "effective_verdict_class": "blocked",
                    "honest_verdict": "complete_blocked_upstream_artifact_not_object",
                    "integrity_errors": ["artifact_not_object"],
                    "source_hash_rows": [],
                }
            )
            continue
        artifacts[task_id] = loaded
        declared = loaded.get("verdict_class")
        honest = loaded.get("honest_verdict")
        derived = _class_from_verdict(honest)
        source_rows = _source_hash_rows(root, loaded)
        errors = []
        if declared not in CLOSED_CLASSES:
            errors.append("verdict_class_outside_closed_enum")
        if derived is None:
            errors.append("honest_verdict_terminal_prefix_missing")
        elif derived != declared:
            errors.append("honest_verdict_class_mismatch")
        if loaded.get("reproducibility_checksum") != upstream_checksum(loaded):
            errors.append("reproducibility_checksum_mismatch")
        if any(row["matches"] is not True for row in source_rows):
            errors.append("source_artifact_hash_mismatch")
        if loaded.get("flagged_adversarial") is True:
            errors.append("flagged_adversarial")
        rows_value = loaded.get("rows")
        if not isinstance(rows_value, list):
            errors.append("rows_missing_or_not_list")
        if not _headline_consistent(task_id, loaded, str(declared)):
            errors.append("row_to_headline_mismatch")
        if errors:
            effective = "disqualified"
        elif declared == "positive" and loaded.get("verifier_is_oracle") is True:
            effective = "circular_positive"
        else:
            effective = str(declared)
        rows.append(
            {
                "task_id": task_id,
                "artifact_path": relative,
                "artifact_state": "present",
                "artifact_hash": file_hash,
                "declared_verdict_class": declared,
                "effective_verdict_class": effective,
                "honest_verdict": honest,
                "integrity_errors": errors,
                "source_hash_rows": source_rows,
            }
        )
    return rows, artifacts


def _gate_replay(
    yaml_document: Mapping[str, Any], task_rows: list[dict[str, Any]], artifacts: Mapping[str, JsonDict]
) -> list[dict[str, Any]]:
    """Replay gates in roadmap order and reject claims behind failed evidence."""

    by_id = {row["task_id"]: row for row in task_rows}
    result = []
    tasks = yaml_document.get("tasks", [])
    for raw_task in tasks if isinstance(tasks, list) else []:
        if not isinstance(raw_task, Mapping):
            continue
        consumer = str(raw_task.get("id"))
        for upstream, field, op, expected in _canonical_gates(raw_task):
            producer_row = by_id.get(str(upstream), {})
            producer = artifacts.get(str(upstream), {})
            observed = producer.get(str(field))
            value_passed = op == "==" and observed == expected
            evidence_usable = producer_row.get("effective_verdict_class") in {
                "positive",
                "circular_positive",
                "null",
            }
            passed = value_passed and evidence_usable
            consumer_row = by_id.get(consumer)
            consumer_consistent = consumer_row is None or passed or consumer_row.get(
                "effective_verdict_class"
            ) in {"blocked", "disqualified"}
            if consumer_row is not None and not consumer_consistent:
                consumer_row["integrity_errors"].append("upstream_gate_replay_failed")
                consumer_row["effective_verdict_class"] = "disqualified"
            result.append(
                {
                    "consumer": consumer,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": op,
                    "expected_value": expected,
                    "observed_value": observed,
                    "producer_evidence_usable": evidence_usable,
                    "passed": passed,
                    "consumer_consistent": consumer_consistent,
                }
            )
    return result


def _claim_rows(
    task_rows: Sequence[Mapping[str, Any]], artifacts: Mapping[str, JsonDict]
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Keep each scientific claim class on a separate evidence row."""

    by_id = {str(row["task_id"]): row for row in task_rows}

    def claim(task_id: str, claim_name: str, field: str | None = None) -> JsonDict:
        task_row = by_id[task_id]
        source = artifacts.get(task_id, {})
        return {
            "claim": claim_name,
            "task_id": task_id,
            "claim_class": task_row["effective_verdict_class"],
            "evidence_field": field,
            "observed_value": source.get(field) if field else None,
            "artifact_hash": task_row["artifact_hash"],
        }

    identity = [
        claim(EXPECTED_TASK_IDS[2], "typed_identity_transport", "arc_typed_identity_bridge_ready_score"),
        claim(EXPECTED_TASK_IDS[3], "identity_attack_resistance", "arc_identity_report_attack_audit_ready_score"),
        claim(EXPECTED_TASK_IDS[4], "belief_shadow_transport", "belief_shadow_transport_ready_score"),
        claim(EXPECTED_TASK_IDS[5], "belief_shadow_transport_audit", "belief_shadow_trace_audit_ready_score"),
    ]
    uniform = [
        claim(EXPECTED_TASK_IDS[6], "uniform_live_value", "uniform_belief_value_positive_score"),
        claim(EXPECTED_TASK_IDS[7], "uniform_live_value_cold_authority", "uniform_belief_value_audit_positive_score"),
    ]
    learner = artifacts.get(EXPECTED_TASK_IDS[9], {})
    learner_class = by_id[EXPECTED_TASK_IDS[9]]["effective_verdict_class"]
    all_abstain = (
        learner_class == "null"
        and learner.get("selective_belief_policy_safety_score") == 1
        and learner.get("selective_belief_policy_nontrivial_score") == 0
        and int(learner.get("helpful_use_count") or 0) == 0
        and int(learner.get("harmful_use_count") or 0) == 0
    )
    continuous = [
        {
            **claim(EXPECTED_TASK_IDS[9], "selective_learning_safety"),
            "safety_score": learner.get("selective_belief_policy_safety_score"),
            "nontrivial_score": learner.get("selective_belief_policy_nontrivial_score"),
            "all_abstain": all_abstain,
            "continuous_self_learning_task": learner.get("continuous_self_learning_task"),
        }
    ]
    selective = [
        {
            **claim(
                EXPECTED_TASK_IDS[10],
                "selective_live_transfer",
                "selective_belief_live_value_positive_score",
            ),
            "policy_hash": artifacts.get(EXPECTED_TASK_IDS[10], {}).get("final_policy_hash"),
        }
    ]
    solve = []
    for task_id in (EXPECTED_TASK_IDS[4], EXPECTED_TASK_IDS[5], EXPECTED_TASK_IDS[6], EXPECTED_TASK_IDS[10]):
        source = artifacts.get(task_id, {})
        task_row = by_id[task_id]
        claimed = source.get("game_level_solve_claim") is True
        eligible = (
            claimed
            and source.get("solve_provenance") == "live_agent_self_discovery"
            and source.get("verifier_is_oracle") is False
            and task_row["effective_verdict_class"] == "positive"
        )
        solve.append(
            {
                "claim": "incidental_arc_solve",
                "task_id": task_id,
                "claim_class": task_row["effective_verdict_class"] if claimed else "null",
                "game_level_solve_claim": claimed,
                "solve_provenance": source.get("solve_provenance"),
                "headline_eligible": eligible,
            }
        )
    hardware = [
        {
            "claim": "hardware_execution_or_performance",
            "claim_class": "null",
            "evidence": "V617 contains no hardware task or changed authenticated hardware receipt.",
            "headline_eligible": False,
        }
    ]
    return identity, uniform, continuous, selective, solve, hardware


def _dispositions(
    task_rows: Sequence[Mapping[str, Any]], artifacts: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Apply the fixed branch table without pooling unlike measurements."""

    by_id = {str(row["task_id"]): row for row in task_rows}
    identity_rows = [by_id[task_id] for task_id in EXPECTED_TASK_IDS[2:6]]
    identity_classes = {row["effective_verdict_class"] for row in identity_rows}
    if identity_classes == {"positive"}:
        identity = {"branch": "identity", "disposition": "release", "release_mode": "repaired_transport"}
    elif "disqualified" in identity_classes or any(
        any(token in str(row.get("honest_verdict", "")).lower() for token in ("identity", "hash", "checksum", "artifact_invalid"))
        for row in identity_rows
    ):
        identity = {"branch": "identity", "disposition": "repair", "release_mode": None}
    else:
        identity = {"branch": "identity", "disposition": "hold", "release_mode": None}

    uniform_task = by_id[EXPECTED_TASK_IDS[7]]
    uniform_artifact = artifacts.get(EXPECTED_TASK_IDS[7], {})
    uniform_class = uniform_task["effective_verdict_class"]
    if uniform_class == "positive" and uniform_artifact.get("uniform_belief_value_audit_positive_score") == 1:
        uniform = {"branch": "uniform_belief", "disposition": "release", "release_mode": "evidence_only_default_off"}
    elif (
        uniform_class == "null"
        and uniform_artifact.get("uniform_belief_value_audit_complete_score") == 1
        and uniform_artifact.get("uniform_belief_value_audit_positive_score") == 0
    ):
        uniform = {"branch": "uniform_belief", "disposition": "retire", "release_mode": None}
    elif uniform_class == "disqualified":
        uniform = {"branch": "uniform_belief", "disposition": "repair", "release_mode": None}
    else:
        uniform = {"branch": "uniform_belief", "disposition": "hold", "release_mode": None}

    learner_task = by_id[EXPECTED_TASK_IDS[9]]
    transfer_task = by_id[EXPECTED_TASK_IDS[10]]
    learner = artifacts.get(EXPECTED_TASK_IDS[9], {})
    transfer = artifacts.get(EXPECTED_TASK_IDS[10], {})
    policy_hash = transfer.get("final_policy_hash")
    rollback_rows = transfer.get("rollback_rows")
    rollback_ready = isinstance(rollback_rows, list) and bool(rollback_rows) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in rollback_rows
    )
    selective_positive = (
        learner_task["effective_verdict_class"] == "positive"
        and transfer_task["effective_verdict_class"] == "positive"
        and learner.get("selective_belief_policy_safety_score") == 1
        and learner.get("selective_belief_policy_nontrivial_score") == 1
        and transfer.get("selective_belief_live_ab_complete_score") == 1
        and transfer.get("selective_belief_live_value_positive_score") == 1
        and isinstance(policy_hash, str)
        and policy_hash == learner.get("final_policy_hash")
        and rollback_ready
    )
    if selective_positive:
        selective = {
            "branch": "selective_learning",
            "disposition": "release",
            "release_mode": "default_off_canary",
            "policy_hash": policy_hash,
            "rollback_ready": True,
        }
    elif "disqualified" in {
        learner_task["effective_verdict_class"],
        transfer_task["effective_verdict_class"],
    } or transfer_task["effective_verdict_class"] == "positive":
        selective = {
            "branch": "selective_learning",
            "disposition": "repair",
            "release_mode": None,
            "policy_hash": policy_hash,
            "rollback_ready": rollback_ready,
        }
    else:
        selective = {
            "branch": "selective_learning",
            "disposition": "hold",
            "release_mode": None,
            "policy_hash": learner.get("final_policy_hash"),
            "rollback_ready": False,
        }
    return [identity, uniform, selective]


def _next_handoff(dispositions: Sequence[Mapping[str, Any]], input_blocked: bool) -> list[JsonDict]:
    """Choose one bounded action and avoid every forbidden reopened scope."""

    by_branch = {str(row["branch"]): row for row in dispositions}
    if input_blocked:
        action = "restore_the_single_missing_v617_markdown_contract_and_rerun_the_receipt_reducer"
    elif by_branch["identity"]["disposition"] == "repair":
        action = "repair_the_first_typed_identity_evidence_integrity_failure"
    elif by_branch["uniform_belief"]["disposition"] == "repair":
        action = "repair_the_uniform_cold_audit_evidence_integrity_failure"
    elif by_branch["uniform_belief"]["disposition"] == "retire":
        action = "record_the_exact_uniform_influence_retirement_and_evaluate_only_the_frozen_selective_policy"
    elif by_branch["selective_learning"]["disposition"] == "release":
        action = "run_one_default_off_selective_canary_rollback_drill_with_the_frozen_policy_hash"
    else:
        action = "resolve_the_first_blocked_selective_transfer_prerequisite_without_changing_the_default"
    return [{"action": action, "bounded_to_one_action": True}]


def _gate_summary(
    preconditions: Sequence[Mapping[str, Any]],
    contract_conforms: bool,
    overall_class: str,
    task_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Name the first concrete failure for every blocked capstone verdict."""

    failed = next((row for row in preconditions if row.get("available") is not True), None)
    if failed is not None:
        return {
            "passed": False,
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
        }
    if not contract_conforms:
        return {
            "passed": False,
            "failed_check": "v617_markdown_yaml_contract_conformance",
            "expected_value": True,
            "observed_value": False,
        }
    if overall_class == "blocked":
        blocked = [row["task_id"] for row in task_rows if row["effective_verdict_class"] == "blocked"]
        return {
            "passed": False,
            "failed_check": "upstream_terminal_block",
            "expected_value": [],
            "observed_value": blocked,
        }
    return {"passed": overall_class == "positive", "failed_check": None, "expected_value": None, "observed_value": overall_class}


def build_capstone(
    root: Path = REPO_ROOT,
    run_date: str = "20260906",
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> JsonDict:
    """Reduce the V617 repository state into one deterministic artifact."""

    started = time.monotonic()
    root = Path(root)
    preconditions = _preconditions(root, output_path)
    input_blocked = any(row["available"] is not True for row in preconditions)
    yaml_document, yaml_rows, yaml_error = _load_yaml_contract(root / ROADMAP_PATH)
    markdown_document, markdown_rows, markdown_error = _load_markdown_contract(root / DESIGN_PATH)
    contract_rows = _contract_rows(yaml_rows, markdown_rows)
    observed_ids = [row.get("id") for row in yaml_rows]
    contract_conforms = (
        yaml_error is None
        and markdown_error is None
        and yaml_document.get("milestone") == MILESTONE
        and markdown_document.get("milestone") == MILESTONE
        and len(contract_rows) == EXPECTED_TASK_COUNT
        and all(row["matches"] for row in contract_rows)
        and observed_ids == list(EXPECTED_TASK_IDS)
    )

    task_rows, artifacts = _task_rows(root)
    gate_rows = _gate_replay(yaml_document, task_rows, artifacts)
    identity, uniform, continuous, selective, solve, hardware = _claim_rows(task_rows, artifacts)
    dispositions = _dispositions(task_rows, artifacts)
    partial_present = any(row["effective_verdict_class"] == "partial" for row in task_rows)
    complete_score = int(
        not input_blocked
        and contract_conforms
        and len(task_rows) == EXPECTED_TASK_COUNT - 1
        and not partial_present
        and len(dispositions) == 3
    )

    upstream_classes = {row["effective_verdict_class"] for row in task_rows}
    if input_blocked:
        overall_class = "blocked"
        honest_verdict = "complete_blocked_v617_capstone_input_missing"
    elif not contract_conforms:
        overall_class = "disqualified"
        honest_verdict = "complete_disqualified_v617_markdown_yaml_contract_mismatch"
    elif "disqualified" in upstream_classes:
        overall_class = "disqualified"
        honest_verdict = "complete_disqualified_v617_terminal_evidence_disposition"
    elif "partial" in upstream_classes:
        overall_class = "partial"
        honest_verdict = "complete_partial_v617_nonterminal_upstream_evidence"
    elif "blocked" in upstream_classes:
        overall_class = "blocked"
        honest_verdict = "complete_blocked_v617_terminal_evidence_disposition"
    elif "null" in upstream_classes:
        overall_class = "null"
        honest_verdict = "complete_null_v617_terminal_evidence_disposition"
    elif "circular_positive" in upstream_classes:
        overall_class = "circular_positive"
        honest_verdict = "complete_circular_positive_v617_terminal_evidence_disposition"
    else:
        overall_class = "positive"
        honest_verdict = "complete_positive_v617_default_off_disposition"

    self_row = {
        "task_id": EXPECTED_TASK_IDS[-1],
        "artifact_path": TASK_PATHS[EXPECTED_TASK_IDS[-1]],
        "artifact_state": "generated_by_this_run",
        "artifact_hash": None,
        "declared_verdict_class": overall_class,
        "effective_verdict_class": overall_class,
        "honest_verdict": honest_verdict,
        "integrity_errors": [],
        "source_hash_rows": [],
    }
    task_rows.append(self_row)

    exclusion_rows = []
    if dispositions[1]["disposition"] == "retire":
        exclusion_rows.append(
            {
                "action": "record_through_exclusion_manifest_workflow",
                "scope_key": UNIFORM_SCOPE_KEY,
                "experiment_ids": [EXPECTED_TASK_IDS[6], EXPECTED_TASK_IDS[7]],
                "retired_milestone": MILESTONE,
                "retire_if_same_verdict": True,
                "operator_reopen_required": True,
                "preserved_scopes": [
                    "safe_belief_ledger",
                    "bounded_belief_query_api",
                    "exact_outcome_memory_substrate",
                ],
            }
        )

    documentation_rows = []
    for relative in (
        SPEC_PATH,
        Path("_bmad/traceability.md"),
        Path("ops/status.md"),
        Path("ops/changelog.md"),
    ):
        documentation_rows.append(
            {
                "path": str(relative),
                "terminal_evidence_only": True,
                "modified_by_capstone": False,
                "status": "requirement_present" if relative == SPEC_PATH else "deferred_to_conductor_reconciler",
            }
        )

    source_paths = [ROADMAP_PATH, DESIGN_PATH, EXCLUSION_PATH, SPEC_PATH]
    source_paths.extend(Path(TASK_PATHS[task_id]) for task_id in EXPECTED_TASK_IDS[:-1])
    source_hashes = {str(path): sha256_file(root / path) for path in source_paths}
    artifact_hash_rows = [
        {"task_id": row["task_id"], "artifact_path": row["artifact_path"], "sha256": row["artifact_hash"]}
        for row in task_rows
    ]
    verdict_rows = [
        {
            "task_id": row["task_id"],
            "declared_verdict_class": row["declared_verdict_class"],
            "derived_verdict_class": _class_from_verdict(row["honest_verdict"]),
            "effective_verdict_class": row["effective_verdict_class"],
            "terminal_prefix_valid": _terminal_prefix(row["honest_verdict"]),
            "consistent": not any(
                error.startswith("honest_verdict") or error == "verdict_class_outside_closed_enum"
                for error in row["integrity_errors"]
            ),
        }
        for row in task_rows
    ]
    row_consistency = [
        {
            "task_id": row["task_id"],
            "consistent": "row_to_headline_mismatch" not in row["integrity_errors"],
            "integrity_errors": list(row["integrity_errors"]),
        }
        for row in task_rows
    ]
    gate_summary = _gate_summary(preconditions, contract_conforms, overall_class, task_rows)
    next_handoff = _next_handoff(dispositions, input_blocked)

    all_claim_rows = identity + uniform + continuous + selective + solve + hardware
    artifact: JsonDict = {
        "schema": "carnot.exp7049.v617_capstone_disposition.v1",
        "experiment_id": 7049,
        "run_date": run_date,
        "field_principles": _field_principles(),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": source_hashes,
        "rows": task_rows + all_claim_rows + dispositions,
        "roadmap_contract_rows": contract_rows,
        "task_artifact_rows": task_rows,
        "artifact_hash_rows": artifact_hash_rows,
        "verdict_consistency_rows": verdict_rows,
        "gate_replay_rows": gate_rows,
        "row_consistency_rows": row_consistency,
        "identity_claim_rows": identity,
        "uniform_value_claim_rows": uniform,
        "continuous_learning_claim_rows": continuous,
        "selective_transfer_claim_rows": selective,
        "solve_claim_rows": solve,
        "hardware_claim_rows": hardware,
        "blocked_input_rows": [row for row in task_rows if row["effective_verdict_class"] == "blocked"],
        "null_input_rows": [row for row in task_rows if row["effective_verdict_class"] == "null"],
        "disqualified_input_rows": [row for row in task_rows if row["effective_verdict_class"] == "disqualified"],
        "release_hold_repair_retire_rows": dispositions,
        "exclusion_manifest_rows": exclusion_rows,
        "documentation_reconciliation_rows": documentation_rows,
        "production_default_unchanged": True,
        "next_handoff": next_handoff,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": len(yaml_rows),
        "expected_id_order": list(EXPECTED_TASK_IDS),
        "observed_id_order": observed_ids,
        "v617_capstone_complete_score": complete_score,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": overall_class,
        "honest_verdict": honest_verdict,
        "contract_parse_errors": {"yaml": yaml_error, "markdown": markdown_error},
    }
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the final schema invariants before atomic output."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != REQUIRED_FIELDS or not all(
        isinstance(value, str) and value.strip() for value in principles.values()
    ):
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("production_default_unchanged") is not True:
        errors.append("production_default_changed")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("expected_task_count") != EXPECTED_TASK_COUNT:
        errors.append("expected_task_count_invalid")
    if artifact.get("expected_id_order") != list(EXPECTED_TASK_IDS):
        errors.append("expected_id_order_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if artifact.get("verdict_class") not in CLOSED_CLASSES:
        errors.append("verdict_class_invalid")
    if _class_from_verdict(artifact.get("honest_verdict")) != artifact.get("verdict_class"):
        errors.append("honest_verdict_invalid")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s_invalid")
    if artifact.get("v617_capstone_complete_score") not in (0, 1):
        errors.append("complete_score_invalid")
    task_rows = artifact.get("task_artifact_rows")
    if not isinstance(task_rows, list) or len(task_rows) != EXPECTED_TASK_COUNT:
        errors.append("task_artifact_rows_invalid")
    elif artifact.get("v617_capstone_complete_score") == 1 and any(
        row.get("effective_verdict_class") == "partial" for row in task_rows if isinstance(row, Mapping)
    ):
        errors.append("complete_score_contains_partial")
    dispositions = artifact.get("release_hold_repair_retire_rows")
    if not isinstance(dispositions, list) or [row.get("branch") for row in dispositions] != [
        "identity",
        "uniform_belief",
        "selective_learning",
    ]:
        errors.append("disposition_rows_invalid")
    if not isinstance(artifact.get("next_handoff"), list) or len(artifact["next_handoff"]) != 1:
        errors.append("next_handoff_invalid")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or not {
            "failed_check",
            "expected_value",
            "observed_value",
        } <= summary.keys():
            errors.append("blocked_gate_summary_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace only the requested artifact after a complete temporary write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _date_argument(value: str) -> str:
    """Reject ambiguous dates so the artifact names one execution day."""

    if re.fullmatch(r"\d{8}", value) is None:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic reducer and write its validated JSON artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    artifact = build_capstone(args.root, args.date, output_path=args.output)
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit("invalid Exp7049 artifact: " + "; ".join(errors))
    target = args.output if args.output.is_absolute() else args.root / args.output
    _write_json_atomic(target, artifact)
    print(json.dumps({"artifact": str(target), "verdict_class": artifact["verdict_class"], "complete_score": artifact["v617_capstone_complete_score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script wrapper is the CLI path.
    raise SystemExit(main())
