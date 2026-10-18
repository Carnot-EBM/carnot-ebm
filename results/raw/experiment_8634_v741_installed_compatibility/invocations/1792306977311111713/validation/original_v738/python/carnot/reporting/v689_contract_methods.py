"""REQ-REPORT-7940-V689: bind current authority without repeating scientific work.

Private oracle agreement checks administrative mechanics. Exposed source groups
and cited model receipts cannot supply independent benefit or current model calls.
"""

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_methods as shared
from carnot.reporting import v688_contract_methods as prior
from carnot.reporting.current_work_receipt import sha256_file

MILESTONE = "2026.09.689"
SOURCE_SHA256 = shared.SOURCE_SHA256
source_custody = shared.source_custody
primitive_rows = shared.primitive_rows
qualified_custody = prior.qualified_custody


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Check projected design fields and the full active digest as separate operands."""
    try:
        value = shared.lifecycle.assess_authorities(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=7940, count=13
        )
    except (OSError, ValueError, IndexError, KeyError, TypeError, yaml.YAMLError):
        value = shared.assess(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=7940, count=13
        )
    if not value["authority_snapshots"]["staged"]["exists"]:
        preserved = []
        for path in snapshots.glob("staged-*.bin"):
            saved = yaml.safe_load(path.read_bytes())
            if (
                sha256_file(path) == "sha256:" + path.stem.removeprefix("staged-")
                and isinstance(saved, dict)
                and saved.get("milestone") == MILESTONE
                and shared.lifecycle.tasks_digest(saved.get("tasks"))
                == value["canonical_tasks_sha256"]
            ):
                preserved.append(dict(path=str(path), sha256=sha256_file(path)))
        value["preserved_staging_snapshots"] = preserved
        if not preserved:
            value["gate_check_summary"].append(
                shared.operand(staged, "preserved_matching_staging", True, False, "V689_authority")
            )
            value["activated"] = False
    for failure in value["gate_check_summary"]:
        failure["upstream_id"] = "V689_authority"
    value["activation_before_observation"] = False
    return value


def method_freeze(root: Path) -> dict[str, Any]:
    """Preserve accepted methods while binding prospective protocols to thirteen tasks."""
    freeze = prior.method_freeze(root)
    freeze["training"]["arm_names"] = [
        "response_set",
        "local_set",
        "augmented_set",
        "constrained_set",
        "augmented_mlp",
        "constrained_mlp",
        "local_logistic",
        "source_erased_constrained_set",
        "complete_static_constrained_set",
    ]
    freeze["sentence_labels"] = dict(
        task="exp7942",
        target="contains_human_annotated_source_unsupported_span",
        original_offsets="response_relative_character_to_UTF8_verified_against_annotated_text",
        implicit_true="included_primary; exclusion_is_named_sensitivity_only",
        selection_before_labels=True,
        selection="public_hash_of_family_and_byte_interval",
        families=64,
        minimum_clusters=32,
        minimum_per_class=8,
        replacements=False,
        ambiguous_or_missing="excluded_never_negative",
        exposure="exposed_development",
    )
    freeze["typed_loss"] = dict(
        expected={"accept": "5*p", "reject": "1-p", "escalate": ".25"},
        actual={"accept": "5*y", "reject": "1-y", "escalate": ".25"},
        ties="escalate",
    )
    freeze["causal_order"] = [
        "advance_excluded_slots",
        "process_releases_at_tick_plus_20",
        "pin_read_state",
        "seal_prediction",
        "commit_between_queries",
    ]
    freeze["delayed_aci"].update(
        scalar_update="current_alpha+.01*(.10-miss)",
        phase_update="issued_alpha+.01*(.10-miss)",
        quantile="k=ceil((n+1)*(1-alpha)); k<=0:-inf; k>n:+inf; otherwise kth score",
        typed_actions="{0}:accept; {1}:reject; other:escalate",
    )
    freeze["qwen"].update(
        families=64,
        calls_per_family=2,
        tokens_per_call=96,
        total_tokens=12288,
        seed=68945,
        maximum_input_tokens=6000,
        maximum_calls=128,
        claim="original_source_sentence_support_with_human_targets",
    )
    mapping = {
        "exp7933": "exp7946",
        "exp7934": "exp7947",
        "exp7935": "exp7948",
        "exp7937": "exp7950",
        "exp7934/7935": "exp7947/7948",
        "exp7937/7938": "exp7950/7951",
    }
    for decision in freeze["literature_adoption_decisions"]:
        decision["task"] = mapping.get(decision.get("task"), decision.get("task"))
    freeze["literature_adoption_decisions"] += [
        dict(
            source="research-references.md#v689-planning-review",
            decision="adapt",
            task="exp7942/7945",
            method="original human offsets joined to complete sentence bytes",
            reason="query granularity must match the human target",
            measured_benefit=None,
        ),
        dict(
            source="https://arxiv.org/abs/2606.08158",
            decision="adapt",
            task="exp7943/7944",
            method="matched constrained versus augmented views preserving all bytes",
            reason="prediction agreement and label preservation differ; no verified paraphrase claim",
            measured_benefit=None,
        ),
        dict(
            source="research-references.md#v689-planning-review",
            decision="defer",
            method="new KAN, feasibility, guided decoding and Kona training sweeps",
            reason="qualified source decisions and reproducible local recipes are prerequisites",
            measured_benefit=None,
        ),
    ]
    freeze["primary_rechecks"] = dict(
        date="20260930",
        request_count=0,
        status="ingested_existing_V689_review; no_new_external_requests",
        source_path=str(root / "research-references.md"),
        source_sha256=sha256_file(root / "research-references.md"),
    )
    return freeze


def mutations(design: Path, active: Path, source: Path, private: Path) -> list[dict[str, Any]]:
    """Reject twelve edits from a matching private baseline without changing authority."""
    baseline = yaml.safe_load(active.read_bytes())
    rows = []
    started = time.monotonic()
    for index, name in enumerate(
        (
            "count",
            "id",
            "order",
            "phase",
            "deliverable",
            "prompt",
            "gated_on",
            "MODEL_SPECS",
            "inference_substrate_class",
            "prior_failures",
            "digest",
            "authority_date",
        ),
        1,
    ):
        directory = private / name
        directory.mkdir(parents=True, exist_ok=True)
        value, text = deepcopy(baseline), design.read_text()
        if name == "count":
            value["tasks"].pop()
        elif name == "order":
            value["tasks"].reverse()
        elif name == "digest":
            text = text.replace(shared.lifecycle.tasks_digest(baseline["tasks"]), "0" * 64, 1)
        elif name == "authority_date":
            value["milestone"] = "2026.09.688"
        else:
            value["tasks"][0][name] = "changed"
        plan, actual, stage = (
            directory / label for label in ("design.md", "active.yaml", "stage.yaml")
        )
        plan.write_text(text)
        actual.write_text(yaml.safe_dump(value, sort_keys=False))
        stage.write_bytes(active.read_bytes())
        observed = assess(plan, stage, actual, directory / "snapshots")["activated"]
        rows.append(
            dict(
                unit_id=name,
                arm="private_oracle_mutation",
                seed=None,
                status="completed",
                expected_activation=False,
                observed_activation=observed,
                passed=not observed,
                claim_scope="circular_positive",
            )
        )
        print(
            f"[exp7940] phase=mutations completed_units={index}/12 elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    return rows


def candidate(root: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Add current custody while keeping historical fits outside current work counters."""
    value = shared.candidate(
        root, *args, **kwargs, experiment_id=7940, milestone=MILESTONE, count=13
    )
    custody = qualified_custody(root)
    value["qualified_evidence_custody"] = custody
    value["source_artifact_hashes"] += custody["hashes"]
    value["gate_check_summary"] += custody["gate_check_summary"]
    if not custody["ready"] and value["verdict_class"] != "disqualified":
        value.update(
            honest_verdict="complete_blocked_qualified_evidence_custody",
            verdict_class="blocked",
            contract_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
    if kwargs.get("fixture"):
        honest, classification, _ = shared.verdict(
            value["activation_confirmed"], value["source_custody_ready"] and custody["ready"], True
        )
        value.update(
            honest_verdict="complete_null_private_fixture"
            if classification == "circular_positive"
            else honest,
            verdict_class="null" if classification == "circular_positive" else classification,
        )
    value["preconditions_checked"]["qualified_evidence_custody"] = custody["ready"]
    value.update(
        historical_fixture_date="20260929",
        execution_date="20260930",
        lifecycle_task_count=13,
        random_seed=6897940,
        coverage_statement_counts={},
        activation_before_observation=False,
        sample_size_unit="ordered_authority_tasks",
        scientific_independent_count=0,
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["acceptance_gate_results"]["calibration"] = None
    value["resolved_imports"]["carnot.reporting.v689_contract_methods"] = str(
        Path(__file__).resolve()
    )
    manifest = Path(value["validation_command_manifest_path"])
    for report in manifest.parent.glob("coverage.json-*"):
        value["coverage_statement_counts"] = {
            name: data["summary"] for name, data in json.loads(report.read_text())["files"].items()
        }
    value["primary_resolution_receipt"] = dict(
        path=str(manifest.parent / "primary_resolution_receipt.json"),
        binding="external receipt records actual consumer-selected final bytes",
    )
    value["terminal_validation_sidecar_path"] = str(
        manifest.parent / "terminal_validation/terminal_validation.json"
    )
    old_path = root / "results/experiment_7929_v688_contract_methods.json"
    old = json.loads(old_path.read_text())
    value["historical_required_failures"].append(
        dict(
            experiment_id=7929,
            path=str(old_path),
            sha256=sha256_file(old_path),
            honest_verdict=old["honest_verdict"],
            required_failures=old["historical_required_failures"],
            resolved=False,
        )
    )
    value["retire_if_same_verdict"].update(
        action="retire unchanged runs; new thirteen-task authority has distinct producer identity"
    )
    value["reproducibility_checksum"] = shared.canonical_hash(
        dict(
            manifest=sha256_file(manifest),
            sources=value["source_artifact_hashes"],
            methods=value["method_freeze"],
        )
    )[7:23]
    value["field_principles"].update(
        {
            key: "Keep current producer, exact custody and unmeasured scope explicit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"]["sample_size_budget"] = (
        "Count thirteen authority tasks with zero independent scientific samples."
    )
    return value


def cold_replay(path: Path, raw: Path) -> bool:
    """Reduce primitive rows before authenticating current custody and execution dates."""
    if not shared.cold_replay(path, raw, experiment_id=7940, count=13):
        return False
    value = json.loads(path.read_text())
    return (
        value["milestone"] == MILESTONE
        and value["execution_date"] == "20260930"
        and value["historical_fixture_date"] == "20260929"
        and value["lifecycle_task_count"] == 13
        and value["qualified_evidence_custody"]
        == qualified_custody(Path(__file__).resolve().parents[3])
        and value["duration_s"]
        == (value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]) / 1e9
    )
