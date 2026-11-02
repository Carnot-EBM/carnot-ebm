"""REQ-REPORT-7992-V693: immutable evidence keeps administrative and science claims separate."""

from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v686_contract_methods as shared
from carnot.reporting import v690_authority as authority
from carnot.reporting import v691_contract_methods as producer
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

MILESTONE = "2026.10.693"
SOURCE_SHA256 = shared.SOURCE_SHA256
source_custody = shared.source_custody
MUTATIONS = (
    "count",
    "order",
    "id",
    "title",
    "phase",
    "deliverable",
    "model",
    "substrate",
    "gate_upstream",
    "gate_field",
    "prompt",
    "retirement_flag",
)


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Reuse independent readers so consumed staging never requires invented authority."""
    try:
        value = authority.assess(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=7992
        )
    except yaml.YAMLError:
        value = shared.assess(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=7992, count=13
        )
        value.update(observed_activation=False, staging_custody_status="unparseable")
    for failure in value["gate_check_summary"]:
        failure["upstream_id"] = "V693_authority"
    return value


def historical_custody(root: Path, snapshots: Path) -> dict[str, Any]:
    """Freeze reported producer bytes; current files cannot reconstruct missing original code."""
    capstone = root / "results/experiment_7991_v692_capstone.json"
    if not capstone.is_file():
        return dict(
            ready=False,
            rows=[],
            hashes=[],
            gate_check_summary=[
                shared.operand(capstone, "producer_exists", True, False, "exp7991-capstone")
            ],
            publication={},
        )
    value = json.loads(capstone.read_text())
    saved = value["authority_snapshots"]["active"]
    authority_path = Path(saved["snapshot_path"])
    if not authority_path.is_file():
        return dict(
            ready=False,
            rows=[],
            hashes=[],
            gate_check_summary=[
                shared.operand(authority_path, "snapshot_exists", True, False, "V692_authority")
            ],
            publication=value["publication_gate_results"],
        )
    refs, rows, failures = [], [], []
    authority_ref = authority.lifecycle._snapshot(
        authority_path, authority_path.read_bytes(), snapshots, "completed_v692"
    )
    refs.append(authority_ref)
    if authority_ref["sha256"] != saved["sha256"]:
        failures.append(
            shared.operand(
                authority_path, "sha256", saved["sha256"], authority_ref["sha256"], "V692_authority"
            )
        )
    for task, old in zip(
        yaml.safe_load(authority_path.read_bytes())["tasks"], value["outcome_rows"], strict=True
    ):
        path = root / task["deliverable"]
        frozen = producer.freeze_invocation(path)
        artifact = json.loads(path.read_text()) if path.is_file() else {}
        dispatch = old.get("conductor_gate_receipt", {})
        for original in (path, Path(dispatch.get("path") or path)):
            if original.is_file():
                ref = authority.lifecycle._snapshot(
                    original, original.read_bytes(), snapshots, original.stem
                )
                refs.append(
                    {
                        **ref,
                        **producer.freeze_invocation(original),
                        "reported_code_hashes": json.loads(original.read_text()).get(
                            "code_config_hashes", {}
                        ),
                        "original_code_bytes": "not asserted; reported producer identities preserved",
                        "custody_basis": "immutable producer artifact bytes; no mutable checkout comparison",
                    }
                )
        rows.append(
            dict(
                task_id=task["id"],
                producer_invocation=frozen,
                verdict_class=artifact.get("verdict_class"),
                honest_verdict=artifact.get("honest_verdict"),
                disposition="preserve_historical_verdict" if artifact else "dispatch_blocked",
                dispatch_receipt=dispatch,
                dispatch_receipt_status="recorded"
                if dispatch.get("sha256")
                else "absent_in_original_capstone",
                gate_check_summary=artifact.get(
                    "gate_check_summary", old.get("gate_check_summary", [])
                ),
                scientific_benefit_transferred=False,
                original_outcome=old,
            )
        )
    return dict(
        ready=not failures,
        rows=rows,
        hashes=refs,
        gate_check_summary=failures,
        publication=value["publication_gate_results"],
    )


def method_freeze(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Whole prompts retain every budget and statistical choice without stale registry defaults."""
    return dict(
        task_contracts=tasks,
        canonical_tasks_sha256=authority.lifecycle.tasks_digest(tasks),
        literature_adoption_decisions=[],
        science_pre_gate=False,
        exposure="source-disjoint development; bounded recorded exposure knowledge",
        statistical_choices_source="complete immutable task prompts; no defaults imported from V692",
        registry=[
            dict(
                task_id=t["id"],
                prompt_sha256=canonical_hash(t["prompt"]),
                frozen_methods=t["prompt"],
                planned_models=t["MODEL_SPECS"],
                trained_heads="declared separately by each numerical producer",
                prospective_code_snapshot_required=True,
            )
            for t in tasks
        ],
    )


def mutations(design: Path, active: Path, private: Path) -> list[dict[str, Any]]:
    """Change one frozen operand at a time so agreement cannot hide an uncovered route."""
    baseline = yaml.safe_load(active.read_bytes())
    rows = []
    for name in MUTATIONS:
        changed = deepcopy(baseline)
        tasks = changed["tasks"]
        if name == "count":
            tasks.pop()
        elif name == "order":
            tasks.reverse()
        elif name in {"gate_upstream", "gate_field"}:
            task = next(t for t in tasks if t["gated_on"])
            task["gated_on"][0]["upstream" if name == "gate_upstream" else "artifact_field"] = (
                "changed"
            )
        elif name == "retirement_flag":
            task = next(t for t in tasks if t["prior_failures"])
            task["prior_failures"][0]["retire_if_same_verdict"] = False
        else:
            key = {"model": "MODEL_SPECS", "substrate": "inference_substrate_class"}.get(name, name)
            tasks[0][key] = "changed"
        directory = private / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "active.yaml"
        path.write_text(yaml.safe_dump(changed, sort_keys=False))
        result = assess(design, directory / "absent", path, directory / "snapshots")
        rows.append(
            dict(
                unit_id=name,
                arm="private_oracle_mutation",
                status="completed",
                passed=not result["activated"],
                expected_activation=False,
                observed_activation=result["activated"],
                gate_check_summary=result["gate_check_summary"],
                raw_numerator=int(not result["activated"]),
                raw_denominator=1,
                eligible=True,
                failed=result["activated"],
                censored=False,
                excluded=False,
                claim_scope="circular_positive",
            )
        )
        atomic_json(private / "checkpoint.json", {"rows": rows})
        print(f"[exp7992] phase=mutations completed_units={len(rows)}/12", flush=True)
    return rows


def candidate(
    root: Path,
    assessment: dict[str, Any],
    custody: dict[str, Any],
    freeze: dict[str, Any],
    controls: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    manifest_path: Path,
    started_ns: int,
    ended_ns: int,
) -> dict[str, Any]:
    """Owned validation alone qualifies this audit; historical science retains its failed outcome."""
    manifest = json.loads(manifest_path.read_text())
    history = manifest.get("historical_inputs") or historical_custody(
        root, manifest_path.parent / "producer_snapshots"
    )
    value = shared.candidate(
        root,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        manifest_path,
        started_ns,
        ended_ns,
        experiment_id=7992,
        milestone=MILESTONE,
        count=13,
    )
    if not history["ready"] and value["verdict_class"] != "disqualified":
        value.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_historical_custody",
            contract_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
    value["gate_check_summary"] += history["gate_check_summary"]
    value.update(
        run_date="20261001",
        execution_date="20261001",
        current_run_id="exp7992-20261001",
        finished_at=datetime.now(UTC).isoformat(),
        random_seed=6937992,
        task_digest=freeze["canonical_tasks_sha256"],
        task_contract=freeze["task_contracts"],
        observed_activation=assessment["observed_activation"],
        staging_custody_status=assessment["staging_custody_status"],
        historical_failure_rows=history["rows"],
        historical_custody_ready=history["ready"],
        producer_snapshot_rows=history["hashes"]
        + manifest.get("code_snapshot_rows", [])
        + manifest.get("source_snapshot_rows", []),
        publication_gate_results=history["publication"],
        code_config_hashes=manifest["dependency_hashes"],
        coverage_statement_counts={},
        positive_control_results=dict(
            twelve_mutations=all(r["passed"] for r in controls),
            scope="protocol fixtures; no natural science headroom claim",
        ),
        terminal_validation_sidecar_path=str(
            manifest_path.parent / "terminal_validation/terminal_validation.json"
        ),
    )
    value["cited_upstream_artifacts"] = history["hashes"]
    value["raw_shard_hashes"] = custody["hashes"]
    value["preconditions_checked"]["historical_custody_ready"] = history["ready"]
    value["claim_scope"].update(
        science_pre_gate=False,
        independent_benefit=False,
        human_annotations="fallible source support",
        natural_evidence="not measured",
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            freeze=freeze,
            manifest=sha256_file(manifest_path),
            sources=value["producer_snapshot_rows"],
        )
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["field_principles"].update(
        {k: "Current audit and immutable historical evidence have separate claims." for k in value}
    )
    return value


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """Seal every aggregate so cold replay never trusts a summary changed after validation."""
    return dict(
        artifact_sha256=canonical_hash(value),
        rows=value["rows"],
        historical_failure_rows=value["historical_failure_rows"],
        mutation_rows=value["mutation_rows"],
    )


def cold_replay(path: Path, raw: Path) -> bool:
    """Replay saved evidence, allowing later capability or manifest appends without changing history."""
    if not path.is_file() or not raw.is_file():
        return False
    value = json.loads(path.read_text())
    if value.get("experiment_id") != 7992:
        return False
    snapshots = [s for s in value["authority_snapshots"].values() if s["exists"]]
    snapshots += value["producer_snapshot_rows"]
    return bool(
        primitive_rows(value) == json.loads(raw.read_text())
        and value["task_id"] == "exp7992-contract-methods"
        and value["task_digest"] == authority.lifecycle.tasks_digest(value["task_contract"])
        and value["run_date"] == value["execution_date"] == "20261001"
        and value["sample_size_budget"] == shared.contract_budget(value["rows"], count=13)
        and value["contract_ready_score"]
        == int(
            value["activation_confirmed"]
            and value["source_custody_ready"]
            and value["historical_custody_ready"]
            and value["required_checks_passed"]
            and value["verdict_class"] == "circular_positive"
        )
        and value["duration_s"]
        == (value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]) / 1e9
        and all(
            Path(s["snapshot_path"]).is_file()
            and sha256_file(Path(s["snapshot_path"])) == s["sha256"]
            for s in snapshots
        )
    )
