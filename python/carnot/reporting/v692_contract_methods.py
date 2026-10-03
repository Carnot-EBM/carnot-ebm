"""REQ-REPORT-7979-V692: retain historical authority without importing its claims.

An authority fixture proves agreement between documents. It supplies no new
scientific source, fitted head, or model call for this invocation.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v690_authority as authority
from carnot.reporting import v691_contract_methods as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

MILESTONE = "2026.10.692"
SOURCE_SHA256 = prior.SOURCE_SHA256
source_custody = prior.source_custody
PINS = {
    7966: "b0952033f8c04fdba25657a85acd42f91e2f984cdcdefe436fd8e9b7a818e8fc",
    7967: "630e22ac967c45ecdf69304b9a62656f15ab4c0ff94a19a767744a4f99c67e71",
    7968: "5978a0302945b4111afd06ee5f756ff0d8d810a3d84e2cf4fec4d88acb8f06d1",
    7969: "3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51",
    7972: "a47d19ca7af500014117ff083249469597cd7e6bc0ff57778098fa39cfa41c63",
    7975: "a28af99cf2de2a4ba93348d7c2fb3ce18879a46a468fb02961f407180fc48a4f",
    7976: "48444e30e0aeeb796ff7a5a1cad04bcc3f6c9eaf8d8c61505aa3ef6efee44190",
    7977: "15147d4b31b7a1c525eda300e7333942346e05e74efd1f28af28ffe4a5c45d7e",
    7978: "184f29fb56fe71f1f7b4f75c6a88861bd7ffe0c9a574f63a4122f5ad799445a0",
}


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Use the qualified full-contract reader so prompts cannot drift unnoticed."""
    return authority.assess(design, staged, active, snapshots, milestone=MILESTONE, first_id=7979)


def historical_custody(root: Path) -> dict[str, Any]:
    """Read failed history as evidence about history, never as a present benefit gate."""
    failures, hashes, values = [], [], {}
    for number, digest in PINS.items():
        paths = list((root / "results").glob(f"experiment_{number}_v691_*.json"))
        path = (
            paths[0] if len(paths) == 1 else root / f"results/experiment_{number}_v691_missing.json"
        )
        receipt = prior.freeze_invocation(path)
        hashes.append(dict(path=str(path), sha256=receipt["sha256"], role="historical_primary"))
        if receipt["sha256"] != "sha256:" + digest:
            failures.append(
                prior.prior.shared.operand(
                    path, "sha256", "sha256:" + digest, receipt["sha256"], f"exp{number}"
                )
            )
        else:
            values[number] = json.loads(path.read_text())
    rows = []
    if 7966 in values and 7978 in values:
        snapshot = Path(values[7966]["authority_snapshots"]["active"]["snapshot_path"])
        expected = values[7966]["authority_snapshots"]["active"]["sha256"]
        actual = sha256_file(snapshot) if snapshot.is_file() else None
        hashes.append(dict(path=str(snapshot), sha256=actual, role="historical_authority"))
        if actual != expected:
            failures.append(
                prior.prior.shared.operand(snapshot, "sha256", expected, actual, "V691_authority")
            )
        else:
            for task, old_row in zip(
                yaml.safe_load(snapshot.read_bytes())["tasks"], values[7978]["rows"], strict=True
            ):
                number = int(task["id"][3:7])
                old = values.get(number, {})
                dispatch = old_row["conductor_gate_receipt"]
                if dispatch.get("sha256"):
                    path = Path(dispatch["path"])
                    found = sha256_file(path) if path.is_file() else None
                    hashes.append(dict(path=str(path), sha256=found, role="conductor_skip_receipt"))
                    if found != dispatch["sha256"]:
                        failures.append(
                            prior.prior.shared.operand(
                                path, "sha256", dispatch["sha256"], found, task["id"]
                            )
                        )
                rows.append(
                    dict(
                        task_id=task["id"],
                        planned_producer_path=str(root / task["deliverable"]),
                        producer_invocation=prior.freeze_invocation(root / task["deliverable"]),
                        verdict_class=old.get("verdict_class"),
                        honest_verdict=old.get("honest_verdict"),
                        readiness_fields={
                            k: v for k, v in old.items() if k.endswith("ready_score")
                        },
                        disposition="dispatch_blocked"
                        if not old
                        else "preserve_historical_verdict",
                        dispatch_receipt=dispatch,
                        gate_check_summary=old.get("gate_check_summary", [])
                        if old
                        else [
                            prior.prior.shared.operand(
                                root / task["deliverable"],
                                "producer_exists",
                                True,
                                False,
                                task["id"],
                            ),
                            *[
                                g
                                for g in values[7978]["gate_check_summary"]
                                if g.get("upstream_id") == task["id"]
                            ],
                        ],
                        scientific_benefit_transferred=False,
                    )
                )
    return dict(ready=not failures, rows=rows, hashes=hashes, gate_check_summary=failures)


def method_freeze(root: Path, *, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Complete prompts retain every future budget even when a short registry omits it."""
    value = dict(
        role_allocation=prior.method_freeze(root, tasks=tasks)["role_allocation"],
        task_contracts=tasks,
        canonical_tasks_sha256=authority.lifecycle.tasks_digest(tasks),
        literature_adoption_decisions=[],
    )
    value.update(
        features=dict(source=8, qwen_probability=1),
        reserved_evaluation_sources=96,
        training=dict(
            seeds=[69201, 69202, 69203],
            fit_minimum=128,
            fit_per_class=16,
            tune_minimum=32,
            tune_per_class=4,
            primary_width=8,
            controls=["logistic", "quadratic_logistic", "MLP16"],
            steps=200,
            learning_rate=0.01,
            L2=0.001,
        ),
        decisions=dict(
            costs=dict(accept="5*y", reject="1-y", escalate=0.25),
            primary_comparisons=6,
            correction="Holm",
            evaluation_minimum=64,
            evaluation_per_class=12,
            cost_gain_min=0.02,
            Brier_degradation_upper_max=0.01,
            automation_min=0.50,
            extra_false_accepts_max=0,
            bootstrap_draws=10000,
            seed_reduction="mean_within_original_source",
        ),
        learning=dict(
            initial_predicates=16,
            candidate_conjunctions=8,
            blocks=8,
            update_slots=12,
            admission_slots=8,
            delay_events=20,
            future_minimum=32,
            future_per_class=8,
            retention_minimum=16,
            retention_per_class=4,
            retention_degradation_upper_max=0.01,
            cost_gain_min=0.02,
            minimum_inferential_blocks=8,
            available_suffix_blocks=6,
            generalized_learning_benefit_score=0,
            crash_block=4,
        ),
        delayed_aci=dict(
            delays=[20, 24, 36],
            alpha=0.10,
            gamma=0.01,
            rolling_window=32,
            fixed_windows=[[33, 48], [65, 80], [97, 112], [129, 144]],
            minimum_groups=64,
            minimum_per_class=8,
            minimum_complete_blocks=8,
            primary_delay=20,
            local_error_gain_min=0.02,
            set_size_increase_max=0.10,
            bootstrap_draws=10000,
            point_Brier="identical_across_arms",
            update="alpha[t+tau]=issued_alpha[t]+gamma*(.10-error[t])",
        ),
        deferred_scope=[
            "foundation_model_training",
            "energy_guided_generation",
            "unchanged_importance_anchoring",
            "new_sampler_sweeps",
            "hardware_speedup_without_compatible_kernel",
        ],
        science_pre_gate=False,
        registry=[
            dict(
                task_id=t["id"],
                prompt_sha256=canonical_hash(t["prompt"]),
                frozen_methods=t["prompt"],
                planned_models=t["MODEL_SPECS"],
                gates=t.get("gated_on", []),
            )
            for t in tasks
        ],
        literature_map=[
            dict(
                name=name,
                source=url,
                mechanism=mechanism,
                adaptation=adaptation,
                deferred_scope=deferred,
                measured_benefit=None,
            )
            for name, url, mechanism, adaptation, deferred in (
                (
                    "EBT",
                    "https://arxiv.org/abs/2507.02092",
                    "input-candidate energy compatibility",
                    "small two-label conditional head",
                    "foundation-model training and independent truth guarantees",
                ),
                (
                    "ARM-EBM",
                    "https://arxiv.org/abs/2512.15605",
                    "function-space equivalence and normalized energies",
                    "exact two-label probability with conversion control",
                    "equivalence is not correctness",
                ),
                (
                    "evidence alignment",
                    "https://arxiv.org/abs/2608.15804",
                    "align output with input evidence",
                    "eight public lexical window mean/max features",
                    "masked encoder training and semantic entailment guarantees",
                ),
                (
                    "HallDetect",
                    "https://arxiv.org/abs/2608.05823",
                    "decomposed evidence over chunks",
                    "complete-response evidence ablations",
                    "lexical mismatch is not contradiction",
                ),
                (
                    "delayed ACI",
                    "https://arxiv.org/abs/2609.07251",
                    "state-at-issuance delayed updates",
                    "fixed scalar probability trajectory and pending labels",
                    "deployment coverage theorem",
                ),
                (
                    "sparse spline learning",
                    "https://arxiv.org/abs/2602.02056v4",
                    "local coefficient updates",
                    "descriptive spline and touched-state accounting",
                    "vendor FPGA speedup transfer",
                ),
            )
        ],
    )
    return value


def mutations(design: Path, active: Path, source: Path, private: Path) -> list[dict[str, Any]]:
    """Keep qualified mutations and add source custody and producer-date challenges."""
    rows = prior.prior.mutations(
        design, active, source, private, milestone=MILESTONE, first_id=7979
    )
    duplicate = yaml.safe_load(active.read_bytes())
    duplicate["tasks"][1]["id"] = duplicate["tasks"][0]["id"]
    path = private / "duplicate.yaml"
    path.write_text(yaml.safe_dump(duplicate, sort_keys=False))
    rejected = not assess(design, private / "absent", path, private / "duplicate-snapshots")[
        "activated"
    ]
    producer = private / "producer.json"
    atomic_json(
        producer,
        dict(
            experiment_id=7966,
            task_id="exp7966-contract-methods",
            milestone="2026.10.691",
            run_date="20260930",
            execution_date="20260930",
            started_at="2026-09-30T23:59:59Z",
            finished_at="2026-10-01T00:00:01Z",
        ),
    )
    receipt = prior.freeze_invocation(producer)
    rollover = prior.validate_invocation(producer, receipt)["passed"]
    changed = json.loads(producer.read_text()) | {"run_date": "20261001"}
    atomic_json(producer, changed)
    for name, passed in (
        ("duplicate_id", rejected),
        ("source_hash", not source_custody(source, "sha256:changed")["ready"]),
        ("producer_rollover", rollover),
        ("forged_date", not prior.validate_invocation(producer, receipt)["passed"]),
    ):
        rows.append(
            dict(
                unit_id=name,
                arm="private_oracle_mutation",
                status="completed",
                passed=passed,
                claim_scope="circular_positive",
            )
        )
        print(f"[exp7979] phase=mutations completed_units={len(rows)}/16", flush=True)
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
    """Separate measured audit validity from historical and planned science."""
    value = prior.prior.shared.candidate(
        root,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        manifest_path,
        started_ns,
        ended_ns,
        experiment_id=7979,
        milestone=MILESTONE,
        count=13,
    )
    history = historical_custody(root)
    value["gate_check_summary"] += history["gate_check_summary"]
    if not history["ready"] and value["verdict_class"] != "disqualified":
        value.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_historical_custody",
            contract_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
    manifest = json.loads(manifest_path.read_text())
    for attempt in manifest["prior_owned_attempts"]:
        old = json.loads(Path(attempt["path"]).read_text())
        value["historical_required_failures"].append(
            dict(
                **attempt,
                honest_verdict=old["honest_verdict"],
                required_failures=old["gate_check_summary"],
                original_verdict_class=old["verdict_class"],
                resolved_by="new frozen validation with wrong-ID rejection coverage",
            )
        )
        value["source_artifact_hashes"].append(attempt)
    value.update(
        run_date="20261001",
        execution_date="20261001",
        current_run_id="exp7979-20261001",
        finished_at=datetime.now(UTC).isoformat(),
        random_seed=6927979,
        task_contract=freeze["task_contracts"],
        method_registry=freeze["registry"],
        literature_map=freeze["literature_map"],
        historical_dispositions=history["rows"],
        historical_custody_ready=history["ready"],
        MODEL_SPECS=[],
        observed_activation=assessment["observed_activation"],
        staging_custody_status=assessment["staging_custody_status"],
        planning_ready_score=assessment["planning_ready_score"],
        code_config_hashes=manifest["dependency_hashes"],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=str(
            manifest_path.parent / "terminal_validation/terminal_validation.json"
        ),
    )
    value["source_artifact_hashes"] += history["hashes"]
    value["cited_upstream_artifacts"] = history["hashes"]
    value["raw_shard_hashes"] = custody["hashes"]
    value["preconditions_checked"]["historical_custody_ready"] = history["ready"]
    value["claim_scope"].update(
        science_pre_gate=False,
        independent_benefit=False,
        gap_oracle_distinct="open_after_20260928_corrigendum",
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            freeze=freeze,
            manifest=sha256_file(manifest_path),
            sources=value["source_artifact_hashes"],
        )
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["field_principles"].update(
        {
            k: "Bind current audit evidence; preserve historical failures and planned methods separately."
            for k in value
        }
    )
    return value


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """Seal all aggregate fields so a replay cannot trust an edited summary."""
    return dict(
        artifact_sha256=canonical_hash(value),
        rows=value["rows"],
        historical_dispositions=value["historical_dispositions"],
        mutation_rows=value["mutation_rows"],
    )


def cold_replay(path: Path, raw: Path) -> bool:
    """Recheck exact bytes, source identities and reductions from outside the checkout."""
    if not path.is_file() or not raw.is_file():
        return False
    value = json.loads(path.read_text())
    if value.get("experiment_id") != 7979:
        return False
    snapshots = [
        Path(s["snapshot_path"]) for s in value["authority_snapshots"].values() if s["exists"]
    ]
    hashes = {
        s["snapshot_path"]: s["sha256"]
        for s in value["authority_snapshots"].values()
        if s["exists"]
    }
    refs = [
        s
        for s in value["source_artifact_hashes"]
        if s.get("role") not in {"active", "staged", "design"}
    ]
    return bool(
        primitive_rows(value) == json.loads(raw.read_text())
        and value["task_id"] == "exp7979-contract-methods"
        and value["run_date"] == value["execution_date"] == "20261001"
        and value["canonical_tasks_sha256"]
        == authority.lifecycle.tasks_digest(value["task_contract"])
        and value["sample_size_budget"]
        == prior.prior.shared.contract_budget(value["rows"], count=13)
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
        and all(p.is_file() and sha256_file(p) == hashes[str(p)] for p in snapshots)
        and all(
            (sha256_file(Path(s["path"])) if Path(s["path"]).is_file() else None) == s["sha256"]
            for s in refs
        )
        and all(
            sha256_file(Path(__file__).resolve().parents[3] / p) == digest
            for p, digest in value["code_config_hashes"].items()
        )
    )
