"""REQ-REPORT-7966-V691: bind each producer to its own issued identity.

Milestones can cross UTC days. Frozen producer bytes decide valid dates;
the consumer date cannot change an upstream record's historical identity.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v690_authority as authority
from carnot.reporting import v690_contract_methods as prior
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

MILESTONE = "2026.10.691"
SOURCE_SHA256 = prior.SOURCE_SHA256
source_custody = prior.source_custody
IDENTITY_FIELDS = (
    "experiment_id",
    "task_id",
    "milestone",
    "run_date",
    "execution_date",
    "current_run_id",
)


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Reuse the qualified reader with current identity, without writing authorities."""
    result = authority.assess(design, staged, active, snapshots, milestone=MILESTONE, first_id=7966)
    for failure in result["gate_check_summary"]:
        failure["upstream_id"] = "V691_authority"
    return result


def mutations(design: Path, active: Path, source: Path, private: Path) -> list[dict[str, Any]]:
    """Keep the twelve existing mutations and apply them to current task bytes."""
    return prior.mutations(design, active, source, private, milestone=MILESTONE, first_id=7966)


def freeze_invocation(path: Path) -> dict[str, Any]:
    """Capture original identity before a consumer can observe changed bytes."""
    value = json.loads(path.read_text()) if path.is_file() else {}
    return dict(
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        identity={key: value.get(key) for key in IDENTITY_FIELDS},
        historical_timestamps={key: value.get(key) for key in ("started_at", "finished_at")},
    )


def validate_invocation(path: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """Reject hash or identity drift while accepting honest different producer dates."""
    observed = freeze_invocation(path)
    failures = [
        prior.shared.operand(path, key, expected, observed.get(key), "producer_invocation")
        for key, expected in frozen.items()
        if observed.get(key) != expected
    ]
    if observed["sha256"] is None:
        failures.append(
            prior.shared.operand(path, "producer_exists", True, False, "producer_invocation")
        )
    return dict(**frozen, passed=not failures, gate_check_summary=failures)


def method_freeze(root: Path, *, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze complete prompts because short method names omit budgets and controls."""
    value = prior.method_freeze(root)
    value.update(
        task_contracts=tasks,
        canonical_tasks_sha256=authority.lifecycle.tasks_digest(tasks),
        role_allocation=dict(
            fit=256,
            tune=64,
            policy_design=32,
            calibration_replay=32,
            online_update=96,
            online_admission=64,
            evaluation=64,
            retention=32,
        ),
        scientific_branches=dict(
            source_energy=[7970, 7971, 7973, 7974], qwen_calibration=[7968, 7969, 7972]
        ),
        literature_adoption_decisions=[
            dict(
                source="https://arxiv.org/abs/2606.16667",
                status="rechecked",
                task="exp7969/7972",
                method="complete-pipeline calibration; same full-source prompt, score and action rule",
                decision="adapt",
            ),
            dict(
                source="https://arxiv.org/abs/2602.02056",
                status="rechecked",
                task="exp7972/7977",
                method="sparse cubic spline control; touched coefficients and state bytes",
                decision="adapt",
            ),
            dict(
                source="https://arxiv.org/abs/2609.07251",
                status="rechecked",
                task="exp7974",
                method="delayed issued-state feedback on a fixed prediction trajectory",
                decision="adapt",
            ),
            dict(
                source="https://arxiv.org/abs/2602.15985",
                status="rechecked",
                task="exp7976/7977",
                method="full-service cost including preprocessing, transfer and readout",
                decision="adapt",
            ),
        ],
        new_review_context=dict(
            source="https://arxiv.org/abs/2609.23742",
            status="new_to_V691_review",
            method="separate schema validity from target semantics; no transferred 27B guarantee",
        ),
        review_access_limits=dict(
            EBT="HTTP 429",
            ARM_EBM_returned_records=9,
            complete_citation_census=False,
            new_external_requests=0,
        ),
        preserved_v690_design=dict(
            path=str(
                root / "openspec/change-proposals/research-roadmap-v690-preserved-20261001.md"
            ),
            sha256=sha256_file(
                root / "openspec/change-proposals/research-roadmap-v690-preserved-20261001.md"
            ),
        ),
    )
    return value


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
    *,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Reuse the terminal schema while keeping historical science and dates separate."""
    value = prior.shared.candidate(
        root,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        manifest_path,
        started_ns,
        ended_ns,
        experiment_id=7966,
        milestone=MILESTONE,
        count=13,
    )
    manifest = json.loads(manifest_path.read_text())
    date_rows = [validate_invocation(Path(r["path"]), r) for r in manifest["producer_invocations"]]
    qualified = prior.qualified_custody(root)
    failures = qualified["gate_check_summary"] + [
        f for row in date_rows for f in row["gate_check_summary"]
    ]
    value["gate_check_summary"] += failures
    if failures and value["verdict_class"] != "disqualified":
        value.update(
            honest_verdict="complete_blocked_upstream_custody",
            verdict_class="blocked",
            contract_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
    value.update(
        execution_date="20261001",
        run_date="20261001",
        current_run_id="exp7966-20261001",
        started_at=started_at or datetime.now(UTC).isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        lifecycle_task_count=13,
        historical_fixture_date="20260929",
        random_seed=6917966,
        observed_activation=assessment["observed_activation"],
        staging_custody_status=assessment["staging_custody_status"],
        planning_ready_score=assessment["planning_ready_score"],
        lineage_applicability_rows=assessment["lineage_applicability_rows"],
        preserved_staging_snapshots=assessment["preserved_staging_snapshots"],
        qualified_evidence_custody=qualified,
        producer_date_rows=date_rows,
        coverage_statement_counts=manifest.get("coverage_statement_counts", {}),
        current_dependency_hashes={
            str(root / p): digest for p, digest in manifest["dependency_hashes"].items()
        },
        primary_resolution_receipt=dict(
            path=str(manifest_path.parent / "primary_resolution_receipt.json"),
            binding="external receipt binds final primary hash",
        ),
        terminal_validation_sidecar_path=str(
            manifest_path.parent / "terminal_validation/terminal_validation.json"
        ),
        scratch_root_receipt=dict(
            path=manifest["scratch_root"], private=True, mutable_results_paths=False
        ),
        cited_upstream_artifacts=[
            dict(
                path=r["path"],
                sha256=r["sha256"],
                experiment_id=r["identity"]["experiment_id"],
                fields_imported=list(IDENTITY_FIELDS),
            )
            for r in date_rows
        ],
    )
    value["acceptance_gate_results"]["calibration"] = None
    value["sample_size_unit"] = "ordered_authority_tasks"
    value["source_sample_size_unit"] = "source_family_role_slot"
    value["retirement_decisions"] = [
        dict(
            task_id=t["id"],
            prior_failures=t.get("prior_failures", []),
            decision="preserve_unchanged_prior_verdicts; retire_same_scope_without_changed_prerequisites",
        )
        for t in freeze["task_contracts"]
    ]
    value["source_artifact_hashes"] += (
        qualified["hashes"]
        + [
            dict(path=r["path"], sha256=r["sha256"], role="frozen_producer_invocation")
            for r in date_rows
        ]
        + [dict(**freeze["preserved_v690_design"], role="preserved_design_provenance")]
    )
    for row in date_rows:
        old = json.loads(Path(row["path"]).read_text()) if Path(row["path"]).is_file() else {}
        value["historical_required_failures"].append(
            dict(
                experiment_id=old.get("experiment_id"),
                path=row["path"],
                sha256=row["sha256"],
                honest_verdict=old.get("honest_verdict"),
                required_failures=old.get("gate_check_summary", []),
                resolved=False,
            )
        )
    value["claim_scope"].update(
        gap_oracle_distinct="open_after_20260928_corrigendum",
        human_annotations="model-independent_but_fallible",
        science_pre_gate=False,
    )
    value["resolved_imports"].update(
        {
            __name__: str(Path(__file__).resolve()),
            authority.__name__: str(Path(authority.__file__).resolve()),
        }
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            manifest=sha256_file(manifest_path),
            freeze=freeze,
            sources=value["source_artifact_hashes"],
        )
    )
    value["field_principles"].update(
        {
            key: "Bind original identity, exact bytes and separate measured validity from scientific benefit."
            for key in value
        }
    )
    value["field_principles"].update(
        sample_size_budget="Thirteen authority tasks supply zero independent scientific units.",
        producer_date_rows="A milestone may cross UTC days; each producer retains its issued identity.",
        staging_custody_status="Missing consumed staging remains unknown custody.",
    )
    return value


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """Seal added operands so replay cannot trust a changed aggregate claim."""
    return {
        **prior.shared.primitive_rows(value),
        **{
            key: value[key]
            for key in (
                "producer_date_rows",
                "planning_ready_score",
                "observed_activation",
                "staging_custody_status",
                "lineage_applicability_rows",
                "canonical_tasks_sha256",
                "method_freeze",
                "current_dependency_hashes",
            )
        },
    }


def cold_replay(path: Path, raw: Path) -> bool:
    """Cold-reduce primitives and authenticate upstreams against their own receipts."""
    if not path.is_file() or not raw.is_file():
        return False
    value = json.loads(path.read_text())
    if value.get("experiment_id") != 7966:
        return False
    for snapshot in value["authority_snapshots"].values():
        if snapshot["exists"]:
            saved = Path(snapshot["snapshot_path"])
            if not saved.is_file() or sha256_file(saved) != snapshot["sha256"]:
                return False
    return (
        value["task_id"] == "exp7966-contract-methods"
        and len(value["rows"]) == 13
        and value["contract_rows"] == value["rows"]
        and all(row["absolute_metric"] == int(all(row["checks"].values())) for row in value["rows"])
        and value["sample_size_budget"] == prior.shared.contract_budget(value["rows"], count=13)
        and value["source_sample_size_budget"]
        == prior.shared.boundary.budget(value["source_custody_rows"], 640)
        and value["contract_ready_score"]
        == int(
            value["activation_confirmed"]
            and value["source_custody_ready"]
            and value["required_checks_passed"]
            and value["verdict_class"] == "circular_positive"
            and value["qualified_evidence_custody"]["ready"]
            and all(r["passed"] for r in value["producer_date_rows"])
        )
        and primitive_rows(value) == json.loads(raw.read_text())
        and value["canonical_tasks_sha256"]
        == authority.lifecycle.tasks_digest(value["method_freeze"]["task_contracts"])
        and value["run_date"] == value["execution_date"] == "20261001"
        and value["observed_activation"] == value["activation_confirmed"]
        and all(
            validate_invocation(
                Path(r["path"]),
                {k: r[k] for k in ("path", "sha256", "identity", "historical_timestamps")},
            )["passed"]
            == r["passed"]
            for r in value["producer_date_rows"]
        )
        and value["duration_s"]
        == (value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]) / 1e9
        and all(
            (sha256_file(Path(p)) if Path(p).is_file() else None) == digest
            for p, digest in value["current_dependency_hashes"].items()
        )
        and all(
            (sha256_file(Path(s["path"])) if Path(s["path"]).is_file() else None) == s["sha256"]
            for s in value["source_artifact_hashes"]
            if s.get("role") not in {"active", "staged", "design", "frozen_producer_invocation"}
        )
    )
