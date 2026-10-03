"""REQ-REPORT-7953-V690: freeze methods with exact upstream evidence identity.

This audit reuses numerical qualifications. It makes no current model calls and
cannot transfer historical eligibility or claim independent scientific benefit.
"""

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_methods as shared
from carnot.reporting import v688_contract_methods as qualified
from carnot.reporting import v689_contract_methods as prior
from carnot.reporting import v690_authority as authority
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

MILESTONE = authority.MILESTONE
SOURCE_SHA256 = shared.SOURCE_SHA256
source_custody = shared.source_custody
assess = authority.assess
PUBLICATION_SHA256 = "sha256:70102318c925d07d5ea26957302dc9fafa0aecf4c4afaeaa9e346060e103f6c5"


def qualified_custody(root: Path) -> dict[str, Any]:
    """Keep historical numerical qualification and publication custody explicit."""
    value = qualified.qualified_custody(root)
    path = root / "results/experiment_7941_v689_training_publication.json"
    actual = sha256_file(path) if path.is_file() else None
    value["hashes"].append(
        dict(
            path=str(path),
            sha256=actual,
            expected_sha256=PUBLICATION_SHA256,
            role="publication_primary",
        )
    )
    observed = json.loads(path.read_text()) if actual == PUBLICATION_SHA256 else {}
    for field, expected, found in (
        ("sha256", PUBLICATION_SHA256, actual),
        ("experiment_id", 7941, observed.get("experiment_id")),
        ("runtime_ready_score", 1, observed.get("runtime_ready_score")),
        ("flagged_adversarial", False, observed.get("flagged_adversarial")),
        ("verdict_class", "circular_positive", observed.get("verdict_class")),
    ):
        if found != expected:
            value["gate_check_summary"].append(
                shared.operand(path, field, expected, found, "exp7941-training-publication")
            )
    value["rows"].append(
        dict(
            unit_id="exp7941-training-publication",
            path=str(path),
            sha256=actual,
            runtime_ready_score=observed.get("runtime_ready_score"),
        )
    )
    value["ready"] = not value["gate_check_summary"]
    return value


def method_freeze(root: Path) -> dict[str, Any]:
    """Bind reviewed methods to the current prompts before any result exists."""
    freeze = prior.method_freeze(root)
    freeze["sentence_labels"]["historical_only"] = True
    freeze["response_targets"] = dict(
        task="exp7955",
        families=64,
        minimum_clusters=32,
        minimum_per_class=8,
        target="any_original_human_unsupported_span_in_complete_response",
        implicit_true=True,
        incomplete="excluded_never_negative",
        replacements=False,
        offsets="verified_unicode_and_UTF8",
        exposure="exposed_development",
    )
    freeze["confidence_formula"] = "abs(2*p-1)"
    freeze["qwen"].update(
        seed=69058,
        claim="complete_response_support_against_original_human_spans",
        total_tokens=12288,
        maximum_calls=128,
    )
    freeze["authority_status"] = "observed_active_document_identity; historical_staging_separate"
    freeze["science_pre_gate"] = False
    freeze["oracle_distinct_corrigendum"] = "2026-09-28; GAP-ORACLE-DISTINCT remains open"
    for decision in freeze["literature_adoption_decisions"]:
        decision.pop("task", None)
    freeze["literature_adoption_decisions"].extend(
        [
            dict(
                source="research-references.md#v690-planning-review",
                decision="adapt",
                task="exp7955/7958",
                method="EAEV source erasure and original spans joined to complete responses",
                limit="alignment and source sensitivity are not correctness certificates",
                measured_benefit=None,
            ),
            dict(
                source="research-references.md#v690-planning-review",
                decision="adapt",
                task="exp7958",
                method="fixed grammar with matched human targets",
                measured_benefit=None,
            ),
            dict(
                source="research-references.md#v690-planning-review",
                decision="defer",
                method="KAN retention sweeps; sparse hardware gains until full service cost measurement",
                measured_benefit=None,
            ),
        ]
    )
    freeze["review_access_limits"] = dict(
        Semantic_Scholar="HTTP 429",
        OpenReview="browser challenges",
        complete_citation_census=False,
        new_external_requests=0,
    )
    return freeze


def mutations(
    design: Path,
    active: Path,
    source: Path,
    private: Path,
    *,
    milestone: str = MILESTONE,
    first_id: int = 7953,
) -> list[dict[str, Any]]:
    """Reject all twelve private contract changes without modifying live roadmaps."""
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
            text = text.replace(authority.lifecycle.tasks_digest(baseline["tasks"]), "0" * 64, 1)
        elif name == "authority_date":
            value["milestone"] = "2026.09.689"
        else:
            value["tasks"][0][name] = [] if name == "prior_failures" else "changed"
        plan, actual, stage = (directory / n for n in ("design.md", "active.yaml", "stage.yaml"))
        plan.write_text(text)
        actual.write_text(yaml.safe_dump(value, sort_keys=False))
        stage.write_bytes(active.read_bytes())
        observed = authority.assess(
            plan, stage, actual, directory / "snapshots", milestone=milestone, first_id=first_id
        )["activated"]
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
            f"[exp7953] phase=mutations completed_units={index}/12 elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    return rows


def reproduce_v689(root: Path, private: Path) -> dict[str, bool]:
    """Reproduce the two recorded defects from private original V689 bytes."""
    private.mkdir(parents=True, exist_ok=True)
    design = root / "openspec/change-proposals/research-roadmap-v689-preserved-20260930.md"
    old = json.loads((root / "results/experiment_7940_v689_contract_methods.json").read_text())
    snapshot = Path(old["authority_snapshots"]["active"]["snapshot_path"])
    actual = private / "active.yaml"
    actual.write_bytes(snapshot.read_bytes())
    value = prior.assess(design, private / "absent.yaml", actual, private / "snapshots")
    return dict(
        empty_history_rejected=not value["contract_rows"][2]["checks"]["prior"],
        absent_consumed_staging_rejected=any(
            f["artifact_field"] == "preserved_matching_staging" for f in value["gate_check_summary"]
        ),
    )


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
    fixture: bool = False,
) -> dict[str, Any]:
    """Build current producer evidence while preserving historical required failures."""
    if fixture:
        receipts = [
            dict(
                name="private_authority_and_mutations",
                argv=["in_process_private_fixture"],
                passed=all(r["passed"] for r in controls),
                actual_exit=0,
                expected_exit=0,
                deadline_s=0,
            )
        ]
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
        experiment_id=7953,
        milestone=MILESTONE,
        count=13,
    )
    qualified_value = qualified_custody(root)
    value["qualified_evidence_custody"] = qualified_value
    value["source_artifact_hashes"] += qualified_value["hashes"]
    value["gate_check_summary"] += qualified_value["gate_check_summary"]
    if not qualified_value["ready"] and value["verdict_class"] != "disqualified":
        value.update(
            honest_verdict="complete_blocked_qualified_evidence_custody",
            verdict_class="blocked",
            contract_ready_score=0,
        )
        value["acceptance_gate_results"]["readiness"] = 0
    value.update(
        historical_fixture_date="20260929",
        execution_date="20260930",
        lifecycle_task_count=13,
        MODEL_SPECS=[],
        random_seed=6907953,
        staging_custody_status=assessment["staging_custody_status"],
        planning_ready_score=assessment["planning_ready_score"],
        observed_activation=assessment["observed_activation"],
        lineage_applicability_rows=assessment["lineage_applicability_rows"],
        preserved_staging_snapshots=assessment["preserved_staging_snapshots"],
        historical_authority_failures=reproduce_v689(
            root, manifest_path.parent / "v689-reproduction"
        ),
        coverage_statement_counts={},
        sample_size_unit="ordered_authority_tasks",
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["acceptance_gate_results"]["calibration"] = None
    value["resolved_imports"].update(
        {
            "carnot.reporting.v690_contract_methods": str(Path(__file__).resolve()),
            "carnot.reporting.v690_authority": str(Path(authority.__file__).resolve()),
        }
    )
    manifest = json.loads(manifest_path.read_text())
    value["current_dependency_hashes"] = manifest.get("dependency_hashes", {})
    value["primary_resolution_receipt"] = dict(
        path=str(manifest_path.parent / "primary_resolution_receipt.json"),
        binding="external receipt selects final primary hash",
    )
    value["terminal_validation_sidecar_path"] = str(
        manifest_path.parent / "terminal_validation/terminal_validation.json"
    )
    for report in manifest_path.parent.glob("coverage.json-*"):
        value["coverage_statement_counts"] = {
            name: row["summary"] for name, row in json.loads(report.read_text())["files"].items()
        }
    value["cited_upstream_artifacts"] = [
        dict(
            experiment_id=identity, fields_imported=fields, path=str(path), sha256=sha256_file(path)
        )
        for identity, fields, path in (
            (
                7892,
                [
                    "rows",
                    "sample_size_budget",
                    "source_boundary_ready_score",
                    "cohort_manifest_path",
                ],
                root / "results/experiment_7892_v685_source_boundary.json",
            ),
            (
                7916,
                ["training_runtime_ready_score", "validation_receipts"],
                root / "results/experiment_7916_v687_training_qualification.json",
            ),
            (
                7917,
                ["intervention_protocol_ready_score", "validation_receipts"],
                root / "results/experiment_7917_v687_intervention_qualification.json",
            ),
            (
                7941,
                ["runtime_ready_score", "validation_receipts"],
                root / "results/experiment_7941_v689_training_publication.json",
            ),
        )
    ]
    old_path = root / "results/experiment_7940_v689_contract_methods.json"
    old = json.loads(old_path.read_text())
    value["historical_required_failures"].append(
        dict(
            experiment_id=7940,
            path=str(old_path),
            sha256=sha256_file(old_path),
            honest_verdict=old["honest_verdict"],
            required_failures=old["gate_check_summary"],
            resolved=False,
        )
    )
    value["repository_health"]["current_full_suite"] = [
        r for r in receipts if r.get("name") == "repository_full_suite"
    ]
    value["preconditions_checked"]["qualified_evidence_custody"] = qualified_value["ready"]
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            manifest=sha256_file(manifest_path),
            sources=value["source_artifact_hashes"],
            methods=freeze,
        )
    )[7:23]
    value["field_principles"].update(
        {
            k: "Keep current producer, measured custody and unmeasured science separate."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        staging_custody_status="Absent consumed staging means unknown custody, not mismatched activation.",
        planning_ready_score="Staging proves planning only.",
        observed_activation="Matching active/document identity proves observed activation.",
        lineage_applicability_rows="Unchanged readers determine whether empty history is legitimate.",
        sample_size_budget="Count thirteen authority tasks and zero independent scientific samples.",
    )
    return value


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """Bind additional lifecycle operands so a changed custody claim fails replay."""
    return {
        **shared.primitive_rows(value),
        **{
            key: value[key]
            for key in (
                "observed_activation",
                "staging_custody_status",
                "planning_ready_score",
                "lineage_applicability_rows",
                "preserved_staging_snapshots",
            )
        },
    }


def cold_replay(path: Path, raw: Path) -> bool:
    """Reduce primitive rows and validate dates without rerunning historical experiments."""
    if not path.is_file() or not raw.is_file():
        return False
    value = json.loads(path.read_text())
    if value.get("experiment_id") != 7953 or value.get("task_id") != "exp7953-contract-methods":
        return False
    primitive = json.loads(raw.read_text())
    for snapshot in value["authority_snapshots"].values():
        if snapshot["exists"]:
            saved = Path(snapshot["snapshot_path"])
            if not saved.is_file() or sha256_file(saved) != snapshot["sha256"]:
                return False
    return (
        primitive_rows(value) == primitive
        and len(value["rows"]) == 13
        and value["contract_rows"] == value["rows"]
        and all(r["absolute_metric"] == int(all(r["checks"].values())) for r in value["rows"])
        and value["sample_size_budget"] == shared.contract_budget(primitive["rows"], count=13)
        and value["source_sample_size_budget"]
        == shared.boundary.budget(primitive["source_custody_rows"], 640)
        and value["contract_ready_score"]
        == int(
            value["activation_confirmed"]
            and value["source_custody_ready"]
            and value["qualified_evidence_custody"]["ready"]
            and value["required_checks_passed"]
            and value["verdict_class"] == "circular_positive"
        )
        and value["milestone"] == MILESTONE
        and value["execution_date"] == "20260930"
        and value["historical_fixture_date"] == "20260929"
        and value["observed_activation"] == value["activation_confirmed"]
        and value["qualified_evidence_custody"]
        == qualified_custody(Path(__file__).resolve().parents[3])
        and value["duration_s"]
        == (value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]) / 1e9
    )
