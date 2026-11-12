"""REQ-REPORT-9054-V773: retain qualified methods and exposed history without refitting."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v772_branch_evidence as old
from carnot.reporting import v772_branch_contract as historical

Json = dict[str, Any]
ROOT, LEDGER = old.ROOT, old.LEDGER
ARCHIVE = "openspec/change-proposals/research-roadmap-v772-preserved-20261112.md"
QUEUE = "openspec/change-proposals/research-roadmap-v772-queue-preserved-20261112.yaml"
BRANCH = "results/experiment_9041_v772_branch_contract.json"
PROVENANCE = "results/experiment_9028_v771_historical_provenance_qualification.json"
A = "results/experiment_9030_v771_retained_shard_a.json"
B = "results/experiment_9031_v771_graph_corpus_b.json"
ROSTER = "results/experiment_9042_v772_fresh_seal_methods.json"
IDENTITY = "results/experiment_9043_v772_identity_obstruction_audit.json"
ANNOTATION = "results/experiment_9044_v772_annotation_feasibility.json"
CAPSTONE = "results/experiment_9053_v772_capstone.json"
PRIMARIES = dict(
    old.PRIMARIES,
    **{
        BRANCH: ("null", True),
        ROSTER: ("blocked", True),
        IDENTITY: ("blocked", True),
        ANNOTATION: ("blocked", True),
        CAPSTONE: ("blocked", True),
    },
)
PINS = dict(
    old.PINS,
    **{
        ARCHIVE: "sha256:b8114003aa7be73393edd3e70a731df947e89176f69fbce764cb4be8014897fc",
        QUEUE: "sha256:5cf26fc9e212725527f9243707a92e9f2b7257944de19aef9f522e1f0695f6ab",
        BRANCH: "sha256:1837f2e88e964d63bb25c890dc3b4bad0292df90e4690c6c28d54c803476aba2",
        ROSTER: "sha256:1cb85452aabf4e851823a675f6137ff5f51571768cae6881b8ba2b47c8ae7a12",
        IDENTITY: "sha256:644e3db049e45b9fb9d80984a7083bf4afd32b53a0cb7eb4b9214227a758e324",
        ANNOTATION: "sha256:288ad02b0b47ff820a9afef66269a640a42a6465d389af002368fe6e9cfeec3a",
        CAPSTONE: "sha256:1cf9e996fe6bb325db1df53c2fb8ffc9c31f373e2b74f1bba6bf1b3cf9c812ac",
    },
)
GRAPH, ONLINE, PROTOCOL, ONLINE_PROTOCOL = old.GRAPH, old.ONLINE, old.PROTOCOL, old.ONLINE_PROTOCOL
reference, check, science, health = old.reference, old.check, old.science, old.health


def load(root: Path, name: str) -> Json:
    """Independent byte pins prevent a successor from replacing completed evidence."""
    path = root / name
    return json.loads(path.read_bytes()) if reference(path)["sha256"] == PINS[name] else {}


def historical_references(root: Path) -> list[Json]:
    """Retain primitive custody and failed logs without copying protected truth sidecars."""
    refs = []
    for name in PRIMARIES:
        value = load(root, name)
        refs += value.get("raw_shard_hashes", [])[:1]
        for receipt in value.get("validation_receipts", []):
            refs += [
                dict(path=receipt[k], sha256=receipt[k.replace("_path", "_sha256")])
                for k in ["log_path", "stdout_path", "stderr_path"]
                if k in receipt
            ]
    branch = load(root, BRANCH)
    refs += [
        r
        for r in branch.get("raw_shard_hashes", [])
        if r.get("source", {}).get("path") == str(ROOT / LEDGER)
    ]
    capstone = load(root, CAPSTONE)
    refs += [
        r["primitive_reference"] for r in capstone.get("natural_feasibility_summary", {}).values()
    ]
    return refs


def operands(root: Path) -> list[Json]:
    """Reuse original fitted states and retain historical failures as immutable evidence."""
    return old.operands(root) + historical_references(root)


def history(root: Path) -> Json:
    """Preservation checks accept original blocked science only while it stays blocked."""
    rows, checks = [], []
    for name, (verdict, required) in PRIMARIES.items():
        value = load(root, name)
        for field, expected in [("verdict_class", verdict), ("required_checks_passed", required)]:
            checks.append(
                check(
                    root / name,
                    field,
                    expected,
                    value.get(
                        field, "missing_field" if (root / name).is_file() else "missing_file"
                    ),
                    "preserved_v772",
                )
            )
        rows.append(
            dict(
                experiment_id=int(Path(name).name.split("_")[1]),
                honest_verdict=value.get("honest_verdict", "missing_file"),
                verdict_class=value.get("verdict_class", "missing_file"),
                validation_receipts=value.get("validation_receipts", []),
                failed_gate_operands=[
                    r for r in value.get("gate_check_summary", []) if not r["passed"]
                ],
            )
        )
    for ref in historical_references(root):
        path = root / Path(ref["path"]).relative_to(ROOT)
        checks.append(
            check(
                path,
                "sha256",
                ref["sha256"],
                reference(path)["sha256"] or "missing_file",
                "preserved_v772",
            )
        )
    with patch.multiple(historical, DESIGN=ARCHIVE, ACTIVE=QUEUE, STAGED=QUEUE):
        authority = historical.authority(root)
    checks.append(
        check(
            root / ARCHIVE, "original_full_task_digest", True, authority["ready"], "preserved_v772"
        )
    )
    provenance = load(root, PROVENANCE)
    for field in ["provenance_ready_score", "validation_scope_ready_score"]:
        checks.append(
            check(
                root / PROVENANCE,
                field,
                1,
                provenance.get(
                    field, "missing_field" if (root / PROVENANCE).is_file() else "missing_file"
                ),
                "methods_provenance",
            )
        )
    a, b = load(root, A), load(root, B)
    exposed = set(a.get("nonexposure_audit", {}).get("exposed_unit_ids", []))
    exposure = [
        dict(
            unit_id=r["unit_id"],
            shard="A",
            exposure_status="exposed" if r["unit_id"] in exposed else "unknown",
            eligible_as_new_test=False,
        )
        for r in a.get("response_rows", [])
    ]
    exposure += [
        dict(unit_id=identity, shard="B", exposure_status="unknown", eligible_as_new_test=False)
        for identity in b.get("prior_B_inventory", {}).get("completed_unit_ids", [])
    ]
    roster, capstone = load(root, ROSTER), load(root, CAPSTONE)
    summary = capstone.get("natural_feasibility_summary", {})
    graph = load(root, GRAPH)
    frozen = dict(
        state_hashes=roster.get("state_hashes", {}),
        trained_state_reference=graph.get("trained_state_reference"),
        feature_references=graph.get("feature_references", []),
    )
    for name, field, expected, observed in [
        (ROSTER, "completed_count", 414, roster.get("completed_count", "missing_field")),
        (ROSTER, "intended_count", 1024, roster.get("intended_count", "missing_field")),
        (
            ROSTER,
            "construction.proposal_count",
            20000,
            roster.get("construction", {}).get("proposal_count", "missing_field"),
        ),
        (
            IDENTITY,
            "source_identity_ready_score",
            0,
            load(root, IDENTITY).get("source_identity_ready_score", "missing_field"),
        ),
        (
            CAPSTONE,
            "natural_feasibility_summary.annotation.participation_mismatch_count",
            1006,
            summary.get("annotation", {}).get("participation_mismatch_count", "missing_field"),
        ),
        (
            CAPSTONE,
            "natural_feasibility_summary.annotation.current_participation_authority_ready_score",
            0,
            summary.get("annotation", {}).get(
                "current_participation_authority_ready_score", "missing_field"
            ),
        ),
    ]:
        if not (root / name).is_file():
            observed = "missing_file"
        checks.append(check(root / name, field, expected, observed, "preserved_v772"))
    return dict(
        frozen_state_references=frozen,
        fresh_source_summary=capstone.get("fresh_source_summary", {}),
        natural_feasibility_summary=summary,
        mdh_source_authority="Separate future qualification required; closed Mu-SHROOM authority is not repaired",
        rows=rows,
        checks=checks,
        passed=all(r["passed"] for r in checks),
        methods_provenance_ready=all(
            r["passed"] for r in checks if r["check"] == "methods_provenance"
        ),
        original_task_digest=historical.PIN,
        exposure_rows=exposure,
        historical_model_receipts="Cited original primaries only; no current invocation credit",
    )


def frontier(root: Path) -> Json:
    """Compare a single saved ledger to Exp9041; zero new support stays scientific zero."""
    with patch.multiple(old.old, BRANCH=BRANCH, PINS=PINS):
        result = old.old.frontier(root)
    result["baseline_experiment_id"] = 9041
    return result
