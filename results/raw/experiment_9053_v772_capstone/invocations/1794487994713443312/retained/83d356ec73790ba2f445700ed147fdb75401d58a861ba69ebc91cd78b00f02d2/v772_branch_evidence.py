"""REQ-REPORT-9041-V772: retain qualified methods and exposed history without refitting."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v771_branch_evidence as old
from carnot.reporting import v771_branch_contract as historical

Json = dict[str, Any]
ROOT, LEDGER = old.ROOT, old.LEDGER
ARCHIVE = "openspec/change-proposals/research-roadmap-v771-preserved-20261112.md"
QUEUE = "openspec/change-proposals/research-roadmap-v771-queue-preserved-20261112.yaml"
BRANCH = "results/experiment_9027_v771_branch_contract.json"
PROVENANCE = "results/experiment_9028_v771_historical_provenance_qualification.json"
A = "results/experiment_9030_v771_retained_shard_a.json"
B = "results/experiment_9031_v771_graph_corpus_b.json"
PRIMARIES = {
    BRANCH: ("null", True),
    PROVENANCE: ("positive", True),
    "results/experiment_9029_v771_source_identity_resolution.json": ("blocked", True),
    A: ("blocked", True),
    B: ("blocked", True),
    "results/experiment_9040_v771_capstone.json": ("blocked", True),
}
PINS = dict(
    old.PINS,
    **{
        ARCHIVE: "sha256:645327851ea953d7cfd0961838dcc36eb7e10dd8dbabd32a81c52adadb76d3fd",
        QUEUE: "sha256:2f18f190413146ad123228723a2e9d0780981a5c69a471473b5d06b757041d42",
        BRANCH: "sha256:3c233de7321f62d1286e8e94bd045da059bc1d65a33f503b637dbd904d084ff7",
        PROVENANCE: "sha256:a2cb53fd67a42e3bb9614849c42f27fc1d694a65bdc5d508ac5943a77e774067",
        "results/experiment_9029_v771_source_identity_resolution.json": "sha256:a9ca396d7bc115c221a948bb835a3da3e6221feba16dbdf65bf0779fc19a2120",
        A: "sha256:9755b2dd55459cc48361deb85cbc2bae16e6fc80f231373d56d7033486033944",
        B: "sha256:be48f8994445432da78d1daa0b0058501fa227e2d3e49c11392ad62369488367",
        "results/experiment_9040_v771_capstone.json": "sha256:bfad023f742ed825b7cb98eb155ae6e672a8c0e7c346b76b33624ebe1e0e724b",
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
                    "preserved_v771",
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
                "preserved_v771",
            )
        )
    with patch.multiple(historical, DESIGN=ARCHIVE, ACTIVE=QUEUE, STAGED=QUEUE):
        authority = historical.authority(root)
    checks.append(
        check(
            root / ARCHIVE, "original_full_task_digest", True, authority["ready"], "preserved_v771"
        )
    )
    provenance = load(root, PROVENANCE)
    for field in ["provenance_ready_score", "validation_scope_ready_score"]:
        checks.append(
            check(
                root / PROVENANCE,
                field,
                1,
                provenance.get(field, "missing_field"),
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
    return dict(
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
    """Compare a single saved ledger to Exp9027; zero new support stays scientific zero."""
    with patch.multiple(old.old, BRANCH=BRANCH, PINS=PINS):
        result = old.old.frontier(root)
    result["baseline_experiment_id"] = 9027
    return result
