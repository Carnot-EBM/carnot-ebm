"""REQ-REPORT-9027-V771: retain original methods and failures without new science."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v770_branch_evidence as old

Json = dict[str, Any]
ROOT, LEDGER = old.ROOT, old.LEDGER
ARCHIVE = "openspec/change-proposals/research-roadmap-v770-preserved-20261111.md"
QUEUE = "openspec/change-proposals/research-roadmap-v770-queue-preserved-20261111.yaml"
BRANCH = "results/experiment_9013_v770_branch_contract.json"
PRIMARIES = {
    BRANCH: ("null", True),
    "results/experiment_9014_v770_scoped_provenance_qualification.json": ("disqualified", False),
    "results/experiment_9015_v770_human_disagreement_source.json": ("blocked", True),
    "results/experiment_9026_v770_capstone.json": ("blocked", True),
}
PINS = dict(
    old.PINS,
    **{
        ARCHIVE: "sha256:7f07a9250aa8cba0ad21ce9164f87fc380c069dc7e1c26430534cfe8cb28ecfb",
        QUEUE: "sha256:4b6641d72a5a0878d5d3098e61d715385f23413a5eb58ac0c9879401fe9dab53",
        BRANCH: "sha256:dc94513d17071575dcb0d2f93610b7978bda98b361b00e82764a49174ed6dbe8",
        "results/experiment_9014_v770_scoped_provenance_qualification.json": "sha256:ac6037edc4bcb9fad3d7d62610153962090da35bd1b0cc597c68e7b3bd41dcdb",
        "results/experiment_9015_v770_human_disagreement_source.json": "sha256:94fefec6e66dbbca4974fbfdab9f5c470bfc515a62d1d0314f1793169872d96a",
        "results/experiment_9026_v770_capstone.json": "sha256:91e86a467cede0d45cfb73a0ad8625f14f53652b99b56111ac6d8919bb342c27",
    },
)
GRAPH, ONLINE, PROTOCOL, ONLINE_PROTOCOL = old.GRAPH, old.ONLINE, old.PROTOCOL, old.ONLINE_PROTOCOL
reference, check, science, health = old.reference, old.check, old.science, old.health


def load(root: Path, name: str) -> Json:
    """Independent pins keep changed bytes from silently replacing historical authority."""
    path = root / name
    return json.loads(path.read_bytes()) if reference(path)["sha256"] == PINS[name] else {}


def historical_references(root: Path) -> list[Json]:
    """Keep actual failed logs and source primitives, rather than only their summaries."""
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
    """Reuse the existing fitted-state and failed-health custody without rerunning either."""
    return old.operands(root) + historical_references(root)


def history(root: Path) -> Json:
    """A historical failure passes a preservation check only when it stays a failure."""
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
                    "preserved_v770",
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
                "preserved_v770",
            )
        )
    return dict(rows=rows, checks=checks, passed=all(r["passed"] for r in checks))


def frontier(root: Path) -> Json:
    """Read the saved ledger once against Exp9013; zero new support remains scientific zero."""
    with patch.multiple(old, BRANCH=BRANCH, PINS=PINS):
        result = old.frontier(root)
    result["baseline_experiment_id"] = 9013
    return result
