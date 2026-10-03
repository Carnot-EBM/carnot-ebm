"""Bind current task authority without repeating numerical qualification.

REQ-REPORT-7929-V688: agreement checks administrative custody. Reused
development data and private mutation oracles cannot prove scientific benefit.
"""

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_methods as shared
from carnot.reporting import v687_contract_methods as prior
from carnot.reporting.current_work_receipt import sha256_file

MILESTONE = "2026.09.688"
SOURCE_SHA256 = shared.SOURCE_SHA256
source_custody = shared.source_custody
primitive_rows = shared.primitive_rows
QUALIFIED = (
    (7916, "training", "a5d6cb1464966a818f3a85bc448da3adcb1bbc0660834ae44826a7b17e9a5049"),
    (7917, "intervention", "2617fd18e5cf5cd2fd3763249c8805f40b4a17e74347c401ce9c64791e515386"),
)


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Read each actual authority; consumed staging never borrows active bytes."""
    try:
        return shared.lifecycle.assess_authorities(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=7928, count=12
        )
    except (OSError, ValueError, IndexError, KeyError, TypeError, yaml.YAMLError):
        return shared.assess(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=7928, count=12
        )


def qualified_custody(root: Path) -> dict[str, Any]:
    """Authenticate durable qualification dependencies before reusing their scores."""
    hashes, failures, rows = [], [], []

    def check(path: Path, field: str, expected: Any, observed: Any, upstream: str) -> None:
        if expected != observed:
            failures.append(shared.operand(path, field, expected, observed, upstream))

    def authenticate(path: Path, expected: str, role: str, upstream: str) -> bool:
        observed = sha256_file(path) if path.is_file() else None
        hashes.append(
            dict(
                path=str(path),
                sha256=observed,
                expected_sha256=expected,
                role=role,
                exposure="exposed_development",
            )
        )
        check(path, "sha256", expected, observed, upstream)
        return observed == expected

    def references(value: Any, upstream: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_path") and key[:-5] + "_sha256" in value:
                    authenticate(Path(item), value[key[:-5] + "_sha256"], key, upstream)
            if "path" in value and "sha256" in value:
                path = Path(value["path"])
                if path.is_relative_to("/tmp"):
                    hashes.append(
                        {
                            **value,
                            "role": "historical_private_fixture_reference",
                            "required": False,
                            "reason": "durable sealed checkpoint and predictions authenticated separately",
                        }
                    )
                else:
                    authenticate(path, value["sha256"], value.get("role", "dependency"), upstream)
            for key, item in value.items():
                if key != "field_principles":
                    references(item, upstream)
        elif isinstance(value, list):
            for item in value:
                references(item, upstream)

    for identity, kind, digest in QUALIFIED:
        upstream = f"exp{identity}-{kind}-qualification"
        path = root / f"results/experiment_{identity}_v687_{kind}_qualification.json"
        if not authenticate(path, "sha256:" + digest, "qualified_primary", upstream):
            continue
        value = json.loads(path.read_text())
        field = (
            "training_runtime_ready_score"
            if kind == "training"
            else "intervention_protocol_ready_score"
        )
        for key, expected in {
            "experiment_id": identity,
            field: 1,
            "flagged_adversarial": False,
            "verdict_class": "circular_positive",
        }.items():
            check(path, key, expected, value.get(key), upstream)
        rows.append(
            dict(
                unit_id=upstream,
                path=str(path),
                sha256=sha256_file(path),
                score_field=field,
                score=value.get(field),
                flagged_adversarial=value.get("flagged_adversarial"),
                verdict_class=value.get("verdict_class"),
            )
        )
        references(value, upstream)
        check(
            path, "validation_receipts.nonempty", True, bool(value["validation_receipts"]), upstream
        )
        for receipt in value["validation_receipts"]:
            if receipt.get("classification", receipt.get("scope")) != "diagnostic":
                check(
                    path,
                    f"validation_receipts.{receipt['name']}.passed",
                    True,
                    receipt["passed"],
                    upstream,
                )
        for label, expected in value.get("training_dependency_hashes", {}).items():
            authenticate(root / label, expected, "training_code_dependency", upstream)
        if kind == "training":
            sidecar = Path(value["terminal_validation_sidecar_path"])
            terminal = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            check(
                sidecar,
                "candidate_sha256",
                "sha256:" + digest,
                terminal.get("candidate_sha256"),
                upstream,
            )
            references(terminal, upstream)
            checkpoint = json.loads(Path(value["checkpoint_manifest_path"]).read_text())
            references(checkpoint, upstream)
    return dict(ready=not failures, hashes=hashes, rows=rows, gate_check_summary=failures)


def method_freeze(root: Path) -> dict[str, Any]:
    """Retain accepted budgets and bind prospective protocols to current task IDs."""
    freeze = prior.method_freeze(root)
    tasks = yaml.safe_load((root / "research-roadmap.yaml").read_text())["tasks"]
    freeze["future_validation_scopes"] = shared.method_freeze(root, tasks=tasks)[
        "future_validation_scopes"
    ]
    for decision in freeze["literature_adoption_decisions"]:
        decision["task"] = {
            "exp7921": "exp7933",
            "exp7923": "exp7935",
            "exp7925": "exp7937",
            "exp7922/7923": "exp7934/7935",
            "exp7925/7926": "exp7937/7938",
        }.get(decision.get("task"), decision.get("task"))
    freeze["literature_adoption_decisions"].append(
        dict(
            source="https://arxiv.org/html/2607.20792v1",
            decision="adapt",
            task="exp7934",
            method="read-only prediction and delayed between-query commits with no-write control",
            reason="matched memory writes need a causal timing control",
            limit="procedural recall does not establish natural-source benefit",
            measured_benefit=None,
        )
    )
    freeze["primary_rechecks"].update(
        request_count=3,
        concurrency=1,
        sources=[
            "https://arxiv.org/html/2609.00366v1",
            "https://arxiv.org/html/2609.07251v1",
            "https://arxiv.org/html/2607.20792v1",
        ],
    )
    freeze["custody_repeat_required_by_science_agents"] = True
    freeze["qualified_evidence_custody"] = qualified_custody(root)
    return freeze


def mutations(design: Path, active: Path, source: Path, private: Path) -> list[dict[str, Any]]:
    """Reject twelve authority mutations while retaining all negative outcomes."""
    baseline = yaml.safe_load(active.read_text())
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
            value["milestone"] = "2026.09.687"
        else:
            value["tasks"][0][name] = "changed"
        plan, actual = directory / "design.md", directory / "active.yaml"
        plan.write_text(text)
        actual.write_text(yaml.safe_dump(value, sort_keys=False))
        observed = assess(plan, directory / "stage.yaml", actual, directory / "snapshots")[
            "activated"
        ]
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
            f"[exp7929] phase=mutations completed_units={index}/12 elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    return rows


def candidate(root: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Add current custody without counting historical head fits as current work."""
    value = shared.candidate(
        root, *args, **kwargs, experiment_id=7929, milestone=MILESTONE, count=12
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
    value["preconditions_checked"]["qualified_evidence_custody"] = custody["ready"]
    value.update(
        historical_fixture_date="20260929",
        execution_date="20260930",
        lifecycle_task_count=12,
        random_seed=6887929,
        coverage_statement_counts={},
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["acceptance_gate_results"]["calibration"] = None
    value["resolved_imports"]["carnot.reporting.v688_contract_methods"] = str(
        Path(__file__).resolve()
    )
    manifest = Path(value["validation_command_manifest_path"])
    for report in manifest.parent.glob("coverage.json-*"):
        value["coverage_statement_counts"] = {
            name: data["summary"] for name, data in json.loads(report.read_text())["files"].items()
        }
    raw = manifest.parent
    value["primary_resolution_receipt"] = {
        "path": str(raw / "primary_resolution_receipt.json"),
        "binding": "external receipt records reader-selected final bytes",
    }
    value["terminal_validation_sidecar_path"] = str(
        raw / "terminal_validation/terminal_validation.json"
    )
    for identity, label in (
        (7915, "contract_methods"),
        (7916, "training_qualification"),
        (7917, "intervention_qualification"),
    ):
        path = root / f"results/experiment_{identity}_v687_{label}.json"
        old = json.loads(path.read_text())
        value["historical_required_failures"].append(
            dict(
                experiment_id=identity,
                path=str(path),
                sha256=sha256_file(path),
                honest_verdict=old["honest_verdict"],
                required_failures=old.get("historical_required_failures", []),
                resolved=False,
            )
        )
    value["retire_if_same_verdict"].update(
        identical_prior_failure=False,
        action="retire unchanged qualification-only runs; bind new twelve-task authority",
    )
    value["reproducibility_checksum"] = shared.canonical_hash(
        {
            "manifest": sha256_file(manifest),
            "sources": value["source_artifact_hashes"],
            "methods": value["method_freeze"],
        }
    )[7:23]
    value["field_principles"].update(
        {
            key: "Keep current producer, byte custody and unmeasured scientific scope explicit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    return value


def cold_replay(path: Path, raw: Path) -> bool:
    """Recompute task and source counts before checking current producer dates."""
    if not shared.cold_replay(path, raw, experiment_id=7929, count=12):
        return False
    value = json.loads(path.read_text())
    return (
        value["milestone"] == MILESTONE
        and value["execution_date"] == "20260930"
        and value["historical_fixture_date"] == "20260929"
        and value["lifecycle_task_count"] == 12
        and value["qualified_evidence_custody"]
        == qualified_custody(Path(__file__).resolve().parents[3])
        and value["duration_s"]
        == (value["ended_monotonic_timestamp_ns"] - value["started_monotonic_timestamp_ns"]) / 1e9
    )
