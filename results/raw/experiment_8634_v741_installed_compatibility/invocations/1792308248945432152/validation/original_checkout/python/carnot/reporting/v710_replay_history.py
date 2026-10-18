"""REQ-VERIFY-8218: retain original historical bytes and unavailable producer seals.

The original audit remains disqualified. Its repaired CLI receives a separate
custody receipt using copied original rows and unchanged statistical operands.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v709_qualification as old
from carnot.reporting import v709_execution as x
from carnot.reporting import v710_contract_replay as q
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_decision_audit_8210 as audit

Json = dict[str, Any]
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    q.DESIGN,
    q.HISTORY,
    "research-roadmap.yaml",
    "ops/conductor-log.md",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/v709_qualification.py",
    "python/carnot/reporting/v709_capstone.py",
    "python/carnot/verify/restricted_decision_audit_8210.py",
    "results/experiment_8217_v709_capstone.json",
    "results/experiment_8210_v709_restricted_decision_audit.json",
]


def bind(path: Path, raw: Path, work: Json, expected: str | None = None) -> Json:
    """Every missing or changed input names a real unavailable operand, never a zero."""
    ref = q.snapshot(path, raw / "inputs", str(len(work["source_artifact_hashes"])))
    work["source_artifact_hashes"].append(ref)
    observed = ref["sha256"] if expected else ref["exists"]
    wanted = expected if expected else True
    check = q.failure(path, "sha256" if expected else "exists", wanted, observed, ref["sha256"])
    check["passed"] = observed == wanted
    work["preconditions_checked"].append(check)
    if not check["passed"]:
        work["failures"].append(check)
    return ref


def recover_code(ref: Json, raw: Path, cache: dict[str, Json]) -> Json:
    """Committed bytes qualify only at the producer's seal; live files are never substitutes."""
    key = ref["sha256"]
    if key in cache:
        return cache[key]
    path = Path(ref["path"])
    result = dict(ref, exists=False, recovery="original_bytes_unavailable")
    if path.is_relative_to(q.ROOT):
        relative = str(path.relative_to(q.ROOT))
        receipt = x.child(
            "git_" + key[7:19],
            ["git", "log", "-20", "--format=%H", "--", relative],
            raw / "code_logs",
            deadline=30,
            scope="historical_code_search",
        )
        commits = Path(receipt["stdout_path"]).read_text().splitlines()
        for index, commit in enumerate(commits):
            x.progress("immutable_code_search", index, len(commits) - index)
            row = x.child(
                "blob_" + key[7:19] + "_" + str(index),
                ["git", "show", commit + ":" + relative],
                raw / "code_logs",
                deadline=30,
                scope="historical_code_search",
            )
            saved = Path(row["stdout_path"])
            if row["passed"] and sha256_file(saved) == key:
                snap = q.snapshot(saved, raw / "code", key[7:])
                result = dict(
                    ref,
                    exists=True,
                    snapshot_path=snap["snapshot_path"],
                    recovery="immutable_git_commit",
                    git_commit=commit,
                )
                break
    cache[key] = result
    return result


def audit_copy(value: Json, raw: Path, work: Json) -> None:
    """Cold replay the repaired audit on original copies without rerunning any training."""
    original = value["measurement_reference"]
    work["h1_custody"] = dict(
        agreement=None, scope="unavailable_original_primitives", scientific_gate=False
    )
    bound = bind(Path(original["path"]), raw, work, original["sha256"])
    if not bound["exists"] or bound["sha256"] != original["sha256"]:
        return
    source = json.loads(Path(original["path"]).read_bytes())
    private = raw / "audit_replay"
    private.mkdir(parents=True, exist_ok=True)
    path = private / "candidate.json"
    argv = [
        str(q.ROOT / ".venv/bin/python"),
        "-u",
        str(q.ROOT / audit.CLI),
        "--cold-replay",
        str(path),
    ]
    tampered_path = private / "tampered_candidate.json"
    tampered_argv = [*argv[:-1], str(tampered_path)]
    control_plan = x.pytest_plan(raw / "audit_controls")
    atomic_json(
        private / "replay_manifest.json",
        dict(
            argv=argv,
            CLI=audit.CLI,
            expected_exits=[0, 1],
            tampered_argv=tampered_argv,
            controls=control_plan,
            frozen_before_measurement_ns=time.monotonic_ns(),
        ),
    )
    copied = deepcopy(source)
    copied["code_config_hashes"] = []
    for name in audit.OWNED:
        snap = q.snapshot(q.ROOT / name, raw / "repaired_code", Path(name).name)
        work["immutable_code_snapshots"].append(
            dict(snap, purpose="repaired_audit_current_invocation")
        )
        copied["code_config_hashes"].append(dict(path=snap["snapshot_path"], sha256=snap["sha256"]))
    for ref in copied["refs"]:
        snap = bind(Path(ref["path"]), raw, work, ref["sha256"])
        if not snap["exists"] or snap["sha256"] != ref["sha256"]:
            return
        target = private / "inputs" / Path(ref["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(snap["snapshot_path"]).read_bytes())
        ref["path"] = str(target)
    x.progress("8218_original_H1_reduction_before", 0, 128)
    reduced = audit.n.reduce(copied["evidence"])
    x.progress("8218_original_H1_reduction_after", 128, 0)
    work["h1_custody"] = dict(
        scope="administrative_custody_only",
        scientific_gate=False,
        H1=reduced["H1"],
        original_H1=value["H1"],
        energy_increment_gain=reduced["energy_increment_gain"],
        agreement=reduced["H1"] == value["H1"],
        intended_count=reduced["intended_count"],
        completed_count=reduced["completed_count"],
    )
    if not work["h1_custody"]["agreement"]:
        work["failures"].append(q.failure(Path(original["path"]), "H1", value["H1"], reduced["H1"]))
    for name, data in [
        ("measurement", copied),
        ("primitive_evidence", copied["evidence"]),
        ("independent_reduction", reduced),
        ("frozen_configuration", audit.CONFIG),
    ]:
        atomic_json(private / (name + ".json"), data)
    # New measured controls are administrative evidence, not historical passed flags.
    controls = x.execute(control_plan, raw / "audit_control_logs")
    candidate = audit.build(copied, private, controls)
    candidate.pop("reproducibility_checksum")
    candidate["reproducibility_checksum"] = canonical_hash(candidate)
    atomic_json(path, candidate)
    receipts = [x.child("original_8210_replay", argv, private / "logs", deadline=90)]
    candidate = deepcopy(candidate)
    candidate["H1"]["passed"] = not candidate["H1"]["passed"]
    candidate.pop("reproducibility_checksum")
    candidate["reproducibility_checksum"] = canonical_hash(candidate)
    atomic_json(tampered_path, candidate)
    receipts.append(
        x.child("tampered_8210_replay", tampered_argv, private / "logs", deadline=90, expected=1)
    )
    work["replay_controls"].extend(controls + receipts)
    for r in receipts:
        if not r["passed"]:
            work["failures"].append(
                q.failure(
                    Path(r["stdout_path"]),
                    r["name"] + ".exit",
                    r["expected_exit"],
                    r["exit_code"],
                    r["stdout_sha256"],
                )
            )


def measure(root: Path, raw: Path) -> Json:
    """Preserve every original task slot before computing any current replay readiness."""
    began = time.monotonic_ns()
    work: Json = dict(
        failures=[],
        preconditions_checked=[],
        source_artifact_hashes=[],
        historical_dispositions=[],
        replay_controls=[],
        immutable_code_snapshots=[],
        h1_custody={},
    )
    x.progress("8218_preconditions_before", 0, len(INPUTS))
    for i, name in enumerate(INPUTS):
        bind(root / name, raw, work)
        x.progress("8218_preconditions", i + 1, len(INPUTS) - i - 1)
    work["contract"] = q.assess(
        root / q.DESIGN,
        root / "research-roadmap-next.yaml",
        root / "research-roadmap.yaml",
        raw / "authority",
    )
    if work["failures"]:
        return work
    d = Path(work["contract"]["authority_snapshots"]["design"]["snapshot_path"])
    tasks = work["contract"]["tasks"]
    active = raw / "full_private_authority.yaml"
    active.write_text(yaml.safe_dump(dict(milestone=q.MILESTONE, tasks=tasks)))
    work["mutation_controls"] = q.mutations(d, active, raw / "mutations") if tasks else []
    for control in work["mutation_controls"]:
        if not control["rejected"]:
            work["failures"].append(q.failure(d, "mutation." + control["control"], True, False))
    _, historical = q.parse_design((root / q.HISTORY).read_text(), milestone="2026.10.709")
    cache: dict[str, Json] = {}
    for index, task in enumerate(historical):
        x.progress("8218_historical_before", index, 13 - index)
        path = root / task["deliverable"]
        ref = bind(path, raw, work)
        if not ref["exists"]:
            work["historical_dispositions"].append(
                dict(task=task, original_primary=None, disposition="missing_original_bytes")
            )
            continue
        value = json.loads(Path(ref["snapshot_path"]).read_bytes())
        code = old.hash_references(value.get("code_config_hashes", []))
        receipt_rows = old.receipts(value.get("validation_receipts", []))
        codes = [recover_code(c, raw, cache) for c in code]
        work["immutable_code_snapshots"].extend(codes)
        for c in codes:
            if not c["exists"]:
                work["failures"].append(
                    q.failure(
                        Path(c["path"]), "original_code_bytes", c["sha256"], None, c["sha256"]
                    )
                )
        side = Path(
            value.get(
                "terminal_validation_sidecar_path",
                path.parent / "raw" / path.stem / "validators" / (ref["sha256"][7:] + ".json"),
            )
        )
        if value.get("terminal_validation_sidecar_path"):
            observation = old.terminal(path, value, raw / "terminal" / str(index))
        else:
            report = read_bound_sidecar(path, side)
            observation = dict(passed=report["report"]["passed"], report=report, references=[])

        for terminal_ref in observation["references"]:
            bind(Path(terminal_ref["path"]), raw, work, terminal_ref["sha256"])
        terminal_ref = bind(side, raw, work)
        fields = [
            "experiment_id",
            "task_id",
            "honest_verdict",
            "verdict_class",
            "required_checks_passed",
            "flagged_adversarial",
            "MODEL_SPECS",
            "model_invocation_counts",
            "measurement_reference",
            "terminal_validation_sidecar_path",
        ]
        if index == 5:
            fields.extend(["H1", "energy_increment_gain"])
        projection = {key: value[key] for key in fields if key in value}
        work["historical_dispositions"].append(
            dict(
                original_primary_reference=ref,
                fields_imported=list(projection),
                task=task,
                path=str(path),
                sha256=ref["sha256"],
                original_primary=projection,
                honest_verdict=value["honest_verdict"],
                verdict_class=value["verdict_class"],
                historical_required_checks_passed=value["required_checks_passed"],
                receipt_schema="named-mapping-v1"
                if isinstance(value["validation_receipts"], dict)
                else "named-list-v1",
                code_schema="mapping-v1"
                if isinstance(value["code_config_hashes"], dict)
                else "reference-list-v1",
                original_receipts=receipt_rows,
                terminal_sidecar=terminal_ref,
                terminal_observation=observation,
                immutable_code_qualified=all(c["exists"] for c in codes),
                historical_misconduct_inferred=False,
            )
        )
        if index == 5:
            audit_copy(value, raw, work)
        if index == 12:
            retained = next(r for r in receipt_rows if r["name"] == "affected_reducers_E2E015_019")
            for stream in ("stdout", "stderr"):
                bind(Path(retained[stream + "_path"]), raw, work, retained[stream + "_sha256"])
            original_text = Path(retained["stdout_path"]).read_text()
            work["retained_affected_failure"] = dict(
                original_receipt=retained,
                errno122_retained="[Errno 122] Disk quota exceeded" in original_text,
                original_exit=retained["exit_code"],
            )
            reproduction = (
                root
                / "results/raw"
                / q.NAME
                / "retained_failure/affected_reducers_reproduction.receipt.json"
            )
            if reproduction.is_file():
                bind(reproduction, raw, work)
                r = json.loads(reproduction.read_bytes())
                for stream in ("stdout", "stderr"):
                    bind(Path(r[stream + "_path"]), raw, work, r[stream + "_sha256"])
                exact = "[Errno 122] Disk quota exceeded" in Path(r["stdout_path"]).read_text()
                work["retained_affected_failure"].update(
                    reproduction_receipt=r, exact_errno_reproduced=exact
                )
            else:
                exact = False
            if not exact:
                work["failures"].append(
                    q.failure(
                        reproduction,
                        "retained_quota_failure_reproduction",
                        "[Errno 122] Disk quota exceeded",
                        None,
                    )
                )
        x.progress("8218_historical_after", index + 1, 12 - index)
    work["phase_spans"] = [
        dict(
            phase="authenticate_and_replay",
            started_monotonic_ns=began,
            ended_monotonic_ns=time.monotonic_ns(),
        )
    ]
    return work
