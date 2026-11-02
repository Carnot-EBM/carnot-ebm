"""REQ-REPORT-8439: engineering support is separate from policy refinement.

This retrospective reader keeps exposed games and censored events visible. A
publication defect closes readiness without deleting valid historical operands.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting.arc_supervisor_live_panel_8384 import freeze
from carnot.reporting import arc_supervisor_support_evidence_8439 as evidence
from carnot.reporting import v726_contract_methods as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8439_v726_arc_supervisor_support", "exp8439-arc-supervisor-support"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_supervisor_support_8439.py"
OWNED = [
    "python/carnot/reporting/arc_supervisor_support_evidence_8439.py",
    "python/carnot/reporting/arc_supervisor_support_8439.py",
    "python/carnot/reporting/arc_supervisor_support_runner_8439.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "ops/exclusion_manifest.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/arc_panel_qualification_8412.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_solver_kit.py",
]
CANDIDATE_PIN = evidence.CANDIDATE_PIN
reference, checksum = authority.reference, authority.checksum
reduce = evidence.reduce


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real boundary counts let the caller distinguish bounded work from a stall."""
    print(f"[exp8439] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """A missing external operand stays null rather than becoming a measured zero."""
    return dict(
        check="source_or_precondition",
        upstream="exp8412-arc-panel-qualification" if "8412" in str(path) else TASK,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def producer(root: Path) -> Json:
    """The independent pin remains fixed even when a supplied local hash changes."""
    return evidence.producer(root, CANDIDATE_PIN)


def frontier(path: Path | None, source: Json, raw: Path) -> Json:
    """A bounded optional read cannot authorize new evidence without a qualified seal."""
    started = time.monotonic()
    if path is None:
        return dict(status="not_supplied", sha256=None, new_authenticated_live_outcomes=0)
    try:
        if path.stat().st_size > 2_000_000:
            raise ValueError("frontier_size_budget")
        value = json.loads(path.read_bytes())
        digest = sha256_file(path)
        atomic_json(raw / "supplied_frontier.json", value)
        consumed = digest == CANDIDATE_PIN or value.get("supervisor_outcome_rows") == source.get(
            "supervisor_outcome_rows"
        )
        return dict(
            status="already_consumed" if consumed else "unqualified_new_frontier",
            sha256=digest,
            new_authenticated_live_outcomes=0,
            elapsed_s=time.monotonic() - started,
            read_cap_s=60,
        )
    except (OSError, ValueError, TypeError) as error:
        return dict(
            status="unavailable_or_invalid",
            sha256=None,
            new_authenticated_live_outcomes=0,
            reason=str(error),
            read_cap_s=60,
        )


def measure(root: Path, raw: Path, private: Path, supplied: Path | None = None) -> Json:
    """Authenticate only actual dependencies before reconstructing cached outcomes."""
    progress("preconditions_before")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    probe = private / "probe"
    probe.write_bytes(b"exp8439-private")
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    filesystem = max(
        (m for m in mounts if private.resolve().is_relative_to(m[1])), key=lambda m: len(m[1])
    )[2]
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    resources: Json = dict(
        private_scratch=str(private),
        disk_backed=filesystem not in ["tmpfs", "ramfs"],
        filesystem=filesystem,
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage(private).free,
        task_cap_s=4800,
        validation_reserve_s=900,
        no_model_resources_required=True,
        tools={
            n: (ROOT / ".venv/bin" / n).is_file()
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
    )
    contract = authority.authority(root, raw / "authority")
    failures = (
        []
        if contract["activated"] and any(t["id"] == TASK for t in contract["tasks"])
        else [gate(root / authority.ACTIVE, "activated_complete_V726_authority", True, False)]
    )
    if (
        not resources["disk_backed"]
        or memory < 512_000_000
        or resources["free_disk_bytes"] < 512_000_000
        or probe.read_bytes() != b"exp8439-private"
        or private.resolve().is_relative_to(ROOT)
    ):
        failures.append(gate(private, "private_disk_memory_resources", True, resources))
    for name, present in resources["tools"].items():
        if not present:
            failures.append(gate(ROOT / ".venv/bin" / name, "tool_available", True, None))
    named_hashes = {
        str(root / name): sha256_file(root / name) if (root / name).is_file() else None
        for name in INPUTS
    }
    for path, digest in named_hashes.items():
        if digest is None:
            failures.append(gate(Path(path), "named_input_present", True, None))
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    if exclusion_path.is_file():
        retired = yaml.safe_load(exclusion_path.read_bytes()).get("retired", [])
        if any(str(r.get("experiment_id")) == "8439" for r in retired):
            failures.append(gate(exclusion_path, "task_not_retired", True, False))
    refs = [
        dict(
            path=s["source_path"], sha256=s["sha256"], snapshot_path=s["snapshot_path"], exists=True
        )
        for s in contract["authority_snapshots"].values()
        if s.get("exists")
    ]
    registry_path = root / "ops/arc_solve_registry.yaml"
    registry = yaml.safe_load(registry_path.read_bytes()) if registry_path.is_file() else None
    if registry is None:
        failures.append(gate(registry_path, "registry_precheck", "present", None))
    else:
        shutil.copyfile(registry_path, raw / "registry.yaml")
    rows, operands, source = [], [], {}
    source_ref: Json | None = None
    observed = None
    try:
        source = producer(root)
        shutil.copyfile(root / evidence.CANDIDATE, raw / "upstream_candidate.json")
        source_ref = reference(raw / "upstream_candidate.json")
        if registry is not None:
            checked_registry = freeze(source["frozen_panel"]["roster"], registry["games"])[
                "registry_precheck"
            ]
            if checked_registry != source["registry_precheck"]:
                failures.append(
                    gate(
                        registry_path,
                        "unchanged_registry_precheck",
                        source["registry_precheck"],
                        checked_registry,
                    )
                )
        current = root / evidence.UPSTREAM
        observed = sha256_file(current) if current.is_file() else None
        if current.is_file():
            shutil.copyfile(current, raw / "observed_upstream_primary.json")
        if observed != CANDIDATE_PIN:
            failures.append(gate(current, "validated_primary_sha256", CANDIDATE_PIN, observed))
        progress("primitive_snapshot_before", 0, 8)
        for index, (unit, row) in enumerate(
            zip(source["frozen_panel"]["units"], source["rows"], strict=True)
        ):
            target = raw / "episodes" / str(index)
            target.mkdir(parents=True)
            for name in ["steps.jsonl", "episode.json"]:
                old = Path(row["primitive_directory"]) / name
                if sha256_file(old) != source["raw_shard_hashes"][str(old)]:
                    raise ValueError("primitive_seal")
                shutil.copyfile(old, target / name)
            operands.append(
                dict(
                    unit=unit,
                    steps=reference(target / "steps.jsonl"),
                    terminal=reference(target / "episode.json"),
                )
            )
            progress("primitive_snapshot", index + 1, 7 - index)
        rows = reduce(source, operands)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            gate(
                root / evidence.CANDIDATE,
                "qualified_terminal_and_primitives",
                CANDIDATE_PIN,
                str(error),
            )
        )
    inspected = frontier(supplied, source, raw)
    work = dict(
        root=str(root),
        rows=rows,
        operands=operands,
        producer_reference=source_ref,
        failures=failures,
        authority=contract,
        authority_refs=refs,
        registry_precheck=source.get("registry_precheck", {}),
        registry_reference=reference(raw / "registry.yaml") if registry else None,
        frontier=inspected,
        preconditions_checked=resources,
        historical_model_provenance=source.get("historical_model_provenance", []),
        source_artifact_hashes={
            str(root / evidence.CANDIDATE): CANDIDATE_PIN if source_ref else None,
            str(registry_path): sha256_file(registry_path) if registry else None,
            str(root / evidence.UPSTREAM): observed if source_ref else None,
            **named_hashes,
            **{r["path"]: r["sha256"] for r in refs},
        },
        code_config_hashes={n: sha256_file(ROOT / n) for n in OWNED},
        raw_shard_hashes={str(p): sha256_file(p) for p in raw.rglob("*") if p.is_file()},
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[
            dict(
                phase="preconditions_and_primitive_replay",
                started_monotonic_ns=started,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(rows), 8 - len(rows))
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A support null is terminal; upstream defects and owned failures stay distinct."""
    rows = work["rows"] or [
        dict(
            episode_id=f"missing:{g}:{s}:{a}",
            game=None,
            seed=s,
            arm=a,
            status="unstarted",
            absolute_metric=None,
            missing_reason="required_external_operand_unavailable",
            censored=True,
        )
        for g in range(2)
        for s in [11, 22]
        for a in ["off", "on"]
    ]
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
    )
    verdict = "disqualified" if not qualified else "blocked" if work["failures"] else "null"
    ready = int(qualified and not work["failures"] and len(work["rows"]) == 8)
    summary = evidence.summarize(work["rows"])
    value = dict(
        experiment_id=8439,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v726_arc_supervisor_support",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, loads=0, generations=0, llm_calls=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=8,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(bool(r.get("censoring") or r.get("censored")) for r in rows),
        excluded_count=0,
        independent_count=len({r["game"] for r in work["rows"]}),
        sample_size_budget=dict(
            games=2,
            seeds=[11, 22],
            policy_arms=["off", "on"],
            supervisor_arms=evidence.ARMS,
            intended_episodes=8,
            intended_arm_cells=32,
            minimum_overlapping_games=3,
            minimum_resolved_firings_per_shared_cell=5,
            independent_unit="exposed_public_game",
            hidden_games=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="two_historically_exposed_public_games_adapter_withheld",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_validation=qualified,
            external_operands=not work["failures"],
            eight_episodes=len(work["rows"]) == 8,
            refinement_support=False,
            scientific_generalization=False,
        ),
        acceptance_gate_principles=dict(
            owned_validation="All real owned commands and consumer findings must qualify.",
            external_operands="Missing or changed required source bytes block readiness.",
            eight_episodes="All eight sealed episodes remain available without increasing independent games.",
            refinement_support="Three overlapping games and five resolved firings per shared cell are required.",
            scientific_generalization="Exposed engineering evidence supplies no hidden-game or causal learning claim.",
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=[11, 22],
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        cited_upstream_artifacts=[
            dict(
                path=p,
                sha256=h,
                fields_imported=[
                    "frozen_panel",
                    "rows",
                    "supervisor_outcome_rows",
                    "historical_model_provenance",
                    "registry_precheck",
                ]
                if "8412" in p
                else ["authority_or_registry"],
            )
            for p, h in sorted(work["source_artifact_hashes"].items())
        ],
        arc_support_ready_score=ready,
        solve_provenance="live_agent_self_discovery",
        registry_precheck=work["registry_precheck"],
        duplicate_solve_credit=0,
        new_frontier_hash=work["frontier"]["sha256"],
        frontier_inspection=work["frontier"],
        work_reference=reference(raw / "measurement.json"),
        **summary,
    )
    value["methodology_note"] = (
        "Reconstruct original live-wrapper engineering receipts from independently pinned primitive observations. Shadow redirects are counterfactual diagnostics; all pending outcomes remain right-censored. Two exposed games cannot prove hidden-game generalization. No justified policy change."
    )
    value["field_principles"] = {
        key: "Bind "
        + key
        + " to sealed primitive evidence; missing operands are distinct from zero."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        arc_support_ready_score="Qualification requires unchanged upstream publication and all owned checks; support nulls grant no refinement.",
        overlap_by_arm="Three games and five resolved events per shared cell are required; seeds do not multiply games.",
        censoring_counts="Pending windows never become failures; completion and censoring may overlap.",
        solve_provenance="Only imported runtime attempts are self-discovery; no new solve was scheduled.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(candidate: Path) -> bool:
    """Fresh reconstruction defeats rehashed summaries using independent producer seals."""
    try:
        value = json.loads(candidate.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        work = evidence.read(value["work_reference"])
        if any(
            sha256_file(Path(path)) != digest for path, digest in work["raw_shard_hashes"].items()
        ):
            return False
        if work["producer_reference"]:
            if work["producer_reference"]["sha256"] != CANDIDATE_PIN:
                return False
            source = evidence.read(work["producer_reference"])
            if reduce(source, work["operands"]) != work["rows"]:
                return False
        with TemporaryDirectory(prefix="exp8439-cold-", dir="/var/tmp") as directory:
            (Path(directory) / "writes").mkdir()
            with frozen_inputs(work["authority_refs"], Path(directory) / "writes"):
                actual = authority.authority(Path(work["root"]), Path(directory) / "authority")
                if (
                    actual["activated"] != work["authority"]["activated"]
                    or actual["tasks"] != work["authority"]["tasks"]
                ):
                    return False
        for receipt in value["validation_receipts"]:
            for channel in ["stdout", "stderr"]:
                if (
                    channel + "_path" in receipt
                    and sha256_file(Path(receipt[channel + "_path"]))
                    != receipt[channel + "_sha256"]
                ):
                    return False
        raw = Path(value["work_reference"]["path"]).parent
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return bool(value == build(work, value["validation_receipts"], raw, output))
    except (OSError, ValueError, KeyError, TypeError):
        return False
