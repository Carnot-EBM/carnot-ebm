"""REQ-REPORT-8402: dependency custody preserves historical science and policy."""

from __future__ import annotations

from contextlib import contextmanager
import ast
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import parse_design as parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = (
    "experiment_8402_v724_historical_consumer_roots",
    "exp8402-historical-consumer-roots",
    "2026.10.724",
)
DESIGN, ACTIVE, STAGED = (
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-roadmap.yaml",
    "research-roadmap-next.yaml",
)
PROTOCOL = "openspec/change-proposals/v724-qualified-service-protocol.json"
METHODS = "docs/research-notes/v724-methods.md"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_historical_consumer_roots_8402.py"
OWNED = [
    "python/carnot/reporting/historical_consumer_roots_8402.py",
    "python/carnot/reporting/historical_consumer_runner_8402.py",
    "python/carnot/testing/historical_roots_8402.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASKS_PIN = "82778c8abdbdbf2a69db3c7382e054515098e71056bb91d4521afe47792b2c34"
PRIMARY_PINS = {
    8388: "sha256:17031d48bdd178087c4d3843225c5456925e5348b594fcc3d1b80baaf43c36c7",
    8390: "sha256:fb57dfc3b0fcce2931a231133ba857aec7ee6dc3d6c4a7b8279fca767c7f9610",
    8396: "sha256:ca568e7525204e9f68600076cf42fbc6ed395b48c78ec5f7a1bed30646b42e0b",
    8401: "sha256:ba5b47cf9281d1614f8dec412c5fbdedf099bf40b8a5de9ef1e62558d038d4fd",
}
HISTORY = {
    720: (
        "research-roadmap-v720-preserved-20261009.md",
        "76e61897ab2e89523108c7402a5d9d6349417b4ff1bbc69cf9f839a154733ddb",
        "experiment_8347_v720_local_consumer_qualification/invocations/1791554502661113165/authority/assessment/active-e2ff70fb6329444b05d7794d1fc75d6ec5e12fd0e93d5397d9c5a6dd4c4f6a3b.bin",
        "e2ff70fb6329444b05d7794d1fc75d6ec5e12fd0e93d5397d9c5a6dd4c4f6a3b",
    ),
    721: (
        "research-roadmap-v721-preserved-20261010.md",
        "9e36f4a57a73f80acd443e52afa25caeed5e7f91fdbe72b4ff8c750b388f0ccd",
        "experiment_8360_v721_contract_methods/invocations/1791594915296581265/authority/2-0e9c838f83aa7d2664bc1b7b5db241efeaf50c4408bd33d9c4f2e5bb9986c161.bin",
        "0e9c838f83aa7d2664bc1b7b5db241efeaf50c4408bd33d9c4f2e5bb9986c161",
    ),
    722: (
        "research-roadmap-v722-preserved-20261010.md",
        "cc6def57b4614b4016f456a28895c3033701ad72f8f10e6b712cb3db2929efa1",
        "experiment_8374_v722_contract_methods/invocations/1791622850199168604/authority/2-10c257eb52c30ff5761889551579ae8bcebaac7c9ef0a7a0feb925a525d2cdc3.bin",
        "10c257eb52c30ff5761889551579ae8bcebaac7c9ef0a7a0feb925a525d2cdc3",
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work boundaries because a quiet child can hide a stalled task."""
    print(f"[exp8402] phase={phase} completed={completed} pending={pending}", flush=True)


def checked(path: Path, digest: str) -> Path:
    """An independently fixed hash rejects replacement bytes even after a manifest rehash."""
    if sha256_file(path) != digest:
        raise ValueError("pinned_source_sha256")
    return path


def reference(path: Path) -> Json:
    """Name the bytes a fresh reader must see, rather than an inferred source version."""
    return dict(path=str(path), sha256=sha256_file(path))


def assertion_sources(nodes: list[str]) -> Json:
    """A family seal binds actual test bytes and assertions, not only a reported exit."""
    result = {}
    for node in nodes:
        path = ROOT / node.split("::")[0]
        result[str(path)] = dict(
            source=reference(path),
            assertions_sha256=canonical_hash(
                [
                    ast.dump(n, include_attributes=False)
                    for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.Assert)
                ]
            ),
        )
    return result


def historical_tasks(version: int) -> list[Json]:
    """Original activated bytes supply fixtures; missing design authority is never synthesized."""
    _, _, source, digest = HISTORY[version]
    return list(
        yaml.safe_load(checked(ROOT / "results/raw" / source, "sha256:" + digest).read_bytes())[
            "tasks"
        ]
    )


def fixtures(directory: Path) -> Json:
    """Build separate immutable roots from preserved designs and original activation receipts."""
    result = {}
    for version, (design, design_hash, active, active_hash) in HISTORY.items():
        root = directory / str(version)
        refs = []
        for name, source, digest in [
            (DESIGN, ROOT / "openspec/change-proposals" / design, design_hash),
            (ACTIVE, ROOT / "results/raw" / active, active_hash),
            (STAGED, ROOT / "results/raw" / active, active_hash),
        ]:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(checked(source, "sha256:" + digest), destination)
            destination.chmod(0o400)
            refs.append(
                dict(
                    reference(destination),
                    source_path=str(ROOT / name),
                    exists=True,
                    snapshot_path=str(destination),
                )
            )
        tasks = historical_tasks(version)
        result[str(version)] = dict(
            root=str(root),
            refs=refs,
            tasks=tasks,
            tasks_sha256=tasks_digest(tasks),
            independent_design_available=version != 722,
        )
    atomic_json(directory / "manifest.json", result)
    return result


def check_fixtures(value: Json) -> bool:
    """The manifest cannot authorize its own aliases, task mutations or an unrelated root."""
    try:
        for version, (_, design_hash, _, active_hash) in HISTORY.items():
            row = value[str(version)]
            if row["tasks"] != historical_tasks(version) or row["tasks_sha256"] != tasks_digest(
                row["tasks"]
            ):
                return False
            if row["independent_design_available"] != (version != 722):
                return False
            for name, ref, digest in zip(
                [DESIGN, ACTIVE, STAGED],
                row["refs"],
                [design_hash, active_hash, active_hash],
                strict=True,
            ):
                if (
                    ref["source_path"] != str(ROOT / name)
                    or Path(ref["path"]) != Path(row["root"]) / name
                    or ref["snapshot_path"] != ref["path"]
                ):
                    return False
                checked(Path(ref["path"]), "sha256:" + digest)
                if ref["sha256"] != "sha256:" + digest:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


@contextmanager
def inject(fixture: Json, scratch: Path) -> Iterator[None]:
    """Redirect only declared dependency aliases; private writes preserve old input bytes."""
    if not fixture.get("refs"):
        raise ValueError("explicit_historical_root_required")
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    mapping = {r["source_path"]: r["path"] for r in fixture["refs"]}
    copy = shutil.copyfile

    def copied(source: Any, target: Any, **kwargs: Any) -> Any:
        return copy(mapping.get(str(Path(source).absolute()), source), target, **kwargs)

    with frozen_inputs(fixture["refs"], scratch), patch.object(shutil, "copyfile", copied):
        yield


def cost_tape(ratio: int) -> list[Json]:
    """Release the last prediction of each cycle after eight ticks; drain without sleeping."""
    predictions = [
        dict(kind="prediction", tick=t, cycle=(t - 1) // ratio, designated=t % ratio == 0)
        for t in range(1, 8 * ratio + 1)
    ]
    feedback = [
        dict(
            kind="feedback",
            tick=c * ratio + 8,
            issued_tick=c * ratio,
            cycle=c - 1,
            source_policy="missing_or_invalid_never_updates",
        )
        for c in range(1, 9)
    ]
    return sorted(predictions + feedback, key=lambda r: (r["tick"], r["kind"] == "feedback"))


def authority(root: Path, raw: Path) -> Json:
    """Current activation binds full objects; old activation cannot grant new task authority."""
    try:
        text = (root / DESIGN).read_text()
        tasks = parse_design(text, milestone=MILESTONE)[1]
        if tasks_digest(tasks) != TASKS_PIN:
            raise ValueError("independent_current_tasks_sha256")
        # The shipped validator expects a plain digest label. Keep exact source
        # bytes sealed separately and adapt only Markdown decoration for it.
        view = raw / "canonical_design.md"
        view.parent.mkdir(parents=True, exist_ok=True)
        view.write_text(
            text.replace("**Canonical full-task SHA-256:**", "Canonical full-task SHA256:")
        )
        result = dict(
            assess_authorities(
                view,
                root / STAGED if (root / STAGED).exists() else root / ACTIVE,
                root / ACTIVE,
                raw,
                milestone=MILESTONE,
                first_id=8402,
                count=14,
            )
        )
        for row, task in zip(result["contract_rows"], tasks, strict=True):
            row.update(
                matched=all(row["checks"].values()),
                absolute_metric=int(all(row["checks"].values())),
                excluded=not all(row["checks"].values()),
                full_task_sha256=canonical_hash(task),
            )
        failures = [
            g
            for g in result["gate_check_summary"]
            if g["artifact_field"] != "contract_rows.matched"
            or not all(r["matched"] for r in result["contract_rows"])
        ]
        result.update(tasks=tasks, activated=not failures, gate_check_summary=failures)
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError):
        return dict(
            tasks=[],
            activated=False,
            canonical_tasks_sha256=None,
            contract_rows=[],
            gate_check_summary=[failure(root / DESIGN, "complete_current_design", True, None)],
        )


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name missing external fields explicitly so absence cannot be mistaken for zero."""
    return dict(
        check=field,
        upstream_id=path.stem,
        artifact_path=str(path),
        artifact_hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def parse_outcomes(path: Path) -> Json:
    """Read actual verbose pytest terminal outcomes while retaining every failed assertion."""
    text = path.read_text()
    result: Json = {key: [] for key in ["passed", "failed", "error", "skipped"]}
    for node, status in re.findall(
        r"^(tests/python/\S+) (PASSED|FAILED|ERROR|SKIPPED)", text, re.M
    ):
        result[status.lower()].append(node)
    result["summary"] = [
        line for line in text.splitlines() if re.search(r"\d+ (?:passed|failed|errors?)", line)
    ][-1:]
    return result


def family(name: str, receipts: list[Json], nodes: list[str]) -> Json:
    """Matching failed outcomes remain a failed family, even when the reader classifies them correctly."""
    outcomes = [parse_outcomes(Path(r["stdout_path"])) for r in receipts]
    equal = len(receipts) == 2 and all(
        outcomes[0][key] == outcomes[1][key] for key in ["passed", "failed", "error", "skipped"]
    )
    ready = (
        equal
        and all(r["actual_exit"] == 0 and not r["timed_out"] for r in receipts)
        and bool(outcomes[0]["passed"])
        and not any(outcomes[0][key] for key in ["failed", "error", "skipped"])
    )
    return dict(
        family=name,
        ready=ready,
        status="passed" if ready else "failed" if receipts else "blocked",
        identical_outcomes=equal,
        consumer_node_ids=nodes,
        runs=receipts,
        outcomes=outcomes,
    )


def historical_logs(raw: Path) -> list[Json]:
    """Reproduce the three observed failure summaries from original sealed logs, without recursive milestone replay."""
    rows = []
    for number, filename, expected in [
        (8390, "private_E2E020_direct_consumers", (3, 15)),
        (8396, "private_E2E018", (2, 0)),
        (8401, "qualified_branch_consumers", (5, 0)),
    ]:
        primary = next((ROOT / "results").glob(f"experiment_{number}_*.json"))
        value = json.loads(checked(primary, PRIMARY_PINS[number]).read_bytes())
        receipt = next(r for r in value["validation_receipts"] if r["name"] == filename)
        source = checked(Path(receipt["stdout_path"]), receipt["stdout_sha256"])
        destination = raw / f"historical_{number}.stdout"
        shutil.copyfile(source, destination)
        text = destination.read_text()
        failures = re.findall(r"^FAILED (\S+)", text, re.M)
        errors = re.findall(r"^ERROR (\S+)", text, re.M)
        if (len(failures), len(errors)) != expected:
            raise ValueError("historical_log_counts")
        rows.append(
            dict(
                experiment_id=number,
                primary=reference(primary),
                log=reference(destination),
                original_receipt=receipt,
                failed_node_ids=failures,
                error_node_ids=errors,
                failed_count=len(failures),
                error_count=len(errors),
            )
        )
    return rows


PROTOCOL_PIN = "sha256:7de0337029fdf040efa16ce81ce1df4467597bea9dbba9bc86ed01a1c8eeb89d"
METHODS_PIN = "sha256:cd746863f0c3ba6eb31222a4963d1e16fb2f94872d14574cd6d81ed54b6ebbd9"


def measure(root: Path, raw: Path) -> Json:
    """Authenticate before starting consumers; future producers stay dependencies rather than observations."""
    import os
    import time
    from carnot.reporting.v721_capstone_evidence import freeze

    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if raw.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mounted[mounted.index("-") + 1]
    memory = (
        int(
            next(
                s.split()[1]
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    resources: Json = dict(
        private_disk_backed_scratch=filesystem not in ["tmpfs", "ramfs"],
        filesystem=filesystem,
        available_memory_bytes=memory,
        available_disk_bytes=shutil.disk_usage(raw).free,
        minimum_memory_bytes=536870912,
        task_cap_s=4800,
        child_poll_s=30,
        scratch_path=str(raw),
        tools={
            n: os.access(ROOT / ".venv/bin" / n, os.X_OK)
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
    )
    contract = authority(root, raw / "authority")
    failures = list(contract["gate_check_summary"])
    if (
        not resources["private_disk_backed_scratch"]
        or memory < 536870912
        or resources["available_disk_bytes"] < 536870912
        or not all(resources["tools"].values())
    ):
        failures.append(failure(raw, "private_disk_memory_tools", True, resources))
    refs = [freeze(root / name, raw / "custody") for name in [DESIGN, ACTIVE, STAGED]]
    work: Json = dict(
        root=str(root),
        contract=contract,
        failures=failures,
        preconditions=resources,
        refs=refs,
        fixtures={},
        families={},
        historical_failure_reproduction=[],
        current_file_substitution_rows=[],
        historical_model_provenance=[],
        protocol={},
        started_monotonic_ns=start,
    )
    if not failures:
        progress("authenticate_pinned_methods_before")
        protocol = json.loads(checked(ROOT / PROTOCOL, PROTOCOL_PIN).read_bytes())
        checked(ROOT / METHODS, METHODS_PIN)
        for ref in protocol["preserved_protocols"]:
            checked(Path(ref["path"]), ref["sha256"])
        work.update(
            protocol=protocol,
            fixtures=fixtures(raw / "fixtures"),
            historical_failure_reproduction=historical_logs(raw),
        )
        history = json.loads(
            checked(
                ROOT / "results/experiment_8388_v723_contract_methods.json", PRIMARY_PINS[8388]
            ).read_bytes()
        )
        work["global_health"] = [
            dict(r, evidence_scope="historical_global_health_not_current_pass")
            for r in history["validation_receipts"]
            if r.get("scope") == "global"
        ]
        work["historical_model_provenance"] = [
            dict(
                experiment_id=r["experiment_id"],
                MODEL_SPECS=r["MODEL_SPECS"],
                imported_counts=r.get("imported_counts"),
            )
            for r in history["historical_model_provenance"]
        ]
        for ref in [
            reference(ROOT / PROTOCOL),
            reference(ROOT / METHODS),
            *protocol["preserved_protocols"],
            *[r["primary"] for r in work["historical_failure_reproduction"]],
        ]:
            refs.append(freeze(Path(ref["path"]), raw / "custody", ref["sha256"]))
        missing = failure(
            ROOT / "openspec/change-proposals" / HISTORY[722][0],
            "independent_V722_machine_design",
            True,
            None,
        )
        missing["upstream_id"] = "exp8374-contract-methods"
        failures.append(missing)
        progress("authenticate_pinned_methods_after", 14, 0)
    work["code_refs"] = [reference(ROOT / p) for p in OWNED]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Administrative checks qualify faithful classification; a historical failure never becomes a pass."""
    owned = [r for r in receipts if r.get("scope") != "global"]
    qualified = bool(owned) and all(r["passed"] for r in owned)
    families = work["families"]
    kind = (
        "disqualified"
        if not qualified
        else "blocked"
        if work["failures"]
        or any(f.get("status") == "blocked" for f in families.values())
        or not families
        else "circular_positive"
        if all(f["ready"] for f in families.values())
        else "null"
    )
    rows = work["contract"]["contract_rows"] or [
        dict(
            unit_id=f"exp{8402 + i}",
            family=f"exp{8402 + i}",
            arm="current_authority",
            seed=None,
            order=i + 1,
            status="unstarted",
            absolute_metric=None,
            raw_numerator=0,
            raw_denominator=1,
            censored=True,
            excluded=True,
            effective_independent_groups=0,
            missing_reason="complete_current_design_absent",
        )
        for i in range(14)
    ]
    value: Json = dict(
        experiment_id=8402,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261011",
        honest_verdict="complete_" + kind + "_historical_consumer_roots",
        verdict_class=kind,
        gate_check_summary=work["failures"]
        + [
            failure(Path(r["stdout_path"]), r["name"], True, False)
            for r in owned
            if not r["passed"]
        ],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        no_model_load=True,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=14,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r.get("matched") is False for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        sample_size_budget=dict(
            intended_tasks=14, independent_sources=0, historical_tests_are_qualification=True
        ),
        verifier_is_oracle=True,
        exposure_scope="administrative custody and exposed historical development; no independent generalization",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_classification=qualified,
            current_contract=work["contract"]["activated"],
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work.get("phase_spans", []),
        random_seed=7248402,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[reference(raw / "measurement.json"), *work.get("coverage_shards", [])],
        cited_upstream_artifacts=[
            dict(
                producer=r["experiment_id"],
                producer_hash=r["primary"]["sha256"],
                imported_fields=["failed_node_ids", "error_node_ids"],
                evidence=r["log"],
            )
            for r in work["historical_failure_reproduction"]
        ],
        current_contract_ready_score=int(qualified and work["contract"]["activated"]),
        historical_direct_consumers_ready_score=int(
            qualified and families.get("direct", {}).get("ready", False)
        ),
        historical_runtime_consumers_ready_score=int(
            qualified and families.get("runtime", {}).get("ready", False)
        ),
        historical_family_receipts=work.get("family_refs", {}),
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        historical_fixture_hashes=work["fixtures"],
        current_file_substitution_rows=work["current_file_substitution_rows"],
        consumer_node_ids={k: f["consumer_node_ids"] for k, f in families.items()},
        missing_historical_authority=[
            f for f in work["failures"] if f["artifact_field"] == "independent_V722_machine_design"
        ],
        frozen_protocol_path=str(ROOT / PROTOCOL),
        frozen_protocol_sha256=PROTOCOL_PIN,
        scope_reduction_compliance=work["protocol"].get("scope_reduction_compliance", {}),
        historical_failure_reproduction=work["historical_failure_reproduction"],
        historical_families=families,
        future_dependencies=[
            dict(task_id=t["id"], path=t["deliverable"], status="dependency_not_input")
            for t in work["contract"]["tasks"][1:]
        ],
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        invocation_argv=work.get("invocation_argv", []),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        repository_health=[
            *work.get("global_health", []),
            *[r for r in receipts if r.get("scope") == "global"],
        ],
        methodology_note="Full-object oracle custody and independent family classification only. Historic failed assertions remain failed. No inference, generator update or semantic benefit. Missing V722 machine authority stays blocked.",
    )
    value["field_principles"] = {
        k: "Bind this administrative field to sealed operands and this invocation; qualification grants no scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    purposes = {
        "experiment_id task_id milestone run_date publication_output invocation_argv": "Bind the actual invocation and selected current task identity.",
        "rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget future_dependencies": "Account for fourteen administrative objects; tests, repetitions and future dependencies are not independent scientific sources.",
        "inference_substrate inference_substrate_class MODEL_SPECS no_model_load model_invocation_counts historical_model_provenance": "Record zero current LLM loads/generations; cached names and counts retain only historical provenance.",
        "current_contract_ready_score canonical_tasks_sha256": "Require complete current objects, visible table and fixed canonical digest.",
        "historical_direct_consumers_ready_score historical_runtime_consumers_ready_score historical_family_receipts historical_families consumer_node_ids": "Authenticate families independently through identical test identities, assertions, actual exits, logs and terminal seals; a failed family zeros only itself.",
        "historical_fixture_hashes current_file_substitution_rows missing_historical_authority": "Keep original dependencies separate from current rotations; missing V722 machine authority stays absent.",
        "required_checks_passed flagged_adversarial acceptance_gates honest_verdict verdict_class gate_check_summary": "Owned checks qualify custody and classification; external absence blocks, owned failure disqualifies all scores, and historical failed assertions remain failed.",
        "validation_receipts terminal_validation_sidecar_path adversarial_findings execution_manifest_reference owned_coverage_reference repository_health": "Bind bounded checks and genuine coverage; preserve findings and separate historical global health from current owned checks.",
        "preconditions_checked duration_s phase_spans random_seed reproducibility_checksum source_artifact_hashes code_config_hashes raw_shard_hashes work_reference": "Bind resources, clocks, source bytes, code and genuine child shards to replayable evidence.",
        "cited_upstream_artifacts historical_failure_reproduction": "Import failed/error node IDs only from the pinned producer and original hashed log.",
        "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score methodology_note": "Administrative oracle qualification and opened targets grant zero generalization or learning-benefit credit.",
        "frozen_protocol_path frozen_protocol_sha256 scope_reduction_compliance": "Freeze service variants and causal cost tapes; preserve earlier controls and operator-only priorities.",
    }
    value["field_principles"].update(
        {field: purpose for fields, purpose in purposes.items() for field in fields.split()}
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh classification and pinned operands defeat summary tampering even after repaired hashes."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            checked(Path(ref["path"]), ref["sha256"])
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                checked(Path(work[field]["path"]), work[field]["sha256"])
        if work["contract"]["tasks"] and tasks_digest(work["contract"]["tasks"]) != TASKS_PIN:
            return False
        for ref in work["refs"]:
            if ref.get("source_path", ref["path"]) != ref["path"]:
                return False
            if ref["exists"]:
                checked(Path(ref["snapshot_path"]), ref["sha256"])
        with TemporaryDirectory(prefix="exp8402-cold-", dir="/var/tmp") as directory:
            root = Path(directory)
            for name, ref in zip([DESIGN, ACTIVE, STAGED], work["refs"][:3], strict=True):
                if ref["exists"]:
                    destination = root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ref["snapshot_path"], destination)
            current = authority(root, root / "assessment")
            if any(
                current[key] != work["contract"][key]
                for key in ["tasks", "activated", "canonical_tasks_sha256", "contract_rows"]
            ):
                return False
        if work["fixtures"] and not check_fixtures(work["fixtures"]):
            return False
        if work["protocol"] and work["protocol"] != json.loads(
            checked(ROOT / PROTOCOL, PROTOCOL_PIN).read_bytes()
        ):
            return False
        for name, row in work["families"].items():
            if family(name, row["runs"], row["consumer_node_ids"]) != row:
                return False
        for receipt in [
            *value["validation_receipts"],
            *[r for f in work["families"].values() for r in f["runs"]],
        ]:
            for stream in ["stdout", "stderr"]:
                checked(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
        for name, ref in work.get("family_refs", {}).items():
            sealed = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_bytes())
            if (
                sealed["classification"] != work["families"][name]
                or sealed["terminal_hash"] != canonical_hash(sealed["classification"])
                or sealed["assertion_sources"]
                != assertion_sources(work["families"][name]["consumer_node_ids"])
                or sealed["historical_fixture_hashes"] != work["fixtures"]
                or sealed["canonical_tasks_sha256"] != TASKS_PIN
            ):
                return False
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def authenticate_family(primary: Path, name: str) -> Json:
    """Downstream tasks must authenticate the administrative terminal and their separate family seal."""
    from carnot.reporting.primary_publication import read_bound_sidecar

    value = json.loads(primary.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    report = read_bound_sidecar(primary, Path(terminal["publication"]["sidecar_path"]))
    if (
        report["report"]["passed"] is not True
        or not value["required_checks_passed"]
        or value["flagged_adversarial"]
        or not replay(primary)
    ):
        raise ValueError("qualified_administrative_terminal")
    ref = value["historical_family_receipts"][name]
    seal = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_bytes())
    row = seal["classification"]
    if (
        not row["ready"]
        or seal["terminal_hash"] != canonical_hash(row)
        or family(name, row["runs"], row["consumer_node_ids"]) != row
    ):
        raise ValueError("qualified_family_terminal")
    return dict(row)
