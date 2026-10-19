"""REQ-REPORT-8358: separate replay qualification from scientific promotion.

Workers use existing readers with frozen paths. A reproducible disqualification
keeps its original verdict; unavailable history remains a named external block.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v720_replay_closure as c
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v719_capstone_evidence import memory

Json = dict[str, Any]
ROOT = c.ROOT
NAME = "experiment_8358_v720_terminal_replay_qualification"
TASK, MILESTONE = "exp8358-terminal-replay-qualification", "2026.10.720"
TASK_PIN = "sha256:38871044b72b2c89ca9b6081508651268cf06f82dae67cf54d47889775b020b1"
CAPSTONE_PIN = "sha256:510acd6ffd49f51f32711a098cd6c0035a020c961942deedd23f880868570c77"
CLI, TEST = "scripts/experiments/" + NAME + ".py", "tests/python/test_v720_terminal_replay_8358.py"
OWNED = [
    "python/carnot/reporting/v720_replay_closure.py",
    "python/carnot/reporting/v720_terminal_replay.py",
    "python/carnot/reporting/v720_replay_execution.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
progress = c.progress
PRODUCERS = {
    8342: "experiment_8342_v719_arc_supervisor_frontier",
    8344: "experiment_8344_v719_gatemate_change_ledger",
    8355: "experiment_8355_v720_arc_supervisor_frontier",
    8357: "experiment_8357_v720_gatemate_change_ledger",
}


def gate(path: str, field: str, expected: Any, observed: Any, digest: str | None = None) -> Json:
    """A missing operand keeps None so a reader cannot mistake absence for zero."""
    return dict(
        upstream=Path(path).stem,
        path=path,
        sha256=digest,
        artifact_field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def private_producer(output: Path, failed: bool) -> Json:
    """Publish a real private mechanical control with an intentionally failed child.

    The child prints its intended exit before exiting. Replay independently
    joins that log with the exit receipt, rather than trusting a claimed verdict.
    """
    if not str(output.absolute()).startswith("/tmp/"):
        raise ValueError("private_control_requires_tmp")
    raw = output.parent / "raw" / output.stem
    code = 7 if failed else 0
    receipt = child(
        "private_outcome",
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "-c",
            f"print('private_exit={code}', flush=True); raise SystemExit({code})",
        ],
        raw / "logs",
        deadline=5,
    )
    primitive = raw / "primitive.json"
    atomic_json(primitive, dict(receipt=receipt))
    kind = "disqualified" if failed else "circular_positive"
    value = dict(
        experiment_id=8358,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        honest_verdict="complete_" + kind + "_private_control",
        verdict_class=kind,
        primitive_reference=reference(primitive),
        source_artifact_hashes=[
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ],
        code_config_hashes=[],
        raw_shard_hashes=[],
        terminal_validation_sidecar_path=str(raw / "terminal.json"),
    )
    publication = publish_primary(
        output, value, lambda p: dict(passed=True, private_mechanical_control=True)
    )
    atomic_json(Path(value["terminal_validation_sidecar_path"]), dict(publication=publication))
    return value


def worker(request: Path, output: Path) -> int:
    """Authenticate before executing a known reader; report child RSS even on errors."""
    progress("worker_before")
    before, accesses = memory(), dict(count=0)
    result: Json = dict(
        passed=False, replay_passed=False, first_mismatch=None, status="disqualified"
    )
    args: Json = {}
    try:
        args = json.loads(request.read_bytes())
        if args["authority"] != TASK_PIN:
            raise ValueError("task_authority_drift")
        bundle = args["bundle"]
        value = c.verify(bundle)
        if args["disposition"] != value["verdict_class"]:
            raise ValueError("disposition_drift")
        result.update(
            experiment_id=value["experiment_id"],
            verdict_class=value["verdict_class"],
            honest_verdict=value["honest_verdict"],
        )
        missing = [r for r in bundle["rows"] if not r["available"]]
        if missing:
            result.update(
                status="authenticated-history-unavailable",
                first_mismatch=gate(
                    missing[0]["original_path"],
                    "source_closure.sha256",
                    missing[0]["expected_sha256"],
                    missing[0]["observed_sha256"],
                ),
            )
            result["passed"] = value["experiment_id"] in {8342, 8344}
        else:
            identity = value["experiment_id"]
            if identity == 8358:
                saved = next(
                    r["reference"]
                    for r in bundle["rows"]
                    if r["original_path"] == value["primitive_reference"]["path"]
                )
                primitive = json.loads(checked(saved).read_bytes())
                receipt = primitive["receipt"]
                log = next(
                    r["reference"]
                    for r in bundle["rows"]
                    if r["original_path"] == receipt["stdout_path"]
                )
                marker = checked(log).read_text().strip()
                expected = "disqualified" if receipt["exit_code"] != 0 else "circular_positive"
                replayed = (
                    marker == f"private_exit={receipt['exit_code']}"
                    and value["verdict_class"] == expected
                )
            else:
                with TemporaryDirectory(prefix="exp8358-reader-") as directory:
                    adapter = c.adapter(bundle, Path(directory) / "source")
                    with c.frozen_reads(bundle, Path(directory) / "scratch", accesses):
                        answer = adapter.replay(
                            value
                            if identity in {8342, 8355}
                            else Path(bundle["rows"][0]["reference"]["path"])
                        )
                        replayed = bool(
                            answer.get("passed") if isinstance(answer, dict) else answer
                        )
            if not replayed:
                raise ValueError("primitive_reduction_drift")
            result.update(passed=True, replay_passed=True, status="recorded_disposition_replayed")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        StopIteration,
        ImportError,
        AttributeError,
        SyntaxError,
    ) as error:
        result.update(
            error=str(error),
            first_mismatch=gate(
                str(args.get("bundle", {}).get("original_path", request)),
                "authenticated_primitive_reduction",
                True,
                str(error),
            ),
        )
    after = memory()
    result.update(
        memory=dict(
            before=before,
            after=after,
            growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
            passed=after["peak_rss_mb"] - before["peak_rss_mb"] <= 500,
        ),
        mutable_authority_access_count=accesses["count"],
    )
    result["passed"] = result["passed"] and result["memory"]["passed"] and accesses["count"] == 0
    atomic_json(output, result)
    progress("worker_after", 1, 0)
    return int(not result["passed"])


def invoke(request: Json, raw: Path, name: str, *, expected: int = 0) -> tuple[Json, Json]:
    """Keep exact argv and both streams from bounded real child CLIs."""
    source, output = raw / (name + "_request.json"), raw / (name + "_result.json")
    atomic_json(source, request)
    receipt = child(
        name,
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--worker-request",
            str(source),
            "--worker-output",
            str(output),
        ],
        raw / "logs",
        expected=expected,
        deadline=180,
        heartbeat=20,
    )
    result = (
        json.loads(output.read_bytes())
        if output.is_file()
        else dict(
            passed=False,
            replay_passed=False,
            error="owned_child_no_output",
            status="disqualified",
            first_mismatch=gate(
                str(request.get("bundle", {}).get("original_path", source)),
                "child_result.exists",
                True,
                None,
            ),
            mutable_authority_access_count=0,
        )
    )
    receipt["result_reference"] = reference(output) if output.is_file() else None
    return result, receipt


def measure(raw: Path, *, private: bool = False) -> Json:
    """Freeze natural operands once; private controls never stand in for producers."""
    began, before = time.monotonic_ns(), memory()
    bundles, rows, receipts, diagnostics, imported = [], [], [], [], []
    if private:
        output = raw / "private" / (NAME + ".json")
        private_producer(output, False)
        paths = [(output, sha256_file(output))]
    else:
        paths = [
            (
                ROOT / "results" / (PRODUCERS.get(i, f"experiment_{i}_missing") + ".json"),
                pin,
            )
            for i, pin in c.PINS.items()
        ]
    for i, (path, pin) in enumerate(paths):
        progress("producer_before", i, len(paths) - i)
        index: dict[str, list[Path]] = {}
        for source in (path.parent / "raw" / path.stem).rglob("*"):
            match = re.search(r"[a-f0-9]{64}", source.name)
            if match and source.is_file():
                index.setdefault("sha256:" + match[0], []).append(source)
        try:
            bundle = c.capture(path, pin, raw / "closures" / str(i), index)
            value = c.verify(bundle)
            if value.get("historical_model_provenance"):
                imported.append(
                    dict(
                        experiment_id=value["experiment_id"],
                        source_sha256=pin,
                        provenance=value["historical_model_provenance"],
                        current_invocations=0,
                    )
                )
            result, receipt = invoke(
                dict(bundle=bundle, authority=TASK_PIN, disposition=value["verdict_class"]),
                raw / "workers" / str(i),
                "producer_replay",
            )
            bundles.append(bundle)
            receipts.append(receipt)
            rows.append(
                dict(
                    result,
                    source_reference=bundle["rows"][0]["reference"],
                    historical_disposition=value["verdict_class"],
                    task_id=value["task_id"],
                )
            )
            for operand in bundle["rows"]:
                if not operand["available"]:
                    diagnostics.append(
                        gate(
                            operand["original_path"],
                            "source_closure.sha256",
                            operand["expected_sha256"],
                            operand["observed_sha256"],
                            operand["observed_sha256"],
                        )
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing = not path.is_file()
            rows.append(
                dict(
                    experiment_id=int(path.name.split("_")[1]),
                    passed=False,
                    replay_passed=False,
                    status="external_blocked" if missing else "disqualified",
                    historical_disposition=None,
                    first_mismatch=gate(
                        str(path),
                        "authenticated_terminal_producer",
                        pin,
                        None if missing else str(error),
                    ),
                    mutable_authority_access_count=0,
                )
            )
        progress("producer_after", i + 1, len(paths) - i - 1)
    historical: list[Json] = []
    source_refs: list[Json] = []
    if not private:
        cap = ROOT / "results/experiment_8345_v719_capstone.json"
        if sha256_file(cap) != CAPSTONE_PIN:
            raise ValueError("capstone_anchor_drift")
        source_refs.append(reference(cap))
        previous = json.loads(cap.read_bytes())
        for receipt in previous["branch_replay_receipts"]:
            if receipt["name"] in {"branch_8342_replay", "branch_8344_replay"}:
                for stream in ["stdout", "stderr"]:
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
                primary = Path(receipt["argv"][-1])
                value = json.loads(primary.read_bytes())
                bundle = next(
                    (b for b in bundles if b["experiment_id"] == value["experiment_id"]),
                    dict(rows=[]),
                )
                drift = next(
                    (r for r in bundle["rows"][3:] if r["observed_sha256"] != r["expected_sha256"]),
                    None,
                )
                stdout = Path(receipt["stdout_path"]).read_text()
                named = re.search(r'"error": "hash:([^"\n]+)', stdout)
                if named:
                    drift = next(
                        (r for r in bundle["rows"] if r["original_path"] == named[1]), drift
                    )
                historical.append(
                    dict(
                        experiment_id=value["experiment_id"],
                        verdict_class=value["verdict_class"],
                        honest_verdict=value["honest_verdict"],
                        failed_receipt=receipt,
                        failed_argv_source_references=[
                            r for r in bundle["rows"] if r["original_path"] in receipt["argv"]
                        ],
                        failed_primary_reference=reference(primary),
                        failed_stdout=stdout,
                        first_mismatch=drift,
                        gate_check_summary=value.get("gate_check_summary", []),
                        diagnosis_scope="first currently mismatched authenticated operand; original failure text preserved",
                        science_promoted=False,
                    )
                )
    return dict(
        rows=rows,
        bundles=bundles,
        branch_replay_receipts=receipts,
        historical_dispositions=historical,
        diagnostics=diagnostics,
        source_refs=source_refs,
        started_monotonic_ns=began,
        ended_monotonic_ns=time.monotonic_ns(),
        parent_before=before,
        parent_after=memory(),
        private_control=private,
        historical_model_provenance=imported,
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Only qualified mechanics earn readiness; absent history keeps a blocked verdict."""
    rows = work["rows"]
    owned_receipts = [r for r in receipts if r.get("scope", "owned") == "owned"]
    owned = (
        bool(owned_receipts)
        and all(r["passed"] for r in owned_receipts)
        and all(
            r["status"] != "disqualified" and (r["passed"] or r["status"] == "external_blocked")
            for r in rows
        )
    )
    blocked = bool(work["diagnostics"]) or any(not r["replay_passed"] for r in rows)
    memory_ok = work["parent_after"]["peak_rss_mb"] - work["parent_before"][
        "peak_rss_mb"
    ] <= 500 and all(r.get("memory", {}).get("passed", True) for r in rows)
    owned = owned and memory_ok
    kind = "disqualified" if not owned else "blocked" if blocked else "circular_positive"
    primitive = raw / "measurement.json"
    atomic_json(primitive, work)
    refs = [*work["source_refs"], *work.get("authority_refs", [])]
    codes = work.get("code_refs", [])
    value: Json = dict(
        experiment_id=8358,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        honest_verdict="complete_" + kind + "_terminal_replay_qualification",
        verdict_class=kind,
        gate_check_summary=work["diagnostics"]
        + [r["first_mismatch"] for r in rows if r.get("first_mismatch")],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_model_calls=0,
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(producer_slots=len(rows), independent_scientific_sources=0),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned and all(r["passed"] for r in receipts),
        flagged_adversarial=False,
        acceptance_gates=dict(owned_adapters=owned, closures=not blocked, memory=memory_ok),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work.get("preconditions_checked", []),
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="producer_closure_replay",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        ],
        random_seed=7208358,
        source_artifact_hashes=refs,
        code_config_hashes=codes,
        raw_shard_hashes=[reference(primitive)],
        cited_upstream_artifacts=[
            dict(r, fields_imported=["branch_replay_receipts", "historical_dispositions"])
            for r in work["source_refs"]
        ]
        + [
            dict(
                b["rows"][0]["reference"],
                experiment_id=b["experiment_id"],
                fields_imported=[
                    "honest_verdict",
                    "verdict_class",
                    "source_artifact_hashes",
                    "code_config_hashes",
                    "raw_shard_hashes",
                ],
            )
            for b in work["bundles"]
        ],
        branch_replay_ready_score=int(owned and not blocked),
        branch_replay_receipts=work["branch_replay_receipts"],
        first_mismatch=next((r["first_mismatch"] for r in rows if r.get("first_mismatch")), None),
        source_closure_manifest=work["bundles"],
        historical_dispositions=work["historical_dispositions"],
        parent_child_memory_rows=[
            dict(role="parent", before=work["parent_before"], after=work["parent_after"]),
            *[dict(role="child", **r["memory"]) for r in rows if r.get("memory")],
        ],
        mutable_authority_access_count=sum(r["mutable_authority_access_count"] for r in rows),
        work_reference=reference(primitive),
        publication_output=str(output.absolute()),
        authority_task_sha256=TASK_PIN,
        private_control=work["private_control"],
        repository_suite_attempt=work.get("repository_suite_attempt"),
    )
    value["field_principles"] = {
        k: "Authenticate exact invocation and primitive bytes; mechanical replay preserves failed dispositions and grants no independent science credit."
        for k in value
    }
    value["field_principles"]["field_principles"] = (
        "Explain every field so a reader can assess the claim's limit."
    )
    value["field_principles"]["reproducibility_checksum"] = (
        "Bind every emitted claim to this immutable reconstruction."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold children rejoin primitive meaning; repaired aggregate hashes do not suffice."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(checked(value["work_reference"]).read_bytes())
        for ref in [*value["source_artifact_hashes"], *value["code_config_hashes"]]:
            checked(ref)
        if value["authority_task_sha256"] != TASK_PIN:
            return False
        if work["frozen_validation_receipts"] != value["validation_receipts"]:
            return False
        for recorded in value["validation_receipts"] + work["branch_replay_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in recorded:
                    checked(
                        dict(path=recorded[stream + "_path"], sha256=recorded[stream + "_sha256"])
                    )
        if not work["private_control"]:
            from carnot.reporting import v720_frozen_input_contract as contract

            with TemporaryDirectory(prefix="exp8358-authority-") as directory:
                frozen = Path(directory)
                for ref in work["authority_refs"]:
                    target = frozen / Path(ref["original_path"]).relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(checked(ref).read_bytes())
                actual_authority = contract.authority(frozen, frozen / "assessment")
                task = next(t for t in actual_authority["tasks"] if t["id"] == TASK)
                if (
                    canonical_hash(task) != TASK_PIN
                    or dict(activated=actual_authority["activated"], task=task) != work["authority"]
                ):
                    return False
        with TemporaryDirectory(prefix="exp8358-cold-") as directory:
            for bundle, row in zip(
                work["bundles"], [r for r in work["rows"] if r.get("source_reference")], strict=True
            ):
                original = c.verify(bundle)
                actual, receipt = invoke(
                    dict(bundle=bundle, authority=TASK_PIN, disposition=original["verdict_class"]),
                    Path(directory) / str(bundle["experiment_id"]),
                    "cold_producer",
                    expected=int(not row["passed"]),
                )
                if not receipt["passed"] or {k: v for k, v in actual.items() if k != "memory"} != {
                    k: v
                    for k, v in row.items()
                    if k not in {"memory", "source_reference", "historical_disposition", "task_id"}
                }:
                    return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False
