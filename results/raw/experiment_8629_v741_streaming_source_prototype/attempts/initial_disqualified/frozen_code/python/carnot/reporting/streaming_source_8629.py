"""REQ-VERIFY-8629; REQ-REPORT-V741-SOURCE: qualify fixtures, never natural truth."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory, mkdtemp
import time
import tracemalloc
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.verify import evidence_features_7980 as legacy
from carnot.verify import source_alignment as alignment
from carnot.verify import streaming_evidence as streaming

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8629_v741_streaming_source_prototype", "exp8629-streaming-source-prototype"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_streaming_evidence_8629.py"
OWNED = [
    "python/carnot/verify/streaming_evidence.py",
    "python/carnot/reporting/streaming_source_8629.py",
    "python/carnot/reporting/streaming_source_runner_8629.py",
    CLI,
]
INPUTS = [
    "python/carnot/verify/source_alignment.py",
    "python/carnot/verify/evidence_features_7980.py",
    "ops/exclusion_manifest.yaml",
]
MODEL_SPECS: list[Json] = []
SEED = 8629


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase and work counts so a supervisor can distinguish work from stalls."""
    print(f"[exp8629] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Keep exact operand identity separate from any interpretation of its contents."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def fixtures() -> list[Json]:
    """Code-defined original fixtures are authority, not a mutable derived manifest."""
    pairs = [
        (b"", b"Yes."),
        (b" \t", b"Yes."),
        (b"Yes.", b""),
        ("Élan ３.\u2003猫?! \nEnd.tail".encode(), "猫 ３?".encode()),
        (b"Same. \nSame. Same. ", b"Same."),
        (b"A. B. C. ", b"Yes. " * 17),
        (b"Birds have 12 eggs. No birds remain. ", b"Not 13. Birds? " * 8),
    ]
    pairs += [
        (b"".join(f"Unit {i}. ".encode() for i in range(n)), b"Unit 4.") for n in (65, 66, 160)
    ]
    pairs += [(b"Same. Same. " + b"".join(f"Unit {i}. ".encode() for i in range(64)), b"Unit 4.")]
    return [
        dict(family_id=f"fixture-{i}", source_bytes=s.hex(), answer_bytes=a.hex())
        for i, (s, a) in enumerate(pairs)
    ]


def eager(row: Json) -> tuple[Json, bytes]:
    """Use original eager parsing and132-feature arithmetic as an independent check."""
    source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
    sentences, units = alignment.sentence_spans(source), alignment.sentence_spans(answer)
    starts, cursor = [], 0
    for sentence in sentences:
        starts.append(cursor)
        cursor += len(sentence)
    windows = list(zip(starts, sentences, strict=True))
    windows += [
        (starts[i], b"".join(sentences[i : i + 3])) for i in range(max(0, len(sentences) - 2))
    ]
    seen: dict[bytes, int] = {}
    records = []
    for i, (start, window) in enumerate(windows):
        key = window.strip()
        fresh = key not in seen
        seen.setdefault(key, len(seen))
        records.append(
            streaming.record_bytes(
                dict(
                    index=i,
                    start=start,
                    end=start + len(window),
                    window_hash=hashlib.sha256(key).hexdigest(),
                    group_id=seen[key],
                    first_occurrence=fresh,
                    answer_unit_ids=list(range(len(units))),
                )
            )
        )
    reason = (
        "answer_units_over_budget"
        if len(units) > 16
        else ("empty_answer" if not units else "empty_source" if not source.strip() else None)
    )
    pairs = [alignment.pair_features(w, units)[-4:] for w in seen] if not reason else []
    values = (
        [
            v
            for i in range(4)
            for v in (sum(p[i] for p in pairs) / len(pairs), max(p[i] for p in pairs))
        ]
        if pairs
        else None
    )
    value = dict(
        family_id=row["family_id"],
        values=values,
        abstention=reason,
        window_count=len(seen),
        source_normalized_hash=legacy.normalized(source),
    )
    return dict(value, feature_hash=canonical_hash(value)), b"".join(records)


def measure(root: Path, raw: Path, date: str) -> Json:
    """Authenticate only CPU operands, then retain all fixture windows and real costs."""
    started = time.monotonic_ns()
    progress("start_preconditions")
    raw.mkdir(parents=True, exist_ok=True)
    scratch = Path(mkdtemp(prefix="exp8629-", dir="/var/tmp"))
    probe = scratch / "probe"
    probe.write_bytes(b"disk")
    probe.unlink()
    mount = max(
        (
            line.split()
            for line in Path("/proc/mounts").read_text().splitlines()
            if scratch.is_relative_to(Path(line.split()[1]))
        ),
        key=lambda r: len(r[1]),
    )
    preconditions = dict(
        memory=Path("/proc/meminfo").read_text().splitlines()[:3],
        scratch=str(scratch),
        scratch_free_bytes=shutil.disk_usage(scratch).free,
        writable_disk_backed_scratch=mount[2] not in {"tmpfs", "ramfs"},
        optional_gpu_required=False,
        optional_board_required=False,
        tools={n: reference(ROOT / ".venv/bin" / n) for n in ("pytest", "ruff", "mypy")},
    )
    gates, sources = [], []
    for name in INPUTS:
        path = root / name
        if path.is_file():
            sources.append(reference(path))
        else:
            gates.append(
                dict(
                    check="required_input",
                    upstream="original_lexical_extractor",
                    path=str(path),
                    sha256=None,
                    field="exists",
                    operator="==",
                    expected=True,
                    observed=None,
                )
            )
    chunks, resources = [], []
    if not gates:
        for i, row in enumerate(fixtures()):
            progress("before_fixture_benchmark", i, len(fixtures()) - i)
            expected, records = eager(row)
            for chunk in (1, 32, 128):
                log = raw / f"{row['family_id']}-chunk-{chunk}.jsonl"
                tracemalloc.start()
                value = streaming.extract(row, log, chunk_size=chunk)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                parity = value["features"] == expected and log.read_bytes() == records
                old = len(records.splitlines()) <= 128
                parity = parity and (not old or value["features"] == legacy.extract(row))
                chunks.append(
                    dict(
                        family_id=row["family_id"],
                        chunk_size=chunk,
                        **value,
                        primitive=reference(log),
                        parity=parity,
                        traced_peak_bytes=peak,
                        old_domain=old,
                        expected_digest=hashlib.sha256(records).hexdigest(),
                    )
                )
            progress("after_fixture_benchmark", i + 1, len(fixtures()) - i - 1)
        for key, reason in [
            ("source_bytes", "source_bytes_over_budget"),
            ("answer_bytes", "answer_bytes_over_budget"),
            ("working_bytes", "working_memory_over_budget"),
            ("seconds", "response_time_over_budget"),
            ("disk_bytes", "dedup_disk_over_budget"),
        ]:
            limits = streaming.Limits(**{key: 0})
            value = streaming.extract(fixtures()[3], scratch / "resource.jsonl", limits=limits)
            resources.append(
                dict(
                    limit=key,
                    expected=reason,
                    observed=value["features"]["abstention"],
                    passed=value["features"]["abstention"] == reason and not value["complete"],
                )
            )
    ended = time.monotonic_ns()
    work = dict(
        raw=str(raw.absolute()),
        run_date=date,
        chunk_rows=chunks,
        resource_rejection_rows=resources,
        source_refs=sources,
        code_refs=[reference(ROOT / p) for p in OWNED],
        gate_check_summary=gates,
        preconditions_checked=preconditions,
        duration_s=(ended - started) / 1e9,
        phase_spans=[
            dict(
                phase="fixture_measurement", started_monotonic_ns=started, ended_monotonic_ns=ended
            )
        ],
        limits=asdict(streaming.Limits()),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_sealed", len(chunks), 33 - len(chunks))
    return work


def build(work: Json, validation: Json, output: Path) -> Json:
    """Keep readiness separate from scientific benefit and preserve failed gates."""
    chunks, receipts = work["chunk_rows"], validation["receipts"]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    parity = len(chunks) == 33 and all(
        r["parity"] and r["complete"] and r["complete_window_digest"] == r["expected_digest"]
        for r in chunks
    )
    memory = all(r["traced_peak_bytes"] <= work["limits"]["working_bytes"] for r in chunks)
    resources = len(work["resource_rejection_rows"]) == 5 and all(
        r["passed"] for r in work["resource_rejection_rows"]
    )
    gates = list(work["gate_check_summary"])
    if not owned:
        gates.append(
            dict(
                check="owned_validation",
                upstream=TASK,
                path=work["raw"],
                sha256=None,
                field="all_required_checks",
                operator="==",
                expected=True,
                observed=False,
            )
        )
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if gates
        else ("null" if parity and memory and resources else "disqualified")
    )
    ready = int(verdict == "null")
    acceptance = dict(
        owned_checks=owned,
        original_inputs=not work["gate_check_summary"],
        exact_parity=parity,
        bounded_memory=memory,
        resource_rejections=resources,
    )
    value = dict(
        experiment_id=8629,
        task_id=TASK,
        milestone="2026.10.741",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v741_streaming_source_prototype",
        verdict_class=verdict,
        streaming_extractor_ready_score=ready,
        branch_gate_values=dict(streaming_extractor_ready_score=ready),
        gate_check_summary=gates,
        inference_substrate="no_model_load",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[],
        target_model=dict(
            kind="none",
            small_head_inference_calls=0,
            numerical_work="fixed lexical arithmetic only",
        ),
        rows=[
            dict(
                family_id=r["family_id"],
                chunk_size=r["chunk_size"],
                status="completed",
                metric=int(r["parity"]),
                missing_reason=r["features"]["abstention"],
            )
            for r in chunks
        ],
        intended_count=33,
        completed_count=len(chunks),
        failed_count=sum(not r["parity"] for r in chunks),
        censored_count=33 - len(chunks),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            deterministic_fixtures=11, chunk_arms=3, natural_questions=0, repeated_measures=True
        ),
        verifier_is_oracle=False,
        exposure_scope="deterministic qualification fixtures only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(
            validation.get("adversarial_findings", {}).get("flagged_count", 0)
        ),
        acceptance_gates=acceptance,
        validation_receipts=receipts,
        adversarial_findings=validation.get("adversarial_findings", {}),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["source_refs"],
        cited_upstream_artifacts=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[r["primitive"] for r in chunks],
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        publication_output=str(output.absolute()),
        chunk_parity_rows=[
            {k: r[k] for k in ("family_id", "chunk_size", "parity")} for r in chunks
        ],
        bounded_memory_rows=[
            {
                k: r[k]
                for k in (
                    "family_id",
                    "chunk_size",
                    "traced_peak_bytes",
                    "reserved_working_bytes",
                    "latency_s",
                )
            }
            for r in chunks
        ],
        window_offset_rows=[
            {
                k: r[k]
                for k in (
                    "family_id",
                    "chunk_size",
                    "primitive",
                    "total_window_count",
                    "complete_window_digest",
                )
            }
            for r in chunks
        ],
        complete_window_digest=canonical_hash([r["complete_window_digest"] for r in chunks]),
        feature_parity_rows=[
            dict(
                family_id=r["family_id"],
                chunk_size=r["chunk_size"],
                old_domain=r["old_domain"],
                features=r["features"],
                exact=r["parity"],
            )
            for r in chunks
        ],
        resource_rejection_rows=work["resource_rejection_rows"],
        resource_limits=work["limits"],
        methodology_note="Exact deterministic lexical parity and complete window custody; no natural coverage, fit, score selection, entailment, calibration or learning claim. Exp8632 owns natural measurement.",
    )
    value["field_principles"] = {
        k: "Invocation-bound fixture qualification; not scientific benefit." for k in value
    }
    value["field_principles"].update(
        bounded_memory_rows="Traced Python peak, conservative reservation and real latency are descriptive; SQLite cache is bounded separately.",
        window_offset_rows="Disk logs retain every original occurrence, full UTF-8 interval and all contributing answer IDs.",
        complete_window_digest="Ordered records must equal complete independent eager enumeration.",
        streaming_extractor_ready_score="Only exact parity, bounded resources and all rejection/validation checks authorize readiness.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute immutable fixture authority so repaired manifests cannot hide drift."""
    progress("cold_replay_start")
    try:
        value = json.loads(path.read_bytes())
        ref = value["primitive_reference"]
        if reference(Path(ref["path"])) != ref:
            raise ValueError("primitive_hash")
        work = json.loads(Path(ref["path"]).read_bytes())
        if work["code_refs"] != [reference(ROOT / p) for p in OWNED]:
            raise ValueError("code_hash")
        for ref in work["source_refs"]:
            if reference(Path(ref["path"])) != ref:
                raise ValueError("source_hash")
        chunks = work["chunk_rows"]
        if not work["gate_check_summary"]:
            roster = [(r["family_id"], c) for r in fixtures() for c in (1, 32, 128)]
            if [(r["family_id"], r["chunk_size"]) for r in chunks] != roster:
                raise ValueError("missing_chunk")
            with TemporaryDirectory(prefix="exp8629-replay-", dir="/var/tmp") as directory:
                for i, row in enumerate(fixtures()):
                    expected, records = eager(row)
                    for result in chunks[i * 3 : i * 3 + 3]:
                        ref = result["primitive"]
                        if (
                            reference(Path(ref["path"])) != ref
                            or Path(ref["path"]).read_bytes() != records
                        ):
                            raise ValueError("window_custody")
                        actual = streaming.extract(
                            row, Path(directory) / "windows.jsonl", chunk_size=result["chunk_size"]
                        )
                        if (
                            not result["parity"]
                            or result["features"] != expected
                            or any(
                                actual[k] != result[k]
                                for k in (
                                    "features",
                                    "complete",
                                    "complete_window_digest",
                                    "total_window_count",
                                    "reserved_working_bytes",
                                )
                            )
                        ):
                            raise ValueError("feature_drift")
                    progress("cold_fixture", i + 1, len(fixtures()) - i - 1)
        validation = dict(
            receipts=value["validation_receipts"],
            adversarial_findings=value["adversarial_findings"],
        )
        if build(work, validation, Path(value["publication_output"])) != value:
            raise ValueError("reduction_drift")
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("replay_rejected_" + str(error)[:100])
        return False
    progress("replay_passed")
    return True
