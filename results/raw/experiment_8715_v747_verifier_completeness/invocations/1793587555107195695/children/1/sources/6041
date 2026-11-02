"""REQ-VERIFY-8153: capture fixed source pairs before any evaluator access.

The qualified worker owns CUDA and its process. This adapter changes only the
frozen prompt schedule and evidence parser; it makes no learning benefit claim.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot import experiment_7969_v691_qwen_calibration_capture as legacy
from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import evidence_protocol_8124 as protocol
from carnot.verify import source_method_custody_8151 as source
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
ROOT = source.ROOT
NAME = "experiment_8153_v705_fit_evidence_capture"
TASK = "exp8153-fit-evidence-capture"
MODULE = "python/carnot/verify/fit_evidence_capture_8153.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_fit_evidence_capture_8153.py"
OWNED = [MODULE, CLI]
ROLES = dict(fit=128, tune=64)
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
UPSTREAM = "results/experiment_8151_v705_source_method_custody.json"
PIN = "sha256:29d912b64552af30b1742e8c80cdf93585cdf9e81520986215bbee3317af351f"
RUN_DATE = "20261005"
CONFIG = dict(
    seed=70553,
    main_calls=384,
    diagnostic_calls=48,
    max_tokens=128,
    input_tokens=6000,
    load_timeout_s=300,
    call_timeout_s=120,
    latest_launch_s=3000,
    closure_s=3120,
    checkpoint_source_slots=16,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual counters so a waiting child cannot look like finished work."""
    print(f"[exp8153] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(plan: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """Keep the exact operand; an unavailable upstream is a terminal block."""
    plan["checks"].append(
        dict(
            check=field,
            upstream=UPSTREAM,
            path=str(path),
            hash=observed if field.endswith("sha256") else None,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=expected == observed,
        )
    )


def reference(path: Path) -> Json:
    """Bind raw files by bytes rather than relying on a mutable filename."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def freeze(rows: list[Json]) -> list[Json]:
    """Preserve original order and make diagnostics from fixed first16 fit slots."""
    selected = [r for r in rows if r["role"] in ROLES]
    if Counter(r["role"] for r in selected) != {k: 2 * v for k, v in ROLES.items()}:
        raise ValueError("role_count")
    slots: list[Json] = []
    clusters: set[str] = set()
    for i in range(0, 384, 2):
        pair = selected[i : i + 2]
        if (
            len({r["unit_id"] for r in pair}) != 1
            or len({r["source_cluster_id"] for r in pair}) != 1
            or pair[0]["source_cluster_id"] in clusters
            or {r["arm"] for r in pair} != {"holistic", "source_span"}
            or [r["order"] for r in pair] != [0, 1]
        ):
            raise ValueError("pair_identity")
        clusters.add(pair[0]["source_cluster_id"])
        for row in pair:
            source_text, answer = (
                bytes.fromhex(row[k]).decode() for k in ("source_bytes", "answer_bytes")
            )
            expected = protocol.protocol()["prompt_prefixes"][row["arm"]] + json.dumps(
                dict(source=source_text, answer=answer), ensure_ascii=False
            )
            if row["prompt"] != expected or row["prompt_sha256"] != canonical_hash(expected):
                raise ValueError("prompt_identity")
            slots.append(
                dict(
                    row,
                    slot=i // 2 + 1 - (128 if row["role"] == "tune" else 0),
                    call_id=f"{row['unit_id']}:{row['arm']}",
                    condition="original",
                    human_target=None,
                )
            )
    first = [next(r for r in slots[i : i + 2] if r["arm"] == "holistic") for i in range(0, 32, 2)]
    for i, original in enumerate(first):
        for condition in ("holistic_duplicate", "source_removed", "source_mismatched"):
            row = deepcopy(original)
            donor = first[(i + 1) % 16]
            source_bytes = (
                b""
                if condition == "source_removed"
                else bytes.fromhex(donor["source_bytes"])
                if condition == "source_mismatched"
                else bytes.fromhex(original["source_bytes"])
            )
            prompt = protocol.protocol()["prompt_prefixes"]["holistic"] + json.dumps(
                dict(
                    source=source_bytes.decode(), answer=bytes.fromhex(row["answer_bytes"]).decode()
                ),
                ensure_ascii=False,
            )
            row.update(
                condition=condition,
                call_id=row["unit_id"] + ":" + condition,
                source_bytes=source_bytes.hex(),
                prompt=prompt,
                prompt_sha256=canonical_hash(prompt),
                donor_unit_id=donor["unit_id"] if condition == "source_mismatched" else None,
            )
            slots.append(row)
    for row in slots:
        row["request"] = dict(
            model=MODEL_SPECS[0],
            messages=[dict(role="user", content=row["prompt"])],
            temperature=0,
            top_p=1,
            seed=CONFIG["seed"],
            max_tokens=128,
            cache_prompt=False,
            chat_template_kwargs=dict(enable_thinking=False),
        )
    return slots


def inputs(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Authenticate public custody before freezing any model-visible request."""
    plan: Json = dict(checks=[], refs=[], slots=[], protocol={}, manifests={})
    path = root / UPSTREAM
    progress("before_upstream_authentication")
    gate(plan, path, "upstream_exists", True, path.is_file())
    if path.is_file():
        gate(
            plan, path, "upstream_sha256", sha256_file(path) if fixture else PIN, sha256_file(path)
        )
        try:
            value = json.loads(path.read_text())
            for field, expected in [
                ("source_protocol_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                gate(plan, path, field, expected, value.get(field))
            if not fixture:
                publication = json.loads(
                    Path(value["terminal_validation_sidecar_path"]).read_text()
                )["publication"]
                gate(
                    plan,
                    path,
                    "terminal_passed",
                    True,
                    read_bound_sidecar(path, Path(publication["sidecar_path"]))["report"]["passed"],
                )
                plan["refs"].extend(reference(Path(publication[k])) for k in ("sidecar_path",))
            refs = [
                value["capture_manifest"],
                value["method_freeze"],
                *value.get("pinned_method_paths", []),
                *[value["source_role_manifests"][role] for role in ROLES],
            ]
            for ref in refs:
                p = Path(ref["path"])
                gate(
                    plan,
                    p,
                    "source_bytes_sha256",
                    ref["sha256"],
                    sha256_file(p) if p.is_file() else None,
                )
            plan["refs"].extend([reference(path), *refs])
            if all(r["passed"] for r in plan["checks"]):
                plan["slots"] = freeze(
                    json.loads(Path(value["capture_manifest"]["path"]).read_text())["rows"]
                )
                plan["protocol"] = value["expected_runtime_identity"]
                plan["manifests"] = value["source_role_manifests"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            gate(plan, path, "authenticated_source_inputs", True, str(error))
    plan["capture_identity"] = canonical_hash([plan["slots"], CONFIG])
    atomic_json(raw / "input_plan.json", plan)
    progress("after_upstream_authentication", len(plan["slots"]), 432 - len(plan["slots"]))
    return plan


def capture(
    slots: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    ledger: Ledger,
    started: float | None = None,
    blocked_reason: str | None = None,
) -> list[Json]:
    """Visit once; exclude whole pairs and cancel the worker after an uncertain call.

    Starting a call is durable before transport. A timeout stops future launches
    so an old server request cannot overlap with a later source judgment.
    """
    began = time.monotonic() if started is None else started
    launch_cutoff = min(
        CONFIG["latest_launch_s"], CONFIG["closure_s"] - CONFIG["call_timeout_s"] - 60
    )
    rows: list[Json] = []
    masks: Json = {}
    cancelled = False
    for i, slot in enumerate(slots):
        progress("before_slot", i, len(slots) - i)
        row = dict(
            slot,
            capture_identity=identity,
            started=False,
            status="censored",
            raw_response={},
            input_tokens=None,
            output_tokens=None,
            duration_s=0.0,
            exclusion_reason="launch_cutoff",
            transcript="",
            probability=None,
        )
        if blocked_reason:
            row.update(status="excluded", exclusion_reason=blocked_reason)
        elif not cancelled and time.monotonic() - began < launch_cutoff:
            try:
                progress("before_token_count", i, len(slots) - i)
                if slot["condition"] == "original" and i % 2 == 0:
                    sizes = [
                        runtime.count(json.dumps(r["request"]["messages"], ensure_ascii=False))
                        for r in slots[i : i + 2]
                    ]
                    masks[slot["unit_id"]] = source.whole_source_mask(sizes)
                    masks[slot["unit_id"]]["sizes"] = sizes
                mask = masks[slot["unit_id"]]
                if mask["status"] == "excluded":
                    row.update(status="excluded", exclusion_reason="whole_source_overlength")
                else:
                    row["input_tokens"] = (
                        mask["sizes"][i % 2]
                        if slot["condition"] == "original"
                        else runtime.count(
                            json.dumps(slot["request"]["messages"], ensure_ascii=False)
                        )
                    )
                    row.update(
                        status="admitted" if row["input_tokens"] <= 6000 else "excluded",
                        exclusion_reason=None
                        if row["input_tokens"] <= 6000
                        else "whole_source_overlength",
                    )
                progress("after_token_count", i, len(slots) - i)
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
                row.update(status="failed", exclusion_reason="tokenizer:" + str(error))
            if row["status"] == "admitted" and time.monotonic() - began < launch_cutoff:
                call_started = time.monotonic()
                row.update(started=True, status="running")
                atomic_json(raw / f"call-{i:03d}.json", row)
                ledger.start("generation", row["call_id"], row["request"])
                progress("before_generation", i, len(slots) - i)
                try:
                    row["raw_response"] = runtime.generate(row["request"])
                    row["status"] = "generated"
                except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                    row.update(status="failed", exclusion_reason=f"{type(error).__name__}:{error}")
                    if isinstance(error, TimeoutError):
                        progress("before_cancel_child", i, len(slots) - i)
                        runtime.close()
                        cancelled = True
                        progress("after_cancel_child", i, len(slots) - i)
                row["duration_s"] = time.monotonic() - call_started
                ledger.finish(
                    row["call_id"],
                    "completed"
                    if row["status"] == "generated"
                    else "cancelled"
                    if cancelled
                    else "failed",
                    row["raw_response"],
                )
                usage = row["raw_response"].get("usage", {})
                row["output_tokens"] = usage.get("completion_tokens")
                ledger.rows[-1].update(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
                ledger.save()
                progress("after_generation", i + 1, len(slots) - i - 1)
        if row["status"] == "admitted":
            row.update(status="censored", exclusion_reason="launch_cutoff")
        response = row["raw_response"]
        if response:
            row["transcript"] = (
                response.get("choices", [{}])[0].get("message", {}).get("content") or ""
            )
        parsed = protocol.parse(row["transcript"], bytes.fromhex(row["source_bytes"]), row["arm"])
        row.update(
            parsed=parsed,
            probability=parsed["p_hallucination"],
            numerator=int(parsed["status"] == "parsed"),
            denominator=1,
            metric="finite_probability",
        )
        if row["status"] == "generated":
            usage = response.get("usage", {})
            qualified = (
                type(usage.get("prompt_tokens")) is int
                and 0 <= usage["prompt_tokens"] <= 6000
                and type(usage.get("completion_tokens")) is int
                and 0 <= usage["completion_tokens"] <= 128
            )
            row.update(
                status="completed" if row["numerator"] and qualified else "failed",
                exclusion_reason=None
                if row["numerator"] and qualified
                else "invalid_probability_or_usage",
            )
        atomic_json(raw / f"call-{i:03d}.json", row)
        rows.append(row)
        if (i < 384 and (i + 1) % 32 == 0) or i + 1 == len(slots):
            atomic_json(
                raw / "checkpoint.json",
                dict(completed_slots=i + 1, pending_slots=len(slots) - i - 1, rows=rows),
            )
            progress("checkpoint", i + 1, len(slots) - i - 1)
    return rows


def reduce(calls: list[Json]) -> Json:
    """Reparse primitives and count pairs rather than multiplying source units."""
    groups: Json = {}
    quotes, interventions = [], []
    for row in calls:
        if (
            row["request"]["messages"] != [dict(role="user", content=row["prompt"])]
            or canonical_hash(row["prompt"]) != row["prompt_sha256"]
            or row["request"]["max_tokens"] != 128
            or row["request"]["model"] != MODEL_SPECS[0]
        ):
            raise ValueError("request_identity")
        parsed = protocol.parse(row["transcript"], bytes.fromhex(row["source_bytes"]), row["arm"])
        if (
            parsed != row["parsed"]
            or row["probability"] != parsed["p_hallucination"]
            or row["human_target"] is not None
        ):
            raise ValueError("parse_or_target_drift")
        if row["condition"] == "original":
            groups.setdefault(row["unit_id"], {})[row["arm"]] = row
            if row["arm"] == "source_span":
                quotes.append(
                    dict(
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        role=row["role"],
                        **parsed,
                    )
                )
    rows: list[Json] = []
    for pair in groups.values():
        a, b = pair["holistic"], pair["source_span"]
        statuses = [a["status"], b["status"]]
        status = (
            "completed"
            if statuses == ["completed", "completed"]
            else "excluded"
            if "excluded" in statuses
            else "failed"
            if "failed" in statuses
            else "censored"
        )
        rows.append(
            dict(
                unit_id=a["unit_id"],
                source_cluster_id=a["source_cluster_id"],
                role=a["role"],
                arm="paired_evidence",
                condition="complete_original_source",
                metric="paired_finite_probability",
                numerator=int(status == "completed"),
                denominator=1,
                status=status,
                exclusion_reason=None
                if status == "completed"
                else a["exclusion_reason"] or b["exclusion_reason"],
                holistic_probability=a["probability"],
                span_probability=b["probability"],
            )
        )
    for row in calls:
        if row["condition"] != "original":
            original = groups[row["unit_id"]]["holistic"]
            valid = row["status"] == original["status"] == "completed"
            interventions.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=row["condition"],
                    condition="fixed_first16_fit",
                    metric="absolute_probability_difference",
                    numerator=abs(row["probability"] - original["probability"]) if valid else 0,
                    denominator=1,
                    status="completed"
                    if valid
                    else row["status"]
                    if row["status"] != "completed"
                    else original["status"],
                    exclusion_reason=None if valid else "unavailable_pair",
                    human_target=None,
                )
            )
    if not calls:
        rows = [
            dict(
                unit_id=f"{role}-missing-{i}",
                source_cluster_id=f"{role}-missing-{i}",
                role=role,
                arm="paired_evidence",
                condition="missing_source_slot",
                metric="paired_finite_probability",
                numerator=0,
                denominator=1,
                status="excluded",
                exclusion_reason="upstream_custody_unavailable",
                holistic_probability=None,
                span_probability=None,
            )
            for role, n in ROLES.items()
            for i in range(n)
        ]
    counts = Counter(r["status"] for r in rows)
    support = {
        role: sum(r["role"] == role and r["status"] == "completed" for r in rows) for role in ROLES
    }
    return dict(
        rows=rows,
        source_pair_rows=rows,
        quote_validity_rows=quotes,
        source_intervention_rows=interventions,
        intended_count=192,
        eligible_count=sum(r["status"] != "excluded" for r in rows),
        independent_count=len(groups),
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=counts["failed"],
        pair_support=support,
        transport_ready=int(support["fit"] >= 96 and support["tune"] >= 48),
    )


def trainability(rows: list[Json], labels: Json) -> Json:
    """Class support affects fitting alone and never chooses replacement sources."""
    support = {}
    for role in ROLES:
        counts = Counter(
            labels.get(r["unit_id"])
            for r in rows
            if r["role"] == role and r["status"] == "completed"
        )
        support[role] = {str(k): counts[k] for k in (0, 1)}
    return dict(
        class_support=support,
        fit_trainable_score=int(all(min(c.values()) >= 12 for c in support.values())),
    )


class FixtureRuntime:
    """Scripted transport is circular evidence with no generator invocation credit."""

    def count(self, text: str) -> int:
        return 3

    def generate(self, request: Json) -> Json:
        return dict(
            choices=[dict(message=dict(content='{"p_hallucination":0.2}'), finish_reason="stop")],
            usage=dict(prompt_tokens=3, completion_tokens=9),
        )


def live(plan: Json, raw: Path, scratch: Path) -> Json:
    """Reuse the qualified lease, CUDA receipts and child cleanup with one load."""
    ledger = Ledger(raw / "ledger.json")
    runtime_class = legacy.QwenRuntime

    class Recorded(runtime_class):  # type: ignore[misc,valid-type]
        def load(self) -> Json:
            """Record failed loads and reject served-template drift before capture."""
            ledger.start("model_load", "owned-model-load", dict(model=str(self.model)))
            progress("before_model_load", 0, 432)
            try:
                result: Json = legacy.bounded(super().load, 300)
                if (
                    canonical_hash(result["props"]["chat_template"])
                    != plan["protocol"]["chat_template_sha256"]
                ):
                    raise ValueError("served_template_drift")
            except (OSError, RuntimeError, TimeoutError, ValueError):
                ledger.finish("owned-model-load", "failed", {})
                progress("after_model_load_failed", 0, 432)
                raise
            ledger.finish("owned-model-load", "completed", result)
            progress("after_model_load", 0, 432)
            return result

    adapter = SimpleNamespace(
        freeze=lambda _: plan["slots"],
        capture=lambda frozen, runtime, path, identity, *, started: capture(
            frozen,
            runtime,
            path,
            identity,
            ledger=ledger,
            started=plan.get("capture_started_monotonic", started),
        ),
    )
    with (
        patch.object(legacy, "TASK", TASK),
        patch.object(legacy, "QwenRuntime", Recorded),
        patch.object(legacy, "capture", adapter),
        patch.object(legacy, "load_public", lambda _: {}),
        patch.object(
            legacy,
            "progress",
            lambda phase, started, units=0: progress(
                phase,
                sum(
                    r["operation"] == "generation" and r["status"] != "running" for r in ledger.rows
                ),
                432
                - sum(
                    r["operation"] == "generation" and r["status"] != "running" for r in ledger.rows
                ),
            ),
        ),
    ):
        result = dict(
            legacy.live_capture(dict(plan, public_role_manifests=plan["manifests"]), raw, scratch)
        )
    for check in result["checks"]:
        check.update(
            check=check["upstream_id"] + "_" + check["field"],
            upstream=check["upstream_id"],
            artifact_field=check["field"],
        )
    result["ledger"] = ledger.rows
    return result


def runtime_preflight(plan: Json, raw: Path, private: Path) -> None:
    """Freeze cache, embedded tokenizer, template and native build before loading."""
    expected = plan["protocol"]
    model = Path(expected["model_path"])
    binary = Path(expected["native_binary"]["path"])
    progress("before_runtime_freeze", 0, 432)
    spec = legacy.cached_current_model() or {}
    for path, field, wanted, observed in [
        (model, "hf_id", MODEL_SPECS[0], spec.get("hf_id")),
        (
            model,
            "cache_revision",
            expected["model_revision"],
            Path(spec.get("model_path", "/missing")).parent.name,
        ),
        (
            model,
            "gguf_sha256",
            expected["gguf_sha256"],
            sha256_file(model) if model.is_file() else None,
        ),
        (
            binary,
            "llama_build_sha256",
            expected["runtime_sha256"],
            sha256_file(binary) if binary.is_file() else None,
        ),
        (binary, "runtime_executable", True, os.access(binary, os.X_OK)),
        (ROOT, "CARNOT_FORCE_LIVE", "1", os.environ.get("CARNOT_FORCE_LIVE")),
    ]:
        gate(plan, path, field, wanted, observed)
    if all(r["passed"] for r in plan["checks"]):
        metadata = legacy.read_gguf_metadata(model)
        offset = metadata["field_provenance"]["metadata_keys"]["tokenizer.chat_template"][
            "value_offset"
        ]
        with model.open("rb") as stream:
            stream.seek(offset)
            template = stream.read(struct.unpack("<Q", stream.read(8))[0]).decode()
        gate(
            plan,
            model,
            "embedded_chat_template_sha256",
            expected["chat_template_sha256"],
            canonical_hash(template),
        )
        command = dict(
            name="native_cuda_support",
            argv=[str(binary), "--list-devices"],
            expected_exit=0,
            deadline_s=15,
        )
        progress("before_subprocess_native_cuda_support", 0, 1)
        receipt = execution.run_check(ROOT, command, private, raw / "preflight", heartbeat_s=5)
        progress("after_subprocess_native_cuda_support", 1, 0)
        gate(
            plan,
            binary,
            "cuda_offload_support",
            True,
            receipt["passed"] and "CUDA" in receipt["output_tail"],
        )
        plan["refs"].append(dict(path=receipt["log_path"], sha256=receipt["log_sha256"]))
        plan["runtime_freeze"] = dict(
            metadata=metadata,
            tokenizer="embedded_GGUF",
            chat_template_sha256=canonical_hash(template),
            cuda_offload_layers=99,
            command=legacy.QwenRuntime(model, private, 0).command,
            native_build_files=[
                reference(p) for p in sorted(binary.parent.glob("*.so*")) if p.is_file()
            ],
        )
        plan["refs"].append(
            source.qualified.cohort.immutable(raw / "runtime_freeze.json", plan["runtime_freeze"])
        )
    progress("after_runtime_freeze", len(plan["checks"]), 432)


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Seal predictions before reading only fit/tune targets for trainability."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    plan = inputs(root, raw, fixture=fixture)
    plan["capture_started_monotonic"] = began
    manifest = source.qualified.cohort.immutable(
        raw / "capture_manifest.json", dict(rows=plan["slots"], config=CONFIG)
    )
    result: Json = dict(rows=[], checks=[], ledger=[])
    with TemporaryDirectory(prefix="carnot8153-model-") as directory:
        private = Path(directory)
        if plan["slots"] and not fixture:
            runtime_preflight(plan, raw, private)
        ready = bool(plan["slots"]) and all(r["passed"] for r in plan["checks"])
        if ready and not fixture:
            progress("before_live_capture", 0, 432)
            result = live(plan, raw, private)
            progress("after_live_capture", len(result["rows"]), 432 - len(result["rows"]))
        if fixture or (plan["slots"] and not result["rows"]):
            failures = [r["check"] for r in [*plan["checks"], *result["checks"]] if not r["passed"]]
            ledger = Ledger(raw / ("fixture_ledger.json" if fixture else "unstarted_ledger.json"))
            result["rows"] = capture(
                plan["slots"],
                FixtureRuntime(),
                raw / "slots",
                plan["capture_identity"],
                ledger=ledger,
                blocked_reason=failures[0] if failures else None,
            )
            if fixture:
                result["ledger"] = ledger.rows
    primitive = source.qualified.cohort.immutable(
        raw / "primitive_calls.json", dict(rows=result["rows"])
    )
    labels: Json = {}
    if ready and not fixture:
        progress("predictions_sealed_before_fit_tune_targets", len(result["rows"]), 0)
        b = source.qualified.Custody(raw)
        original = b.read(root / source.qualified.COHORT, source.qualified.UPSTREAM_HASHES[8098])
        for role in ROLES:
            ref = original["evaluator_label_manifests"][role]
            evaluator = source.qualified.read_ref(b, ref)
            labels.update({r["unit_id"]: r["y"] for r in evaluator["rows"]})
        plan["refs"].extend(b.refs)
    if fixture:
        labels = {
            r["unit_id"]: (i // 2) % 2
            for i, r in enumerate(result["rows"])
            if r["condition"] == "original" and r["arm"] == "holistic"
        }
    label_ref = source.qualified.cohort.immutable(raw / "fit_tune_targets.json", labels)
    work: Json = dict(
        plan=plan,
        result=result,
        labels=labels,
        owned_failure=bool(mutation),
        capture_manifest=manifest,
        raw_shard_hashes=[manifest, primitive, label_ref],
        duration_s=time.monotonic() - began,
        phase_spans=[dict(phase="fixed_capture", start_s=0, duration_s=time.monotonic() - began)],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                protocol.MODULE,
                source.MODULE,
                "python/carnot/inference/qwen_sufficiency_7920.py",
                "python/carnot/experiment_7969_v691_qwen_calibration_capture.py",
            ]
        },
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(result["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness needs usable source pairs and normal checks, never a favorable effect."""
    plan, result = work["plan"], work["result"]
    reduced = reduce(result["rows"])
    checks = [*plan["checks"], *result["checks"]]
    failures = [r for r in checks if not r["passed"]]
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    klass = (
        "disqualified"
        if not owned
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    ledger = Ledger(raw / "not_written.json")
    ledger.rows = [] if fixture else result["ledger"]
    counts = ledger.counts()
    call_rows = {r["call_id"]: r for r in result["rows"] if r["started"]}
    for call in ledger.rows:
        if call["operation"] == "generation":
            row = call_rows[call["call_id"]]
            if call["request_sha256"] != canonical_hash(row["request"]):
                raise ValueError("ledger_request_binding")
            if call["response_sha256"] != canonical_hash(row["raw_response"]):
                raise ValueError("ledger_response_binding")
    generated = counts["generation_calls_completed"] > 0
    substrate_class = (
        "model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load"
    )
    cuda = bool(
        result.get("model_identity_receipt", {}).get("authenticated")
        and result.get("resolved_library")
        and result.get("gpu_lease_receipt")
    )
    ready = int(
        owned and not failures and reduced["transport_ready"] and (fixture or (generated and cuda))
    )
    trainable = trainability(reduced["rows"], work["labels"])
    value: Json = dict(
        work,
        **reduced,
        experiment_id=8153,
        task_id=TASK,
        milestone="2026.10.705",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + klass
        + "_"
        + (
            failures[0]["check"]
            if klass == "blocked"
            else "owned_validation"
            if klass == "disqualified"
            else "fit_evidence_capture"
        ),
        verdict_class=klass,
        required_checks_passed=owned,
        flagged_adversarial=False,
        fixture_protocol_only=fixture,
        verifier_is_oracle=fixture,
        claim_scope="source evidence transport and descriptive sensitivity; no entailment, calibration or learning benefit",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        preconditions_checked=checks,
        gate_check_summary=checks,
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=[],
        model_invocation_counts=counts,
        call_ledger=ledger.rows,
        inference_substrate="live_llm_inference"
        if generated
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class=substrate_class,
        planned_inference_substrate_class="model_bounded_generation",
        inference_mode="fixture"
        if fixture
        else "live_gpu"
        if generated
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "blocked_no_run",
        fit_capture_ready_score=ready,
        fit_trainable_score=int(ready and trainable["fit_trainable_score"]),
        class_support=trainable["class_support"],
        random_seed=CONFIG["seed"],
        sample_size_budget=dict(
            intended=192,
            fit=128,
            tune=64,
            main_calls=384,
            diagnostic_calls=48,
            independent_unit="original_source_cluster",
            output_token_budget=432 * 128,
        ),
        model_receipt={k: v for k, v in result.items() if k not in ("rows", "ledger", "checks")},
        source_artifact_hashes=plan["refs"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        acceptance_gates=dict(
            transport="96 fit pairs and48 tune pairs; quote validity and source sensitivity are descriptive",
            trainability="12 original targets per class per role",
            owned="normal validation and 100% new statement coverage",
        ),
        field_principles=dict(
            verdict="External blocks are terminal; owned failures disqualify.",
            independence="Two arms and fixed interventions create no new source units.",
            parser="An invalid quote preserves a valid probability; quotes do not prove entailment.",
            science="Exposed development and scripted fixtures earn no independent generalization credit.",
            inference="Only current owned calls count; no imported or simulated model credit.",
            duration="Measured elapsed time is never padded.",
        ),
    )
    value["measurement_reference"] = reference(raw / "measurement.json")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Authenticate exact inputs and recompute all headlines from sealed calls."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["raw_shard_hashes"],
            *value["source_artifact_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if (
            json.loads(Path(work["raw_shard_hashes"][1]["path"]).read_text())["rows"]
            != work["result"]["rows"]
        ):
            return False
        return (
            build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze the exact owned commands and scope coverage to new module and CLI."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["deadline_s"] = 600
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][1]["name"] = "qualified_parser_and_E2E016"
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_experiment_7868_v683_intervention_protocol.py",
        "tests/python/test_evidence_protocol_8124.py",
    ]
    e2e = str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    for name, flag in [("E2E016_fixture", "--fixture-e2e"), ("E2E016_replay", "--cold-replay")]:
        specs["commands"].append(
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    e2e,
                    "--date",
                    "20260929",
                    flag,
                    str(private / "intervention.json"),
                ],
                expected_exit=0,
                deadline_s=60,
            )
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    """Run the dated capture or private CLI fixture, then validate terminal bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutation", choices=["", "source"], default="")
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "replay_rejected")
        return 0 if passed else 1
    fixture = args.fixture_output is not None
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.is_relative_to(ROOT / "results"):
        parser.error("private fixture output required")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8153-validation-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        atomic_json(raw / "validation_commands.json", specs)
        work = measure(args.root, raw, fixture=fixture, mutation=args.mutation)
        receipts = [dict(name="private_fixture_transport", passed=True)] if fixture else []
        if not fixture:
            for spec in specs["commands"]:
                progress(
                    "before_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
                receipts.append(
                    execution.run_check(ROOT, spec, private, raw / "logs", heartbeat_s=20)
                )
                progress(
                    "after_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
            health = Path("/tmp/carnot8153health.json")
            if health.is_file():
                work["repository_health"] = json.loads(health.read_text())
                log = Path(work["repository_health"]["log_path"])
                shutil.copy2(log, raw / "repository_health.log")
                work["repository_health"]["log_path"] = str(raw / "repository_health.log")
                atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture)
        with patch.object(execution, "e", sys.modules[__name__]):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
