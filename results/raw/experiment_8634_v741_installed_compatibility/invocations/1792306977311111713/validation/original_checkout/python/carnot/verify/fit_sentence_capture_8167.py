"""REQ-VERIFY-8167: capture intact local evidence before evaluator access.

A source is usable only when every original sentence has valid transport.
Byte quotes remain mechanical evidence. No model judgment establishes truth.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import fit_evidence_capture_8153 as qualified
from carnot.verify import sentence_evidence_8166 as sentence
from carnot.verify import sentence_methods_8166 as methods
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8167_v706_fit_sentence_capture"
TASK = "exp8167-fit-sentence-capture"
MODULE = "python/carnot/verify/fit_sentence_capture_8167.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_fit_sentence_capture_8167.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261005"
ROLES = dict(fit=128, tune=64)
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
UPSTREAM = "results/experiment_8166_v706_sentence_evidence_methods.json"
HISTORICAL = "results/experiment_8153_v705_fit_evidence_capture.json"
PINS = {
    UPSTREAM: "sha256:339aa6acd1f91ca6a5f262b82e0f71716646c992c093cdb1a35d111f4c4d6405",
    HISTORICAL: methods.PINS[HISTORICAL],
}
CONFIG = dict(
    seed=70667,
    main_calls=384,
    diagnostic_calls=16,
    max_tokens=256,
    input_tokens=6000,
    load_timeout_s=300,
    call_timeout_s=120,
    latest_launch_s=3000,
    closure_s=4800,
    checkpoint_source_slots=8,
)
reference = qualified.reference
progress = lambda phase, completed=0, pending=0: print(
    f"[exp8167] phase={phase} completed={completed} pending={pending}", flush=True
)


def freeze(sources: list[Json], historical: Json) -> list[Json]:
    """Retain original roles and select diagnostic donors before outcomes exist."""
    if Counter(r["role"] for r in sources) != ROLES:
        raise ValueError("role_count")
    seen: set[str] = set()
    slots = []
    for role, count in ROLES.items():
        for i, row in enumerate(r for r in sources if r["role"] == role):
            if row["slot"] != i + 1 or row["source_cluster_id"] in seen:
                raise ValueError("original_slot_or_cluster")
            seen.add(row["source_cluster_id"])
            old = historical.get(row["unit_id"])
            if old and any(
                row[k] != old[k]
                for k in ("source_bytes", "answer_bytes", "role", "source_cluster_id")
            ):
                raise ValueError("historical_source_identity")
            slots.append(
                dict(row, condition="original", human_target=None, historical_present=bool(old))
            )
    first = [
        r
        for r in slots
        if r["role"] == "fit"
        and r["historical_present"]
        and sentence.requests(r, lambda _: 0)["status"] == "completed"
    ][:8]
    for i, original in enumerate(first):
        donor = first[(i + 1) % len(first)]
        for condition in ("source_removed", "source_mismatched"):
            row = deepcopy(original)
            row.update(
                condition=condition,
                source_bytes="" if condition == "source_removed" else donor["source_bytes"],
                donor_unit_id=None if condition == "source_removed" else donor["unit_id"],
            )
            slots.append(row)
    for row in slots:
        for key in ("source", "answer"):
            row[key + "_sha256"] = (
                "sha256:" + hashlib.sha256(bytes.fromhex(row[key + "_bytes"])).hexdigest()
            )
        row["group_sha256"] = canonical_hash(
            [row["unit_id"], row["source_cluster_id"], row["role"], row["slot"]]
        )
    return slots


def inputs(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Authenticate sealed public manifests without opening human targets."""
    plan: Json = dict(
        checks=[], refs=[], slots=[], protocol={}, manifests={}, upstream=[], label_ref=None
    )

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        plan["checks"].append(
            dict(
                check=field,
                upstream=str(path),
                path=str(path.absolute()),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
        if expected != observed:
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        path = Path(ref["path"])
        require(path, "input_sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
        target = raw / "inputs" / (ref["sha256"].split(":")[-1] + "-" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        plan["refs"].append(reference(target))
        return dict(json.loads(target.read_text()))

    progress("before_input_authentication")
    try:
        values = {}
        for name, pin in PINS.items():
            path = root / name
            require(path, "upstream_exists", True, path.is_file())
            value = bind(reference(path) if fixture else dict(path=str(path), sha256=pin))
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                require(path, field, expected, value.get(field))
            if name == UPSTREAM:
                require(
                    path,
                    "sentence_protocol_ready_score",
                    1,
                    value.get("sentence_protocol_ready_score"),
                )
            if not fixture:
                terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
                sidecar = Path(terminal["publication"]["sidecar_path"])
                require(
                    path,
                    "terminal_passed",
                    True,
                    read_bound_sidecar(path, sidecar)["report"]["passed"],
                )
                plan["refs"].append(reference(sidecar))
            values[name] = value
            plan["upstream"].append(
                dict(
                    experiment_id=int(Path(name).name.split("_")[1]),
                    fields_imported=["source_manifest", "capture_manifest", "runtime_identity"],
                    sha256=sha256_file(path),
                )
            )
        protocol = values[UPSTREAM]
        require(
            Path(protocol["protocol_path"]),
            "protocol_sha256",
            methods.PROTOCOL_HASH,
            sha256_file(Path(protocol["protocol_path"])),
        )
        bind(dict(path=protocol["protocol_path"], sha256=methods.PROTOCOL_HASH))
        require(
            ROOT / methods.METHOD,
            "immutable_v705_methods",
            methods.METHOD_HASH,
            sha256_file(ROOT / methods.METHOD),
        )
        plan["refs"].append(reference(ROOT / methods.METHOD))
        sources = []
        for role in ROLES:
            view = bind(protocol["source_manifest"][role])
            plan["manifests"][role] = plan["refs"][-1]
            public = {r["family_id"]: r for r in view["request_rows"]}
            for row in view["roster"]:
                item = public[row["unit_id"]]
                require(
                    root,
                    "public_fields",
                    ["answer_bytes", "family_id", "source_bytes"],
                    sorted(item),
                )
                sources.append(dict(row, **item))
        historical = values[HISTORICAL]
        old = bind(historical["capture_manifest"])
        matched = {
            r["unit_id"]: r
            for r in old["rows"]
            if r["condition"] == "original" and r["arm"] == "holistic"
        }
        plan["slots"] = freeze(sources, matched)
        plan["protocol"] = historical["plan"]["protocol"]
        plan["label_ref"] = next(
            (
                r
                for r in historical["raw_shard_hashes"]
                if r["path"].endswith("fit_tune_targets.json")
            ),
            None,
        )
        if not fixture:
            require(
                root / HISTORICAL, "original_label_reference", True, plan["label_ref"] is not None
            )
            target = Path(plan["label_ref"]["path"])
            require(
                target,
                "original_target_custody_sha256",
                plan["label_ref"]["sha256"],
                sha256_file(target) if target.is_file() else None,
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        plan["slots"] = []
        if all(r["passed"] for r in plan["checks"]):
            plan["checks"].append(
                dict(
                    check="input_structure",
                    upstream=str(root),
                    path=str(root),
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="authenticated inputs",
                    observed=str(error),
                    passed=False,
                )
            )
    plan["capture_identity"] = canonical_hash([plan["slots"], CONFIG])
    atomic_json(raw / "input_plan.json", plan)
    progress("after_input_authentication", len(plan["slots"]), 208 - len(plan["slots"]))
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
    """Seal attempts before dispatch; preserve every slot when transport fails."""
    began = time.monotonic() if started is None else started
    rows: list[Json] = []
    attempts = Counter()
    cancelled = False
    original_ready: Json = {}
    for i, slot in enumerate(slots):
        progress("before_source", i, len(slots) - i)
        row = dict(
            slot,
            capture_identity=identity,
            calls=[],
            sentences=sentence.partition(bytes.fromhex(slot["answer_bytes"])),
            status="excluded",
            exclusion_reason=blocked_reason,
            local_features=None,
        )
        original = slot["condition"] == "original"
        reason = blocked_reason or (
            "absent_historical_source" if not slot["historical_present"] else None
        )
        schedule: Json = dict(requests=[], sentences=row["sentences"], exclusion_reason=reason)
        if not reason:
            progress("before_token_count", i, len(slots) - i)
            try:
                counter = lambda payload: runtime.count(
                    json.dumps([dict(role="user", content=payload)], ensure_ascii=False)
                )
                if original:
                    schedule = sentence.requests(slot, counter)
                else:
                    payload = dict(
                        instruction="Return JSON array: sentence_index, p_unsupported, relation (entailed/contradicted/baseless), quote, byte_start, byte_end. Null quotes have null offsets.",
                        source=bytes.fromhex(slot["source_bytes"]).decode(),
                        answer=bytes.fromhex(slot["answer_bytes"]).decode(),
                        sentences=[
                            dict(r, text=bytes.fromhex(r["sentence_bytes"]).decode())
                            for r in row["sentences"][:4]
                        ],
                    )
                    prompt = json.dumps(payload, ensure_ascii=False)
                    n = counter(prompt)
                    reason = "input_token_limit" if n > 6000 else None
                    schedule = dict(
                        exclusion_reason=reason,
                        requests=[
                            dict(
                                payload=prompt,
                                input_upper_bound=n,
                                sentence_indices=[
                                    r["sentence_index"] for r in row["sentences"][:4]
                                ],
                            )
                        ],
                    )
                reason = schedule["exclusion_reason"]
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError):
                reason = "tokenizer_failure"
            progress("after_token_count", i, len(slots) - i)
        if not original and not original_ready.get(slot["unit_id"], False):
            reason = reason or "original_ineligible"
        row["exclusion_reason"] = reason
        if reason == "tokenizer_failure":
            row["status"] = "failed"
        if not reason:
            row.update(status="completed", exclusion_reason=None)
            kind = "main" if original else "diagnostic"
            for group in schedule["requests"]:
                if (
                    cancelled
                    or time.monotonic() - began
                    >= CONFIG["latest_launch_s"] - CONFIG["call_timeout_s"]
                    or attempts[kind] >= CONFIG[kind + "_calls"]
                ):
                    row.update(
                        status="censored",
                        exclusion_reason="launch_cutoff"
                        if attempts[kind] < CONFIG[kind + "_calls"]
                        else "call_budget",
                    )
                    break
                request = dict(
                    model=MODEL_SPECS[0],
                    messages=[dict(role="user", content=group["payload"])],
                    temperature=0,
                    top_p=1,
                    seed=CONFIG["seed"],
                    max_tokens=256,
                    cache_prompt=False,
                    chat_template_kwargs=dict(enable_thinking=False),
                )
                for retry in range(2):
                    call_id = f"{slot['unit_id']}:{slot['condition']}:{len(row['calls'])}"
                    call = dict(
                        call_id=call_id,
                        request=request,
                        prompt=group["payload"],
                        prompt_sha256=canonical_hash(group["payload"]),
                        sentence_indices=group["sentence_indices"],
                        input_tokens=group["input_upper_bound"],
                        output_tokens=None,
                        raw_response={},
                        transcript="",
                        status="running",
                        attempt=retry + 1,
                        model_sha256=None,
                        source_sha256=slot["source_sha256"],
                        answer_sha256=slot["answer_sha256"],
                        group_sha256=slot["group_sha256"],
                        duration_s=0.0,
                    )
                    attempts[kind] += 1
                    row["calls"].append(call)
                    atomic_json(raw / f"slot-{i:03d}.json", row)
                    ledger.start("generation", call_id, request)
                    progress(
                        "before_generation",
                        sum(attempts.values()) - 1,
                        400 - sum(attempts.values()) + 1,
                    )
                    at = time.monotonic()
                    transport = False
                    try:
                        call["raw_response"] = runtime.generate(request)
                        call["status"] = "completed"
                    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                        transport = isinstance(error, OSError) and not isinstance(
                            error, TimeoutError
                        )
                        call["status"] = (
                            "cancelled" if isinstance(error, TimeoutError) else "failed"
                        )
                        row.update(
                            status="failed",
                            exclusion_reason="transport_timeout"
                            if isinstance(error, TimeoutError)
                            else "transport_failure",
                        )
                        if isinstance(error, TimeoutError):
                            runtime.close()
                            cancelled = True
                    call["duration_s"] = time.monotonic() - at
                    ledger.finish(call_id, call["status"], call["raw_response"])
                    response = call["raw_response"]
                    usage = response.get("usage", {})
                    call["output_tokens"] = usage.get("completion_tokens")
                    call["transcript"] = (
                        response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                    )
                    call["parsed"] = sentence.parse(
                        call["transcript"],
                        bytes.fromhex(slot["source_bytes"]),
                        group["sentence_indices"],
                    )
                    progress(
                        "after_generation", sum(attempts.values()), 400 - sum(attempts.values())
                    )
                    if (
                        transport
                        and retry == 0
                        and attempts[kind] < CONFIG[kind + "_calls"]
                        and time.monotonic() - began
                        < CONFIG["latest_launch_s"] - CONFIG["call_timeout_s"]
                    ):
                        row.update(status="completed", exclusion_reason=None)
                        continue
                    break
                if call["status"] != "completed":
                    break
                valid_usage = (
                    type(usage.get("prompt_tokens")) is int
                    and 0 <= usage["prompt_tokens"] <= 6000
                    and type(usage.get("completion_tokens")) is int
                    and 0 < usage["completion_tokens"] <= 256
                )
                if response.get("choices", [{}])[0].get("finish_reason") == "length":
                    row.update(status="censored", exclusion_reason="output_token_budget")
                    break
                if not valid_usage or call["parsed"]["status"] != "completed":
                    row.update(
                        status="failed",
                        exclusion_reason="invalid_usage" if not valid_usage else "parser_failure",
                    )
                    break
            parsed = [
                r for c in row["calls"] if c["status"] == "completed" for r in c["parsed"]["rows"]
            ]
            if row["status"] == "completed" and original:
                row["local_features"] = sentence.local_features(parsed, len(row["sentences"]))
                if row["local_features"] is None:
                    row.update(status="failed", exclusion_reason="incomplete_sentence_coverage")
        original_ready[slot["unit_id"]] = (
            (not reason) if original else original_ready.get(slot["unit_id"], False)
        )
        row.update(
            arm="sentence_evidence",
            metric="full_sentence_transport",
            numerator=int(row["status"] == "completed"),
            denominator=1,
        )
        atomic_json(raw / f"slot-{i:03d}.json", row)
        rows.append(row)
        if (i + 1) % 8 == 0 or i + 1 == len(slots):
            atomic_json(
                raw / "checkpoint.json",
                dict(completed_slots=i + 1, pending_slots=len(slots) - i - 1, rows=rows),
            )
            progress("checkpoint", i + 1, len(slots) - i - 1)
    return rows


def reduce(captured: list[Json]) -> Json:
    """Reparse exact transcripts and count independent original source clusters."""
    rows, sentence_rows, diagnostic_rows = [], [], []
    originals: Json = {}
    for row in captured:
        if row["human_target"] is not None or row["sentences"] != sentence.partition(
            bytes.fromhex(row["answer_bytes"])
        ):
            raise ValueError("sentence_or_target_drift")
        if any(
            row[key + "_sha256"]
            != "sha256:" + hashlib.sha256(bytes.fromhex(row[key + "_bytes"])).hexdigest()
            for key in ("source", "answer")
        ):
            raise ValueError("source_or_answer_hash_drift")
        parsed = []
        for call in row["calls"]:
            payload = json.loads(call["prompt"])
            observed = sentence.parse(
                call["transcript"], bytes.fromhex(row["source_bytes"]), call["sentence_indices"]
            )
            if (
                call["parsed"] != observed
                or canonical_hash(call["prompt"]) != call["prompt_sha256"]
                or call["request"]["messages"] != [dict(role="user", content=call["prompt"])]
                or call["request"]["max_tokens"] != 256
                or call["request"]["model"] != MODEL_SPECS[0]
                or payload["source"].encode().hex() != row["source_bytes"]
                or payload["answer"].encode().hex() != row["answer_bytes"]
                or call["transcript"]
                != (
                    call["raw_response"].get("choices", [{}])[0].get("message", {}).get("content")
                    or ""
                )
            ):
                raise ValueError("request_or_parse_drift")
            if call["status"] == "completed":
                parsed.extend(observed["rows"])
        if row["condition"] == "original":
            features = (
                sentence.local_features(parsed, len(row["sentences"]))
                if row["status"] == "completed"
                else None
            )
            if features != row["local_features"]:
                raise ValueError("feature_drift")
            originals[row["unit_id"]] = parsed
            rows.append(
                {
                    k: row[k]
                    for k in (
                        "unit_id",
                        "source_cluster_id",
                        "role",
                        "slot",
                        "arm",
                        "condition",
                        "metric",
                        "numerator",
                        "denominator",
                        "status",
                        "exclusion_reason",
                        "local_features",
                    )
                }
            )
            for part in parsed:
                sentence_rows.append(
                    dict(
                        part,
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        role=row["role"],
                        sentence_sha256=canonical_hash(row["sentences"][part["sentence_index"]]),
                        source_sha256=row["source_sha256"],
                        answer_sha256=row["answer_sha256"],
                        group_sha256=row["group_sha256"],
                        model_sha256=row["calls"][-1].get("model_sha256"),
                        runtime_sha256=row["calls"][-1].get("runtime_sha256"),
                        source_complete=row["status"] == "completed",
                        human_target=None,
                    )
                )
        else:
            indices = [p["sentence_index"] for p in parsed]
            baseline = [
                p for p in originals.get(row["unit_id"], []) if p["sentence_index"] in indices
            ]
            valid = bool(parsed) and len(parsed) == len(baseline) and row["status"] == "completed"
            delta = (
                sum(abs(p["p_unsupported"] - b["p_unsupported"]) for p, b in zip(parsed, baseline))
                / len(parsed)
                if valid
                else None
            )
            diagnostic_rows.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=row["condition"],
                    condition="fixed_first8_eligible_fit",
                    metric="absolute_probability_sensitivity",
                    numerator=delta if valid else 0,
                    denominator=1,
                    status="completed" if valid else "excluded",
                    exclusion_reason=None if valid else "unavailable_pair",
                    probability_sensitivity=delta,
                    byte_valid_evidence_count=sum(p["quote_valid"] for p in parsed),
                    evidence_denominator=len(parsed),
                    human_target=None,
                    gasp_token_likelihood=False,
                )
            )
    if not captured:
        rows = [
            dict(
                unit_id=f"unavailable-{role}-slot-{i + 1}",
                source_cluster_id=None,
                role=role,
                slot=i + 1,
                arm="sentence_evidence",
                condition="unavailable_original_slot",
                metric="full_sentence_transport",
                numerator=0,
                denominator=1,
                status="excluded",
                exclusion_reason="external_inputs_unavailable",
                local_features=None,
            )
            for role, count in ROLES.items()
            for i in range(count)
        ]
    statuses = Counter(r["status"] for r in rows)
    support = {
        role: sum(r["role"] == role and r["status"] == "completed" for r in rows) for role in ROLES
    }
    return dict(
        rows=rows,
        sentence_rows=sentence_rows,
        diagnostic_rows=diagnostic_rows,
        pair_support=support,
        intended_count=192,
        eligible_count=192 - statuses["excluded"],
        independent_count=len(
            {r["source_cluster_id"] for r in rows if r["source_cluster_id"] is not None}
        ),
        completed_count=statuses["completed"],
        excluded_count=statuses["excluded"],
        censored_count=statuses["censored"],
        failed_count=statuses["failed"],
        transport_ready=int(support["fit"] >= 96 and support["tune"] >= 48),
    )


def trainability(rows: list[Json], labels: Json) -> Json:
    """Require both complete source support and original class support."""
    support = qualified.trainability(rows, labels)
    support["fit_trainable_score"] *= int(
        all(
            sum(r["role"] == role and r["status"] == "completed" for r in rows) >= n
            for role, n in [("fit", 96), ("tune", 48)]
        )
    )
    return support


class FixtureRuntime:
    """Exercise sentence transport privately with zero live invocation credit."""

    def count(self, text: str) -> int:
        return 30

    def generate(self, request: Json) -> Json:
        payload = json.loads(request["messages"][0]["content"])
        rows = [
            dict(
                sentence_index=r["sentence_index"],
                p_unsupported=0.2,
                relation="baseless",
                quote=None,
                byte_start=None,
                byte_end=None,
            )
            for r in payload["sentences"]
        ]
        return dict(
            choices=[dict(message=dict(content=json.dumps(rows)), finish_reason="stop")],
            usage=dict(prompt_tokens=30, completion_tokens=120),
        )


def live(plan: Json, raw: Path, private: Path) -> Json:
    """Adapt only sentence requests while retaining the qualified CUDA owner."""
    ledger = Ledger(raw / "ledger.json")
    runtime_class = qualified.legacy.QwenRuntime

    class Recorded(runtime_class):  # type: ignore[misc,valid-type]
        def load(self) -> Json:
            ledger.start("model_load", "owned-model-load", dict(model=str(self.model)))
            progress("before_model_load", 0, len(plan["slots"]))
            try:
                receipt: Json = qualified.legacy.bounded(super().load, 300)
                if (
                    canonical_hash(receipt["props"]["chat_template"])
                    != plan["protocol"]["chat_template_sha256"]
                ):
                    raise ValueError("served_template_drift")
            except (OSError, RuntimeError, TimeoutError, ValueError):
                ledger.finish("owned-model-load", "failed", {})
                progress("after_model_load_failed", 0, len(plan["slots"]))
                raise
            ledger.finish("owned-model-load", "completed", receipt)
            progress("after_model_load", 0, len(plan["slots"]))
            return receipt

    adapter = SimpleNamespace(
        freeze=lambda _: plan["slots"],
        capture=lambda frozen, runtime, path, identity, *, started: capture(
            frozen,
            runtime,
            path,
            identity,
            ledger=ledger,
            started=plan["capture_started_monotonic"],
        ),
    )

    def pulse(phase: str, started: float, units: int = 0) -> None:
        finished = sum(
            r["operation"] == "generation" and r["status"] != "running" for r in ledger.rows
        )
        progress(phase, finished, 400 - finished)

    with (
        patch.object(qualified.legacy, "TASK", TASK),
        patch.object(qualified.legacy, "QwenRuntime", Recorded),
        patch.object(qualified.legacy, "capture", adapter),
        patch.object(qualified.legacy, "load_public", lambda _: {}),
        patch.object(qualified.legacy, "progress", pulse),
    ):
        result = dict(
            qualified.legacy.live_capture(
                dict(plan, public_role_manifests=plan["manifests"]), raw, private
            )
        )
    for check in result["checks"]:
        check.update(
            check=check["upstream_id"] + "_" + check["field"],
            upstream=check["upstream_id"],
            artifact_field=check["field"],
        )
    result["ledger"] = ledger.rows
    return result


def measure(root: Path, raw: Path, *, fixture: bool = False, started: float | None = None) -> Json:
    """Seal current predictions before opening original fit and tune targets."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    plan = inputs(root, raw, fixture=fixture)
    plan["capture_started_monotonic"] = began if started is None else started
    manifest_ref = qualified.source.qualified.cohort.immutable(
        raw / "capture_manifest.json", dict(rows=plan["slots"], config=CONFIG)
    )
    result: Json = dict(rows=[], checks=[], ledger=[])
    with TemporaryDirectory(prefix="carnot8167-model-") as directory:
        private = Path(directory)
        if plan["slots"] and not fixture:
            with patch.object(qualified, "progress", progress):
                qualified.runtime_preflight(plan, raw, private)
        failures = [r["check"] for r in plan["checks"] if not r["passed"]]
        if plan["slots"] and not failures and not fixture:
            result = live(plan, raw, private)
        if fixture or (plan["slots"] and not result["rows"]):
            failures += [r["check"] for r in result["checks"] if not r["passed"]]
            ledger = Ledger(raw / "unstarted_or_fixture_ledger.json")
            result["rows"] = capture(
                plan["slots"],
                FixtureRuntime(),
                raw / "slots",
                plan["capture_identity"],
                ledger=ledger,
                blocked_reason=failures[0]
                if failures
                else None
                if fixture
                else "no_owned_live_rows",
            )
            if fixture:
                result["ledger"] = ledger.rows
    for row in result["rows"]:
        for call in row["calls"]:
            call.update(
                model_sha256=plan["protocol"].get("gguf_sha256"),
                runtime_sha256=plan["protocol"].get("runtime_sha256"),
            )
    primitive = qualified.source.qualified.cohort.immutable(
        raw / "primitive_calls.json", dict(rows=result["rows"])
    )
    progress("predictions_sealed_before_evaluator", len(result["rows"]), 0)
    labels: Json = {}
    if fixture:
        labels = {
            r["unit_id"]: i % 2
            for i, r in enumerate(result["rows"])
            if r["condition"] == "original"
        }
    elif plan["label_ref"] and all(r["passed"] for r in [*plan["checks"], *result["checks"]]):
        ref = plan["label_ref"]
        path = Path(ref["path"])
        observed = sha256_file(path) if path.is_file() else None
        qualified.gate(plan, path, "original_fit_tune_target_sha256", ref["sha256"], observed)
        if observed == ref["sha256"]:
            labels = json.loads(path.read_text())
            labels = {
                r["unit_id"]: labels[r["unit_id"]]
                for r in plan["slots"][:192]
                if r["unit_id"] in labels
            }
            plan["refs"].append(reference(path))
    label_ref = qualified.source.qualified.cohort.immutable(raw / "fit_tune_targets.json", labels)
    work: Json = dict(
        plan=plan,
        result=result,
        labels=labels,
        owned_failure=False,
        capture_manifest=manifest_ref,
        raw_shard_hashes=[manifest_ref, primitive, label_ref],
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticated_sentence_capture",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                qualified.MODULE,
                methods.NUMERIC,
                "python/carnot/verify/qwen_development_capture_7995.py",
                "python/carnot/inference/qwen_sufficiency_7920.py",
                "python/carnot/experiment_7969_v691_qwen_calibration_capture.py",
            ]
        },
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(result["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Require normal validation, observed CUDA and source coverage for readiness."""
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
    calls = {c["call_id"]: c for r in result["rows"] for c in r["calls"]}
    for call in ledger.rows:
        if call["operation"] == "generation":
            row = calls[call["call_id"]]
            if (
                call["request_sha256"] != canonical_hash(row["request"])
                or call["response_sha256"] != canonical_hash(row["raw_response"])
                or call["status"] != row["status"]
            ):
                raise ValueError("ledger_binding")
    generated = counts["generation_calls_completed"] > 0
    cuda = bool(
        result.get("model_identity_receipt", {}).get("authenticated")
        and result.get("resolved_library")
        and result.get("gpu_lease_receipt")
        and result.get("cleanup", {}).get("leak_free")
    )
    ready = int(
        owned
        and not failures
        and reduced["transport_ready"]
        and (fixture or (generated and cuda and work["duration_s"] >= 10))
    )
    trainable = trainability(reduced["rows"], work["labels"])
    value: Json = dict(
        **reduced,
        experiment_id=8167,
        task_id=TASK,
        milestone="2026.10.706",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + klass
        + "_"
        + (
            failures[0]["check"]
            if klass == "blocked"
            else "owned_validation"
            if klass == "disqualified"
            else "fit_sentence_capture"
        ),
        verdict_class=klass,
        required_checks_passed=owned,
        flagged_adversarial=False,
        fixture_protocol_only=fixture,
        verifier_is_oracle=fixture,
        claim_scope="Sentence transport and original source coverage; unlabeled probability sensitivity; no semantic truth or benefit claim",
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
        inference_substrate_class="model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load",
        planned_inference_substrate="live_llm_inference",
        planned_inference_substrate_class="model_bounded_generation",
        inference_mode="fixture" if fixture else "live_gpu" if generated else "blocked_no_run",
        fit_capture_ready_score=ready,
        fit_trainable_score=int(ready and trainable["fit_trainable_score"]),
        class_support=trainable["class_support"],
        random_seed=CONFIG["seed"],
        sample_size_budget=dict(
            intended=192,
            fit=128,
            tune=64,
            main_calls=384,
            diagnostic_calls=16,
            independent_unit="original_source_cluster",
            output_token_budget=400 * 256,
        ),
        model_receipt={k: v for k, v in result.items() if k not in ("rows", "ledger", "checks")},
        gpu_receipts={
            k: v
            for k, v in result.items()
            if "gpu" in k or k in ("model_identity_receipt", "resolved_library", "cleanup")
        },
        model_revision=plan["protocol"].get("model_revision"),
        gguf_sha256=plan["protocol"].get("gguf_sha256"),
        runtime_sha256=plan["protocol"].get("runtime_sha256"),
        capture_manifest=work["capture_manifest"],
        source_artifact_hashes=plan["refs"],
        cited_upstream_artifacts=plan["upstream"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        acceptance_gates=dict(
            transport="96 complete fit and48 complete tune sources",
            trainability="12 original targets per class per role",
            owned="normal required exits and100 percent added statements",
            budgets=CONFIG,
        ),
        field_principles=dict(
            verdict="External blocks terminate; owned failures disqualify.",
            independence="Sentences and repeats add no independent sources.",
            quotes="Byte-valid evidence is not entailment.",
            inference="Only current owned attempts count.",
            science="Exposed development grants no generalization or benefit.",
            duration="Elapsed work is measured and never padded.",
        ),
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health", {}),
        methodology_note="Frozen V706 sentence protocol; historical V705 matched controls stay historical. GASP-inspired elicited probability diagnostic, not token likelihood.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reject changed custody bytes and rebuild every headline independently."""
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
        manifest_rows = json.loads(Path(work["capture_manifest"]["path"]).read_text())
        if (
            manifest_rows["config"] != CONFIG
            or len(manifest_rows["rows"]) != len(work["result"]["rows"])
            or any(
                any(actual.get(key) != expected for key, expected in frozen.items())
                for frozen, actual in zip(
                    manifest_rows["rows"], work["result"]["rows"], strict=True
                )
            )
        ):
            return False
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
    """Freeze file paths and required exit codes before any current model load."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 600
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_sentence_evidence_8166.py",
        "tests/python/test_sentence_methods_8166.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_experiment_7868_v683_intervention_protocol.py",
    ]
    specs["commands"][1]["name"] = "qualified_sentence_parser_and_E2E016"
    return specs


def main(argv: list[str] | None = None) -> int:
    """Run bounded capture after owned checks, or replay a sealed terminal result."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
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
    with TemporaryDirectory(prefix="carnot8167-validation-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        atomic_json(raw / "validation_commands.json", specs)
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
        if fixture or all(r["passed"] for r in receipts):
            work = measure(args.root, raw, fixture=fixture, started=started)
        else:
            work = measure(
                args.root / "owned-validation-refused", raw, fixture=True, started=started
            )
            work["owned_failure"] = True
        if not fixture:
            progress("before_repository_health_subprocess", 0, 1)
            work["repository_health"] = execution.run_check(
                ROOT, specs["repository_health"], private, raw / "repository_health", heartbeat_s=20
            )
            progress("after_repository_health_subprocess", 1, 0)
        atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture)
        with patch.object(execution, "e", sys.modules[__name__]):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
