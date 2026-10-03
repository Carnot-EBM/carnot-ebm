"""REQ-REPORT-8059: own new fixed-answer scores without importing old model calls.

Repeated native contexts qualify numerical measurements. They cannot establish
correctness or a detection benefit, and a failed group is never retried.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot import experiment_8058_v698_sealed_evidence_methods as methods
from carnot.inference import fixed_answer_likelihood_8022 as base
from carnot.inference import likelihood_runtime_8022 as runtime
from carnot.inference.scoring_isolation_8033 import NativeController
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.reporting.v698_fixture_consumer_contract import (
    clean_terminal,
    failure as operand_failure,
)
from carnot.verify.evidence_features_7980 import normalized

Json = dict[str, Any]
ROOT = runtime.ROOT
NAME = "experiment_8059_v698_fit_source_scoring"
TASK = "exp8059-fit-source-scoring"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_fit_source_scoring_8059.py"
OWNED = [MODULE, CLI]
LIMITS = dict(forwards=384, tokens=147456, seconds=1800, context=4096, answer=384)
SEED = 6988059
START = time.monotonic()
INPUTS = methods.INPUTS[:9] + [
    "ops/exclusion_manifest.yaml",
    "python/carnot/inference/scoring_isolation_8033.py",
    "python/carnot/inference/likelihood_isolation_runtime_8033.py",
    "python/carnot/experiment_8033_v696_scoring_isolation.py",
    "python/carnot/inference/sota_models.py",
    "python/carnot/inference/fixed_answer_likelihood_8022.py",
    "python/carnot/inference/likelihood_runtime_8022.py",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a waiting child cannot invent progress."""
    print(
        f"[exp8059] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def failure(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Reuse operand formatting while attributing current failures to this task."""
    return operand_failure(path, field, expected, observed, upstream)


def authenticate(root: Path, raw: Path) -> Json:
    """Bind public roles and terminal gates before opening any evaluator labels."""
    plan: Json = dict(
        originals=[], failures=[], references=[], target_references={}, repository_health=[]
    )

    def bind(path: Path, expected: str | None = None) -> Json:
        if not path.is_file():
            plan["failures"].append(failure(path, "resource_exists", True, False))
            return {}
        ref = reference(path)
        if expected and ref["sha256"] != expected:
            plan["failures"].append(failure(path, "sha256", expected, ref["sha256"]))
        snapshot = raw / "inputs" / (ref["sha256"][7:] + path.suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        plan["references"].append(dict(ref, snapshot_path=str(snapshot)))
        return ref

    progress("preconditions_before")
    for name in INPUTS:
        bind(root / name)
    for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
        bind(ROOT / ".venv/bin" / tool)
    parents = {}
    for name, field in [
        ("experiment_8057_v698_fixture_consumer_contract", "fixture_consumer_ready_score"),
        ("experiment_8058_v698_sealed_evidence_methods", "source_protocol_ready_score"),
        ("experiment_8045_v697_scorer_workspace", "scorer_fixture_ready_score"),
    ]:
        path = root / "results" / (name + ".json")
        if not bind(path):
            continue
        value = json.loads(path.read_text())
        for key, expected in [(field, 1), ("flagged_adversarial", False)]:
            if value.get(key, "MISSING_FIELD") != expected:
                plan["failures"].append(
                    failure(path, key, expected, value.get(key, "MISSING_FIELD"), name)
                )
        if value.get("verdict_class") not in ["positive", "null", "circular_positive"]:
            plan["failures"].append(
                dict(
                    failure(
                        path,
                        "verdict_class",
                        ["positive", "null", "circular_positive"],
                        value.get("verdict_class"),
                        name,
                    ),
                    op="in",
                )
            )
        try:
            terminal = clean_terminal(path)
            for key in ["terminal_path", "validator_path"]:
                bind(Path(terminal[key]))
        except (OSError, ValueError, KeyError) as error:
            plan["failures"].append(failure(path, "terminal_hash_clean", True, str(error), name))
        parents[name] = value
    upstream = parents.get("experiment_8058_v698_sealed_evidence_methods", {})
    meta = {r["unit"]: r for r in upstream.get("rows", [])}
    for role in ["fit", "tune"]:
        ref = upstream.get("role_manifests", {}).get(role)
        if ref and bind(Path(ref["path"]), ref["sha256"]):
            rows = json.loads(Path(ref["path"]).read_text())["rows"]
            for r in rows:
                row = dict(r, role=role, eligible=meta[f"{role}/{r['slot']}"]["eligible"])
                plan["originals"].append(row)
    prior = root / "results/experiment_8019_v695_eligible_targets.json"
    if bind(prior):
        value = json.loads(prior.read_text())
        plan["target_references"] = {
            r: {
                s: value[k][r]
                for s, k in [("public", "public_manifests"), ("evaluator", "evaluator_manifests")]
            }
            for r in ["fit", "tune"]
        }
    plan["repository_health"] = upstream.get("repository_health", [])
    health_log = Path("/tmp/carnot-8059-health/full-suite.log")
    if health_log.is_file():
        ref = bind(health_log)
        plan["repository_health"].append(
            dict(
                name="current_bounded_full_suite",
                argv=[
                    "timeout",
                    "--kill-after=5s",
                    "60s",
                    str(ROOT / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                ],
                exit_code=124,
                timeout_s=60,
                passed=False,
                duration_s=None,
                duration_note="The outer timeout was observed; exact subprocess duration was not separately timed.",
                log_sha256=ref["sha256"],
                log_path=plan["references"][-1]["snapshot_path"],
                classification="repository_health_only",
            )
        )
    for ref in parents.get("experiment_8057_v698_fixture_consumer_contract", {}).get(
        "scorer_code_hashes", []
    ):
        bind(Path(ref["path"]), ref["sha256"])
    progress("preconditions_after", len(plan["originals"]), len(plan["failures"]))
    return plan


def freeze(originals: list[Json], tokenizer: Any) -> list[Json]:
    """Keep complete byte views or an explicit exclusion; never shorten answers."""
    panel = []
    seen: set[str] = set()
    for original in originals:
        r = dict(original, views={}, target_tokens=[], response_token_offsets=[])
        source, answer = (bytes.fromhex(r[k]) for k in ["source_bytes", "answer_bytes"])
        identity = normalized(source)
        if identity in seen or identity != r["source_cluster_id"]:
            raise ValueError("source_role_overlap_or_identity")
        seen.add(identity)
        r.update(
            source_sha256="sha256:" + hashlib.sha256(source).hexdigest(),
            answer_sha256="sha256:" + hashlib.sha256(answer).hexdigest(),
        )
        if r["eligible"]:
            try:
                targets = tokenizer.tokenize(answer, add_bos=False, special=False)
                if not source or not 1 <= len(targets) <= LIMITS["answer"]:
                    raise ValueError("complete_answer_or_source_budget")
                if tokenizer.detokenize(targets) != answer:
                    raise ValueError("answer_byte_roundtrip")
                for arm, blob in [("full", source), ("no_source", b"")]:
                    prompt = tokenizer.render(blob, base.QUESTION)
                    prefix = tokenizer.tokenize(prompt, add_bos=True, special=True)
                    combined = tokenizer.tokenize(prompt + answer, add_bos=True, special=True)
                    if combined != prefix + targets:
                        raise ValueError("prompt_answer_boundary")
                    if len(combined) > LIMITS["context"]:
                        raise ValueError("complete_context_budget")
                    r["views"][arm] = dict(
                        tokens=combined, response_start=len(prefix), prompt_bytes=prompt.hex()
                    )
                r["target_tokens"] = targets
                stops = [len(tokenizer.detokenize(targets[: i + 1])) for i in range(len(targets))]
                r["response_token_offsets"] = list(
                    map(list, zip([0] + stops[:-1], stops, strict=True))
                )
            except (ValueError, UnicodeError) as error:
                r.update(eligible=False, exclusion_reason=str(error), views={})
        panel.append(r)
        progress("token_view_seal", len(panel), len(originals) - len(panel))
    return panel


def schedule(panel: list[Json]) -> list[Json]:
    """Finish the original pilot first; repeat sweeps separate duplicate groups."""
    pilot = [r for r in panel if r["role"] == "fit" and r["slot"] < 8]
    rest = [r for r in panel if r not in pilot]
    return [
        dict(
            id=f"pass-{i:03d}",
            family_id=r["family_id"],
            role=r["role"],
            slot=r["slot"],
            arm=arm,
            repeat=repeat,
            pilot=block == 0,
        )
        for i, (block, repeat, r, arm) in enumerate(
            (b, a, r, k)
            for b, rows in enumerate([pilot, rest])
            for a in ["A", "B"]
            for r in rows
            for k in ["full", "no_source"]
        )
    ]


def capture(panel: list[Json], controller: Any, raw: Path, *, deadline: float) -> list[Json]:
    """Persist uncertain calls before execution and stop once without retries."""
    items = {r["family_id"]: r for r in panel}
    rows: list[Json] = []
    stopped, tokens, calls = False, 0, 0
    for slot in schedule(panel):
        if not slot["pilot"] and rows and rows[-1]["pilot"]:
            stopped = stopped or not reduce(panel, rows, partial=True)["pilot_passed"]
            progress("pilot_complete", calls, len(schedule(panel)) - len(rows))
        item = items[slot["family_id"]]
        row = dict(
            slot,
            unit=slot["id"],
            source=item["source_cluster_id"],
            seed=SEED,
            denominator=len(item["target_tokens"]),
            numerator=None,
            token_rows=[],
            status="censored",
            exclusion_reason=item.get("exclusion_reason"),
            censor_reason="budget_or_pilot_stop",
            failure_reason=None,
        )
        if not item["eligible"]:
            row.update(status="excluded", censor_reason=None)
        elif (
            not stopped
            and time.monotonic() < deadline
            and tokens + row["denominator"] <= LIMITS["tokens"]
            and calls < LIMITS["forwards"]
        ):
            row.update(
                status="running", started_monotonic_ns=time.monotonic_ns(), censor_reason=None
            )
            atomic_json(raw / (row["id"] + ".json"), row)
            calls += 1
            progress("before_teacher_forcing", calls - 1, len(schedule(panel)) - len(rows))
            try:
                result = controller.score(item["views"][slot["arm"]], "fresh_full")
                row.update(result)
                row["token_rows"] = [
                    dict(
                        token_id=t,
                        offset=o,
                        log_probability=float(p),
                        probability=math.exp(float(p)),
                        logit_position=z,
                    )
                    for t, o, p, z in zip(
                        item["target_tokens"],
                        item["response_token_offsets"],
                        result["target_logprobs"],
                        result["conditional_logit_positions"],
                        strict=True,
                    )
                ]
                row["numerator"] = math.fsum(-t["log_probability"] for t in row["token_rows"])
                row.update(status="completed", mean_nll=row["numerator"] / row["denominator"])
                tokens += row["denominator"]
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                row.update(status="failed", failure_reason=f"{type(error).__name__}:{error}")
                stopped = True
            row["ended_monotonic_ns"] = time.monotonic_ns()
            progress("after_teacher_forcing", calls, len(schedule(panel)) - len(rows) - 1)
        atomic_json(raw / (row["id"] + ".json"), row)
        rows.append(row)
    return rows


def reduce(panel: list[Json], rows: list[Json], *, partial: bool = False) -> Json:
    """Recompute features from scalar tokens rather than trusting stored means."""
    slots = schedule(panel)
    if (not partial and len(slots) != len(rows)) or any(
        any(r[k] != s[k] for k in s) for r, s in zip(rows, slots)
    ):
        raise ValueError("slot_roster")
    items = {r["family_id"]: r for r in panel}
    means: dict[tuple[str, str, str], float] = {}
    scored = 0
    for r in rows:
        item = items[r["family_id"]]
        if r["status"] != "completed":
            continue
        ts, view = r["token_rows"], item["views"][r["arm"]]
        numerator = math.fsum(-t["log_probability"] for t in ts)
        if (
            [t["token_id"] for t in ts] != item["target_tokens"]
            or [t["offset"] for t in ts] != item["response_token_offsets"]
            or [t["logit_position"] for t in ts]
            != list(range(view["response_start"] - 1, len(view["tokens"]) - 1))
            or not ts
            or r["denominator"] != len(ts)
            or abs(r["numerator"] - numerator) > 1e-10
            or abs(r["mean_nll"] - numerator / len(ts)) > 1e-10
            or any(
                not math.isfinite(t["log_probability"])
                or t["log_probability"] > 0
                or abs(t["probability"] - math.exp(t["log_probability"])) > 1e-12
                for t in ts
            )
            or r["normalization_max_error"] > 1e-10
            or not r["context_closed"]
            or not r["context_identity"].startswith("fresh-")
        ):
            raise ValueError("token_alignment_aggregate_or_normalization")
        means[(r["family_id"], r["arm"], r["repeat"])] = numerator / len(ts)
        scored += len(ts)
    drift, features = [], []
    for item in panel:
        averages, valid = {}, True
        for arm in ["full", "no_source"]:
            pair = [means.get((item["family_id"], arm, repeat)) for repeat in ["A", "B"]]
            d = abs(pair[0] - pair[1]) if all(x is not None for x in pair) else None
            passed = d is not None and d <= 1e-6
            drift.append(
                dict(
                    family_id=item["family_id"],
                    arm=arm,
                    drift=d,
                    passed=passed,
                    captures=sum(x is not None for x in pair),
                )
            )
            valid = valid and passed
            if passed:
                averages[arm] = math.fsum(pair) / 2
        if valid:
            features.append(
                dict(
                    family_id=item["family_id"],
                    source_cluster_id=item["source_cluster_id"],
                    role=item["role"],
                    slot=item["slot"],
                    full_mean_nll=averages["full"],
                    no_source_minus_full=averages["no_source"] - averages["full"],
                )
            )
    pilot = [r["family_id"] for r in panel if r["role"] == "fit" and r["slot"] < 8]
    return dict(
        feature_rows=features,
        duplicate_drift_rows=drift,
        scored_tokens=scored,
        forward_pass_counts=sum(r["status"] in ["completed", "failed"] for r in rows),
        pilot_passed=len(pilot) == 8
        and all(any(f["family_id"] == i for f in features) for i in pilot),
        token_alignment_checks=dict(passed=True, complete_answer_coverage=True),
    )


def worker(plan_path: Path, output: Path) -> None:
    """Reuse the existing CUDA lease and loader only inside this owned child."""
    import llama_cpp

    plan = json.loads(plan_path.read_text())
    deadline = time.monotonic() + LIMITS["seconds"]
    original_bound, original_lease = runtime.bounded, runtime.GpuLease.acquire
    panel: list[Json] = []

    def bound(call: Any, timeout: float) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model_work_budget")
        return original_bound(call, min(timeout, remaining))

    def prepare(public: Json, tokenizer: Any) -> Json:
        nonlocal panel
        panel = freeze(plan["originals"], tokenizer)
        atomic_json(output.parent / "frozen_panel.json", dict(rows=panel))
        return dict(rows=panel)

    def score(model: Any, checkpoints: Path) -> Json:
        rows = capture(panel, NativeController(model, deadline), checkpoints, deadline=deadline)
        reduced = reduce(panel, rows)
        return dict(reduced, rows=rows, passed=reduced["pilot_passed"], generated_tokens=0)

    class Lease:
        @staticmethod
        def acquire(**kwargs: Any) -> Any:
            return original_lease(**dict(kwargs, ttl_s=LIMITS["seconds"] + 120))

    with (
        patch.object(runtime, "TASK", TASK),
        patch.object(runtime, "GpuLease", Lease),
        patch.object(runtime, "bounded", bound),
        patch.object(base, "freeze_panel", prepare),
        patch.object(base, "qualify", score),
    ):
        progress("before_model_subprocess_work")
        runtime.worker(plan_path, output)
        progress("after_model_subprocess_work")
    result = json.loads(output.read_text())
    result.update(
        llama_cpp_build=dict(
            version=llama_cpp.__version__,
            native_library_sha256=sha256_file(Path(llama_cpp.llama_cpp._lib._name)),
        ),
        process=dict(pid=os.getpid(), executable=str(ROOT / ".venv/bin/python")),
        quantization="Q4_K_M",
        execution_shape="fresh_full",
        kernel_configuration=dict(
            n_ctx=6384, n_batch=256, n_ubatch=256, flash_attn=False, n_gpu_layers=-1, padding=False
        ),
    )
    atomic_json(output, result)


def byte_hash(value: bytes) -> str:
    """Bind actual bytes rather than their hexadecimal JSON spelling."""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def eligibility(plan: Json, seal: Json) -> Json:
    """Read complete original targets only after verifying the public feature seal."""
    if canonical_hash(seal["feature_rows"]) != seal["sha256"]:
        raise ValueError("public_feature_seal")
    items = {r["family_id"]: r for r in seal["panel"]}
    support = {}
    for role, refs in plan["target_references"].items():
        public = {
            r["family_id"]: r for r in json.loads(checked(refs["public"]).read_text())["rows"]
        }
        targets = {
            r["family_id"]: r for r in json.loads(checked(refs["evaluator"]).read_text())["rows"]
        }
        counts = {"0": 0, "1": 0}
        eligible_ids = []
        for f in (r for r in seal["feature_rows"] if r["role"] == role):
            item, p, y = items[f["family_id"]], public[f["family_id"]], targets[f["family_id"]]
            if any(p[k] != item[k] for k in ["source_bytes", "answer_bytes"]) or y[
                "response_sha256"
            ] != byte_hash(bytes.fromhex(item["answer_bytes"])):
                raise ValueError("original_answer_or_source_changed")
            target = y["eligible_y"] if y["completely_annotated"] and y["custody_passed"] else None
            if target in [0, 1]:
                counts[str(target)] += 1
                eligible_ids.append(f["family_id"])
        minimum, per_class = (48, 8) if role == "fit" else (24, 4)
        support[role] = dict(
            qualified_count=len(eligible_ids),
            class_counts=counts,
            eligible_ids=eligible_ids,
            minimum=minimum,
            minimum_per_class=per_class,
            passed=len(eligible_ids) >= minimum and min(counts.values()) >= per_class,
        )
    return support


def manifest(private: Path) -> list[Json]:
    """Reuse bounded validation with coverage limited to this producer and CLI."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(methods, "OWNED", OWNED), patch.object(methods, "TEST", TEST):
        specs = methods.manifest(private)
    for spec in specs:
        if spec["name"] == "consumer_tests":
            spec["argv"] += [
                "tests/python/test_fixture_consumer_contract_8057.py",
                "tests/python/test_scorer_workspace_8045.py",
                "tests/python/test_scoring_isolation_8033.py",
            ]
    return specs


def owned_receipt(plan: Json, measured: Json, raw: Path, work: Json) -> Json:
    """Keep canonical counters in a bound sidecar and separate operation semantics."""
    ledger = measured.get("current_invocation_ledger", [])
    owner = ledger[0]["owner_pid"] if ledger else plan["owner_pid"]
    events = [
        dict(
            call_id=r["call_id"],
            operation=r["operation"],
            scope="current",
            transport="owned_runtime",
            run_id=str(raw),
            owner_pid=owner,
            state=state,
            monotonic_ns=r[key],
        )
        for r in ledger
        for state, key in [
            ("attempted", "started_monotonic_ns"),
            (r["status"], "ended_monotonic_ns"),
        ]
    ]
    return build_current_work_receipt(
        run_id=str(raw),
        owner_pid=owner,
        events=events,
        inference_substrate="live_llm_embedding_extraction"
        if ledger
        else "aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            operation="teacher_forced_token_scoring", generated_tokens=0
        ),
        inference_substrate_class="model_load_no_generation" if ledger else "no_model_load",
        execution_venue="host",
        started_monotonic_ns=plan["started_monotonic_ns"],
        ended_monotonic_ns=work["ended_monotonic_ns"],
        phase_spans=work["phase_spans"],
    )


def build(plan: Json, measured: Json, raw: Path, validation: Json, work: Json) -> Json:
    """Qualification combines current tokens, independent support and owned checks."""
    panel = measured.get("panel", {}).get("rows", [])
    rows = measured.get("qualification", {}).get("rows", [])
    if not panel:
        panel = [dict(r, target_tokens=[], views={}) for r in plan["originals"]]
        rows = [
            dict(
                s,
                unit=s["id"],
                source=next(
                    r["source_cluster_id"] for r in panel if r["family_id"] == s["family_id"]
                ),
                seed=SEED,
                numerator=None,
                denominator=0,
                status="censored",
                exclusion_reason=None,
                token_rows=[],
            )
            for s in schedule(panel)
        ]
    reduced = reduce(panel, rows)
    support = work.get("support", {})
    receipts, coverage = validation["receipts"], validation["coverage"]
    checks_ok = (
        [r["name"] for r in receipts] == plan["validation_manifest"]
        and bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(
            p in coverage
            and coverage[p]["summary"]["num_statements"] > 0
            and not coverage[p]["missing_lines"]
            for p in OWNED
        )
    )
    gates = deepcopy(plan["failures"])
    for r in receipts:
        if not r["passed"]:
            gates.append(
                failure(Path(r["log_path"]), "exit_code", r["expected_exit"], r["exit_code"])
            )
    drifted = any(
        r["drift"] is not None and not r["passed"] for r in reduced["duplicate_drift_rows"]
    )
    for i, r in enumerate(reduced["duplicate_drift_rows"]):
        if r["drift"] is not None and not r["passed"]:
            gates.append(
                dict(
                    failure(
                        raw / "runtime.json",
                        f"qualification.duplicate_drift_rows[{i}].drift",
                        1e-6,
                        r["drift"],
                    ),
                    check="duplicate_mean_nll_drift",
                    op="<=",
                    family_id=r["family_id"],
                    arm=r["arm"],
                )
            )
    runtime_failed = bool(measured.get("checks"))
    for check in measured.get("checks", []):
        gates.append(
            failure(
                raw / "runtime.json", check["artifact_field"], check["expected"], check["observed"]
            )
        )
    ready = (
        checks_ok
        and not gates
        and reduced["pilot_passed"]
        and not drifted
        and all(support.get(r, {}).get("passed") for r in ["fit", "tune"])
        and not work["fixture"]
        and measured.get("model_identity_receipt", {}).get("hf_id") == runtime.MODEL
        and measured.get("offload_evidence", {}).get("supported") is True
        and measured.get("offload_evidence", {}).get("resident_memory_mb", 0) >= 10000
    )
    kind = (
        "disqualified"
        if drifted or (receipts and not checks_ok and not work["fixture"])
        else "blocked"
        if gates or runtime_failed
        else "circular_positive"
        if work["fixture"]
        else "null"
    )
    groups = []
    for item in panel:
        owned = [r for r in rows if r["family_id"] == item["family_id"]]
        status = (
            "excluded"
            if not item["eligible"]
            else "failed"
            if any(r["status"] == "failed" for r in owned)
            else "completed"
            if all(r["status"] == "completed" for r in owned)
            else "censored"
        )
        groups.append(
            dict(
                family_id=item["family_id"],
                source=item["source_cluster_id"],
                role=item["role"],
                slot=item["slot"],
                status=status,
                eligible=item["eligible"],
                exclusion_reason=item.get("exclusion_reason"),
            )
        )
    sizes = dict(
        intended_count=96,
        eligible_count=sum(r["eligible"] for r in groups),
        independent_count=len(reduced["feature_rows"]),
        completed_count=sum(r["status"] == "completed" for r in groups),
        censored_count=96 - len(groups) + sum(r["status"] == "censored" for r in groups),
        excluded_count=sum(r["status"] == "excluded" for r in groups),
        failed_count=sum(r["status"] == "failed" for r in groups),
    )
    current = owned_receipt(plan, measured, raw, work)
    ledger = measured.get("current_invocation_ledger", [])
    substrate = current["inference_substrate"]
    value = dict(
        experiment_id=8059,
        task_id=TASK,
        schema="carnot.v698.fit_source_scoring.v1",
        milestone="2026.10.698",
        run_date="20261003",
        honest_verdict="complete_blocked_" + str(gates[0]["field"]).replace(".", "_")
        if kind == "blocked" and gates
        else "complete_" + kind + "_fit_source_scoring",
        verdict_class=kind,
        verifier_is_oracle=work["fixture"],
        claim_scope="Current fixed-answer measurement qualification on historically exposed development sources. Repeatability adds no detection benefit. No generalized learning or hypothesis gain claimed.",
        flagged_adversarial=False,
        required_checks_passed=checks_ok,
        validation_receipts=receipts,
        gate_check_summary=gates,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        fit_capture_ready_score=int(ready),
        rows=rows,
        group_status_rows=groups,
        **sizes,
        sample_size_budget=dict(
            **sizes,
            unit="original independent source groups",
            limits=LIMITS,
            fit=64,
            tune=32,
            pilot=8,
        ),
        **reduced,
        support_by_role=support,
        model_identity_receipt=measured.get("model_identity_receipt", {}),
        gpu_lease_receipt=measured.get("gpu_lease_receipt", {}),
        offload_evidence=measured.get("offload_evidence", {}),
        model_invocation_counts={
            **{
                k: v
                for k, v in current["invocation_counts"].items()
                if not k.startswith("generation_calls_")
            },
            "generation": {
                k.removeprefix("generation_calls_"): v
                for k, v in current["invocation_counts"].items()
                if k.startswith("generation_calls_")
            },
        },
        model_invocation_counts_schema="carnot.operation_counters.separate_generation.v1",
        current_work_receipt=dict(
            reference(raw / "current_work_receipt.json"),
            receipt_payload_sha256=canonical_hash(current),
        ),
        preconditions_checked=[
            dict(
                ref,
                field="resource_exists_and_bound_hash",
                expected=True,
                observed=True,
                passed=True,
            )
            for ref in plan["references"]
        ]
        + measured.get("checks", []),
        inference_substrate=substrate,
        inference_substrate_class=current["inference_substrate_class"],
        MODEL_SPECS=[runtime.MODEL] if ledger else [],
        generated_tokens=0,
        substrate_declaration=dict(
            substrate=substrate,
            operation="teacher_forced_token_scoring",
            inference_mode="live_gpu" if ledger else "no_model_load",
            mode=current["inference_substrate_class"],
            duration_floor_s=2 if ledger else 0.0001,
        ),
        public_feature_seal=work["public_feature_seal"],
        source_artifact_hashes=plan["references"],
        raw_shard_hashes=[
            reference(raw / n)
            for n in [
                "plan.json",
                "runtime.json",
                "validation.json",
                "work.json",
                "validation_commands.json",
                "feature_seal.json",
                "current_work_receipt.json",
            ]
        ]
        + [reference(p) for p in sorted((raw / "forwards").glob("pass-*.json"))]
        + [
            reference(raw / n)
            for n in [
                "support.json",
                "model_child_receipt.json",
                "eligibility_receipt.json",
                "frozen_panel.json",
                "ledger.json",
            ]
            if (raw / n).is_file()
        ],
        code_config_hashes=plan["code_hashes"],
        measurement_code_hashes=plan.get("measurement_code_hashes", plan["code_hashes"]),
        reproducibility_checksum=canonical_hash(
            dict(plan=plan, reduced=reduced, work=work, validation=validation)
        ),
        random_seed=SEED,
        phase_spans=work["phase_spans"],
        duration_s=(work["ended_monotonic_ns"] - plan["started_monotonic_ns"]) / 1e9,
        coverage_statement_counts=coverage,
        repository_health=plan["repository_health"],
        generalized_learning_benefit_score=0,
        historical_evidence_policy="Exp8033 remains disqualified; imported current calls=0. Exp8057 fixture credit authorizes measurement only.",
        execution_configuration={
            k: measured.get(k)
            for k in [
                "llama_cpp_build",
                "process",
                "quantization",
                "execution_shape",
                "kernel_configuration",
            ]
        },
        methodology_note="Complete fixed supplied answers; independent fresh_full contexts, separated duplicates, exact preceding-token full-vocabulary float64 scoring. First eight fit groups are the pilot within the original384-call budget. No answer generation or retries.",
    )
    value["field_principles"] = {
        k: f"Bind {k} to current primitive evidence; prevent history, duplicate calls or private fixtures from adding scientific credit."
        for k in value
    }
    value["field_principles"].update(
        model_invocation_counts="Separate model loads from generation operations; zero sampled generation never denies observed model loading or supplied-token forwards.",
        current_work_receipt="Retain the canonical owned event ledger unchanged in an authenticated sidecar; operation-specific counters remain independently readable.",
        field_principles="Name the inference error each field prevents instead of turning metadata into scientific credit.",
        fit_capture_ready_score="Require qualified current groups, complete original targets and owned validation; never promote fixture readiness to science.",
        public_feature_seal="Prevent evaluator target access from changing selected sources or features.",
        generalized_learning_benefit_score="Finite historically exposed development sources cannot establish generalized lifelong learning.",
    )
    return value


def replay(path: Path) -> bool:
    """Cold-rebuild every public claim and authenticate its primitive dependencies."""
    try:
        v = json.loads(path.read_text())
        raw = Path(v["terminal_validation_sidecar_path"]).parent
        for ref in v["source_artifact_hashes"] + v["raw_shard_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        for name, digest in v["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        for r in v["validation_receipts"]:
            if sha256_file(Path(r["log_path"])) != r["log_sha256"]:
                return False
        plan, measured, validation, work = [
            json.loads((raw / n).read_text())
            for n in ["plan.json", "runtime.json", "validation.json", "work.json"]
        ]
        seal = json.loads((raw / "feature_seal.json").read_text())
        if (
            canonical_hash(json.loads(Path(v["current_work_receipt"]["path"]).read_text()))
            != v["current_work_receipt"]["receipt_payload_sha256"]
        ):
            return False
        if measured.get("panel"):
            calls = [
                json.loads(p.read_text()) for p in sorted((raw / "forwards").glob("pass-*.json"))
            ]
            if (
                calls != measured["qualification"]["rows"]
                or reduce(measured["panel"]["rows"], calls)["feature_rows"] != seal["feature_rows"]
            ):
                return False
        if (
            not work["fixture"]
            and not plan["failures"]
            and eligibility(plan, seal) != work["support"]
        ):
            return False
        return build(plan, measured, raw, validation, work) == v
    except (OSError, ValueError, KeyError, TypeError):
        return False


class FixtureController:
    """Known private scores exercise CLI custody without pretending to load Qwen."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, view: Json, condition: str) -> Json:
        self.calls += 1
        start = view["response_start"]
        return dict(
            target_logprobs=[-1.0] * (len(view["tokens"]) - start),
            conditional_logit_positions=list(range(start - 1, len(view["tokens"]) - 1)),
            normalization_max_error=0,
            context_identity=f"fresh-{self.calls}",
            context_closed=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Own bounded children, then publish one terminal identity after their exit."""
    os.environ.update(PYTHONUNBUFFERED="1", CARNOT_FORCE_LIVE="1")
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--eligibility", type=Path)
    parser.add_argument("--private-validation", action="store_true")
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    if args.worker:
        worker(args.worker, args.output)
        return 0
    if args.eligibility:
        plan = json.loads((args.eligibility.parent / "plan.json").read_text())
        atomic_json(args.output, eligibility(plan, json.loads(args.eligibility.read_text())))
        return 0
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem
    fixture = bool(args.fixture_input)
    private_route = fixture or args.private_validation
    if private_route and output.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("private_route_in_protected_results")
    if (raw / "plan.json").exists():
        progress("existing_run_preserved")
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="carnot-8059-") as temporary:
            private = Path(temporary)
            specs = manifest(private)
            py = str(ROOT / ".venv/bin/python")
            model_spec = dict(
                name="current_model_capture",
                argv=[
                    py,
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(raw / "plan.json"),
                    "--output",
                    str(raw / "runtime.json"),
                ],
                deadline_s=LIMITS["seconds"] + 60,
                expected_exit=0,
            )
            eligibility_spec = dict(
                name="independent_eligibility",
                argv=[
                    py,
                    "-u",
                    str(ROOT / CLI),
                    "--eligibility",
                    str(raw / "feature_seal.json"),
                    "--output",
                    str(raw / "support.json"),
                ],
                deadline_s=60,
                expected_exit=0,
            )
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=specs,
                    model_command=model_spec,
                    eligibility_command=eligibility_spec,
                    terminal_commands=[
                        dict(
                            name="cold_replay",
                            argv=[
                                py,
                                str(ROOT / CLI),
                                "--cold-replay",
                                str(raw / "terminal_candidate.json"),
                            ],
                        ),
                        dict(
                            name="adversarial",
                            argv=[
                                py,
                                str(ROOT / "scripts/adversarial_verify.py"),
                                "--json",
                                str(raw / "terminal_candidate.json"),
                            ],
                        ),
                        dict(
                            name="strict_rows",
                            argv=[
                                py,
                                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                                "--strict",
                                str(raw / "terminal_candidate.json"),
                            ],
                        ),
                    ],
                    terminal_deadline_s=60,
                ),
            )
            plan = (
                authenticate(args.root, raw)
                if not fixture
                else dict(
                    originals=json.loads(args.fixture_input.read_text())["panel"],
                    failures=[],
                    references=[],
                    target_references={},
                    repository_health=[],
                )
            )
            plan.update(
                owner_pid=os.getpid(),
                started_monotonic_ns=time.monotonic_ns(),
                validation_manifest=[s["name"] for s in specs],
                code_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST, *INPUTS[9:]]},
            )
            atomic_json(raw / "plan.json", plan)
            measured: Json = {}
            if fixture:
                p = plan["originals"]
                rows = capture(
                    p, FixtureController(), raw / "forwards", deadline=time.monotonic() + 60
                )
                if args.mutate:
                    rows[0]["token_rows"][0]["logit_position"] += 1
                measured = dict(panel=dict(rows=p), qualification=dict(reduce(p, rows), rows=rows))
            elif not plan["failures"]:
                progress("before_model_child", 0, 384)
                result = run_check(ROOT, model_spec, private, raw / "runtime_logs", heartbeat_s=30)
                measured = (
                    json.loads((raw / "runtime.json").read_text())
                    if (raw / "runtime.json").is_file()
                    else {}
                )
                if not result["passed"]:
                    plan["failures"].append(
                        failure(
                            Path(result["log_path"]), "model_child_exit", 0, result["exit_code"]
                        )
                    )
                atomic_json(raw / "model_child_receipt.json", result)
                progress(
                    "after_model_child",
                    measured.get("qualification", {}).get("forward_pass_counts", 0),
                    0,
                )
            atomic_json(raw / "runtime.json", measured)
            reduced = measured.get("qualification", {})
            feature_seal = dict(
                panel=measured.get("panel", {}).get("rows", []),
                feature_rows=reduced.get("feature_rows", []),
                sha256=canonical_hash(reduced.get("feature_rows", [])),
                sealed_monotonic_ns=time.monotonic_ns(),
            )
            atomic_json(raw / "feature_seal.json", feature_seal)
            support = {}
            if not private_route and not plan["failures"]:
                progress("before_independent_eligibility")
                result = run_check(ROOT, eligibility_spec, private, raw / "eligibility_logs")
                if result["passed"]:
                    support = json.loads((raw / "support.json").read_text())
                else:
                    plan["failures"].append(
                        failure(
                            Path(result["log_path"]), "eligibility_exit", 0, result["exit_code"]
                        )
                    )
                atomic_json(raw / "eligibility_receipt.json", result)
                progress("after_independent_eligibility")
            atomic_json(raw / "plan.json", plan)
            progress("owned_validation_before", 0, len(specs))
            os.environ["CARNOT_8059_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            receipts = (
                []
                if private_route
                else [run_check(ROOT, s, private, raw / "validation_logs") for s in specs]
            )
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            validation = dict(receipts=receipts, coverage=coverage)
            atomic_json(raw / "validation.json", validation)
            work = dict(
                fixture=fixture,
                support=support,
                public_feature_seal=reference(raw / "feature_seal.json"),
                ended_monotonic_ns=time.monotonic_ns(),
                phase_spans=[
                    dict(
                        name="current_capture_and_owned_validation",
                        duration_s=time.monotonic() - START,
                    )
                ],
            )
            atomic_json(raw / "work.json", work)
            atomic_json(raw / "current_work_receipt.json", owned_receipt(plan, measured, raw, work))
            value = build(plan, measured, raw, validation, work)
            progress("publication_before", len(value["feature_rows"]), 0)
            with patch.object(methods, "CLI", CLI):
                publication = publish_primary(output, value, methods.terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, owned_invocation_exit=0),
            )
            progress("complete", len(value["feature_rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error))
        return 1
