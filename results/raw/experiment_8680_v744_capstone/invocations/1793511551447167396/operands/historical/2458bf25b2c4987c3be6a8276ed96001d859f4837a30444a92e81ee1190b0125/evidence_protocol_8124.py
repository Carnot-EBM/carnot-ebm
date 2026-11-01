"""REQ-VERIFY-8124: seal source evidence without treating a quote as a proof.

The existing custody worker authenticates original roles and missing masks. This
module adds a public protocol freeze and independently expected runtime identity
before optional cache checks. It never loads a generator or assigns entailment.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.verify import methods_stream_custody_8111 as qualified

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8124_v703_evidence_protocol"
TASK = "exp8124-evidence-protocol"
MODULE = "python/carnot/verify/evidence_protocol_8124.py"
CLI = f"scripts/experiments/{NAME}.py"
RUNNER = MODULE
TEST = "tests/python/test_evidence_protocol_8124.py"
DESIGN = ROOT / "openspec/change-proposals/v703-evidence-protocol.md"
COHORT, STREAM = qualified.COHORT, qualified.STREAM
METHODS = "results/experiment_8111_v702_methods_and_stream_custody.json"
ACQUISITION = "results/experiment_8118_v702_fresh_acquisition_cost.json"
PINS = {
    STREAM: qualified.UPSTREAM_HASHES[8102],
    METHODS: "sha256:9f0c89a0d205488ceb545ca93497e6341bbc3fd8d01a94d69c773ed6535e9ece",
    ACQUISITION: "sha256:ffeff86ac75b20e12dc269c1db33c78764e3e24ccdf6d803cb65b12fde6b44ad",
}
OWNED = [MODULE, CLI]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counters so read-only work cannot masquerade as inference."""
    print(f"[exp8124] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Read the exact design before evaluator access; prompt bytes are evidence."""
    value: Json = json.loads(DESIGN.read_text().split("```json\n")[1].split("```", 1)[0])
    return value


def capture_seal(path: Path, value: Json, *, fixture: bool) -> Json:
    """Allow private mutation tests to reach replay while production stays sealed.

    A private fixture is byte-bound like production but must allow intentional
    damage. Reject repository result paths before granting that write permission.
    """
    if fixture and path.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("private_capture_required")
    reference = qualified.cohort.immutable(path, value)
    if fixture:
        path.chmod(0o600)
    return reference


def parse(transcript: str, source: bytes, arm: str) -> Json:
    """A valid substring shows byte custody; semantic entailment remains unknown.

    Bad quotes keep usable probabilities. A malformed probability makes the pair
    missing, so a parser cannot silently turn missing evidence into a true label.
    """
    result: Json = dict(
        status="missing_probability",
        p_hallucination=None,
        valid_quote=0,
        quote_source_byte_ratio=0.0,
        quote_status="absent",
        entailment_label=None,
    )
    try:
        value = json.loads(transcript)
        probability = value["p_hallucination"]
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            return result
    except (ValueError, KeyError, TypeError):
        return result
    result.update(status="parsed", p_hallucination=probability)
    if arm == "holistic" or value.get("quote") is None:
        return result
    quote, start, end = (value.get(k) for k in ("quote", "byte_start", "byte_end"))
    valid = (
        isinstance(quote, str)
        and bool(quote.strip())
        and len(quote.split()) <= 32
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(source)
        and source[start:end] == quote.encode("utf-8")
    )
    result.update(
        valid_quote=int(valid),
        quote_status="valid" if valid else "invalid",
        quote_source_byte_ratio=(end - start) / len(source) if valid else 0.0,
    )
    return result


def preconditions(
    root: Path, raw: Path, *, fixture: bool = False, model_path: Path | None = None
) -> Json:
    """Authenticate both upstream identities before inspecting any live cache.

    Expected values come solely from authenticated Exp8102 bytes. Exp8118 must
    independently agree, including its nested revision. No observed cache operand
    may fill an absent expected value. Optional failures cannot erase method work.
    """
    b = qualified.Custody(raw)
    values: Json = {}
    for name in PINS:
        b.upstream = name
        try:
            value = b.read(root / name, None if fixture else PINS[name])
            terminal = qualified.historical.terminal(root / name, value, b)
            b.require(root / name, "terminal.report.passed", True, terminal["report"].get("passed"))
            b.require(
                root / name, "required_checks_passed", True, value.get("required_checks_passed")
            )
            b.require(root / name, "flagged_adversarial", False, value.get("flagged_adversarial"))
            values[name] = value
        except (OSError, ValueError, KeyError) as error:
            progress("upstream_unavailable_" + Path(name).stem)
            if not b.failures:
                b.checks.append(
                    dict(
                        check="authenticated_terminal",
                        upstream=name,
                        path=str(root / name),
                        hash=None,
                        artifact_field="authenticated_terminal",
                        op="==",
                        expected=True,
                        observed=str(error),
                        passed=False,
                    )
                )
    history = values.get(STREAM, {})
    identity = history.get("runtime_identity", {})
    expected = dict(
        hf_id="unsloth/Qwen3.8-27B-GGUF",
        model_revision=identity.get("model_revision"),
        model_path=identity.get("model_path"),
        native_binary=identity.get("native_binary", {}),
        **{k: history.get(k) for k in ("gguf_sha256", "runtime_sha256", "chat_template_sha256")},
    )
    progress("expected_identity_frozen", 1)
    for name in (STREAM, ACQUISITION):
        v = values.get(name, {})
        observations = dict(
            model_revision=v.get("runtime_identity", {}).get("model_revision"),
            **{k: v.get(k) for k in ("gguf_sha256", "runtime_sha256", "chat_template_sha256")},
        )
        for key, observed in observations.items():
            field = "runtime_identity.model_revision" if key == "model_revision" else key
            check = dict(
                check=field,
                upstream=name,
                path=str(root / name),
                hash=PINS[name] if not fixture else None,
                artifact_field=field,
                op="==",
                expected=expected[key],
                observed=observed,
                passed=expected[key] is not None and observed == expected[key],
            )
            b.checks.append(check)
    progress("before_cache_identity")
    spec = cached_current_model() if model_path is None and not fixture else {}
    model = model_path or (
        Path(expected["model_path"])
        if fixture and expected["model_path"]
        else Path((spec or {}).get("model_path", raw / "missing-model"))
    )
    for key, observed in [
        ("cache_revision", model.parent.name),
        ("gguf_sha256", sha256_file(model) if model.is_file() else None),
    ]:
        expected_value = expected["model_revision" if key == "cache_revision" else key]
        b.checks.append(
            dict(
                check=key,
                upstream="current_cache",
                path=str(model),
                hash=observed,
                artifact_field=key,
                op="==",
                expected=expected_value,
                observed=observed,
                passed=expected_value is not None and observed == expected_value,
            )
        )
    binary = Path(expected["native_binary"].get("path", raw / "missing-runtime"))
    for key, wanted, observed in [
        ("runtime_executable", True, os.access(binary, os.X_OK)),
        (
            "runtime_sha256",
            expected["runtime_sha256"],
            sha256_file(binary) if binary.is_file() else None,
        ),
    ]:
        b.checks.append(
            dict(
                check=key,
                upstream="current_runtime",
                path=str(binary),
                hash=None,
                artifact_field=key,
                op="==",
                expected=wanted,
                observed=observed,
                passed=wanted is not None and wanted == observed,
            )
        )
    progress("after_cache_identity", 1)
    return dict(
        expected_runtime_identity=expected,
        checks=b.checks,
        references=b.refs,
        role_manifests=values.get(METHODS, {}).get("role_manifests", {}),
        cited_upstream_artifacts=[
            dict(
                path=str(root / n),
                sha256=PINS[n],
                historical_MODEL_SPECS=v.get("MODEL_SPECS", []),
                historical_model_invocation_counts=v.get("model_invocation_counts", {}),
            )
            for n, v in values.items()
        ],
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Freeze public rules first, then use the qualified original-custody worker.

    Runtime readiness is optional and cannot discard a sealed historical learning
    input. Reusing the original worker preserves its missing masks and assertions.
    """
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    config = protocol()
    freeze = qualified.cohort.immutable(
        raw / "v703_methods.json",
        dict(protocol=config, design_text=DESIGN.read_text(), design_sha256=sha256_file(DESIGN)),
    )
    progress("public_protocol_frozen", 1)
    plan = preconditions(root, raw, fixture=fixture)
    conformance = []
    for role in ("fit", "tune", "evaluation"):
        ref = plan["role_manifests"].get(role)
        if ref is None:
            continue
        view = json.loads(Path(ref["path"]).read_text())
        for meta, public in zip(view["roster"], view["request_rows"], strict=True):
            source, answer = (
                bytes.fromhex(public[k]).decode() for k in ("source_bytes", "answer_bytes")
            )
            arms = config["arms"][
                :: 1 if int(meta["source_cluster_id"].split(":")[-1], 16) % 2 == 0 else -1
            ]
            for order, arm in enumerate(arms):
                prompt = config["prompt_prefixes"][arm] + json.dumps(
                    dict(source=source, answer=answer), ensure_ascii=False
                )
                conformance.append(
                    dict(
                        unit_id=meta["unit_id"],
                        source_cluster_id=meta["source_cluster_id"],
                        role=role,
                        arm=arm,
                        order=order,
                        prompt=prompt,
                        prompt_sha256=canonical_hash(prompt),
                        source_bytes=public["source_bytes"],
                        answer_bytes=public["answer_bytes"],
                        max_tokens=128,
                        status="frozen_not_invoked",
                        entailment_label=None,
                    )
                )
        progress("frozen_" + role, len(conformance), 640 - len(conformance))
    manifest = capture_seal(
        raw / "evidence_capture_manifest.json", dict(rows=conformance), fixture=fixture
    )
    progress("source_and_arm_order_frozen", len(conformance))
    work = qualified.measure(root, raw, fixture=fixture, stream_path=stream_path, mutation=mutation)
    work.update(
        expected_runtime_identity=plan["expected_runtime_identity"],
        cited_upstream_artifacts=plan["cited_upstream_artifacts"],
        capture_protocol_ready_score=int(all(r["passed"] for r in plan["checks"])),
        method_freeze=freeze,
        evidence_protocol=config,
    )
    work["gate_check_summary"].extend(plan["checks"])
    work["source_artifact_hashes"].extend(plan["references"])
    work["protocol_conformance_rows"] = conformance
    work["raw_shard_hashes"].extend(
        [freeze, manifest, qualified.cohort.immutable(raw / "runtime_preconditions.json", plan)]
    )
    work["code_config_hashes"].update({n: sha256_file(ROOT / n) for n in [*OWNED, TEST]})
    work["source_artifact_hashes"].append(dict(path=str(DESIGN), sha256=sha256_file(DESIGN)))
    work["phase_spans"].append(
        dict(phase="v703_freeze_and_identity", start_s=0, duration_s=time.monotonic() - started)
    )
    work["duration_s"] = time.monotonic() - started
    atomic_json(raw / "measurement.json", work)
    progress("evidence_protocol_normal_exit", len(work["rows"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Separate three readiness gates; current execution always has zero calls."""
    value = qualified.build(work, raw, receipts, fixture=fixture)
    value.pop("reproducibility_checksum")
    value.update(
        experiment_id=8124,
        task_id=TASK,
        milestone="2026.10.703",
        capture_protocol_ready_score=int(
            value["required_checks_passed"] and work["capture_protocol_ready_score"]
        ),
        claim_scope="V703 protocol and authenticated historical input; no evidence-arm benefit measured",
        method_config=work["evidence_protocol"],
        historical_model_provenance_scope="cited_upstream_artifacts only",
    )
    value.pop("historical_model_provenance", None)
    if value["required_checks_passed"] and not value["capture_protocol_ready_score"]:
        failed = next(r for r in work["gate_check_summary"] if not r["passed"])
        value.update(verdict_class="blocked", honest_verdict="complete_blocked_" + failed["check"])
    elif value["required_checks_passed"] and value["verdict_class"] != "blocked":
        value["honest_verdict"] = (
            "complete_" + value["verdict_class"] + "_v703_evidence_protocol_sealed"
        )
    value["acceptance_gates"]["capture"] = (
        "authenticated expected identity before optional cache comparison"
    )
    value["field_principles"].update(
        capture_protocol_ready_score="Optional cache gate is independent of methods and historical stream.",
        protocol_conformance_rows="Frozen future requests are not model transcripts or current calls.",
        expected_runtime_identity="Authenticated upstream bytes set expected values before observations.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute row accounting and authenticate exact primitive and log bytes."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_text())
        for ref in [*work["source_artifact_hashes"], *work["raw_shard_hashes"]]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        for label, digest in work["code_config_hashes"].items():
            if sha256_file(ROOT / label) != digest:
                return False
        receipts = json.loads((raw / "validation_receipts.json").read_text())["rows"]
        for receipt in receipts:
            if (
                "log_path" in receipt
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        rebuilt = build(work, raw, receipts, fixture=value["fixture_protocol_only"])
        return canonical_hash(rebuilt) == canonical_hash(value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Use the qualified heartbeat, coverage and atomic terminal workflow."""
    import sys

    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        result: int = execution.main(argv)
    return result
