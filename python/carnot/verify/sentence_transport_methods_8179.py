"""REQ-REPORT-8179: freeze bounded evidence methods without neural execution.

Historical failed transport remains in custody. This run prepares the canary
and its gates; the separately scheduled live canary measures transport yield.
Original labels and statistical support remain prerequisites for later science.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import sentence_methods_8166 as methods
from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8179_v707_sentence_transport_methods"
TASK = "exp8179-sentence-transport-methods"
MODULE = "python/carnot/verify/sentence_transport_methods_8179.py"
NUMERIC = "python/carnot/verify/sentence_transport_8179.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_sentence_transport_methods_8179.py"
PARSER_TEST = "tests/python/test_sentence_transport_8179.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
PROTOCOL = "openspec/change-proposals/v707-sentence-transport-protocol.json"
PINS = {
    "results/experiment_8166_v706_sentence_evidence_methods.json": "sha256:339aa6acd1f91ca6a5f262b82e0f71716646c992c093cdb1a35d111f4c4d6405",
    "results/experiment_8167_v706_fit_sentence_capture.json": "sha256:5dc84701520534b187fa607409c1d4f7ed37e6a3b533f639728c6d3f40f2bcb4",
}
TOKENIZER_PATH = Path(
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q4_K_M.gguf"
)
TOKENIZER_PIN = "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
reference = methods.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase and completed source counts for bounded supervision."""
    print(f"[exp8179] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(plan: Json, path: Path, check: str, expected: Any, observed: Any) -> None:
    """Keep each failed external operand visible instead of marking work partial."""
    plan["checks"].append(
        dict(
            check=check,
            upstream=str(path),
            path=str(path.absolute()),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=check,
            op="==",
            expected=expected,
            observed=observed,
            passed=expected == observed,
        )
    )


def tokenizer(plan: Json, *, fixture: bool = False) -> tuple[Callable[[str], int], Json]:
    """Read only the embedded vocabulary; never load transformer weights.

    llama.cpp's vocabulary-only option provides the exact historical tokenizer.
    Fixture byte accounting is explicitly oracle transport plumbing and earns
    no actual tokenizer or model invocation credit.
    """
    if fixture:
        return lambda text: len(text.encode()), dict(
            status="fixture_byte_counter", neural_weights_loaded=False
        )
    progress("before_tokenizer_vocabulary_read")
    observed = sha256_file(TOKENIZER_PATH) if TOKENIZER_PATH.is_file() else None
    gate(plan, TOKENIZER_PATH, "tokenizer_gguf_sha256", TOKENIZER_PIN, observed)
    if observed != TOKENIZER_PIN:
        progress("after_tokenizer_vocabulary_read_blocked")
        return lambda text: len(text.encode()), dict(
            status="blocked", observed=observed, neural_weights_loaded=False
        )
    from llama_cpp import Llama

    try:
        vocab = Llama(
            model_path=str(TOKENIZER_PATH), vocab_only=True, n_gpu_layers=0, verbose=False
        )
    except (OSError, ValueError, RuntimeError) as error:
        gate(plan, TOKENIZER_PATH, "tokenizer_vocabulary_available", True, str(error))
        progress("after_tokenizer_vocabulary_read_blocked")
        return lambda text: len(text.encode()), dict(
            status="blocked", observed=str(error), neural_weights_loaded=False
        )
    progress("after_tokenizer_vocabulary_read", 1)
    return lambda text: len(vocab.tokenize(text.encode(), add_bos=False, special=False)), dict(
        status="actual_embedded_vocabulary",
        path=str(TOKENIZER_PATH),
        sha256=observed,
        vocab_only=True,
        neural_weights_loaded=False,
        add_bos=False,
        special=False,
        tokenizer_only_initializations=1,
    )


def inputs(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Reuse authenticated original-role manifests and preserve failed V706 bytes."""
    # The semantic audit shards contain reserved labels. Source preflight uses
    # public rosters and fit/tune baselines, so those audit shards stay closed.
    public_pins = {name: pin for name, pin in methods.PINS.items() if "8156" not in name}
    with patch.object(methods, "PINS", public_pins):
        plan = methods.inputs(root, raw, fixture=fixture)
    for name, pin in PINS.items():
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        gate(plan, path, "upstream_sha256", observed if fixture and observed else pin, observed)
        if observed is None or (not fixture and observed != pin):
            continue
        try:
            value = json.loads(path.read_text())
            gate(plan, path, "required_checks_passed", True, value.get("required_checks_passed"))
            gate(plan, path, "flagged_adversarial", False, value.get("flagged_adversarial"))
            if not fixture:
                terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
                passed = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))[
                    "report"
                ]["passed"]
                gate(plan, path, "terminal_passed", True, passed)
            for ref in [reference(path), *value["raw_shard_hashes"]]:
                original = Path(ref["path"])
                actual = sha256_file(original) if original.is_file() else None
                gate(plan, path, "primitive_sha256", ref["sha256"], actual)
                if actual == ref["sha256"]:
                    target = raw / "inputs" / (actual.split(":")[-1] + "-" + original.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original.read_bytes())
                    plan["refs"].append(reference(target))
            plan["upstream"].append(
                dict(
                    experiment_id=int(path.name.split("_")[1]),
                    fields_imported=["raw_shard_hashes", "historical_failure"],
                    sha256=observed,
                )
            )
            plan["provenance"].append(
                dict(
                    path=str(path),
                    MODEL_SPECS=value.get("MODEL_SPECS", []),
                    model_invocation_counts=value.get("model_invocation_counts", {}),
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            gate(plan, path, "upstream_structure", "authenticated complete receipt", str(error))
    return plan


PROTOCOL_PIN = "sha256:12af37eda61cdab32f6841afcdf9bafddf7c633f4b90c15106ada9aab2d85540"


def freeze_sources(sources: list[Json], count: Callable[[str], int]) -> list[Json]:
    """Keep every original source, including sources that cannot fit two requests.

    The output bound is checked before any future capture. Counts refer to
    source clusters, so sentence groups cannot create extra statistical support.
    """
    rows = []
    cache: Json = {}
    for i, source in enumerate(sources):
        request = transport.requests(source, count)
        budgets = []
        for call in request["requests"]:
            key = str((call["sentence_indices"], len(request["source_segments"])))
            if key not in cache:
                cache[key] = transport.output_budget(
                    call["sentence_indices"], len(request["source_segments"]), count
                )
            budgets.append(cache[key])
        if any(b["maximum_encoded_output_tokens"] > 256 for b in budgets):
            request.update(status="escalated", exclusion_reason="output_token_limit", requests=[])
        ready = request["status"] == "completed"
        rows.append(
            dict(
                source,
                **request,
                output_budgets=budgets,
                arm="sentence_transport_protocol",
                condition="original",
                metric="lossless_request_ready",
                numerator=int(ready),
                denominator=1,
            )
        )
        rows[-1]["status"] = "completed" if ready else "excluded"
        if (i + 1) % 8 == 0:
            progress("source_preflight", i + 1, len(sources) - i - 1)
    return rows


def reduce_rows(rows: list[Json]) -> Json:
    """Recount original units independently and retain role-specific eligibility."""
    statuses = Counter(r["status"] for r in rows)
    roles = {
        role: dict(
            intended=total,
            observed=sum(r["role"] == role for r in rows),
            eligible=sum(r["role"] == role and r["status"] == "completed" for r in rows),
        )
        for role, total in methods.ROLES.items()
    }
    return dict(
        intended_count=320,
        eligible_count=statuses["completed"],
        independent_count=len({r["source_cluster_id"] for r in rows}),
        completed_count=statuses["completed"],
        excluded_count=statuses["excluded"],
        censored_count=0,
        failed_count=320 - len(rows),
        eligibility_by_role=roles,
        maximum_encoded_output_tokens=max(
            (b["maximum_encoded_output_tokens"] for r in rows for b in r["output_budgets"]),
            default=0,
        ),
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Freeze bounded requests and support before any reserved labels are opened.

    The optional stream argument belongs to the reused command runner. Sentence
    methods use only the authenticated public source manifests, never that stream.
    """
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    progress("before_input_authentication")
    plan = inputs(root, raw, fixture=fixture)
    config = json.loads((ROOT / PROTOCOL).read_text())
    gate(plan, ROOT / PROTOCOL, "protocol_sha256", PROTOCOL_PIN, sha256_file(ROOT / PROTOCOL))
    if mutation:
        gate(plan, root, "private_" + mutation + "_tamper", "unchanged original bytes", mutation)
    count, token_receipt = tokenizer(plan, fixture=fixture)
    progress("before_transport_preflight", 0, len(plan["sources"]))
    rows = freeze_sources(plan["sources"], count)
    reduced = reduce_rows(rows)
    gate(
        plan,
        ROOT / PROTOCOL,
        "reserved_support_reachable",
        True,
        reduced["eligibility_by_role"]["evaluation"]["eligible"] >= config["H1"]["minimum_sources"],
    )
    baseline = methods.select_control(plan["baselines"])
    if not fixture:
        gate(
            plan,
            ROOT / PROTOCOL,
            "original_tune_control",
            config["H1"]["selected_control"],
            baseline["selected_control"],
        )
        ready_ids = {
            r["unit_id"] for r in rows if r["role"] == "fit" and r["status"] == "completed"
        }
        gate(
            plan,
            ROOT / PROTOCOL,
            "canary_preflight_source_count",
            24,
            sum(x in ready_ids for x in config["canary"]["source_ids"]),
        )
    atomic_json(raw / "sentence_requests.json", dict(rows=rows))
    atomic_json(raw / "independent_reduction.json", reduced)
    atomic_json(raw / "baseline_manifest.json", baseline)
    work = dict(
        plan=plan,
        rows=rows,
        config=config,
        baseline=baseline,
        fixture=fixture,
        tokenizer_receipt=token_receipt,
        raw_shard_hashes=[
            reference(raw / p)
            for p in (
                "sentence_requests.json",
                "independent_reduction.json",
                "baseline_manifest.json",
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in (
                *OWNED,
                TEST,
                PARSER_TEST,
                PROTOCOL,
                methods.MODULE,
                methods.NUMERIC,
                "python/carnot/reporting/methods_stream_execution_8111.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v686_contract_validation.py",
            )
        ],
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_tokenize_and_freeze",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_transport_preflight", len(rows), 320 - len(rows))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Grant methods readiness only after owned validation and source preflight.

    Live grammar yield is measured later by Exp8181. Historical failure alone
    cannot block a different bounded method, and syntax never grants science credit.
    """
    value = methods.build(work, raw, receipts, fixture=fixture)
    value.update(
        experiment_id=8179,
        task_id=TASK,
        run_date=RUN_DATE,
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        canary_source_ids=work["config"]["canary"]["source_ids"],
        canary_plan=work["config"]["canary"],
        tokenizer_receipt=work["tokenizer_receipt"],
        source_partition_rows=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                source_segments=r["source_segments"],
                sentences=r["sentences"],
            )
            for r in work["rows"]
        ],
        decision_costs=work["config"]["decision_costs"],
        method_to_task_note=work["config"]["method_to_task_note"],
        acceptance_gates=dict(
            H1=work["config"]["H1"],
            canary=work["config"]["canary"],
            owned="normal required exits and100 percent added statements",
        ),
        methodology_note="Frozen lossless short indexed records and actual vocabulary-only token bounds. Original roles, exposed history, tune-selected V705 control and H1 support retained. Zero current neural model calls; live paired canary and semantic evaluation are separate future tasks.",
        **reduce_rows(work["rows"]),
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild requests from frozen original bytes and reject altered custody.

    Cached headlines are not trusted. Replay recounts source units and redoes
    grammar preflight with the same vocabulary-only tokenizer or private fixture.
    """
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if work["config"] != json.loads((ROOT / PROTOCOL).read_text()):
            return False
        sources = []
        for role in methods.ROLES:
            if role not in work["plan"]["role_refs"]:
                continue
            ref = work["plan"]["role_refs"][role]
            manifest_value = json.loads(Path(ref["path"]).read_text())
            public = {r["family_id"]: r for r in manifest_value["request_rows"]}
            sources += [dict(r, **public[r["unit_id"]]) for r in manifest_value["roster"]]
        if work["plan"]["baseline_ref"]:
            primitive = json.loads(Path(work["plan"]["baseline_ref"]["path"]).read_text())
            if primitive["rows"] != work["plan"]["baselines"]:
                return False
        count, token_receipt = tokenizer(dict(checks=[]), fixture=work["fixture"])
        if (
            freeze_sources(sources, count) != work["rows"]
            or token_receipt != work["tokenizer_receipt"]
        ):
            return False
        if methods.select_control(work["plan"]["baselines"]) != work["baseline"]:
            return False
        return bool(
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


BASE_MANIFEST = execution.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze exact tool paths before measurement; static tools receive files.

    The qualified supervisor seals completed logs and pulses during child waits.
    Full repository health is recorded once, separately from the owned checks.
    """
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].append(PARSER_TEST)
    specs["commands"][1]["name"] = "private_E2E010_015_019"
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        PARSER_TEST,
        "tests/python/test_arc_tool_grammar_transport.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    coverage = candidate.parent / "coverage" / (private.name + ".json")
    coverage.parent.mkdir(parents=True, exist_ok=True)
    specs["commands"][4]["argv"][-1] = str(coverage)
    for index in (5, 6, 8):
        specs["commands"][index]["argv"].append(PARSER_TEST)
    specs["repository_health"]["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """Run the thin dated CLI through qualified no-model checked publication."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
    ):
        return execution.main(argv)
