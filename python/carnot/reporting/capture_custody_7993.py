"""REQ-VERIFY-7993: recover observations without changing producer verdicts.

Exact external bytes establish custody. Model activity belongs to Exp7981;
this reducer only authenticates and joins its already completed observations.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.verify import qwen_stream_capture_7981 as capture
from scripts.adversarial_verify import _classify_current_task_inference_claim as classify

Json = dict[str, Any]


class CustodyError(ValueError):
    """Keep exact missing or conflicting operands for a terminal blocked result."""

    def __init__(self, check: Json):
        self.check = check
        super().__init__(str(check["field"]))


def require(item: Json, field: str, expected: Any, observed: Any) -> None:
    """A missing value stays None so absence cannot masquerade as zero work."""
    if observed != expected:
        raise CustodyError(
            dict(
                upstream_id="exp7981",
                path=item["path"],
                hash=item["sha256"],
                field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=False,
            )
        )


def checked(item: Json) -> Path:
    """Reject absent or replaced files before consuming any imported fields."""
    path = Path(item["path"])
    require(item, "sha256", item["sha256"], sha256_file(path) if path.is_file() else None)
    return path


def load_bundle(history: Json) -> Json:
    """Read all historical files through byte pins, including raw server logs."""
    refs = history["references"]
    data = {
        key: json.loads(checked(refs[key]).read_text())
        for key in ("primary", "candidate", "runtime", "plan", "upstream", "code_archive")
    }
    checked(refs["rejection_log"])
    candidate, upstream = data["candidate"], data["upstream"]
    imported = list(refs.values())
    for item in data["code_archive"]["references"]:
        checked(item)
        imported.append(dict(item, scope="historical"))

    def read(item: Json) -> Json:
        imported.append(dict(item, scope="historical"))
        return dict(json.loads(checked(item).read_text()))

    data["manifest"] = read(candidate["request_manifest"])
    data["rows"] = [read(item) for item in candidate["raw_response_shards"]]
    data["public"] = {role: read(item) for role, item in candidate["public_role_manifests"].items()}
    data["labels"] = {
        role: read(upstream["evaluator_role_manifests"][role]) for role in capture.ROLES
    }
    log = candidate["server_log"]
    data["server_log"] = checked(log).read_text()
    data["lifecycle_rows"] = [
        dict(
            scope="historical",
            logger_timestamp=line.split()[0],
            timestamp_kind="server_relative_display",
            event=line.split(" I ", 1)[1],
            server_log=log,
        )
        for line in data["server_log"].splitlines()
        if any(
            marker in line
            for marker in (
                " I load_tensors: loading model tensors,",
                " I load_tensors: offloaded ",
                " I srv  llama_server: server is listening ",
            )
        )
    ]
    imported.append(dict(log, scope="historical"))
    for key in ("resident_gpu_receipt", "unloaded_gpu_receipt"):
        receipt = candidate[key]
        item = dict(path=receipt["log_path"], sha256=receipt["log_sha256"], scope="historical")
        checked(item)
        imported.append(item)
    data.update(history=history, external_references=imported)
    return data


def reconstruct(history: Json) -> Json:
    """Convert malformed external contracts to exact terminal gate failures."""
    try:
        return reduce_bundle(load_bundle(history))
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, CustodyError):
            raise
        raise CustodyError(
            dict(
                upstream_id="exp7981",
                path="historical_receipt_manifest",
                hash=canonical_hash(history),
                field="contract:" + str(error),
                op="==",
                expected="complete typed custody",
                observed=None,
                passed=False,
            )
        ) from error


def _reduce_bundle(data: Json) -> Json:
    """Join every completion to full bytes, original role and one owned call."""
    history, candidate, runtime = data["history"], data["candidate"], data["runtime"]
    item = history["references"]["candidate"]

    def check(field: str, expected: Any, observed: Any) -> None:
        require(item, field, expected, observed)

    rows = data["rows"]
    check("slot_count", history["expected_slots"], len(rows))
    check("unique_family_ids", len(rows), len({r["family_id"] for r in rows}))
    check("request_manifest", capture.freeze(data["public"]), data["manifest"]["rows"])
    frozen = {r["family_id"]: r for r in data["manifest"]["rows"]}
    receipts = runtime["runtime_receipts"]
    receipt_map = {r["request_sha256"]: r for r in receipts}
    check("unique_receipts", len(receipts), len(receipt_map))
    check("receipt_count", history["expected_completions"], len(receipts))
    check("gguf_sha256", candidate["gguf_sha256"], runtime["gguf_sha256"])
    check("model_revision", candidate["model_revision"], runtime["model_revision"])
    identity = runtime["model_identity_receipt"]
    check("authenticated_model_identity", True, identity["authenticated"])
    check("identity_gguf", runtime["gguf_sha256"], identity["gguf_sha256"])
    check(
        "historical_load",
        (1, 1),
        (runtime["model_loads_attempted"], runtime["model_loads_completed"]),
    )
    check(
        "offload_layers", True, identity["offload_layers"][0] == identity["offload_layers"][1] > 0
    )
    check("server_offload_log", True, "offloaded" in data["server_log"])
    check(
        "producer_dates",
        ("20261001", "20261001"),
        (candidate["run_date"], data["upstream"]["run_date"]),
    )
    public = {r["family_id"]: r for view in data["public"].values() for r in view["request_rows"]}
    labels = {r["family_id"]: r for view in data["labels"].values() for r in view["rows"]}
    check("public_label_roster", set(public), set(labels))
    result, recovered, excluded, lineage = [], [], [], []
    tokens = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    by_role: Json = {role: [] for role in capture.ROLES}
    clusters: dict[str, str] = {}
    start, end = (datetime.fromisoformat(candidate[k]) for k in ("started_at", "finished_at"))
    for index, row in enumerate(rows):
        fid, role = row["family_id"], row["role"]
        check("slot_public_join:" + fid, frozen[fid], {k: row[k] for k in frozen[fid]})
        check("runtime_row:" + fid, runtime["rows"][index], row)
        source, answer = (bytes.fromhex(public[fid][k]) for k in ("source_bytes", "answer_bytes"))
        source_hash = "sha256:" + hashlib.sha256(source).hexdigest()
        answer_hash = "sha256:" + hashlib.sha256(answer).hexdigest()
        label = labels[fid]
        check(
            "label_join:" + fid,
            (role, source_hash, answer_hash),
            (label["role"], label["source_cluster_id"], label["response_sha256"]),
        )
        check("source_role:" + fid, role, clusters.setdefault(source_hash, role))
        parsed = capture.risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        check("parsed:" + fid, parsed, row["parsed"])
        primitive = dict(
            family_id=fid,
            source_cluster_id=source_hash,
            role=role,
            seed=candidate["random_seed"],
            numerator=int(parsed["completed"]),
            denominator=1,
            eligible=row["public_eligible"],
            started=row["started"],
            completed=parsed["completed"],
            failed=row["status"] == "failed",
            censored=row["status"] == "censored",
            excluded=row["status"] == "excluded",
        )
        result.append(primitive)
        if not parsed["completed"]:
            check("excluded_status:" + fid, (False, "excluded"), (row["started"], row["status"]))
            excluded.append(dict(primitive, reason=row["exclusion_reason"]))
        else:
            check(
                "completed_status:" + fid,
                (True, "generated", True, True),
                (
                    row["started"],
                    row["status"],
                    label["custody_passed"],
                    label["completely_annotated"],
                ),
            )
            begin, finish = (
                datetime.fromisoformat(row[k])
                for k in ("invocation_started_at", "invocation_finished_at")
            )
            check("completion_timestamps:" + fid, True, start <= begin <= finish <= end)
            request_hash, response_hash = (
                canonical_hash(row[k]) for k in ("request", "raw_response")
            )
            receipt = receipt_map[request_hash]
            check("response_receipt:" + fid, response_hash, receipt["response_sha256"])
            check("server_identity:" + fid, identity["owner"], receipt["server_identity"])
            check(
                "server_timestamps:" + fid,
                True,
                receipt["started_monotonic_ns"] < receipt["ended_monotonic_ns"],
            )
            usage = row["raw_response"]["usage"]
            check(
                "token_receipt:" + fid,
                (usage["prompt_tokens"], usage["completion_tokens"]),
                (receipt["input_tokens"], receipt["output_tokens"]),
            )
            check(
                "token_total:" + fid,
                usage["prompt_tokens"] + usage["completion_tokens"],
                usage["total_tokens"],
            )
            check("token_budget:" + fid, True, 0 < usage["completion_tokens"] <= 96)
            for key in tokens:
                tokens[key] += usage[key]
            human_request = json.loads(row["request"]["messages"][1]["content"])
            check(
                "complete_source_answer:" + fid,
                (source.decode(), answer.decode()),
                (human_request["complete_source"], human_request["original_answer"]),
            )
            joined = dict(
                primitive,
                probability=parsed["probability"],
                human_label=label["y"],
                response_id=label["response_id"],
                source_id=label["source_id"],
                source_sha256=source_hash,
                answer_sha256=answer_hash,
                public_sha256=canonical_hash(public[fid]),
                request_sha256=request_hash,
                response_sha256=response_hash,
                label_manifest=data["upstream"]["evaluator_role_manifests"][role],
                raw_record=candidate["raw_response_shards"][index],
            )
            recovered.append(joined)
            by_role[role].append(joined)
            lineage.append(
                dict(
                    scope="historical",
                    producer_id=7981,
                    producer_invocation_date=candidate["run_date"],
                    invocation_id=candidate["capture_identity"] + ":" + fid,
                    family_id=fid,
                    role=role,
                    server_pid=receipt["server_identity"]["pid"],
                    server_start_ticks=receipt["server_identity"]["start_time_ticks"],
                    request_sha256=request_hash,
                    response_sha256=response_hash,
                    invocation_started_at=row["invocation_started_at"],
                    invocation_finished_at=row["invocation_finished_at"],
                    input_tokens=receipt["input_tokens"],
                    output_tokens=receipt["output_tokens"],
                    model_revision=runtime["model_revision"],
                    gguf_sha256=runtime["gguf_sha256"],
                )
            )
        if (index + 1) % 32 == 0:
            print(f"[exp7993] phase=custody_rows completed_units={index + 1}", flush=True)
    check(
        "role_completion_counts",
        history["expected_role_counts"],
        dict(Counter(r["role"] for r in recovered)),
    )
    check(
        "historical_generation_counts",
        (len(recovered), len(recovered)),
        tuple(
            candidate["model_invocation_counts"][k]
            for k in ("generation_calls_attempted", "generation_calls_completed")
        ),
    )
    measured = capture.reduce(rows)
    check("historical_budget", measured["sample_size_budget"], candidate["sample_size_budget"])
    check(
        "historical_roles", measured["role_completion_counts"], candidate["role_completion_counts"]
    )
    diagnosis = classify(candidate)
    return dict(
        rows=result,
        recovered_role_rows=recovered,
        excluded_rows=excluded,
        invocation_scope_rows=lineage,
        by_role=by_role,
        historical_token_totals=tokens,
        sample_size_budget=measured["sample_size_budget"],
        role_completion_counts=measured["role_completion_counts"],
        raw_shard_hashes=data["external_references"],
        scope_contradiction_diagnosis=dict(
            scope="diagnostic",
            state=diagnosis["state"],
            negative_evidence=diagnosis["negative_evidence"],
            rejection="unscoped resumed zero counts classified as current task",
            resolution="current audit has zero calls; historical producer receipts remain external",
        ),
        original_verdict=dict(
            primary=data["primary"]["honest_verdict"], candidate=candidate["honest_verdict"]
        ),
        rejection_log_hash=history["references"]["rejection_log"]["sha256"],
        historical_model_custody=dict(
            scope="historical",
            lifecycle_rows=data["lifecycle_rows"],
            model_identity_receipt=identity,
            resident_gpu_receipt=candidate["resident_gpu_receipt"],
            unloaded_gpu_receipt=candidate["unloaded_gpu_receipt"],
            producer_started_at=candidate["started_at"],
            producer_finished_at=candidate["finished_at"],
            code_config_hashes=candidate["code_config_hashes"],
        ),
        historical_required_failures=candidate["historical_required_failures"],
    )


def reduce_bundle(data: Json) -> Json:
    """Name malformed joins as custody errors, including missing typed fields."""
    try:
        return _reduce_bundle(data)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, CustodyError):
            raise
        item = data["history"]["references"]["candidate"]
        raise CustodyError(
            dict(
                upstream_id="exp7981",
                path=item["path"],
                hash=item["sha256"],
                field="contract:" + str(error),
                op="==",
                expected="complete typed custody",
                observed=None,
                passed=False,
            )
        ) from error
