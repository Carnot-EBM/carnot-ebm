"""CPU qualification for a frozen Qwen confidence protocol. REQ-REPORT-7770."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib import request as urlrequest

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7770_v676_qwen_runner_qualification")
PROTOCOL = RAW / "protocol.json"
PANEL = Path("results/raw/experiment_7745_v674_qwen_localization/frozen_panel.json")
OLD_ROWS = Path("results/raw/experiment_7759_v675_qwen_evidence_views/rows.jsonl")
OLD_RESULT = Path("results/experiment_7759_v675_qwen_evidence_views.json")
SERVER_SOURCE = Path.home() / ".cache/llama.cpp-master/tools/server/server-common.cpp"
RESULT = Path("results/experiment_7770_v676_qwen_runner_qualification.json")
ARMS = ("generic", "event")
REQUIRED_CHECKS = {
    "worktree_imports",
    "focused_pytest",
    "full_python_suite",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "private_cli_e2e",
    "independent_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "strict_row_consistency",
}
FIELD_PRINCIPLES = {
    "experiment_id": "One current task owns this artifact.",
    "honest_verdict": "A terminal record must not retry unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence cannot open downstream gates.",
    "gate_check_summary": "Exact operands distinguish missing producers from failed thresholds.",
    "rows": "Raw units make aggregates independently recomputable.",
    "acceptance_gate_results": "Protocol execution is not scientific benefit.",
    "duration_s": "Timing describes measured work without padding.",
    "random_seed": "A third party needs the same inputs and parameters.",
    "sample_size_budget": "Repeated arms do not increase family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with an older result.",
    "preconditions_checked": "Custody and resources precede work.",
    "validation_receipts": "Every registered check must pass before readiness.",
    "verifier_is_oracle": "Fixture truth is circular evidence.",
    "inference_substrate_class": "A fixture server performs no model inference.",
    "qwen_runner_ready_score": "Live capture needs a tested producer.",
    "qwen_protocol_path": "Arm definitions must precede observation.",
}


def sha(path: Path) -> str:
    """Bind each receipt to exact bytes, including historical raw replies."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Show elapsed work at each boundary so a silent child cannot appear healthy."""
    print(
        f"[exp7770] {phase} {event} elapsed_s={time.monotonic() - start:.2f} "
        f"completed_units={units}",
        flush=True,
    )


def make_request(protocol: dict[str, Any], family: dict[str, Any], arm: str) -> dict[str, Any]:
    """Change only the target question; preserve both original input fields."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    question = protocol["prompts"][arm] + "\n" + protocol["common_instruction"]
    visible = {"complete_source": family["source"], "original_answer": family["answer"]}
    return {
        **protocol["request"],
        "messages": [
            {"role": "system", "content": question},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
        ],
    }


def parse_probability(text: str, finish: str, arm: str) -> dict[str, Any]:
    """Invalid output must escalate and must never vanish from the denominator."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    value = data.get("probability") if isinstance(data, dict) else None
    ids = data.get("evidence_sentence_ids", []) if isinstance(data, dict) else []
    valid = (
        finish == "stop"
        and isinstance(data, dict)
        and set(data) in ({"probability"}, {"probability", "evidence_sentence_ids"})
        and type(value) in (int, float)
        and math.isfinite(value)
        and 0 <= value <= 1
        and isinstance(ids, list)
        and all(type(i) is int and i >= 0 for i in ids)
        and len(ids) == len(set(ids))
    )
    risk = (1 - float(value) if arm == "generic" else float(value)) if valid else 0.5
    return {
        "valid": bool(valid),
        "probability": float(value) if valid else None,
        "unsupported_risk": risk,
        "forced_escalation": not valid,
        "evidence_sentence_ids": ids if valid else [],
        "finish_reason": finish,
    }


def old_parse_counts(path: Path) -> dict[str, int]:
    """Read historical dispositions directly; the old aggregate is not an oracle."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    grouped: dict[str, dict[str, bool]] = {}
    counts = {arm: 0 for arm in ("canonical", "paired", "source_withheld")}
    for row in rows:
        parsed = row["metrics"]["parse_valid"]
        counts[row["arm"]] += int(parsed)
        grouped.setdefault(row["family_id"], {})[row["arm"]] = parsed
    counts["both_complete_source_families"] = sum(
        bool(pair.get("canonical") and pair.get("paired")) for pair in grouped.values()
    )
    counts["denominator"] = len(grouped)
    return counts


def save_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    """Publish one complete checkpoint after rejecting duplicate pair keys."""
    keys = [(row["family_id"], row["arm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_checkpoint_key")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(rows, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    """A missing checkpoint represents zero completed units."""
    return json.loads(path.read_text()) if path.exists() else []


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count families once even though each family has two repeated views."""
    keys = [(row["family_id"], row["arm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_pair")
    return {
        "effective_independent_n": len({row["family_id"] for row in rows}),
        "completed_calls": len(rows),
        "forced_escalations": sum(row["metrics"]["forced_escalation"] for row in rows),
        "valid_by_arm": {
            arm: sum(row["arm"] == arm and row["metrics"]["valid"] for row in rows) for arm in ARMS
        },
    }


def check_identity(url: str, model: str) -> bool:
    """A healthy foreign server cannot qualify the pinned model endpoint."""
    with urlrequest.urlopen(url + "/v1/models", timeout=2) as response:
        roster = json.load(response)
    return [item["id"] for item in roster["data"]] == [model]


def chat_stream(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[str, str]:
    """Read actual SSE chunks and require a terminal finish marker."""
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    req = urlrequest.Request(
        url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"}
    )
    parts: list[str] = []
    finish = "missing"
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    break
                event = json.loads(raw)
                if event.get("model") != payload["model"]:
                    raise ValueError("foreign_response_model")
                choice = event["choices"][0]
                parts.append(choice["delta"].get("content", ""))
                finish = choice.get("finish_reason") or finish
    except TimeoutError as exc:
        raise TimeoutError("chat_timeout") from exc
    return "".join(parts), finish


class FixtureHandler(BaseHTTPRequestHandler):
    """CPU server checks the exact schema before returning bounded SSE."""

    def log_message(self, *_args: Any) -> None:
        """Avoid writing raw source text to the service log."""

    def do_GET(self) -> None:
        """Expose the pinned model identifier without loading model bytes."""
        self._send_json({"data": [{"id": "unsloth/Qwen3.8-27B-GGUF"}]})

    def _send_json(self, value: dict[str, Any]) -> None:
        """Respond with complete bytes so identity reads cannot truncate."""
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        """Reject a request if its grammar field could be silently ignored."""
        value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        grammar = value.get("response_format", {})
        schema = grammar.get("json_schema", {}).get("schema", {})
        if (
            grammar.get("type") != "json_schema"
            or schema.get("required") != ["probability"]
            or schema.get("additionalProperties") is not False
        ):
            self.send_error(400, "strict schema required")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in ('{"prob', 'ability":0.7}'):
            event = {
                "model": value["model"],
                "choices": [{"delta": {"content": chunk}, "finish_reason": None}],
            }
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        event = {"model": value["model"], "choices": [{"delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def gate(
    name: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Record both operands so a missing producer has an explicit cause."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha(path) if path.is_file() else None,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(
    root: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the declared producer separate from its conductor panel receipt."""
    producer = root / "results/experiment_7745_v674_qwen_localization.json"
    panel_path = root / PANEL
    old_path = root / OLD_ROWS
    old_result_path = root / OLD_RESULT
    checks = [
        gate("producer_exists", "exp7745", producer, "exists", True, producer.is_file()),
        gate("panel_exists", "exp7745_pre_gate", panel_path, "exists", True, panel_path.is_file()),
        gate("historical_rows_exist", "exp7759_raw", old_path, "exists", True, old_path.is_file()),
        gate(
            "historical_result_exists",
            "exp7759_diagnostic",
            old_result_path,
            "exists",
            True,
            old_result_path.is_file(),
        ),
        gate(
            "server_source_exists",
            "llama_cpp_server",
            SERVER_SOURCE,
            "exists",
            True,
            SERVER_SOURCE.is_file(),
        ),
    ]
    if any(not item["passed"] for item in checks):
        return [], checks
    old = json.loads(producer.read_text())
    diagnostic = json.loads(old_result_path.read_text())
    server_source = SERVER_SOURCE.read_text()
    checks.append(
        gate(
            "schema_request_format",
            "llama_cpp_server",
            SERVER_SOURCE,
            "response_format.json_schema.schema",
            True,
            'response_type == "json_schema"' in server_source
            and 'json_value(schema_wrapper, "schema"' in server_source,
        )
    )
    for field, expected in (
        ("milestone", "2026.09.674"),
        ("qwen_localization_complete_score", 1),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
    ):
        checks.append(gate("producer_field", "exp7745", producer, field, expected, old.get(field)))
    checks.append(
        gate(
            "panel_hash",
            "exp7745_pre_gate",
            panel_path,
            "sha256",
            protocol["panel_sha256"],
            sha(panel_path),
        )
    )
    panel = json.loads(panel_path.read_text())
    checks.append(
        gate(
            "family_ids",
            "exp7745_pre_gate",
            panel_path,
            "family_ids",
            protocol["family_ids"],
            [row["family_id"] for row in panel],
        )
    )
    counts = old_parse_counts(old_path)
    checks.append(
        gate(
            "raw_row_hash",
            "exp7759_diagnostic",
            old_result_path,
            "source_artifact_hashes.pre_gate_receipts.rows_jsonl",
            diagnostic.get("source_artifact_hashes", {})
            .get("pre_gate_receipts", {})
            .get(str(OLD_ROWS)),
            sha(old_path),
        )
    )
    checks.append(
        gate(
            "diagnostic_disposition",
            "exp7759_diagnostic",
            old_result_path,
            "verdict_class",
            "disqualified",
            diagnostic.get("verdict_class"),
        )
    )
    checks.append(
        gate(
            "old_parse_counts",
            "exp7759_raw",
            old_path,
            "counts",
            {
                "canonical": 1,
                "paired": 2,
                "source_withheld": 9,
                "both_complete_source_families": 1,
                "denominator": 24,
            },
            counts,
        )
    )
    return panel, checks


def cold_replay(path: Path) -> dict[str, Any]:
    """A fresh reader reopens request and reply bytes and recomputes metrics."""
    artifact = json.loads(path.read_text())
    unit_rows = [row for row in artifact["rows"] if row.get("arm") in ARMS]
    for row in unit_rows:
        request_path = Path(row["raw_request_path"])
        response_path = Path(row["raw_response_path"])
        if sha(request_path) != row["raw_request_sha256"]:
            raise ValueError("request_hash_changed")
        if sha(response_path) != row["raw_response_sha256"]:
            raise ValueError("response_hash_changed")
        reply = json.loads(response_path.read_text())
        if parse_probability(reply["text"], reply["finish"], row["arm"]) != row["metrics"]:
            raise ValueError("row_metrics_changed")
    reduced = reduce_rows(unit_rows)
    if reduced != artifact["reduced"]:
        raise ValueError("reduction_changed")
    return reduced


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Exercise the frozen request over CPU HTTP and retain every pair row."""
    start = time.monotonic()
    spans: list[dict[str, Any]] = []
    progress(start, "preflight", "before", 0)
    begin = time.monotonic()
    protocol_path = root / PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    panel, checks = preflight(root, protocol)
    spans.append(
        {
            "phase": "preflight",
            "duration_s": time.monotonic() - begin,
            "completed_units": len(checks),
        }
    )
    progress(start, "preflight", "after", len(checks))
    failed = [item for item in checks if not item["passed"]]
    rows: list[dict[str, Any]] = []
    raw = (
        (output.parent / "exp7770-private-raw")
        if output != root / RESULT
        else root / RAW / "runs" / date
    )
    if not failed:
        progress(start, "server", "before", 0)
        begin = time.monotonic()
        server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            if not check_identity(url, protocol["request"]["model"]):
                raise ValueError("fixture_server_identity")
            bad = dict(protocol["request"])
            bad["response_format"] = {"type": "json_object"}
            try:
                chat_stream(url, bad, 2)
            except Exception:
                pass
            else:
                raise ValueError("grammar_field_ignored")
            progress(start, "server", "after", 1)
            spans.append(
                {"phase": "server", "duration_s": time.monotonic() - begin, "completed_units": 1}
            )
            begin = time.monotonic()
            checkpoint = raw / "checkpoint.json"
            rows = load_checkpoint(checkpoint)
            done = {(row["family_id"], row["arm"]) for row in rows}
            for family in panel:
                for arm in ARMS:
                    key = (family["family_id"], arm)
                    if key in done:
                        continue
                    request = make_request(protocol, family, arm)
                    progress(start, "generation", "before", len(rows))
                    reply, finish = chat_stream(url, request, 5)
                    progress(start, "generation", "after", len(rows) + 1)
                    raw.mkdir(parents=True, exist_ok=True)
                    stem = family["family_id"].replace(":", "_") + "_" + arm
                    req_path = raw / (stem + "_request.json")
                    reply_path = raw / (stem + "_response.json")
                    req_path.write_text(json.dumps(request, ensure_ascii=False, sort_keys=True))
                    reply_path.write_text(json.dumps({"text": reply, "finish": finish}))
                    rows.append(
                        {
                            "family_id": family["family_id"],
                            "arm": arm,
                            "seed": protocol["request"]["seed"],
                            "disposition": "completed",
                            "excluded": False,
                            "censored": finish != "stop",
                            "label": bool(family["annotation_types"]),
                            "raw_request_path": str(req_path),
                            "raw_request_sha256": sha(req_path),
                            "raw_response_path": str(reply_path),
                            "raw_response_sha256": sha(reply_path),
                            "metrics": parse_probability(reply, finish, arm),
                        }
                    )
                    save_checkpoint(checkpoint, rows)
            spans.append(
                {
                    "phase": "generation",
                    "duration_s": time.monotonic() - begin,
                    "completed_units": len(rows),
                }
            )
        finally:
            progress(start, "teardown", "before", len(rows))
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            progress(start, "teardown", "after", len(rows))
    reduced = reduce_rows(rows)
    receipt_path = root / RAW / "validation_receipts.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    required = receipts.get("required_commands", [])
    validated = bool(
        {item.get("name") for item in required} == REQUIRED_CHECKS
        and all(item.get("exit_code") == 0 for item in required)
        and receipts.get("coverage_percent") == 100
        and receipts.get("flagged_adversarial") is False
    )
    ready = int(not failed and validated and len(rows) == 48)
    verdict = (
        "complete_blocked_required_upstream"
        if failed
        else "complete_circular_positive_cpu_runner"
        if ready
        else "complete_disqualified_required_checks"
    )
    sources = [
        {
            "path": str(path),
            "sha256": sha(root / path) if (root / path).is_file() else None,
            "date": date,
            "imported_fields": fields,
            "eligible": (root / path).is_file(),
        }
        for path, fields in (
            (PROTOCOL, ["family_ids", "request", "prompts", "reducer"]),
            (PANEL, ["family_id", "source", "answer"]),
            (OLD_ROWS, ["metrics.parse_valid"]),
            (OLD_RESULT, ["verdict_class", "source_artifact_hashes"]),
            (SERVER_SOURCE, ["response_format.json_schema.schema"]),
            (
                Path("results/experiment_7745_v674_qwen_localization.json"),
                ["milestone", "verdict_class", "qwen_localization_complete_score"],
            ),
        )
    ]
    gate_result = lambda passed: {
        "passed": passed,
        "principle": "A CPU fixture proves execution, not natural benefit.",
    }
    artifact = {
        "experiment_id": "exp7770-qwen-runner-qualification",
        "milestone": "2026.09.676",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if failed else "circular_positive" if ready else "disqualified",
        "flagged_adversarial": bool(receipts.get("flagged_adversarial", False)),
        "gate_check_summary": failed,
        "rows": rows
        if not failed
        else [
            {
                "kind": "precondition",
                "disposition": "blocked",
                "check": item["check"],
                "upstream_id": item["upstream_id"],
                "excluded": True,
                "censored": False,
            }
            for item in checks
        ],
        "reduced": reduced,
        "acceptance_gate_results": {
            name: gate_result(value)
            for name, value in (
                ("validity", not failed),
                ("readiness", bool(ready)),
                ("probability_quality", None),
                ("decision_benefit", None),
                ("retention", None),
                ("efficiency", None),
            )
        },
        "duration_s": time.monotonic()
        - start
        + sum(item.get("duration_s", 0) for item in required),
        "phase_spans": spans
        + [
            {
                "phase": item["name"],
                "duration_s": item.get("duration_s", 0),
                "completed_units": int(item.get("exit_code") == 0),
            }
            for item in required
        ],
        "random_seed": protocol["request"]["seed"],
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(
                [sha(protocol_path), sha(Path(__file__)), protocol["request"], protocol["prompts"]],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "sample_size_budget": {
            "intended": 24,
            "eligible": len(panel) if not failed else 0,
            "started": reduced["effective_independent_n"],
            "completed": reduced["effective_independent_n"],
            "excluded": 0,
            "censored": reduced["forced_escalations"],
            "effective_independent_n": reduced["effective_independent_n"],
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "claim_scope": "CPU fixture conformance; exposed development panel; no semantic benefit",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0},
        "qwen_runner_ready_score": ready,
        "qwen_protocol_path": str(PROTOCOL),
        "old_parser_counts": old_parse_counts(root / OLD_ROWS) if not failed else None,
        "server_contract": "CPU SSE fixture; llama.cpp server-common.cpp response_format json_schema wrapper",
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Keep the production entrypoint limited to CPU qualification and replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    output = args.output or ROOT / RESULT
    artifact = run_experiment(ROOT, args.date, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return 0
