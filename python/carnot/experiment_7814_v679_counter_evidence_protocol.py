"""Current CPU counter-evidence protocol and bounded validation dispatcher.

REQ-REPORT-7814 and REQ-REPORT-7814-INTERVENTION. A cited source sentence is
a proposed witness. This task prepares prompts and tests edits without inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from carnot import experiment_7800_v678_counter_evidence_protocol as prior
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify.source_alignment import sentence_spans

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7814_v679_counter_evidence_protocol"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
COMMAND_MANIFEST = RAW / "validation_command_manifest.json"
COMMAND_MANIFEST_SHA256 = "sha256:1c7774d813c614ae53ebe5b3a0af051ef84a1cf94bbc9183245fb62329e223f8"
SEED = 67815
ARMS = prior.ARMS
MODEL_SPECS: list[dict[str, Any]] = []
GGUF_SHA256 = "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
GGUF_PATH = Path(
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q4_K_M.gguf"
)
SYSTEM = prior.SYSTEM


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Expose real elapsed work so a silent child cannot hide a stall."""
    print(
        f"[exp7814] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


digest = prior.digest
sentence_offsets = prior.sentence_offsets
remove_sentence = prior.remove_sentence


def preflight(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Open only authenticated public input before witness selection."""
    rows, checks, hashes = prior.preflight(root)
    manifest_path = root / prior.MANIFEST
    evaluator_path = manifest_path.parent / "evaluation_evaluator.jsonl"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("roles", {}).get("evaluation", {}).get("evaluator_sha256")
        checks.append(
            prior.check(
                "exp7727_evaluator", evaluator_path, "exists", True, evaluator_path.is_file()
            )
        )
        if evaluator_path.is_file():
            checks.append(
                prior.check(
                    "exp7727_evaluator",
                    evaluator_path,
                    "evaluator_sha256",
                    expected,
                    sha256_file(evaluator_path),
                )
            )
    hashes["evaluator"] = {
        "path": str(evaluator_path),
        "sha256": sha256_file(evaluator_path) if evaluator_path.is_file() else None,
        "date": "20260926",
        "imported_fields": ["annotations", "family_id"],
        "eligible": all(item["passed"] for item in checks),
    }
    return (rows if all(item["passed"] for item in checks) else []), checks, hashes


def freeze_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose exposed sources by hash without consulting evaluator labels."""
    if len(rows) != 64 or len({row["source_sha256"] for row in rows}) != 64:
        raise ValueError("evaluation64_invalid")
    return sorted(rows, key=lambda row: digest(f"{SEED}:{row['source_sha256']}".encode()))[:48]


def target_span(answer: bytes) -> dict[str, Any]:
    """Bind the elicited event to the first unchanged answer sentence bytes."""
    parts = sentence_spans(answer)
    if not parts:
        raise ValueError("empty_answer")
    return {"start_byte": 0, "end_byte": len(parts[0]), "text_sha256": digest(parts[0])}


def aligned_label(row: dict[str, Any], evaluator: dict[str, Any]) -> int | None:
    """An overlapping independent error span identifies a positive event only."""
    end = target_span(row["complete_response"].encode())["end_byte"]
    for annotation in evaluator.get("annotations", []):
        if annotation.get("start", end) < end and annotation.get("end", 0) > 0:
            return 1
    return None


def make_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze every model-visible choice before a later GPU capture."""
    return {
        "schema": "carnot.exp7814.counter_evidence_protocol.v1",
        "seed": SEED,
        "family_ids": [row["family_id"] for row in rows],
        "input_hashes": {
            row["family_id"]: {
                "source_sha256": row["source_sha256"],
                "answer_sha256": row["response_sha256"],
            }
            for row in rows
        },
        "system_instruction": SYSTEM,
        "request": {**prior.make_protocol([])["request"], "seed": SEED},
        "context_ceiling_tokens": 8192,
        "context_bound_method": "authenticated GGUF tokenizer and chat template with injected counter",
        "canaries": [{"name": "start", "max_tokens": 32}, {"name": "end", "max_tokens": 32}],
        "arms": list(ARMS),
        "matching_rule": "disjoint original sentence; GGUF token-count difference <=25 percent; minimum seeded SHA-256 order",
        "bootstrap": {
            "unit": "family",
            "families": 48,
            "draws": 10000,
            "seed": SEED,
            "contrast": "p(witness_removed)-p(unrelated_removed)",
        },
        "coverage_rules": {
            "minimum_matched_families": 30,
            "minimum_matched_fraction": 0.8,
            "minimum_mean_shift": 0.05,
            "paired_lower95_above_zero": True,
        },
        "labels_available_during_selection": False,
    }


def visible_after(
    source: bytes, offsets: list[dict[str, Any]], deleted: int
) -> tuple[bytes, list[dict[str, Any]]]:
    """Delete one span and retain each other sentence's original identifier."""
    edited, span = remove_sentence(source, offsets, deleted)
    length = span["end_byte"] - span["start_byte"]
    visible = [
        {
            **item,
            "start_byte": item["start_byte"]
            - (length if item["source_sentence_id"] > deleted else 0),
            "end_byte": item["end_byte"] - (length if item["source_sentence_id"] > deleted else 0),
        }
        for item in offsets
        if item["source_sentence_id"] != deleted
    ]
    return edited, visible


def select_control(
    source: bytes,
    offsets: list[dict[str, Any]],
    witness: int,
    family_id: str,
    token_count: Callable[[str], int],
) -> int | None:
    """Select a disjoint length match from label-free source bytes."""
    if witness not in [item["source_sentence_id"] for item in offsets]:
        return None
    sizes = {
        item["source_sentence_id"]: token_count(
            source[item["start_byte"] : item["end_byte"]].decode()
        )
        for item in offsets
    }
    target = sizes[witness]
    options = [
        item
        for item, size in sizes.items()
        if item != witness and target > 0 and 4 * abs(size - target) <= target
    ]
    return (
        min(options, key=lambda item: digest(f"{SEED}:{family_id}:{item}".encode()))
        if options
        else None
    )


def make_request(
    row: dict[str, Any],
    source: bytes,
    offsets: list[dict[str, Any]],
    arm: str,
    frozen: dict[str, Any],
    token_count: Callable[[str], int],
) -> dict[str, Any]:
    """Build the actual HTTP body and reject offset or context drift."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    if b"".join(source[item["start_byte"] : item["end_byte"]] for item in offsets) != source or any(
        digest(source[item["start_byte"] : item["end_byte"]]) != item["text_sha256"]
        for item in offsets
    ):
        raise ValueError("invalid_offsets")
    body = {
        "complete_source": source.decode(),
        "original_answer": row["complete_response"],
        "source_sentence_offsets": offsets,
        "target_sentence_span": target_span(row["complete_response"].encode()),
    }
    messages = [
        {"role": "system", "content": frozen["system_instruction"]},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
    ]
    payload = {**frozen["request"], "messages": messages}
    prompt_tokens = (
        token_count.count_messages(messages)
        if hasattr(token_count, "count_messages")
        else token_count(json.dumps(messages, ensure_ascii=False))
    )
    if prompt_tokens + payload["max_tokens"] > frozen["context_ceiling_tokens"]:
        raise ValueError("context_budget")
    return payload


def parse_reply(text: str, finish: str, visible_ids: list[int]) -> dict[str, Any]:
    """Require an exact probability object and a visible original source ID."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if (
        finish != "stop"
        or not isinstance(data, dict)
        or set(data) != {"unsupported_probability", "source_sentence_id"}
        or type(data["unsupported_probability"]) not in (int, float)
        or not 0 <= data["unsupported_probability"] <= 1
        or type(data["source_sentence_id"]) is not int
    ):
        return {
            "disposition": "invalid_parse",
            "unsupported_probability": None,
            "source_sentence_id": None,
        }
    witness = data["source_sentence_id"]
    return {
        "disposition": "completed" if witness in visible_ids else "invalid_witness",
        "unsupported_probability": float(data["unsupported_probability"]),
        "source_sentence_id": witness,
    }


def capture_fixture(
    row: dict[str, Any],
    frozen: dict[str, Any],
    transport: Callable[[dict[str, Any]], dict[str, Any]],
    token_count: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Exercise real request construction with scripted HTTP, without inference."""
    source = row["complete_source"].encode()
    offsets = sentence_offsets(source)
    base = {
        "family_id": row["family_id"],
        "source_sha256": row["source_sha256"],
        "answer_sha256": row["response_sha256"],
        "prior_exposure": row.get("previously_exposed", True),
        "original_label": None,
        "excluded": False,
        "censored": False,
    }

    def call(
        arm: str, edited: bytes, visible: list[dict[str, Any]], deleted: int | None
    ) -> dict[str, Any]:
        try:
            payload = make_request(row, edited, visible, arm, frozen, token_count)
        except ValueError as error:
            return {
                **base,
                "arm": arm,
                "disposition": "unstarted_" + str(error),
                "probability": None,
                "deleted_sentence_id": deleted,
            }
        response = transport(payload)
        choice = response["choices"][0]
        parsed = parse_reply(
            choice["message"]["content"],
            choice.get("finish_reason"),
            [item["source_sentence_id"] for item in visible],
        )
        return {
            **base,
            "arm": arm,
            **parsed,
            "probability": parsed["unsupported_probability"],
            "deleted_sentence_id": deleted,
            "request_sha256": digest(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ),
            "response_sha256": digest(
                json.dumps(response, sort_keys=True, ensure_ascii=False).encode()
            ),
            "usage": response.get("usage", {}),
        }

    if not offsets:
        return [
            {
                **base,
                "arm": arm,
                "disposition": "unstarted_empty_source",
                "probability": None,
                "deleted_sentence_id": None,
            }
            for arm in ARMS
        ]
    intact = call("intact", source, offsets, None)
    witness = intact.get("source_sentence_id") if intact["disposition"] == "completed" else None
    if witness is None:
        return [
            intact,
            *[
                {
                    **base,
                    "arm": arm,
                    "disposition": "unstarted_invalid_witness",
                    "probability": None,
                    "deleted_sentence_id": None,
                }
                for arm in ARMS[1:]
            ],
        ]
    treated, treated_offsets = visible_after(source, offsets, witness)
    treatment = call("witness_removed", treated, treated_offsets, witness)
    control_id = select_control(source, offsets, witness, row["family_id"], token_count)
    if control_id is None:
        control = {
            **base,
            "arm": "unrelated_removed",
            "disposition": "unmatched_control",
            "probability": None,
            "deleted_sentence_id": None,
        }
    else:
        controlled, controlled_offsets = visible_after(source, offsets, control_id)
        control = call("unrelated_removed", controlled, controlled_offsets, control_id)
    return [intact, treatment, control]


def reduce_pilot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score only independently labeled intact rows and paired families."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["arm"] != "intact" and row.get("original_label") is not None:
            raise ValueError("modified_source_label")
        pair = grouped.setdefault(row["family_id"], {})
        if row["arm"] in pair:
            raise ValueError("duplicate_arm")
        pair[row["arm"]] = row
    matched = [
        pair
        for pair in grouped.values()
        if all(pair.get(arm, {}).get("disposition") == "completed" for arm in ARMS)
    ]
    labeled = [
        pair["intact"]
        for pair in grouped.values()
        if pair.get("intact", {}).get("disposition") == "completed"
        and pair["intact"].get("original_label") in (0, 1)
    ]
    return {
        "independent_n": len(grouped),
        "matched_n": len(matched),
        "matched_coverage": len(matched) / len(grouped) if grouped else None,
        "aligned_label_count": len(labeled),
        "intact_brier": sum((row["probability"] - row["original_label"]) ** 2 for row in labeled)
        / len(labeled)
        if labeled
        else None,
        "modified_source_labels_applied": 0,
        "paired_mean_shift": sum(
            pair["witness_removed"]["probability"] - pair["unrelated_removed"]["probability"]
            for pair in matched
        )
        / len(matched)
        if matched
        else None,
    }


def load_command_manifest() -> dict[str, Any]:
    """Refuse a command list changed after the task requirement was sealed."""
    if sha256_file(COMMAND_MANIFEST) != COMMAND_MANIFEST_SHA256:
        raise ValueError("validation_manifest_drift")
    return json.loads(COMMAND_MANIFEST.read_text())


def validate_log_receipt(receipt: dict[str, Any]) -> None:
    """One changed byte invalidates a child receipt after it is sealed."""
    path = Path(receipt["log_path"])
    if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
        raise ValueError("validation_log_drift")


def execute_child(command: dict[str, Any], index: int, scope: dict[str, Any]) -> dict[str, Any]:
    """Close the owned child log, then copy its bytes once to durable storage."""
    private = Path(scope["private_root"])
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    (Path(command["private_root"]) / "basetemp").mkdir(parents=True, exist_ok=True)
    for arg in command["argv"]:
        if arg.startswith("--basetemp="):
            Path(arg.partition("=")[2]).parent.mkdir(parents=True, exist_ok=True)
    spec = CommandSpec(
        command["name"],
        tuple(command["argv"]),
        command["classification"],
        timeout_s=float(command["timeout_s"]),
    )
    receipt = run_commands(
        ROOT,
        [spec],
        log_dir=Path(command["private_root"]) / "logs",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )[0]
    source = ROOT / receipt["log_path"]
    digest_value = sha256_file(source)
    destination = (
        Path(scope["raw_root"])
        / "validation_logs"
        / f"{index:02d}_{command['name']}_{digest_value[7:]}.log"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("validation_log_path_reused")
    shutil.copyfile(source, destination)
    receipt.update(
        classification=command["classification"],
        log_path=str(destination),
        log_sha256=sha256_file(destination),
    )
    validate_log_receipt(receipt)
    return receipt


def dispatch(
    scope: dict[str, Any],
    executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Execute the complete frozen list, including terminal readers."""
    if scope != load_command_manifest() or len(scope["commands"]) != 15:
        raise ValueError("validation_manifest_drift")
    (Path(scope["private_root"]) / "basetemp").mkdir(parents=True, exist_ok=True)
    chosen = executor or execute_child
    receipts = []
    for index, command in enumerate(scope["commands"]):
        receipt = chosen(command, index, scope)
        if (receipt["name"], receipt["command_argv"], receipt["classification"]) != (
            command["name"],
            command["argv"],
            command["classification"],
        ):
            raise ValueError("observed_child_command_drift")
        validate_log_receipt(receipt)
        receipts.append(receipt)
    return receipts


class GGUFTokenCounter:
    """Use GGUF vocabulary metadata without loading generator weights."""

    def __init__(self, path: Path, expected_hash: str):
        if sha256_file(path) != expected_hash:
            raise ValueError("gguf_hash_mismatch")
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self.vocab = Llama(model_path=str(path), vocab_only=True, n_gpu_layers=0, verbose=False)
        template = self.vocab.metadata["tokenizer.chat_template"]
        previous = json.loads(
            (ROOT / "results/experiment_7787_v677_qwen_event_confidence.json").read_text()
        )
        expected_template = previous["current_model_receipts"]["server_props"]["chat_template"]
        if (
            hashlib.sha256(template.encode()).digest()
            != hashlib.sha256(expected_template.encode()).digest()
        ):
            raise ValueError("gguf_template_mismatch")
        self.template_sha256 = digest(template.encode())
        self.formatter = Jinja2ChatFormatter(
            template,
            eos_token=self.vocab.metadata.get("tokenizer.ggml.eos_token_id", "<|im_end|>"),
            bos_token="<|im_start|>",
        )

    def __call__(self, text: str) -> int:
        """Count actual GGUF tokens for a candidate source sentence."""
        return len(self.vocab.tokenize(text.encode(), add_bos=False, special=True))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Render the authenticated chat template before context counting."""
        rendered = self.formatter._environment.render(
            messages=messages,
            eos_token=self.formatter.eos_token,
            bos_token=self.formatter.bos_token,
            add_generation_prompt=True,
            raise_exception=lambda message: (_ for _ in ()).throw(ValueError(message)),
            tools=None,
            tool_choice=None,
            functions=None,
            function_call=None,
        )
        return len(self.vocab.tokenize(rendered.encode(), add_bos=False, special=True))


def resource_checks() -> list[dict[str, Any]]:
    """Check CPU and disk before opening a large tokenizer vocabulary."""
    free = shutil.disk_usage(ROOT).free
    checks = [
        prior.check(
            "exp7814_resource", ROOT, "disk_free_at_least_100mb", True, free >= 100_000_000
        ),
        prior.check(
            "exp7814_resource", ROOT, "cpu_backend", "cpu", os.environ.get("JAX_PLATFORMS", "cpu")
        ),
    ]
    checks.append(prior.check("exp7787_tokenizer", GGUF_PATH, "exists", True, GGUF_PATH.is_file()))
    if GGUF_PATH.is_file():
        checks.append(
            prior.check(
                "exp7787_tokenizer", GGUF_PATH, "gguf_sha256", GGUF_SHA256, sha256_file(GGUF_PATH)
            )
        )
    return checks


def build_family_manifest(
    selected: list[dict[str, Any]], counter: Callable[[str], int]
) -> dict[str, Any]:
    """Keep all 48 original source and answer byte strings, even if oversized."""
    families = []
    for row in selected:
        source = row["complete_source"].encode()
        offsets = sentence_offsets(source)
        families.append(
            {
                "family_id": row["family_id"],
                "complete_source": row["complete_source"],
                "complete_response": row["complete_response"],
                "source_sha256": row["source_sha256"],
                "answer_sha256": row["response_sha256"],
                "prior_exposure": True,
                "source_role": "evaluation",
                "source_sentence_offsets": offsets,
                "source_sentence_token_counts": {
                    str(item["source_sentence_id"]): counter(
                        source[item["start_byte"] : item["end_byte"]].decode()
                    )
                    for item in offsets
                },
                "target_sentence_span": target_span(row["complete_response"].encode()),
            }
        )
    return {
        "schema": "carnot.exp7814.family_manifest.v1",
        "seed": SEED,
        "families": families,
        "selection": "minimum 48 seeded source hashes from 64 evaluation families",
        "labels_available_during_selection": False,
    }


def cold_reduce(candidate_path: Path) -> dict[str, Any]:
    """Reopen sealed protocol bytes and recompute all planned family identities."""
    candidate = json.loads(candidate_path.read_text())
    manifest_path = Path(candidate["family_manifest_path"])
    protocol_path = Path(candidate["counter_evidence_protocol_path"])
    family_manifest = json.loads(manifest_path.read_text())
    frozen = json.loads(protocol_path.read_text())
    families = family_manifest["families"]
    if [row["family_id"] for row in families] != frozen["family_ids"] or len(families) != 48:
        raise ValueError("family_manifest_drift")
    for row in families:
        if (
            digest(row["complete_source"].encode()) != row["source_sha256"]
            or digest(row["complete_response"].encode()) != row["answer_sha256"]
            or sentence_offsets(row["complete_source"].encode()) != row["source_sentence_offsets"]
            or target_span(row["complete_response"].encode()) != row["target_sentence_span"]
        ):
            raise ValueError("family_bytes_drift")
    if len(candidate["rows"]) != 144:
        raise ValueError("planned_rows_drift")
    return {
        "families": len(families),
        "arms": len(candidate["rows"]),
        "family_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
    }


def artifact(
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    selected: list[dict[str, Any]],
    protocol_path: Path | None,
    manifest_path: Path | None,
    fixture_rows: list[dict[str, Any]],
    aligned_rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    started: float,
    flagged: bool = False,
) -> dict[str, Any]:
    """Keep scientific gates separate from validation and raw protocol custody."""
    failures = [item for item in checks if not item["passed"]]
    required_failed = any(
        row["classification"] == "required" and not row["passed"] for row in receipts
    )
    blocked = bool(failures)
    ready = (
        bool(selected)
        and not blocked
        and not required_failed
        and not flagged
        and len(receipts) == 15
    )
    verdict_class = "blocked" if blocked else "circular_positive" if ready else "disqualified"
    honest_verdict = (
        "complete_blocked_missing_external_input"
        if blocked
        else "complete_circular_positive_protocol_fixture"
        if ready
        else "complete_disqualified_required_validation"
    )
    rows = [
        {
            "family_id": row["family_id"],
            "arm": arm,
            "seed": SEED,
            "disposition": "unstarted_no_model_invocation",
            "probability": None,
            "original_label": next(
                (
                    label["label"]
                    for label in aligned_rows
                    if label["family_id"] == row["family_id"]
                ),
                None,
            )
            if arm == "intact"
            else None,
            "source_sha256": row["source_sha256"],
            "answer_sha256": row["response_sha256"],
            "prior_exposure": True,
            "excluded": False,
            "censored": False,
        }
        for row in selected
        for arm in ARMS
    ]
    if required_failed:
        failures += [
            prior.check(
                "exp7814_validation",
                Path(row["log_path"]),
                row["name"] + ".exit_code",
                0,
                row["exit_code"],
            )
            for row in receipts
            if row["classification"] == "required" and not row["passed"]
        ]
    identity = {
        "code": sha256_file(Path(__file__)),
        "wrapper": sha256_file(ROOT / "scripts/experiments" / f"{NAME}.py"),
        "inputs": hashes,
        "roles": "evaluation48_of_64",
        "configuration": sha256_file(protocol_path) if protocol_path else None,
        "manifest": sha256_file(COMMAND_MANIFEST),
        "seed": SEED,
    }
    result = {
        "schema": "carnot.exp7814.counter_evidence_protocol_result.v1",
        "experiment_id": "exp7814-counter-evidence-protocol",
        "milestone": "2026.09.679",
        "run_date": "20260928",
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "gate_check_summary": failures,
        "rows": rows,
        "acceptance_gate_results": {
            "validity": 1 if ready else 0,
            "readiness": 1 if ready else 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": SEED,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "sample_size_budget": {
            "intended": 48,
            "eligible": len(selected),
            "started": 0,
            "completed": 0,
            "excluded": 0,
            "censored": 0,
            "independent_n": len(selected),
            "planned_panel_calls": len(selected) * 3,
            "planned_canary_calls": 2,
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "claim_scope": "Exposed development fixture only; no hidden generalization or oracle-distinct benefit.",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "loaded_file_hashes": []},
        "counter_evidence_ready_score": int(ready),
        "counter_evidence_protocol_path": str(protocol_path) if protocol_path else None,
        "family_manifest_path": str(manifest_path) if manifest_path else None,
        "intervention_fixture_rows": fixture_rows,
        "target_event": "unsupported_first_original_answer_sentence",
        "target_sentence_spans": [
            {"family_id": row["family_id"], **target_span(row["complete_response"].encode())}
            for row in selected
        ],
        "aligned_label_rows": aligned_rows,
        "aligned_label_count": sum(row["label"] is not None for row in aligned_rows),
        "validation_command_manifest_path": str(COMMAND_MANIFEST),
        "validation_command_manifest_sha256": sha256_file(COMMAND_MANIFEST),
        "observed_child_commands": [
            {
                "name": row["name"],
                "argv": row["command_argv"],
                "classification": row["classification"],
            }
            for row in receipts
        ],
        "repository_health": {
            "historical_exp7800_verdict": "complete_disqualified_required_checks",
            "historical_full_python_suite_exit": -15,
            "current_diagnostic_exit": next(
                (row["exit_code"] for row in receipts if row["name"] == "repository_health"), None
            ),
            "current_diagnostic_timeout": next(
                (row["timed_out"] for row in receipts if row["name"] == "repository_health"), None
            ),
            "classification": "diagnostic",
        },
    }
    result["field_principles"] = {
        key: rationale
        for key, rationale in {
            "experiment_id": "Each result has one owner.",
            "milestone": "Bind the current task.",
            "run_date": "Bind the current task.",
            "honest_verdict": "External absence has a terminal blocked record.",
            "verdict_class": "Claim strength travels with the record.",
            "flagged_adversarial": "Invalid evidence cannot open a gate.",
            "gate_check_summary": "A failed operand differs from a scientific null.",
            "rows": "Recompute comparisons from units.",
            "acceptance_gate_results": "Fixture success does not establish benefit.",
            "duration_s": "Use measured work.",
            "phase_spans": "Use measured work.",
            "random_seed": "Pairing must replay.",
            "reproducibility_checksum": "Bind exact code and inputs.",
            "sample_size_budget": "Views are not new families.",
            "source_artifact_hashes": "Old files cannot replace producers.",
            "preconditions_checked": "Cheap failures precede compute.",
            "validation_receipts": "Every required check must pass.",
            "verifier_is_oracle": "Fixtures are circular evidence.",
            "claim_scope": "Exposed data do not prove generalization.",
            "inference_substrate": "Floors follow invoked work.",
            "inference_substrate_class": "No generator was loaded.",
            "planned_inference_substrate_class": "Blocked work retains its plan.",
            "MODEL_SPECS": "A cited model is not invoked.",
            "model_specs": "No loaded model is claimed.",
            "model_invocation_counts": "Report actual work.",
            "counter_evidence_ready_score": "Interventions need known input changes.",
            "counter_evidence_protocol_path": "Freeze prompts before outcomes.",
            "family_manifest_path": "Keep every planned family.",
            "intervention_fixture_rows": "Syntax can remove wrong evidence.",
            "target_event": "Probabilities need an exact event.",
            "target_sentence_spans": "Freeze exact answer bytes.",
            "aligned_label_rows": "Only aligned intact labels count.",
            "aligned_label_count": "Unknown labels stay unscored.",
            "validation_command_manifest_path": "Freeze dispatch prospectively.",
            "validation_command_manifest_sha256": "Detect command edits.",
            "observed_child_commands": "Reject hidden children.",
            "repository_health": "Broad failures stay visible and diagnostic.",
        }.items()
    }
    return result


def fixture_e2e(path: Path) -> dict[str, Any]:
    """Drive the same HTTP builder with an explicit fixture token counter."""
    source = "Café one here. Other two now. Third nice here."
    answer = "Café is here. This answer stays."
    row = {
        "family_id": "fixture-e2e",
        "complete_source": source,
        "complete_response": answer,
        "source_sha256": digest(source.encode()),
        "response_sha256": digest(answer.encode()),
        "previously_exposed": True,
    }

    def count(text: str) -> int:
        return {"Café one here. ": 4, "Other two now. ": 4, "Third nice here.": 4}.get(text, 20)

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        body = json.loads(payload["messages"][1]["content"])
        witness = body["source_sentence_offsets"][0]["source_sentence_id"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"unsupported_probability": 0.4, "source_sentence_id": witness}
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        }

    rows = capture_fixture(row, make_protocol([row]), transport, count)
    if [item["disposition"] for item in rows] != ["completed"] * 3 or rows[1][
        "deleted_sentence_id"
    ] != 0:
        raise ValueError("fixture_intervention_failed")
    atomic_json(path, {"rows": rows, "reduced": reduce_pilot(rows)})
    return {"rows": rows, "reduced": reduce_pilot(rows)}


def run_experiment(
    date: str,
    child_executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare real input custody, validate exact children, and publish once."""
    started = time.monotonic()
    progress(started, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    scope = load_command_manifest()
    attempt_raw = Path(scope["raw_root"])
    if attempt_raw.exists():
        raise ValueError("attempt_root_reused")
    attempt_raw.mkdir(parents=True)
    progress(started, "preconditions", "begin")
    phase = time.monotonic()
    rows, checks, hashes = preflight(ROOT)
    checks.extend(resource_checks())
    historical = ROOT / "results/experiment_7800_v678_counter_evidence_protocol.json"
    hashes["historical_exp7800"] = {
        "path": str(historical),
        "sha256": sha256_file(historical) if historical.is_file() else None,
        "date": "20260928",
        "imported_fields": [
            "honest_verdict",
            "verdict_class",
            "validation_receipts.full_python_suite",
        ],
        "eligible": False,
    }
    if historical.is_file():
        old = json.loads(historical.read_text())
        checks.append(
            prior.check(
                "exp7800_historical",
                historical,
                "honest_verdict",
                "complete_disqualified_required_checks",
                old.get("honest_verdict"),
            )
        )
        checks.append(
            prior.check(
                "exp7800_historical",
                historical,
                "validation_receipts.full_python_suite.exit_code",
                -15,
                next(
                    (
                        item.get("exit_code")
                        for item in old.get("validation_receipts", {}).get("full_python_suite", [])
                        if item.get("name") == "full_python_suite"
                    ),
                    None,
                ),
            )
        )
    spans = [
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    ]
    progress(started, "preconditions", "complete", len(checks))
    if any(not item["passed"] for item in checks):
        result = artifact(checks, hashes, [], None, None, [], [], [], spans, started)
        atomic_json(OUTPUT, result)
        progress(started, "publish", "blocked")
        return result
    progress(started, "prepare", "begin")
    phase = time.monotonic()
    selected = freeze_families(rows)
    counter = GGUFTokenCounter(GGUF_PATH, GGUF_SHA256)
    frozen = make_protocol(selected)
    frozen["tokenizer"] = {
        "gguf_path": str(GGUF_PATH),
        "gguf_sha256": GGUF_SHA256,
        "chat_template_sha256": counter.template_sha256,
        "vocabulary_only": True,
    }
    protocol_path = RAW / "counter_evidence_protocol.json"
    if protocol_path.exists():
        raise ValueError("protocol_path_reused")
    atomic_json(protocol_path, frozen)
    family_manifest = build_family_manifest(selected, counter)
    family_path = attempt_raw / "family_manifest.json"
    atomic_json(family_path, family_manifest)
    fixture = fixture_e2e(Path(scope["private_root"]) / "protocol_fixture.json")
    evaluator_path = ROOT / prior.MANIFEST.parent / "evaluation_evaluator.jsonl"
    evaluator_rows = {
        item["family_id"]: item
        for item in (json.loads(line) for line in evaluator_path.read_text().splitlines())
    }
    aligned = [
        {
            "family_id": row["family_id"],
            "label": aligned_label(row, evaluator_rows[row["family_id"]]),
            "target_sentence_span": target_span(row["complete_response"].encode()),
            "annotation_source_sha256": sha256_file(evaluator_path),
        }
        for row in selected
    ]
    spans.append(
        {
            "phase": "prepare",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(selected),
        }
    )
    progress(started, "prepare", "complete", len(selected))
    candidate = artifact(
        checks,
        hashes,
        selected,
        protocol_path,
        family_path,
        fixture["rows"],
        aligned,
        [],
        spans,
        started,
    )
    for item in candidate["rows"]:
        if item["arm"] == "intact":
            row = next(row for row in selected if row["family_id"] == item["family_id"])
            try:
                make_request(
                    row,
                    row["complete_source"].encode(),
                    sentence_offsets(row["complete_source"].encode()),
                    "intact",
                    frozen,
                    counter,
                )
            except ValueError as error:
                item["disposition"] = "unstarted_" + str(error)
    atomic_json(Path(scope["candidate_path"]), candidate)
    progress(started, "validation", "begin")
    phase = time.monotonic()
    receipts = dispatch(scope, child_executor)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    progress(started, "validation", "complete", len(receipts))
    adverse = next(row for row in receipts if row["name"] == "adversarial_verify")
    try:
        flagged = json.loads(Path(adverse["log_path"]).read_text())["flagged_count"] > 0
    except (OSError, ValueError, KeyError):
        flagged = True
    result = artifact(
        checks,
        hashes,
        selected,
        protocol_path,
        family_path,
        fixture["rows"],
        aligned,
        receipts,
        spans,
        started,
        flagged,
    )
    for old_item, new_item in zip(candidate["rows"], result["rows"], strict=True):
        new_item["disposition"] = old_item["disposition"]
    atomic_json(OUTPUT, result)
    progress(started, "publish", "complete", len(result["rows"]))
    return result


def main(
    argv: list[str] | None = None,
    *,
    child_executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] | None = None,
) -> int:
    """Expose dated capture, exact dispatch, fixture E2E, and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--dispatch-check", action="store_true")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260928":
        raise ValueError("run_date_mismatch")
    if args.dispatch_check:
        dispatch(load_command_manifest(), child_executor)
        return 0
    if args.fixture_e2e:
        print(json.dumps(fixture_e2e(args.fixture_e2e)["reduced"], sort_keys=True), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(cold_reduce(args.cold_replay), sort_keys=True), flush=True)
        return 0
    result = run_experiment(args.date, child_executor)
    return int(result["verdict_class"] == "disqualified")
