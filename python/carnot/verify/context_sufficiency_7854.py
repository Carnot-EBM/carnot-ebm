"""Freeze four byte-identifiable source views for a later model experiment.

REQ-REPORT-7854-V682. Fixture completions check the transport contract only.
They do not say whether a source actually supports the answer.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from carnot.verify import source_interventions as source

ARMS = ("full_source", "witness_only", "witness_neighbors", "matched_control")
Json = dict[str, Any]
TokenCount = Callable[[str], int]


def freeze_protocol(*, seed: int) -> Json:
    """Keep the future model budget and every visible request setting fixed."""
    return {
        "schema": "carnot.exp7854.context_sufficiency.v1",
        "seed": seed,
        "arms": list(ARMS),
        "source_family_budget": 48,
        "maximum_model_calls": 192,
        "max_tokens_per_call": 128,
        "context_ceiling_tokens": 8192,
        "call_timeout_s": 120,
        "latest_launch_s": 2400,
        "model_service_cap_s": 3000,
        "cache_resolver": "carnot.inference.sota_models.cached_current_model",
        "cache_miss_disposition": "blocked_no_current_model",
        "tokenizer_requirement": "embedded GGUF tokenizer in real capture; injected counter in fixture",
        "matching_rule": "disjoint original sentence outside immediate neighbors; within 25% witness tokens",
        "request": {
            "model": source.MODEL_ID,
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        },
        "system_instruction": source.SYSTEM,
    }


def choose_control(row: Json, witness: int, count: TokenCount, *, seed: int) -> int | None:
    """Choose an original sentence outside the witness's local context."""
    raw = row["complete_source"].encode()
    offsets = source.sentence_offsets(raw)
    if witness not in range(len(offsets)):
        return None
    sizes = [count(raw[item["start_byte"] : item["end_byte"]].decode()) for item in offsets]
    target = sizes[witness]
    candidates = [
        index
        for index, size in enumerate(sizes)
        if abs(index - witness) > 1 and target > 0 and 4 * abs(size - target) <= target
    ]
    return (
        min(
            candidates,
            key=lambda index: source.digest(f"{seed}:{row['family_id']}:{index}".encode()),
        )
        if candidates
        else None
    )


def build_request(
    row: Json,
    arm: str,
    witness: int | None,
    control: int | None,
    frozen: Json,
    count: TokenCount,
) -> Json:
    """Copy selected original sentence bytes into one bounded HTTP request."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    raw = row["complete_source"].encode()
    offsets = source.sentence_offsets(raw)
    if source.digest(raw) != row["source_sha256"]:
        raise ValueError("source_drift")
    if not offsets:
        raise ValueError("empty_source")
    if arm != "full_source" and (type(witness) is not int or witness not in range(len(offsets))):
        raise ValueError("invalid_witness")
    if arm == "full_source":
        ids = list(range(len(offsets)))
    elif arm == "witness_only":
        ids = [witness]
    elif arm == "witness_neighbors":
        ids = [index for index in range(len(offsets)) if abs(index - witness) <= 1]
    else:
        if (
            type(control) is not int
            or control not in range(len(offsets))
            or abs(control - witness) <= 1
        ):
            raise ValueError("invalid_control")
        ids = sorted((witness, control))
    answer = row["complete_response"].encode()
    if source.digest(answer) != row["response_sha256"]:
        raise ValueError("answer_drift")
    target = source.target_span(answer)
    selected = b"".join(
        raw[offsets[index]["start_byte"] : offsets[index]["end_byte"]] for index in ids
    )
    body = {
        "arm": arm,
        "complete_source": selected.decode(),
        "original_answer": answer[: target["end_byte"]].decode().strip(),
        "visible_source_sentence_ids": ids,
        "source_sentence_offsets": [offsets[index] for index in ids],
        "target_sentence_sha256": target["text_sha256"],
    }
    messages = [
        {"role": "system", "content": frozen["system_instruction"]},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, sort_keys=True)},
    ]
    payload = {**frozen["request"], "messages": messages}
    if (
        count(json.dumps(messages, ensure_ascii=False)) + payload["max_tokens"]
        > frozen["context_ceiling_tokens"]
    ):
        raise ValueError("context_budget")
    return payload


def parse_response(response: Json, model: str, visible_ids: list[int]) -> Json:
    """Accept only a complete strict JSON reply naming a visible source ID."""
    if response.get("model") != model:
        return {"status": "wrong_model", "source_sentence_id": None, "probability": None}
    try:
        choice = response["choices"][0]
        parsed = source.parse_reply(
            choice["message"]["content"], choice["finish_reason"], visible_ids
        )
    except (KeyError, IndexError, TypeError):
        parsed = source.parse_reply("", "", [])
    return {
        "status": parsed["disposition"],
        "source_sentence_id": parsed["source_sentence_id"],
        "probability": parsed["unsupported_probability"],
    }


def capture_family(
    row: Json, frozen: Json, transport: Callable[[Json], Json], count: TokenCount
) -> list[Json]:
    """Record every planned arm, including those stopped by an earlier failure."""
    base = {
        "family_id": row["family_id"],
        "source_sha256": row["source_sha256"],
        "answer_sha256": row["response_sha256"],
        "seed": frozen["seed"],
        "source_family": row["family_id"],
        "original_label": None,
    }

    def empty(arm: str, status: str, *, excluded: bool = False) -> Json:
        return {
            **base,
            "arm": arm,
            "status": status,
            "started": False,
            "completed": False,
            "censored": False,
            "excluded": excluded,
            "probability": None,
            "source_sentence_id": None,
            "control_sentence_id": None,
            "request_bytes": None,
            "request_sha256": None,
            "response_bytes": None,
        }

    def call(arm: str, witness: int | None, control: int | None) -> Json:
        try:
            payload = build_request(row, arm, witness, control, frozen, count)
        except ValueError as error:
            return empty(arm, "excluded_" + str(error), excluded=True)
        request_bytes = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        result = {
            **empty(arm, "unstarted"),
            "started": True,
            "control_sentence_id": control,
            "request_bytes": request_bytes,
            "request_sha256": source.digest(request_bytes.encode()),
        }
        try:
            response = transport(payload)
        except TimeoutError:
            return {**result, "status": "timeout", "censored": True}
        except InterruptedError:
            return {**result, "status": "interrupted", "censored": True}
        body = json.loads(payload["messages"][1]["content"])
        parsed = parse_response(response, payload["model"], body["visible_source_sentence_ids"])
        response_bytes = json.dumps(
            response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return {
            **result,
            **parsed,
            "completed": parsed["status"] == "completed",
            "response_bytes": response_bytes,
            "response_sha256": source.digest(response_bytes.encode()),
            "usage": response.get("usage", {}),
        }

    full = call("full_source", None, None)
    if full["status"].startswith("excluded_"):
        return [full, *[empty(arm, full["status"], excluded=True) for arm in ARMS[1:]]]
    if full["status"] != "completed":
        return [full, *[empty(arm, "unstarted_invalid_witness") for arm in ARMS[1:]]]
    witness = full["source_sentence_id"]
    control = choose_control(row, witness, count, seed=frozen["seed"])
    other = [call("witness_only", witness, None), call("witness_neighbors", witness, None)]
    if control is None:
        other.append(empty("matched_control", "excluded_no_matched_context", excluded=True))
    else:
        other.append(call("matched_control", witness, control))
    return [full, *other]
