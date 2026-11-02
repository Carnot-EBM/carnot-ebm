"""Probe option log probabilities on the scored vLLM server.

REQ-INFRA-7090 keeps this probe bounded and off the server lifecycle path.
"""

from __future__ import annotations

import ipaddress
import json
import math
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


__all__ = ["run_logprob_probe"]

LABELS = tuple(f" {chr(ord('A') + index)}" for index in range(16))
MAX_LOGPROBS = 16
RANDOM_SEED = 7089
DEFAULT_BUDGET_S = 240.0
HTTP_TIMEOUT_S = 30.0

_EMBEDDED_FIXTURE_JSON = r"""
[
  {"id": "p001", "question": "Which shape has three sides?", "options": ["triangle", "square"], "answer": "triangle"},
  {"id": "p002", "question": "Which number is larger?", "options": ["4", "9"], "answer": "9"},
  {"id": "p003", "question": "Which material is transparent?", "options": ["clear glass", "brick"], "answer": "clear glass"},
  {"id": "p004", "question": "Which action makes a room darker?", "options": ["turn off the lamp", "open the curtains"], "answer": "turn off the lamp"},
  {"id": "p005", "question": "Which animal can fly?", "options": ["sparrow", "tortoise"], "answer": "sparrow"},
  {"id": "p006", "question": "Which value equals two plus three?", "options": ["5", "6"], "answer": "5"},
  {"id": "p007", "question": "Which object tells time?", "options": ["clock", "plate"], "answer": "clock"},
  {"id": "p008", "question": "Which direction is opposite east?", "options": ["west", "north"], "answer": "west"},
  {"id": "p009", "question": "Which season follows spring?", "options": ["summer", "winter"], "answer": "summer"},
  {"id": "p010", "question": "Which number is even?", "options": ["7", "8", "9"], "answer": "8"},
  {"id": "p011", "question": "Which item is used for writing?", "options": ["pencil", "spoon", "shoe"], "answer": "pencil"},
  {"id": "p012", "question": "Which color comes from mixing blue and yellow paint?", "options": ["green", "orange", "purple"], "answer": "green"},
  {"id": "p013", "question": "Which word names a day?", "options": ["Monday", "January", "morning"], "answer": "Monday"},
  {"id": "p014", "question": "Which object is usually soft?", "options": ["pillow", "stone", "fork"], "answer": "pillow"},
  {"id": "p015", "question": "Which operation turns six into twelve?", "options": ["multiply by two", "subtract two", "divide by two"], "answer": "multiply by two"},
  {"id": "p016", "question": "Which place holds books for borrowing?", "options": ["library", "garage", "bakery"], "answer": "library"},
  {"id": "p017", "question": "Which state of water is solid?", "options": ["ice", "steam", "rain"], "answer": "ice"},
  {"id": "p018", "question": "Which tool can tighten a screw?", "options": ["screwdriver", "paintbrush", "ruler"], "answer": "screwdriver"},
  {"id": "p019", "question": "Which value is smallest?", "options": ["12", "3", "8", "5"], "answer": "3"},
  {"id": "p020", "question": "Which word is a verb?", "options": ["run", "blue", "table", "quiet"], "answer": "run"},
  {"id": "p021", "question": "Which container is best for hot tea?", "options": ["mug", "envelope", "basket", "folder"], "answer": "mug"},
  {"id": "p022", "question": "Which planet is our home?", "options": ["Earth", "Mars", "Venus", "Mercury"], "answer": "Earth"},
  {"id": "p023", "question": "Which fraction equals one half?", "options": ["2/4", "1/3", "3/4", "2/3"], "answer": "2/4"},
  {"id": "p024", "question": "Which item is magnetic?", "options": ["iron nail", "paper sheet", "wood block", "rubber band"], "answer": "iron nail"},
  {"id": "p025", "question": "Which sign means addition?", "options": ["+", "-", "x", "/"], "answer": "+"},
  {"id": "p026", "question": "Which room is used for cooking?", "options": ["kitchen", "bedroom", "hallway", "garage"], "answer": "kitchen"},
  {"id": "p027", "question": "Which unit measures length?", "options": ["meter", "liter", "second", "degree"], "answer": "meter"},
  {"id": "p028", "question": "Which number is prime?", "options": ["11", "12", "14", "15", "16"], "answer": "11"},
  {"id": "p029", "question": "Which item is a fruit?", "options": ["pear", "carrot", "celery", "onion", "potato"], "answer": "pear"},
  {"id": "p030", "question": "Which surface reflects a clear image?", "options": ["mirror", "cloth", "sand", "cardboard", "cork"], "answer": "mirror"},
  {"id": "p031", "question": "Which value equals three squared?", "options": ["9", "6", "8", "12", "27"], "answer": "9"},
  {"id": "p032", "question": "Which device stores electrical energy?", "options": ["battery", "hinge", "funnel", "ladder", "bowl"], "answer": "battery"},
  {"id": "p033", "question": "Which word is the plural of child?", "options": ["children", "childs", "childes", "child", "childrens"], "answer": "children"},
  {"id": "p034", "question": "Which angle is a right angle?", "options": ["90 degrees", "30 degrees", "45 degrees", "60 degrees", "120 degrees"], "answer": "90 degrees"},
  {"id": "p035", "question": "Which part of a plant absorbs water from soil?", "options": ["roots", "flowers", "fruit", "seeds", "petals"], "answer": "roots"},
  {"id": "p036", "question": "Which item can erase pencil marks?", "options": ["eraser", "stapler", "tape", "clip", "brush"], "answer": "eraser"},
  {"id": "p037", "question": "Which number is a multiple of five?", "options": ["25", "22", "23", "24", "26", "27"], "answer": "25"},
  {"id": "p038", "question": "Which gas do people need to breathe?", "options": ["oxygen", "helium", "neon", "argon", "hydrogen", "methane"], "answer": "oxygen"},
  {"id": "p039", "question": "Which instrument has black and white keys?", "options": ["piano", "drum", "flute", "trumpet", "violin", "cymbal"], "answer": "piano"},
  {"id": "p040", "question": "Which month has fewer than thirty days in a common year?", "options": ["February", "April", "June", "September", "November", "December"], "answer": "February"},
  {"id": "p041", "question": "Which value is the square root of sixty-four?", "options": ["8", "6", "7", "9", "16", "32"], "answer": "8"},
  {"id": "p042", "question": "Which object measures air temperature?", "options": ["thermometer", "compass", "scale", "timer", "ruler", "calendar"], "answer": "thermometer"},
  {"id": "p043", "question": "Which word is a synonym for quick?", "options": ["fast", "late", "heavy", "narrow", "silent", "rough"], "answer": "fast"},
  {"id": "p044", "question": "Which process changes liquid water into vapor?", "options": ["evaporation", "freezing", "melting", "condensation", "crushing", "filtering"], "answer": "evaporation"},
  {"id": "p045", "question": "Which object protects eyes from bright sunlight?", "options": ["sunglasses", "gloves", "scarf", "belt", "boots", "apron"], "answer": "sunglasses"}
]
"""


def _softmax_declared_options(
    scores: Mapping[str, float], declared_options: Sequence[str]
) -> dict[str, float]:
    selected = {option: float(scores[option]) for option in declared_options}
    if not selected:
        raise ValueError("at least one declared option is required")
    if any(math.isnan(score) for score in selected.values()):
        raise ValueError("declared option scores contain NaN")
    peak = max(selected.values())
    if peak == -math.inf:
        raise ValueError("all declared option scores are negative infinity")
    weights = {option: math.exp(score - peak) for option, score in selected.items()}
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("declared option scores cannot be normalized")
    return {option: weight / total for option, weight in weights.items()}


def _resolve_ref(document: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    node: Any = document
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(node, Mapping):
            return {}
        node = node.get(part.replace("~1", "/").replace("~0", "~"), {})
    return node if isinstance(node, Mapping) else {}


def _schema_type(schema: Mapping[str, Any], document: Mapping[str, Any]) -> str:
    if "$ref" in schema:
        return _schema_type(_resolve_ref(document, str(schema["$ref"])), document)
    for union_name in ("anyOf", "oneOf"):
        variants = schema.get(union_name)
        if isinstance(variants, list):
            names = [_schema_type(item, document) for item in variants if isinstance(item, Mapping)]
            return "|".join(dict.fromkeys(name for name in names if name)) or "unknown"
    value_type = schema.get("type")
    if value_type == "array":
        items = schema.get("items")
        item_type = _schema_type(items, document) if isinstance(items, Mapping) else "unknown"
        return f"array[{item_type}]"
    if isinstance(value_type, list):
        return "|".join(str(item) for item in value_type)
    if isinstance(value_type, str):
        return f"{value_type}|null" if schema.get("nullable") else value_type
    if "enum" in schema:
        return "enum"
    return "unknown"


def _schema_properties(
    schema: Mapping[str, Any], document: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if "$ref" in schema:
        return _schema_properties(_resolve_ref(document, str(schema["$ref"])), document)
    properties: dict[str, Mapping[str, Any]] = {}
    direct = schema.get("properties")
    if isinstance(direct, Mapping):
        properties.update(
            {str(name): value for name, value in direct.items() if isinstance(value, Mapping)}
        )
    for branch_name in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(branch_name)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, Mapping):
                    properties.update(_schema_properties(branch, document))
    return properties


def _detect_completion_schema_fields(openapi: Mapping[str, Any]) -> dict[str, Any]:
    paths = openapi.get("paths") if isinstance(openapi, Mapping) else None
    completion = paths.get("/v1/completions", {}) if isinstance(paths, Mapping) else {}
    post = completion.get("post", {}) if isinstance(completion, Mapping) else {}
    request_body = post.get("requestBody", {}) if isinstance(post, Mapping) else {}
    content = request_body.get("content", {}) if isinstance(request_body, Mapping) else {}
    media = content.get("application/json", {}) if isinstance(content, Mapping) else {}
    schema = media.get("schema", {}) if isinstance(media, Mapping) else {}
    properties = _schema_properties(schema, openapi) if isinstance(schema, Mapping) else {}
    property_types = {
        name: _schema_type(value, openapi) for name, value in sorted(properties.items())
    }
    structured = sorted(
        name for name in properties if "guided" in name.lower() or "structured" in name.lower()
    )
    supported = {
        name: name in properties
        for name in (
            "logprobs",
            "prompt_logprobs",
            "allowed_token_ids",
            "logprob_token_ids",
            "logit_bias",
        )
    }
    supported["structured_decoding_fields"] = structured
    supported["guided_or_structured_decoding"] = bool(structured)
    return {
        "properties": property_types,
        "property_count": len(property_types),
        "supported": supported,
    }


def _fixtures() -> list[dict[str, Any]]:
    data = json.loads(_EMBEDDED_FIXTURE_JSON)
    if not isinstance(data, list):
        raise ValueError("embedded fixture root must be a list")
    return data


def _validate_fixtures(fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [str(item.get("id", "")) for item in fixtures]
    counts: list[int] = []
    answers_in_options = True
    for index, item in enumerate(fixtures):
        options = item.get("options")
        if not isinstance(options, list) or not all(isinstance(value, str) for value in options):
            errors.append(f"fixture {index} has invalid options")
            counts.append(0)
            answers_in_options = False
            continue
        counts.append(len(options))
        if item.get("answer") not in options:
            errors.append(f"fixture {index} answer is not an option")
            answers_in_options = False
        if not isinstance(item.get("question"), str) or not str(item.get("question")).strip():
            errors.append(f"fixture {index} has no question")
    unique_ids = len(set(ids)) == len(ids) and all(ids)
    if not unique_ids:
        errors.append("fixture IDs must be non-empty and unique")
    if len(fixtures) < 40:
        errors.append("at least 40 fixtures are required")
    option_count_range = [min(counts), max(counts)] if counts else [0, 0]
    counts_valid = bool(counts) and all(2 <= count <= 6 for count in counts)
    if not counts_valid:
        errors.append("each fixture must have two to six options")
    labels_within_limit = bool(counts) and max(counts) <= len(LABELS)
    if not labels_within_limit:
        errors.append("fixture labels exceed the 16-label limit")
    return {
        "valid": not errors,
        "count": len(fixtures),
        "unique_ids": bool(unique_ids),
        "answers_in_options": answers_in_options,
        "option_count_range": option_count_range,
        "labels_within_limit": labels_within_limit,
        "errors": errors,
    }


def _loopback_http_url(base_url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(base_url)
        host = parsed.hostname
        if parsed.scheme != "http" or not host or parsed.username or parsed.password:
            return False
        return host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except (TypeError, ValueError):
        return False


def _http_json(
    base_url: str,
    path: str,
    deadline: float,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {
            "ok": False,
            "status": None,
            "error": "RuntimeBudgetExhausted: no time remains for the HTTP call",
        }
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    timeout = max(0.001, min(HTTP_TIMEOUT_S, remaining))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode(errors="replace")
            try:
                body: Any = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = {"raw_body": raw}
            return {"ok": True, "status": int(response.status), "body": body}
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read(8192).decode(errors="replace")
        except BaseException as read_exc:
            error_body = f"error body read failed: {type(read_exc).__name__}: {read_exc}"
        return {"ok": False, "status": int(exc.code), "error_body": error_body}
    except BaseException as exc:
        return {"ok": False, "status": None, "error": f"{type(exc).__name__}: {exc}"}


def _format_prompt(fixture: Mapping[str, Any]) -> str:
    lines = [
        "Choose the correct answer. Reply with one letter only.",
        f"Question: {fixture['question']}",
    ]
    for index, option in enumerate(fixture["options"]):
        lines.append(f"{LABELS[index].strip()}. {option}")
    lines.append("Answer:")
    return "\n".join(lines)


def _token_ids_from_response(response: Mapping[str, Any]) -> list[int]:
    body = response.get("body")
    if not isinstance(body, Mapping):
        return []
    raw_tokens = body.get("token_ids")
    if not isinstance(raw_tokens, list):
        raw_tokens = body.get("tokens", [])
    if not isinstance(raw_tokens, list):
        return []
    return [int(token) for token in raw_tokens if isinstance(token, int)]


def _collect_score_candidates(
    node: Any, text_scores: dict[str, float], id_scores: dict[int, float]
) -> None:
    if isinstance(node, Mapping):
        logprob = node.get("logprob")
        if isinstance(logprob, (int, float)):
            token = node.get("token", node.get("decoded_token"))
            token_id = node.get("token_id")
            if isinstance(token, str):
                text_scores[token] = float(logprob)
            if isinstance(token_id, int):
                id_scores[token_id] = float(logprob)
        for key, value in node.items():
            if isinstance(value, (int, float)):
                if isinstance(key, str):
                    text_scores.setdefault(key, float(value))
                    if key.isdigit():
                        id_scores.setdefault(int(key), float(value))
            else:
                if isinstance(key, str) and isinstance(value, Mapping):
                    nested_logprob = value.get("logprob")
                    if isinstance(nested_logprob, (int, float)):
                        text_scores.setdefault(key, float(nested_logprob))
                        if key.isdigit():
                            id_scores.setdefault(int(key), float(nested_logprob))
                _collect_score_candidates(value, text_scores, id_scores)
    elif isinstance(node, list):
        for item in node:
            _collect_score_candidates(item, text_scores, id_scores)


def _option_scores(
    response: Mapping[str, Any], declared_labels: Sequence[str], token_ids: Mapping[str, int]
) -> tuple[dict[str, float], list[str]]:
    text_scores: dict[str, float] = {}
    id_scores: dict[int, float] = {}
    _collect_score_candidates(response.get("body"), text_scores, id_scores)
    scores: dict[str, float] = {}
    missing: list[str] = []
    for label in declared_labels:
        if label in text_scores:
            scores[label] = text_scores[label]
        elif label in token_ids and token_ids[label] in id_scores:
            scores[label] = id_scores[token_ids[label]]
        else:
            missing.append(label)
    return scores, missing


def _request_row(
    base_url: str,
    model_name: str,
    fixture: Mapping[str, Any],
    variant: str,
    temperature: float,
    token_ids: Mapping[str, int],
    deadline: float,
) -> dict[str, Any]:
    labels = LABELS[: len(fixture["options"])]
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": _format_prompt(fixture),
        "max_tokens": 1,
        "temperature": temperature,
        "seed": RANDOM_SEED,
    }
    if variant == "top_n_logprobs":
        payload["logprobs"] = MAX_LOGPROBS
    elif variant == "logprob_token_ids":
        payload["logprobs"] = MAX_LOGPROBS
        payload["logprob_token_ids"] = [token_ids[label] for label in labels]
    elif variant == "allowed_token_ids":
        payload["logprobs"] = len(labels)
        payload["allowed_token_ids"] = [token_ids[label] for label in labels]
    elif variant == "prompt_logprobs":
        payload["prompt_logprobs"] = MAX_LOGPROBS
    else:
        raise ValueError(f"unknown variant: {variant}")
    started = time.monotonic()
    response = _http_json(base_url, "/v1/completions", deadline, payload)
    latency = time.monotonic() - started
    scores, missing = _option_scores(response, labels, token_ids)
    probabilities = None
    normalization_error = None
    if not missing and variant != "prompt_logprobs":
        try:
            probabilities = _softmax_declared_options(scores, labels)
        except ValueError as exc:
            normalization_error = str(exc)
    answer_index = list(fixture["options"]).index(fixture["answer"])
    argmax_label = max(probabilities, key=probabilities.get) if probabilities else None
    return {
        "fixture_id": fixture["id"],
        "option_count": len(labels),
        "known_answer_label": LABELS[answer_index].strip(),
        "temperature": temperature,
        "request": payload,
        "http": response,
        "latency_s": round(latency, 6),
        "option_scores": scores,
        "missing_option_labels": missing,
        "normalization_error": normalization_error,
        "option_probabilities": probabilities,
        "argmax_label": argmax_label.strip() if argmax_label else None,
    }


def _temperature_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_fixture: dict[str, dict[float, Mapping[str, float]]] = {}
    for row in rows:
        probabilities = row.get("option_probabilities")
        if isinstance(probabilities, Mapping):
            by_fixture.setdefault(str(row["fixture_id"]), {})[float(row["temperature"])] = (
                probabilities
            )
    differences: list[float] = []
    exact_matches: list[bool] = []
    for temperatures in by_fixture.values():
        if 0.0 not in temperatures or 1.0 not in temperatures:
            continue
        cold = temperatures[0.0]
        warm = temperatures[1.0]
        keys = set(cold) & set(warm)
        differences.extend(abs(float(cold[key]) - float(warm[key])) for key in keys)
        exact_matches.append(dict(cold) == dict(warm))
    if not exact_matches:
        classification = "unknown_no_paired_option_probabilities"
    elif all(exact_matches):
        classification = "no_temperature_effect_observed_raw_vs_processing_unresolved"
    else:
        classification = "temperature_effect_observed_post_temperature_or_processing"
    return {
        "paired_prompt_count": len(exact_matches),
        "all_probabilities_identical": all(exact_matches) if exact_matches else None,
        "maximum_absolute_probability_difference": max(differences) if differences else None,
        "raw_or_post_temperature": classification,
    }


def _variant_schema_support(result: Mapping[str, Any], variant: str) -> bool:
    openapi = result.get("phases", {}).get("schema", {}).get("openapi", {})
    supported = openapi.get("supported", {}) if isinstance(openapi, Mapping) else {}
    if variant == "top_n_logprobs":
        return bool(supported.get("logprobs"))
    if variant == "logprob_token_ids":
        return bool(supported.get("logprobs") and supported.get("logprob_token_ids"))
    if variant == "allowed_token_ids":
        return bool(supported.get("logprobs") and supported.get("allowed_token_ids"))
    if variant == "prompt_logprobs":
        return bool(supported.get("prompt_logprobs"))
    return False


def _first_http_block(result: Mapping[str, Any]) -> str | None:
    phases = result.get("phases")
    if not isinstance(phases, Mapping):
        return None
    variants_phase = phases.get("logprob_request_variants")
    if not isinstance(variants_phase, Mapping):
        return None
    variants = variants_phase.get("variants")
    if not isinstance(variants, Mapping):
        return None
    for variant in variants.values():
        rows = variant.get("rows", []) if isinstance(variant, Mapping) else []
        for row in rows if isinstance(rows, list) else []:
            http = row.get("http", {}) if isinstance(row, Mapping) else {}
            status = http.get("status") if isinstance(http, Mapping) else None
            error = str(http.get("error", "")) if isinstance(http, Mapping) else ""
            if status == 400:
                return "blocked_logprobs_http_400"
            if "timed out" in error.lower() or "timeout" in error.lower():
                return "blocked_http_timeout"
    return None


def _contains_timeout(node: Any) -> bool:
    if isinstance(node, Mapping):
        error = node.get("error")
        if isinstance(error, str) and (
            "timed out" in error.lower()
            or "timeout" in error.lower()
            or "runtimebudgetexhausted" in error.lower()
        ):
            return True
        return any(_contains_timeout(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_timeout(value) for value in node)
    return False


def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
    schema = result.get("phases", {}).get("schema", {}).get("openapi", {})
    schema_supported = schema.get("supported", {}) if isinstance(schema, Mapping) else {}
    variant_phase = result.get("phases", {}).get("logprob_request_variants", {})
    variants = variant_phase.get("variants", {}) if isinstance(variant_phase, Mapping) else {}
    latency_medians: dict[str, float | None] = {}
    observed_maxima: dict[str, int] = {}
    runtime_supported: dict[str, bool] = {}
    for name, variant in variants.items() if isinstance(variants, Mapping) else []:
        rows = variant.get("rows", []) if isinstance(variant, Mapping) else []
        latencies = [
            float(row["latency_s"])
            for row in rows
            if isinstance(row, Mapping) and row.get("http", {}).get("ok")
        ]
        complete_rows = [
            row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("option_probabilities"), Mapping)
        ]
        latency_medians[str(name)] = statistics.median(latencies) if latencies else None
        observed_maxima[str(name)] = max(
            (int(row["option_count"]) for row in complete_rows), default=0
        )
        runtime_supported[str(name)] = bool(variant.get("effective_supported"))
    expected_rows = 2 * int(variant_phase.get("fixture_validation", {}).get("count", 0) or 0)
    recommended_name = None
    for candidate in ("logprob_token_ids", "allowed_token_ids", "top_n_logprobs"):
        candidate_data = variants.get(candidate, {}) if isinstance(variants, Mapping) else {}
        if int(candidate_data.get("complete_option_distribution_count", 0)) == expected_rows > 0:
            recommended_name = candidate
            break
    if recommended_name:
        request_shape: dict[str, Any] = {
            "status": "recommended",
            "variant": recommended_name,
            "common_fields": {
                "model": result.get("model_name"),
                "max_tokens": 1,
                "temperature": "0 or 1 as measured",
                "seed": RANDOM_SEED,
            },
        }
        if recommended_name == "logprob_token_ids":
            request_shape["variant_fields"] = {
                "logprobs": MAX_LOGPROBS,
                "logprob_token_ids": "declared option token IDs",
            }
        elif recommended_name == "allowed_token_ids":
            request_shape["variant_fields"] = {
                "logprobs": "number of declared options",
                "allowed_token_ids": "declared option token IDs",
            }
        else:
            request_shape["variant_fields"] = {"logprobs": MAX_LOGPROBS}
    else:
        request_shape = {
            "status": "blocked",
            "reason": "no request variant returned every declared option for every frozen prompt",
        }
    direct_max = observed_maxima.get(recommended_name or "", 0)
    maximum_options = direct_max
    maximum_basis = "largest complete declared-option distribution observed"
    server_ready = result.get("phases", {}).get("server", {}).get("status") == "ready"
    blocked = _first_http_block(result)
    if not server_ready:
        verdict = str(result.get("honest_verdict") or "blocked_server_not_alive")
    elif blocked:
        verdict = blocked
    elif _contains_timeout(result.get("phases", {})):
        verdict = "blocked_http_timeout"
    elif recommended_name is None:
        verdict = "blocked_exact_declared_option_scores_unavailable"
    else:
        verdict = "complete_exact_declared_option_score_probe"
    result["honest_verdict"] = verdict
    return {
        "supported_fields": {
            "openapi": schema_supported,
            "runtime_variants": runtime_supported,
        },
        "recommended_request_shape": request_shape,
        "maximum_options_one_request_can_score": maximum_options,
        "maximum_options_basis": maximum_basis,
        "maximum_options_directly_exercised": direct_max,
        "median_latency_per_decision_s": latency_medians,
        "honest_verdict": verdict,
    }


def _record_error(result: dict[str, Any], phase: str, exc: BaseException) -> None:
    result["phases"][phase] = {
        "status": "blocked_exception",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc().splitlines()[-12:],
    }
    result["honest_verdict"] = f"blocked_{phase}_exception"


def _write_result(out_path: Path, result: dict[str, Any], started_at: float) -> None:
    result["duration_s"] = round(time.monotonic() - started_at, 6)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def run_logprob_probe(
    base_url: str,
    model_name: str,
    out_path: str | Path,
    budget_s: float = DEFAULT_BUDGET_S,
    *,
    server_launch_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the REQ-INFRA-7090 probe against an already-running loopback server.

    This function never manages the server process. It catches all failures and
    returns an honest blocked verdict.
    """
    started_at = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "probe": "carnot-vllm-logprob-probe",
        "requirement": "REQ-INFRA-7089",
        "inference_substrate": "hardware_smoke",
        "random_seed": RANDOM_SEED,
        "duration_s": 0.0,
        "honest_verdict": "blocked_not_started",
        "preconditions_checked": [],
        "runtime_budget_s": budget_s,
        "base_url": str(base_url),
        "model_name": str(model_name),
        "phases": {
            "environment": {
                "status": "complete_reusing_already_running_server",
                "python": {"version": sys.version, "executable": sys.executable},
                "platform": sys.platform,
                "base_url": str(base_url),
                "model_name": str(model_name),
                "server_management": "none",
            }
        },
        "server_cleanup": {
            "owned_process_started": False,
            "termination": "not_attempted_already_running_server",
        },
    }
    try:
        output = Path(out_path)
    except BaseException as exc:
        result["honest_verdict"] = "blocked_invalid_output_path"
        result["output_write_error"] = f"{type(exc).__name__}: {exc}"
        result["duration_s"] = round(time.monotonic() - started_at, 6)
        return result

    if server_launch_argv is not None:
        try:
            result["server_launch_argv"] = [str(value) for value in server_launch_argv]
        except BaseException as exc:
            result["honest_verdict"] = "blocked_invalid_server_launch_argv"
            result["server_launch_argv_error"] = f"{type(exc).__name__}: {exc}"
            try:
                _write_result(output, result, started_at)
            except BaseException as write_exc:
                result["output_write_error"] = f"{type(write_exc).__name__}: {write_exc}"
                result["duration_s"] = round(time.monotonic() - started_at, 6)
            return result

    try:
        budget = float(budget_s)
        if not math.isfinite(budget) or budget <= 0:
            result["honest_verdict"] = "blocked_invalid_runtime_budget"
        elif not _loopback_http_url(str(base_url)):
            result["preconditions_checked"].append(
                {
                    "resource": "loopback_http_server",
                    "available": False,
                    "detail": "base_url must use HTTP on a loopback host",
                }
            )
            result["honest_verdict"] = "blocked_non_loopback_base_url"
        else:
            reserve = min(1.0, max(0.05, budget * 0.05))
            deadline = started_at + max(0.0, budget - reserve)
            result["preconditions_checked"].append(
                {
                    "resource": "loopback_http_server",
                    "available": True,
                    "detail": str(base_url),
                }
            )
            health = _http_json(str(base_url), "/health", deadline)
            server_ready = bool(health.get("ok"))
            server_phase: dict[str, Any] = {
                "status": "ready" if server_ready else "blocked_not_alive",
                "startup_time_s": 0.0,
                "scored_launch_difference": {"added_only": []},
                "log_path": None,
                "process_returncode": None,
                "ownership": "caller_managed_already_running",
                "health": health,
            }
            if server_launch_argv is not None:
                server_phase["launch_argv"] = result["server_launch_argv"]
            result["phases"]["server"] = server_phase
            result["preconditions_checked"].append(
                {
                    "resource": "already_running_server_alive",
                    "available": server_ready,
                    "detail": f"GET /health status={health.get('status')}",
                }
            )
            if not server_ready:
                error = str(health.get("error", ""))
                result["honest_verdict"] = (
                    "blocked_http_timeout"
                    if "timed out" in error.lower() or "timeout" in error.lower()
                    else "blocked_server_not_alive"
                )
            else:
                result["inference_substrate"] = "live_llm_inference"
                openapi_response = _http_json(str(base_url), "/openapi.json", deadline)
                if openapi_response.get("ok") and isinstance(openapi_response.get("body"), Mapping):
                    detection = _detect_completion_schema_fields(openapi_response["body"])
                    result["phases"]["schema"] = {
                        "status": "complete",
                        "vllm_serve_help_returncode": None,
                        "vllm_serve_help_logprob_lines": [],
                        "max_logprobs_cap_supported_by_help": None,
                        "max_logprobs_cap_configured": None,
                        "openapi": detection,
                    }
                else:
                    result["phases"]["schema"] = {
                        "status": "blocked_openapi_unavailable",
                        "openapi": None,
                        "openapi_error": openapi_response,
                    }

                token_rows = []
                token_ids: dict[str, int] = {}
                for label in LABELS:
                    response = _http_json(
                        str(base_url),
                        "/tokenize",
                        deadline,
                        {"model": str(model_name), "prompt": label},
                    )
                    ids = _token_ids_from_response(response)
                    body = response.get("body")
                    count = body.get("count") if isinstance(body, Mapping) else None
                    if not isinstance(count, int):
                        count = len(ids)
                    if count == 1 and len(ids) == 1:
                        token_ids[label] = ids[0]
                    token_rows.append(
                        {
                            "label": label,
                            "token_count": count,
                            "token_ids": ids,
                            "one_token": count == 1 and len(ids) == 1,
                            "raw_response": response,
                        }
                    )
                    if time.monotonic() >= deadline:
                        break
                result["phases"]["tokenize_check"] = {
                    "status": (
                        "complete"
                        if len(token_ids) == len(LABELS)
                        else "blocked_non_single_token_label_or_runtime_budget"
                    ),
                    "prompt_style": "raw completion prompt, matching the scored /v1/completions path",
                    "all_16_labels_are_one_token": len(token_ids) == len(LABELS),
                    "labels": token_rows,
                }

                fixtures = _fixtures()
                fixture_validation = _validate_fixtures(fixtures)
                prompt_token_counts: dict[str, Any] = {}
                for fixture in fixtures:
                    if time.monotonic() >= deadline:
                        break
                    response = _http_json(
                        str(base_url),
                        "/tokenize",
                        deadline,
                        {"model": str(model_name), "prompt": _format_prompt(fixture)},
                    )
                    body = response.get("body")
                    prompt_token_counts[str(fixture["id"])] = (
                        body.get("count") if isinstance(body, Mapping) else None
                    )

                variants: dict[str, Any] = {}
                for variant in (
                    "top_n_logprobs",
                    "logprob_token_ids",
                    "allowed_token_ids",
                    "prompt_logprobs",
                ):
                    schema_supported = _variant_schema_support(result, variant)
                    needs_ids = variant in ("logprob_token_ids", "allowed_token_ids")
                    if needs_ids and len(token_ids) != len(LABELS):
                        variants[variant] = {
                            "schema_supported": schema_supported,
                            "effective_supported": False,
                            "status": "blocked_option_token_ids_unavailable",
                            "rows": [],
                            "complete_option_distribution_count": 0,
                            "temperature_comparison": _temperature_comparison([]),
                        }
                        continue
                    rows: list[dict[str, Any]] = []
                    active = schema_supported
                    if not schema_supported and time.monotonic() < deadline:
                        canary = _request_row(
                            str(base_url),
                            str(model_name),
                            fixtures[0],
                            variant,
                            0.0,
                            token_ids,
                            deadline,
                        )
                        canary["schema_absence_canary"] = True
                        rows.append(canary)
                        active = bool(canary["http"].get("ok"))
                    if active:
                        request_failed = False
                        seen = {(str(row["fixture_id"]), float(row["temperature"])) for row in rows}
                        for fixture in fixtures:
                            for temperature in (0.0, 1.0):
                                if time.monotonic() >= deadline:
                                    request_failed = True
                                    break
                                if (str(fixture["id"]), temperature) in seen:
                                    continue
                                row = _request_row(
                                    str(base_url),
                                    str(model_name),
                                    fixture,
                                    variant,
                                    temperature,
                                    token_ids,
                                    deadline,
                                )
                                rows.append(row)
                                if not row["http"].get("ok"):
                                    request_failed = True
                                    break
                            if request_failed:
                                break
                    variants[variant] = {
                        "schema_supported": schema_supported,
                        "effective_supported": any(row["http"].get("ok") for row in rows),
                        "status": (
                            "complete"
                            if active
                            and len(rows) == 2 * len(fixtures)
                            and all(row["http"].get("ok") for row in rows)
                            else "partial_or_blocked"
                        ),
                        "rows": rows,
                        "complete_option_distribution_count": sum(
                            isinstance(row.get("option_probabilities"), Mapping) for row in rows
                        ),
                        "temperature_comparison": _temperature_comparison(rows),
                    }
                result["phases"]["logprob_request_variants"] = {
                    "status": (
                        "complete" if time.monotonic() < deadline else "partial_runtime_budget"
                    ),
                    "fixture_validation": fixture_validation,
                    "prompt_token_counts": prompt_token_counts,
                    "variants": variants,
                }

                prompt_rows = []
                if _variant_schema_support(result, "top_n_logprobs"):
                    for fixture in fixtures[:10]:
                        repetitions = []
                        for _ in range(3):
                            if time.monotonic() >= deadline:
                                break
                            repetitions.append(
                                _request_row(
                                    str(base_url),
                                    str(model_name),
                                    fixture,
                                    "top_n_logprobs",
                                    0.0,
                                    token_ids,
                                    deadline,
                                )
                            )
                        probabilities = [row.get("option_probabilities") for row in repetitions]
                        comparable = len(probabilities) == 3 and all(
                            isinstance(value, Mapping) for value in probabilities
                        )
                        prompt_rows.append(
                            {
                                "fixture_id": fixture["id"],
                                "repetitions": repetitions,
                                "probabilities_identical": (
                                    probabilities[0] == probabilities[1] == probabilities[2]
                                    if comparable
                                    else None
                                ),
                            }
                        )
                        if time.monotonic() >= deadline:
                            break
                comparisons = [
                    row["probabilities_identical"]
                    for row in prompt_rows
                    if row["probabilities_identical"] is not None
                ]
                result["phases"]["determinism"] = {
                    "status": "complete" if len(prompt_rows) == 10 else "partial_runtime_budget",
                    "temperature": 0.0,
                    "seed": RANDOM_SEED,
                    "prompt_rows": prompt_rows,
                    "all_probabilities_identical": all(comparisons) if comparisons else None,
                    "comparable_prompt_count": len(comparisons),
                }
                result["phases"]["summary"] = _build_summary(result)
    except BaseException as exc:
        _record_error(result, "probe", exc)

    if "summary" not in result["phases"]:
        try:
            result["phases"]["summary"] = _build_summary(result)
        except BaseException as exc:
            _record_error(result, "summary", exc)
    try:
        _write_result(output, result, started_at)
    except BaseException as exc:
        result["output_write_error"] = f"{type(exc).__name__}: {exc}"
        if not str(result.get("honest_verdict", "")).startswith("blocked_"):
            result["honest_verdict"] = "blocked_output_write_failed"
        result["duration_s"] = round(time.monotonic() - started_at, 6)
    return result
