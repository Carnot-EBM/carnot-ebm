"""Private offline vLLM option-logprob probe for REQ-INFRA-7089.

The outer loop runs this file on the scored Blackwell machine class.
Importing this module performs no installs, server launches, or inference.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
RESULT_PATH = WORK_DIR / "vllm_logprob_probe.json"
SERVER_LOG_PATH = WORK_DIR / "vllm_server_8919.log"
FIXTURE_PATH = Path(__file__).with_name("fixtures.json")
BASE_URL = "http://127.0.0.1:8919"
LABELS = tuple(f" {chr(ord('A') + index)}" for index in range(16))
MAX_LOGPROBS = 16
RANDOM_SEED = 7089
RUNTIME_BUDGET_S = 40 * 60
SERVER_WAIT_S = 20 * 60
SCORED_MAX_TOKENS = 131072
SCORED_WORST_PROMPT_TOKENS = 22352
SCORED_MAX_MODEL_LEN = SCORED_WORST_PROMPT_TOKENS + SCORED_MAX_TOKENS + 2048
EMBEDDED_FIXTURE_JSON = r"""
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


def softmax_declared_options(
    scores: Mapping[str, float], declared_options: Sequence[str]
) -> dict[str, float]:
    """Normalize only the declared choices, so unrelated tokens cannot take probability mass."""
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


def detect_completion_schema_fields(openapi: Mapping[str, Any]) -> dict[str, Any]:
    """Return every completion request property and the fields needed by this probe."""
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


def embedded_prompt_fixtures() -> list[dict[str, Any]]:
    """Return the upload-safe copy because Kaggle script pushes include only main.py."""
    data = json.loads(EMBEDDED_FIXTURE_JSON)
    if not isinstance(data, list):
        raise ValueError("embedded fixture root must be a list")
    return data


def load_prompt_fixtures(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the frozen synthetic prompts without touching any benchmark input."""
    candidate = path or FIXTURE_PATH
    if candidate.is_file():
        data = json.loads(candidate.read_text(encoding="utf-8"))
    elif path is None:
        data = embedded_prompt_fixtures()
    else:
        raise FileNotFoundError(candidate)
    if not isinstance(data, list):
        raise ValueError("fixture root must be a list")
    return data


def validate_prompt_fixtures(fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the frozen prompt count, IDs, answers, option bounds, and label limit."""
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


def _progress(message: str) -> None:
    print(message, flush=True)


def _write_result(result: dict[str, Any], started_at: float) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    result["duration_s"] = round(time.monotonic() - started_at, 3)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(RESULT_PATH)


def _run_command(args: list[str], timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except BaseException as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def _distribution_metadata(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    metadata = distribution.metadata
    return {
        "installed": True,
        "name": metadata.get("Name", name),
        "version": distribution.version,
        "requires_python": metadata.get("Requires-Python"),
        "requires_dist": metadata.get_all("Requires-Dist") or [],
        "summary": metadata.get("Summary"),
        "location": str(distribution.locate_file("")),
    }


def _vllm_max_seqs() -> int:
    raw = os.environ.get("CARNOT_ARC_VLLM_MAX_SEQS")
    try:
        return max(1, int(raw)) if raw else 8
    except (TypeError, ValueError):
        return 8


def _find_assets() -> dict[str, Any]:
    wheels = list(INPUT_DIR.rglob("vllm-*.whl")) if INPUT_DIR.exists() else []
    model_config = next(
        (
            candidate
            for candidate in INPUT_DIR.rglob("config.json")
            if list(candidate.parent.glob("*.safetensors"))
        ),
        None,
    )
    aot = next(iter(INPUT_DIR.rglob("fp4_gemm_cutlass_sm120.so")), None)
    return {
        "wheels": wheels,
        "selected_wheel": wheels[0] if wheels else None,
        "model_config": model_config,
        "model_dir": model_config.parent if model_config else None,
        "aot": aot,
    }


def _prepare_scored_cuda_and_aot(assets: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    cuda_home = next(
        (
            Path(base) / "nvidia" / "cu13"
            for base in (
                "/usr/local/lib/python3.12/dist-packages",
                "/usr/lib/python3/dist-packages",
            )
            if (Path(base) / "nvidia" / "cu13" / "bin" / "nvcc").exists()
        ),
        None,
    )
    details["cuda_home"] = str(cuda_home) if cuda_home else None
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = str(cuda_home)
        os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
        link_dir = WORK_DIR / "ldlinks"
        link_dir.mkdir(exist_ok=True)
        driver = next((str(path) for path in cuda_home.glob("lib*/stubs/libcuda.so")), None)
        if driver is None:
            ldconfig = _run_command(["ldconfig", "-p"], 20)
            driver = next(
                (
                    line.split()[-1]
                    for line in str(ldconfig.get("stdout", "")).splitlines()
                    if "libcuda.so" in line
                ),
                None,
            )
        if driver is None:
            driver = next(
                (
                    str(path)
                    for pattern in (
                        "/usr/lib/x86_64-linux-gnu/libcuda.so*",
                        "/usr/local/nvidia/lib64/libcuda.so*",
                        "/usr/lib64/libcuda.so*",
                    )
                    for path in sorted(Path("/").glob(pattern.lstrip("/")))
                ),
                None,
            )
        library_dir = next(
            (
                cuda_home / name
                for name in ("lib64", "lib")
                if any((cuda_home / name).glob("libcudart.so*"))
            ),
            cuda_home / "lib64",
        )
        runtime = next(iter(sorted(library_dir.glob("libcudart.so.*"))), None)
        for name, source in (
            ("libcudart.so", runtime),
            ("libcuda.so", Path(driver) if driver else None),
        ):
            target = link_dir / name
            if source and Path(source).exists() and not target.exists():
                target.symlink_to(source)
        all_dirs = [str(link_dir)] + [
            str(path)
            for path in (
                cuda_home / "lib64",
                cuda_home / "lib",
                cuda_home / "lib64" / "stubs",
                cuda_home / "lib" / "stubs",
            )
            if path.is_dir()
        ]
        os.environ["LIBRARY_PATH"] = ":".join(all_dirs) + ":" + os.environ.get("LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            ":".join(all_dirs[1:]) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        details.update(
            {
                "driver": driver,
                "library_dir": str(library_dir),
                "link_names": sorted(path.name for path in link_dir.iterdir()),
                "library_path_dirs": all_dirs,
            }
        )
    aot = assets.get("aot")
    if isinstance(aot, Path):
        try:
            flashinfer = __import__("flashinfer")
            architecture = aot.parents[2]
            destination = (
                Path("/root/.cache/flashinfer") / flashinfer.__version__ / architecture.name
            )
            shutil.copytree(architecture, destination, dirs_exist_ok=True)
            details["aot_staged_to"] = str(destination)
        except BaseException as exc:
            details["aot_stage_error"] = f"{type(exc).__name__}: {exc}"
    else:
        details["aot_stage_error"] = "fp4_gemm_cutlass_sm120.so not found"
    return details


def _filter_logprob_help(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if "logprob" in line.lower()]


def _http_json(
    path: str, payload: Mapping[str, Any] | None = None, timeout: float = 120
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, timeout)) as response:
            raw = response.read().decode(errors="replace")
            try:
                body: Any = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = {"raw_body": raw}
            return {"ok": True, "status": int(response.status), "body": body}
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": int(exc.code),
            "error_body": exc.read().decode(errors="replace"),
        }
    except BaseException as exc:
        return {
            "ok": False,
            "status": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _healthy(timeout: float = 3) -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except BaseException:
        return False


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
    fixture: Mapping[str, Any],
    variant: str,
    temperature: float,
    token_ids: Mapping[str, int],
    deadline: float,
) -> dict[str, Any]:
    labels = LABELS[: len(fixture["options"])]
    payload: dict[str, Any] = {
        "model": "m",
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
    timeout = min(120.0, max(1.0, deadline - time.monotonic()))
    started = time.monotonic()
    response = _http_json("/v1/completions", payload, timeout)
    latency = time.monotonic() - started
    scores, missing = _option_scores(response, labels, token_ids)
    probabilities = None
    normalization_error = None
    if not missing and variant != "prompt_logprobs":
        try:
            probabilities = softmax_declared_options(scores, labels)
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


def _phase_environment(result: dict[str, Any], runtime: dict[str, Any]) -> None:
    assets = _find_assets()
    runtime["assets"] = assets
    gpu = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        20,
    )
    selected_wheel = assets["selected_wheel"]
    install: dict[str, Any]
    if isinstance(selected_wheel, Path):
        install_started = time.monotonic()
        install = _run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={selected_wheel.parent}",
                "vllm",
            ],
            3600,
        )
        install["duration_s"] = round(time.monotonic() - install_started, 3)
    else:
        install = {"returncode": None, "error": "no vllm wheel found"}
    cuda_and_aot = _prepare_scored_cuda_and_aot(assets)
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except BaseException as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    dependency_names = (
        "vllm",
        "torch",
        "flashinfer-python",
        "triton",
        "transformers",
        "tokenizers",
        "fastapi",
        "pydantic",
        "uvicorn",
    )
    checks = [
        {
            "resource": "blackwell_gpu",
            "available": gpu.get("returncode") == 0 and bool(str(gpu.get("stdout", "")).strip()),
            "detail": str(gpu.get("stdout") or gpu.get("stderr") or gpu.get("error") or "").strip(),
        },
        {
            "resource": "offline_vllm_wheel",
            "available": isinstance(selected_wheel, Path),
            "detail": str(selected_wheel) if selected_wheel else "not found",
        },
        {
            "resource": "flashinfer_aot_sm120",
            "available": isinstance(assets["aot"], Path),
            "detail": str(assets["aot"]) if assets["aot"] else "not found",
        },
        {
            "resource": "qwen38_nvfp4_safetensors",
            "available": isinstance(assets["model_dir"], Path),
            "detail": str(assets["model_dir"]) if assets["model_dir"] else "not found",
        },
        {
            "resource": "offline_vllm_install",
            "available": install.get("returncode") == 0,
            "detail": f"pip return code {install.get('returncode')}",
        },
        {
            "resource": "offline_execution",
            "available": True,
            "detail": "external network is not required or used; only the loopback API is called",
        },
    ]
    result["preconditions_checked"] = checks
    result["phases"]["environment"] = {
        "status": "complete" if all(item["available"] for item in checks[:5]) else "blocked",
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "gpu": gpu,
        "wheel_filenames": [path.name for path in assets["wheels"]],
        "selected_wheel": str(selected_wheel) if selected_wheel else None,
        "model_dir": str(assets["model_dir"]) if assets["model_dir"] else None,
        "flashinfer_aot": str(assets["aot"]) if assets["aot"] else None,
        "pip_install": install,
        "vllm_version": _distribution_metadata("vllm").get("version"),
        "torch": torch_info,
        "pip_metadata": {name: _distribution_metadata(name) for name in dependency_names},
        "scored_cuda_and_aot_setup": cuda_and_aot,
    }


def _phase_schema_help(result: dict[str, Any]) -> None:
    help_result = _run_command(["vllm", "serve", "--help"], 120)
    full_text = str(help_result.get("stdout", "")) + "\n" + str(help_result.get("stderr", ""))
    filtered = _filter_logprob_help(full_text)
    result["phases"]["schema"] = {
        "status": "help_captured_openapi_pending_server",
        "vllm_serve_help_returncode": help_result.get("returncode"),
        "vllm_serve_help_logprob_lines": filtered,
        "max_logprobs_cap_supported_by_help": any(
            "max-logprobs" in line.lower() for line in filtered
        ),
        "max_logprobs_cap_configured": MAX_LOGPROBS,
        "openapi": None,
    }


def _phase_server(result: dict[str, Any], runtime: dict[str, Any], deadline: float) -> None:
    assets = runtime.get("assets", {})
    model_dir = assets.get("model_dir") if isinstance(assets, Mapping) else None
    install_ok = (
        result.get("phases", {}).get("environment", {}).get("pip_install", {}).get("returncode")
        == 0
    )
    if not isinstance(model_dir, Path) or not install_ok:
        result["phases"]["server"] = {
            "status": "blocked_missing_model_or_vllm_install",
            "startup_time_s": None,
        }
        return
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_dir),
        "--served-model-name",
        "m",
        "--max-model-len",
        str(SCORED_MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        str(_vllm_max_seqs()),
        "--kv-cache-dtype",
        "fp8",
        "--port",
        "8919",
        "--host",
        "127.0.0.1",
        "--max-logprobs",
        str(MAX_LOGPROBS),
    ]
    SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = SERVER_LOG_PATH.open("ab")
    started = time.monotonic()
    process = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
    runtime["process"] = process
    runtime["server_log_file"] = log_file
    ready = False
    next_progress = 30
    wait_limit = min(SERVER_WAIT_S, max(0.0, deadline - started))
    while time.monotonic() - started < wait_limit:
        time.sleep(5)
        elapsed = time.monotonic() - started
        if elapsed >= next_progress:
            _progress(f"PHASE 3 SERVER waiting {int(elapsed)}s")
            next_progress += 30
        if _healthy():
            ready = True
            break
        if process.poll() is not None:
            break
    startup_time = time.monotonic() - started
    server_phase = {
        "status": "ready" if ready else "blocked_not_ready",
        "startup_time_s": round(startup_time, 3),
        "launch_argv": args,
        "scored_launch_difference": {"added_only": ["--max-logprobs", str(MAX_LOGPROBS)]},
        "log_path": str(SERVER_LOG_PATH),
        "process_returncode": process.poll(),
    }
    if not ready:
        try:
            server_phase["log_tail"] = SERVER_LOG_PATH.read_bytes()[-4000:].decode(errors="replace")
        except OSError as exc:
            server_phase["log_tail_error"] = f"{type(exc).__name__}: {exc}"
    result["phases"]["server"] = server_phase
    runtime["server_ready"] = ready
    if not ready:
        return
    result["inference_substrate"] = "live_llm_inference"
    openapi_response = _http_json(
        "/openapi.json", timeout=min(120.0, max(1.0, deadline - time.monotonic()))
    )
    schema_phase = result["phases"].setdefault("schema", {})
    if openapi_response.get("ok") and isinstance(openapi_response.get("body"), Mapping):
        detection = detect_completion_schema_fields(openapi_response["body"])
        detection["supported"]["max_logprobs_cap"] = True
        detection["max_logprobs_cap_evidence"] = {
            "configured_value": MAX_LOGPROBS,
            "help_lists_option": bool(schema_phase.get("max_logprobs_cap_supported_by_help")),
            "server_started_with_option": True,
        }
        schema_phase["openapi"] = detection
        schema_phase["status"] = "complete"
    else:
        schema_phase["openapi_error"] = openapi_response
        schema_phase["status"] = "blocked_openapi_unavailable"


def _phase_tokenize(result: dict[str, Any], runtime: dict[str, Any]) -> None:
    if not runtime.get("server_ready"):
        result["phases"]["tokenize_check"] = {"status": "blocked_server_not_ready"}
        return
    rows = []
    token_ids: dict[str, int] = {}
    for label in LABELS:
        response = _http_json("/tokenize", {"model": "m", "prompt": label}, timeout=120)
        ids = _token_ids_from_response(response)
        body = response.get("body")
        count = body.get("count") if isinstance(body, Mapping) else None
        if not isinstance(count, int):
            count = len(ids)
        if count == 1 and len(ids) == 1:
            token_ids[label] = ids[0]
        rows.append(
            {
                "label": label,
                "token_count": count,
                "token_ids": ids,
                "one_token": count == 1 and len(ids) == 1,
                "raw_response": response,
            }
        )
    runtime["label_token_ids"] = token_ids
    result["phases"]["tokenize_check"] = {
        "status": "complete" if len(token_ids) == len(LABELS) else "blocked_non_single_token_label",
        "prompt_style": "raw completion prompt, matching the scored /v1/completions path",
        "all_16_labels_are_one_token": len(token_ids) == len(LABELS),
        "labels": rows,
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


def _phase_variants(result: dict[str, Any], runtime: dict[str, Any], deadline: float) -> None:
    if not runtime.get("server_ready"):
        result["phases"]["logprob_request_variants"] = {"status": "blocked_server_not_ready"}
        return
    fixtures = load_prompt_fixtures()
    fixture_validation = validate_prompt_fixtures(fixtures)
    if not fixture_validation["valid"]:
        result["phases"]["logprob_request_variants"] = {
            "status": "blocked_invalid_fixtures",
            "fixture_validation": fixture_validation,
        }
        return
    token_ids = runtime.get("label_token_ids", {})
    prompt_token_counts: dict[str, Any] = {}
    for fixture in fixtures:
        if time.monotonic() >= deadline:
            break
        response = _http_json(
            "/tokenize",
            {"model": "m", "prompt": _format_prompt(fixture)},
            timeout=min(120.0, max(1.0, deadline - time.monotonic())),
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
            }
            continue
        rows: list[dict[str, Any]] = []
        active = schema_supported
        if not schema_supported and time.monotonic() < deadline:
            canary = _request_row(fixtures[0], variant, 0.0, token_ids, deadline)
            canary["schema_absence_canary"] = True
            rows.append(canary)
            active = bool(canary["http"].get("ok"))
        if active:
            seen = {(str(row["fixture_id"]), float(row["temperature"])) for row in rows}
            for fixture in fixtures:
                for temperature in (0.0, 1.0):
                    if time.monotonic() >= deadline:
                        break
                    key = (str(fixture["id"]), temperature)
                    if key not in seen:
                        rows.append(
                            _request_row(fixture, variant, temperature, token_ids, deadline)
                        )
                if time.monotonic() >= deadline:
                    break
        variants[variant] = {
            "schema_supported": schema_supported,
            "effective_supported": any(row["http"].get("ok") for row in rows),
            "status": "complete"
            if active and time.monotonic() < deadline
            else "partial_or_blocked",
            "rows": rows,
            "complete_option_distribution_count": sum(
                isinstance(row.get("option_probabilities"), Mapping) for row in rows
            ),
            "temperature_comparison": _temperature_comparison(rows),
        }
    result["phases"]["logprob_request_variants"] = {
        "status": "complete" if time.monotonic() < deadline else "partial_runtime_budget",
        "fixture_validation": fixture_validation,
        "prompt_token_counts": prompt_token_counts,
        "variants": variants,
    }


def _phase_determinism(result: dict[str, Any], runtime: dict[str, Any], deadline: float) -> None:
    if not runtime.get("server_ready") or not _variant_schema_support(result, "top_n_logprobs"):
        result["phases"]["determinism"] = {"status": "blocked_top_n_logprobs_unavailable"}
        return
    fixtures = load_prompt_fixtures()[:10]
    token_ids = runtime.get("label_token_ids", {})
    prompt_rows = []
    for fixture in fixtures:
        repetitions = []
        for _ in range(3):
            if time.monotonic() >= deadline:
                break
            repetitions.append(_request_row(fixture, "top_n_logprobs", 0.0, token_ids, deadline))
        probabilities = [row.get("option_probabilities") for row in repetitions]
        comparable = len(probabilities) == 3 and all(
            isinstance(value, Mapping) for value in probabilities
        )
        prompt_rows.append(
            {
                "fixture_id": fixture["id"],
                "repetitions": repetitions,
                "probabilities_identical": (
                    probabilities[0] == probabilities[1] == probabilities[2] if comparable else None
                ),
            }
        )
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
    tokenize = result.get("phases", {}).get("tokenize_check", {})
    all_labels = bool(tokenize.get("all_16_labels_are_one_token"))
    if recommended_name:
        request_shape: dict[str, Any] = {
            "status": "recommended",
            "variant": recommended_name,
            "common_fields": {
                "model": "m",
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
    exact_id_variant = recommended_name in ("logprob_token_ids", "allowed_token_ids")
    if exact_id_variant and all_labels and schema_supported.get("max_logprobs_cap"):
        maximum_options = MAX_LOGPROBS
        maximum_basis = "16 single-token labels and the configured max-logprobs cap"
    else:
        maximum_options = direct_max
        maximum_basis = "largest complete declared-option distribution observed"
    server_ready = result.get("phases", {}).get("server", {}).get("status") == "ready"
    if not server_ready:
        verdict = "blocked_server_never_started"
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


def _record_phase_error(result: dict[str, Any], phase: str, exc: BaseException) -> None:
    result["phases"][phase] = {
        "status": "blocked_exception",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc().splitlines()[-12:],
    }


def _stop_server(result: dict[str, Any], runtime: dict[str, Any]) -> None:
    process = runtime.get("process")
    log_file = runtime.get("server_log_file")
    cleanup: dict[str, Any] = {"owned_process_started": process is not None}
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
            cleanup["termination"] = "terminated"
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
            cleanup["termination"] = "killed_after_timeout"
    if process is not None:
        cleanup["returncode"] = process.poll()
    if log_file is not None:
        log_file.close()
    result["server_cleanup"] = cleanup


def main() -> int:
    started_at = time.monotonic()
    deadline = started_at + RUNTIME_BUDGET_S
    result: dict[str, Any] = {
        "schema_version": 1,
        "probe": "carnot-vllm-logprob-probe",
        "requirement": "REQ-INFRA-7089",
        "inference_substrate": "hardware_smoke",
        "random_seed": RANDOM_SEED,
        "duration_s": 0.0,
        "honest_verdict": "blocked_not_started",
        "preconditions_checked": [],
        "runtime_budget_s": RUNTIME_BUDGET_S,
        "phases": {},
    }
    runtime: dict[str, Any] = {"process": None, "server_ready": False}

    def run_phase(number: int, key: str, function: Any, *args: Any) -> None:
        if time.monotonic() >= deadline:
            _progress(f"PHASE {number} {key.upper()} skipped after runtime budget")
            result["phases"][key] = {"status": "skipped_runtime_budget"}
            _write_result(result, started_at)
            return
        _progress(f"PHASE {number} {key.upper()} start")
        try:
            function(result, *args)
        except BaseException as exc:
            _record_phase_error(result, key, exc)
        _write_result(result, started_at)
        _progress(f"PHASE {number} {key.upper()} saved")

    try:
        run_phase(1, "environment", _phase_environment, runtime)
        run_phase(2, "schema", _phase_schema_help)
        run_phase(3, "server", _phase_server, runtime, deadline)
        run_phase(4, "tokenize_check", _phase_tokenize, runtime)
        run_phase(5, "logprob_request_variants", _phase_variants, runtime, deadline)
        run_phase(6, "determinism", _phase_determinism, runtime, deadline)
        if time.monotonic() < deadline:
            _progress("PHASE 7 SUMMARY start")
        else:
            _progress("FINALIZE SUMMARY after runtime budget")
        result["phases"]["summary"] = _build_summary(result)
        _write_result(result, started_at)
        _progress("PHASE 7 SUMMARY saved")
    except BaseException as exc:
        result["fatal"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc().splitlines()[-12:],
        }
        result["phases"]["summary"] = _build_summary(result)
        _write_result(result, started_at)
    finally:
        try:
            _stop_server(result, runtime)
        except BaseException as exc:
            result["server_cleanup"] = {"error": f"{type(exc).__name__}: {exc}"}
        result["phases"]["summary"] = _build_summary(result)
        _write_result(result, started_at)
        _progress(f"PROBE COMPLETE {result['honest_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
