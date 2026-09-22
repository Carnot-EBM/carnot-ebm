#!/usr/bin/env python3
"""Serve one-pass option-logit readouts to the local Jev Tetris client.

The live scorer reuses ``LlamaCppLogitScorer`` from the JevBench reproduction.
Composite mode makes five sequential prompt evaluations per Tetris piece.
Ambient score and noul questions do not affect play, so this server skips them.
Spec: REQ-JEV-TETRIS-001 and SCENARIO-JEV-TETRIS-001-A through D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Protocol

from scripts.jevbench_readout_eval import LlamaCppLogitScorer, Option

DELEGATE_OPTION_ID = "__delegate_to_incumbent_tail__"
MAX_READOUT_OPTIONS = 16
NAMED_OPTIONS_WITH_DELEGATE = 15
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8940
MODEL_IDS = {
    "qwen3.8-27b": "unsloth/Qwen3.8-27B-GGUF",
    "qwen3.5-9b": "unsloth/Qwen3.5-9B-GGUF",
}
PLACEMENT_RANK_FIELDS = (
    "holesCreated",
    "bumpinessDelta",
    "maxHeight",
    "aggregateHeight",
    "cleared",
)


@dataclass(frozen=True)
class TokenLabel:
    """One display label whose leading-space form is one model token."""

    display: str
    token_text: str
    token_id: int


@dataclass(frozen=True)
class OptionSelection:
    """Readout options and their deterministic map back to legal placements."""

    options: tuple[Option, ...]
    original_ids: tuple[str, ...]
    tail_ids: tuple[str, ...]
    delegate_target: str | None


class ReadoutScorer(Protocol):
    """Small surface used by request evaluation and mocked by unit tests."""

    model_spec: Mapping[str, object]

    def score(
        self, state: object, question: str, options: Sequence[Option]
    ) -> dict[str, float]: ...


def assign_single_token_labels(
    tokenize: Callable[[str], Sequence[int]], count: int
) -> tuple[TokenLabel, ...]:
    """Assign stable letter labels after checking their exact tokenization."""

    if count < 1 or count > MAX_READOUT_OPTIONS:
        raise ValueError(f"option count must be between 1 and {MAX_READOUT_OPTIONS}")
    labels: list[TokenLabel] = []
    for index in range(26):
        display = chr(ord("A") + index)
        token_text = f" {display}"
        token_ids = list(tokenize(token_text))
        if len(token_ids) == 1:
            labels.append(TokenLabel(display, token_text, int(token_ids[0])))
        if len(labels) == count:
            return tuple(labels)
    raise RuntimeError(f"tokenizer has fewer than {count} usable letter labels")


def _state_text(state: object) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)


def build_prompt(
    state: object,
    question: str,
    options: Sequence[Option],
    labels: Sequence[TokenLabel],
) -> str:
    """Build the exact prompt used for one option-logit forward pass."""

    if len(options) != len(labels):
        raise ValueError("options and labels must have the same length")
    lines = [
        f"{label.display}: {option.option_id} — {option.description}"
        for label, option in zip(labels, options, strict=True)
    ]
    return (
        "Select one declared option. Use the board description and criterion.\n\n"
        f"Board state:\n{_state_text(state)}\n\n"
        f"Criterion:\n{question}\n\n"
        "Options:\n" + "\n".join(lines) + "\n\nReturn only the option label.\nAnswer:"
    )


def _finite_metric(evaluation: Mapping[str, object], field: str, option_id: str) -> float:
    value = evaluation.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"placement {option_id} needs numeric evaluation field {field}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"placement {option_id} evaluation field {field} must be finite")
    return numeric


def select_readout_options(
    criteria: Mapping[str, object],
    placement_evaluations: Mapping[str, object] | None = None,
) -> OptionSelection:
    """Rank oversized placement sets before applying the 15-plus-delegate cap.

    The exact tie-break is fewest new holes, smallest bumpiness increase,
    lowest maximum height, lowest aggregate height, most cleared lines, then
    the original engine order. The final key makes tied rankings stable.
    """

    pairs = tuple((str(option_id), str(description)) for option_id, description in criteria.items())
    if not pairs:
        raise ValueError("choice criteria must contain at least one placement")
    original_ids = tuple(option_id for option_id, _ in pairs)
    if len(pairs) <= MAX_READOUT_OPTIONS:
        return OptionSelection(
            options=tuple(Option(option_id, description) for option_id, description in pairs),
            original_ids=original_ids,
            tail_ids=(),
            delegate_target=None,
        )
    if placement_evaluations is None:
        raise ValueError("oversized choice criteria need placement_evaluations")

    ranked: list[tuple[tuple[float, ...], tuple[str, str]]] = []
    for original_index, pair in enumerate(pairs):
        option_id, _ = pair
        raw_evaluation = placement_evaluations.get(option_id)
        if not isinstance(raw_evaluation, Mapping):
            raise ValueError(f"placement {option_id} needs object evaluation fields")
        holes_created, bumpiness_delta, max_height, aggregate_height, cleared = (
            _finite_metric(raw_evaluation, field, option_id) for field in PLACEMENT_RANK_FIELDS
        )
        rank_key = (
            holes_created,
            bumpiness_delta,
            max_height,
            aggregate_height,
            -cleared,
            float(original_index),
        )
        ranked.append((rank_key, pair))
    ranked_pairs = tuple(pair for _, pair in sorted(ranked, key=lambda item: item[0]))
    named = ranked_pairs[:NAMED_OPTIONS_WITH_DELEGATE]
    tail = ranked_pairs[NAMED_OPTIONS_WITH_DELEGATE:]
    options = tuple(Option(option_id, description) for option_id, description in named) + (
        Option(
            DELEGATE_OPTION_ID,
            "Delegate to the first remaining placement in the heuristic-ranked tail.",
        ),
    )
    return OptionSelection(
        options=options,
        original_ids=original_ids,
        tail_ids=tuple(option_id for option_id, _ in tail),
        delegate_target=tail[0][0],
    )


def _map_probabilities(
    selection: OptionSelection, readout: Mapping[str, float]
) -> dict[str, float]:
    mapped = dict.fromkeys(selection.original_ids, 0.0)
    for option in selection.options:
        probability = float(readout.get(option.option_id, 0.0))
        if option.option_id == DELEGATE_OPTION_ID:
            if selection.delegate_target is not None:
                mapped[selection.delegate_target] += probability
        else:
            mapped[option.option_id] += probability
    total = sum(mapped.values())
    if total <= 0.0:
        raise ValueError("scorer returned no probability mass")
    return {option_id: probability / total for option_id, probability in mapped.items()}


def evaluate_request(payload: object, scorer: ReadoutScorer) -> dict[str, object]:
    """Evaluate choice questions sequentially and return a Jev-shaped object."""

    if not isinstance(payload, Mapping):
        raise ValueError("request must be a JSON object")
    questions = payload.get("questions")
    if not isinstance(questions, Mapping):
        raise ValueError("questions must be an object")
    state = payload.get("state")
    answers: dict[str, object] = {}
    overflow_counts: dict[str, int] = {}
    forward_passes = 0
    for raw_name, raw_question in questions.items():
        name = str(raw_name)
        if not isinstance(raw_question, Mapping) or raw_question.get("type") != "choice":
            continue
        criteria = raw_question.get("criteria")
        if not isinstance(criteria, Mapping):
            raise ValueError(f"choice question {name} needs object criteria")
        placement_evaluations = raw_question.get("placement_evaluations")
        if placement_evaluations is not None and not isinstance(placement_evaluations, Mapping):
            raise ValueError(f"choice question {name} needs object placement_evaluations")
        selection = select_readout_options(criteria, placement_evaluations)
        instructions = str(raw_question.get("instructions", "Choose the best option"))
        readout = scorer.score(state, instructions, selection.options)
        probabilities = _map_probabilities(selection, readout)
        choice = max(selection.original_ids, key=lambda option_id: probabilities[option_id])
        answers[name] = {
            "choice": choice,
            "probabilities": probabilities,
            "confidence": probabilities[choice],
        }
        overflow_counts[name] = len(selection.tail_ids)
        forward_passes += 1
    return {
        "answers": answers,
        "provenance": {
            "model": dict(scorer.model_spec),
            "readout": "next_token_option_logits",
            "forward_passes": forward_passes,
            "voice_execution": "sequential",
            "ambient_questions": "skipped_not_game_relevant",
            "placement_policy": "evaluate_ranked_top15_plus_delegate_to_first_ranked_tail",
            "placement_rank_tiebreak": [
                "holesCreated ascending",
                "bumpinessDelta ascending",
                "maxHeight ascending",
                "aggregateHeight ascending",
                "cleared descending",
                "original engine order ascending",
            ],
            "tail_sizes": overflow_counts,
        },
    }


class PreconditionFailure(RuntimeError):
    """A named resource gate failed before live inference."""

    def __init__(self, verdict: str, message: str) -> None:
        super().__init__(message)
        self.verdict = verdict


def sha256_file(path: Path) -> str:
    """Hash a model or source file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_root() -> Path:
    configured = os.environ.get("HF_HOME")
    if configured:
        return Path(configured).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_cached_model(model_key: str, explicit_path: Path | None = None) -> Path:
    """Resolve one cached GGUF without downloading anything."""

    if model_key not in MODEL_IDS:
        raise ValueError(f"unknown model {model_key}")
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if path.is_file():
            return path
        raise PreconditionFailure("blocked_model_not_cached", f"model file is missing: {path}")
    cache_dir = _cache_root() / f"models--{MODEL_IDS[model_key].replace('/', '--')}"
    candidates = [
        path for path in cache_dir.glob("snapshots/**/*.gguf") if "mmproj" not in path.name.lower()
    ]
    if not candidates:
        raise PreconditionFailure(
            "blocked_model_not_cached", f"no GGUF found for {MODEL_IDS[model_key]} in {cache_dir}"
        )
    quant_order = ("Q4_K_M", "UD-Q4_K_XL", "Q5_K_M", "Q8_0")

    def rank(path: Path) -> tuple[int, int, str]:
        quant_rank = next(
            (index for index, marker in enumerate(quant_order) if marker in path.name),
            len(quant_order),
        )
        return (quant_rank, path.stat().st_size, path.name)

    return min(candidates, key=rank)


def _command_output(argv: Sequence[str]) -> str:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise PreconditionFailure("blocked_gpu_precondition", f"{' '.join(argv)}: {error}")
    return completed.stdout


def _gpu_identity(index: int) -> tuple[str, float]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and int(fields[0]) == index:
            return fields[1], float(fields[2].split()[0])
    raise PreconditionFailure("blocked_gpu_1_missing", f"GPU index {index} is not visible")


def _compute_memory_by_pid(gpu_uuid: str) -> dict[int, float]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    result: dict[int, float] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or fields[0] != gpu_uuid or fields[2].upper() == "N/A":
            continue
        result[int(fields[1])] = result.get(int(fields[1]), 0.0) + float(fields[2].split()[0])
    return result


def check_preconditions() -> dict[str, object]:
    """Check every required resource before any baseline or live run."""

    node_version = _command_output(["node", "--version"]).strip()
    try:
        node_major = int(node_version.lstrip("v").split(".", maxsplit=1)[0])
    except ValueError as error:
        raise PreconditionFailure("blocked_node_version", node_version) from error
    if node_major < 22:
        raise PreconditionFailure("blocked_node_version", f"Node {node_version} is below 22")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise PreconditionFailure(
            "blocked_cuda_visible_devices",
            f"CUDA_VISIBLE_DEVICES must already equal 1, got {visible!r}",
        )
    model_paths = {key: resolve_cached_model(key) for key in MODEL_IDS}
    gpu_uuid, gpu_used = _gpu_identity(1)
    apps = _compute_memory_by_pid(gpu_uuid)
    own_pid = os.getpid()
    other_memory = sum(memory for pid, memory in apps.items() if pid != own_pid)
    if other_memory >= 500.0:
        raise PreconditionFailure(
            "blocked_gpu_1_busy",
            f"GPU 1 has {other_memory:.0f} MiB used by other compute PIDs: {apps}",
        )
    try:
        from llama_cpp import llama_cpp as backend  # noqa: PLC0415
    except ImportError as error:
        raise PreconditionFailure("blocked_llama_cpp_missing", str(error)) from error
    if not backend.llama_supports_gpu_offload():
        raise PreconditionFailure("blocked_no_gpu_offload_support", "llama.cpp lacks CUDA offload")
    return {
        "node": {"available": True, "version": node_version},
        "models": {
            key: {"available": True, "hf_id": MODEL_IDS[key], "path": str(path)}
            for key, path in model_paths.items()
        },
        "cuda_visible_devices": {"available": True, "value": visible},
        "gpu_1": {
            "available": True,
            "uuid": gpu_uuid,
            "memory_used_mib": gpu_used,
            "other_compute_memory_mib": other_memory,
            "compute_apps": apps,
            "excluded_own_pid": own_pid,
        },
        "llama_cpp_gpu_offload": {"available": True},
    }


@contextmanager
def progress_heartbeat(label: str):  # type: ignore[no-untyped-def]
    """Print during calls that may otherwise leave the task silent."""

    stopped = threading.Event()

    def beat() -> None:
        while not stopped.wait(30.0):
            print(f"[progress server] {label} still running", flush=True)

    print(f"[progress server] {label} starting", flush=True)
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1.0)
        print(f"[progress server] {label} finished", flush=True)


class TetrisLlamaCppLogitScorer(LlamaCppLogitScorer):
    """Use the reproduced JevBench logit reader with the Tetris prompt."""

    def __init__(self, model_path: Path, model_key: str, n_ctx: int = 32768) -> None:
        super().__init__(model_path, n_ctx=n_ctx)
        self.model_key = model_key
        self.model_spec: dict[str, object] = {
            "key": model_key,
            "hf_id": MODEL_IDS[model_key],
            "model_path": str(model_path),
            "filename": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "sha256": None,
        }

    def _option_labels(self, count: int) -> tuple[tuple[str, str, int], ...]:
        cached = self._labels_by_count.get(count)
        if cached is not None:
            return cached
        labels = assign_single_token_labels(lambda text: self._tokenize(text, add_bos=False), count)
        result = tuple((label.display, label.token_text, label.token_id) for label in labels)
        self._labels_by_count[count] = result
        return result

    def _build_prompt(
        self,
        state: object,
        question: str,
        options: Sequence[Option],
        labels: Sequence[tuple[str, str, int]],
    ) -> str:
        typed_labels = tuple(TokenLabel(*label) for label in labels)
        return build_prompt(state, question, options, typed_labels)

    def load_and_verify(self, gpu_index: int = 1) -> None:
        """Load once and prove this process gained memory only on GPU 1."""

        gpu_uuid, _ = _gpu_identity(gpu_index)
        gpu_zero_uuid, _ = _gpu_identity(0)
        before = _compute_memory_by_pid(gpu_uuid).get(os.getpid(), 0.0)
        with progress_heartbeat(f"loading {self.model_key}"):
            self.load()
        after_apps: dict[int, float] = {}
        for _ in range(10):
            after_apps = _compute_memory_by_pid(gpu_uuid)
            if after_apps.get(os.getpid(), 0.0) > before:
                break
            time.sleep(0.5)
        after = after_apps.get(os.getpid(), 0.0)
        gpu_zero_memory = _compute_memory_by_pid(gpu_zero_uuid).get(os.getpid(), 0.0)
        if after <= before + 100.0:
            raise PreconditionFailure(
                "blocked_no_gpu_offload_evidence",
                f"own GPU 1 memory did not rise: before={before}, after={after} MiB",
            )
        if gpu_zero_memory > 0.0:
            raise PreconditionFailure(
                "blocked_touched_conductor_gpu_0",
                f"server PID uses {gpu_zero_memory} MiB on GPU 0",
            )
        self.model_spec.update(
            {
                "load_duration_s": self.load_duration_s,
                "gpu_offload_evidence": {
                    **(self.gpu_offload_evidence or {}),
                    "physical_gpu_index": gpu_index,
                    "own_pid": os.getpid(),
                    "memory_before_mib": before,
                    "memory_after_mib": after,
                    "gpu_0_memory_mib": gpu_zero_memory,
                },
            }
        )


class ReadoutHTTPServer(HTTPServer):
    """A sequential server because one llama.cpp context is stateful."""

    scorer: ReadoutScorer
    request_count: int


class ReadoutHandler(BaseHTTPRequestHandler):
    """Accept one bounded local JSON request at ``/ask``."""

    server: ReadoutHTTPServer

    def _send_json(self, status: int, body: Mapping[str, object]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length < 1 or length > 4 * 1024 * 1024:
                raise ValueError("request body must be between 1 byte and 4 MiB")
            payload = json.loads(self.rfile.read(length))
            self.server.request_count += 1
            request_number = self.server.request_count
            started = time.perf_counter()
            with progress_heartbeat(f"request {request_number}"):
                result = evaluate_request(payload, self.server.scorer)
            result["provenance"]["request_duration_s"] = time.perf_counter() - started  # type: ignore[index]
            self._send_json(200, result)
            answers = result["answers"]
            print(
                f"[progress server] request={request_number} choices={len(answers)} served",
                flush=True,
            )
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001
            self._send_json(500, {"error": f"{type(error).__name__}: {error}"})
            print(f"[progress server] request failed: {error}", flush=True)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_IDS),
        default=os.environ.get("JEV_LOCAL_MODEL", "qwen3.8-27b"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ["JEV_LOCAL_MODEL_PATH"])
        if os.environ.get("JEV_LOCAL_MODEL_PATH")
        else None,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = check_preconditions()
        print(json.dumps({"preconditions_checked": receipt}, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        model_path = resolve_cached_model(args.model, args.model_path)
        scorer = TetrisLlamaCppLogitScorer(model_path, args.model, n_ctx=args.n_ctx)
        with progress_heartbeat(f"hashing {model_path.name}"):
            scorer.model_spec["sha256"] = sha256_file(model_path)
        scorer.load_and_verify()
        server = ReadoutHTTPServer((DEFAULT_HOST, args.port), ReadoutHandler)
        server.scorer = scorer
        server.request_count = 0
        print(
            f"[progress server] ready endpoint=http://{DEFAULT_HOST}:{args.port}/ask "
            f"model={MODEL_IDS[args.model]}",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            print("[progress server] shutdown requested", flush=True)
        finally:
            server.server_close()
        return 0
    except PreconditionFailure as error:
        print(
            json.dumps({"honest_verdict": error.verdict, "error": str(error)}, sort_keys=True),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
