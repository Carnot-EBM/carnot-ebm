"""Exp5567 local SOTA solve-versus-verify asymmetry panel.

Spec refs: REQ-VERIFY-5567, SCENARIO-VERIFY-5567.

This experiment measures whether local frontier GGUF models are better at
checking exact ASP/FSM candidates than generating them.  The LLM verifier is
never treated as the oracle: generated candidates and verifier labels are
scored only against the Exp5566 exact ASP/FSM validators.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as corpus5566
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair


JsonDict = dict[str, Any]
PairResolver = Callable[[], Sequence[Mapping[str, Any]] | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json")
CORPUS_RELATIVE_PATH = corpus5566.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5567.local_sota_solve_verify_asymmetry.v504"
EXPERIMENT = 5567
EXPERIMENT_ID = "exp5567-local-sota-solve-verify-asymmetry"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
RANDOM_SEED = 5567
MIN_INDEPENDENT_INSTANCES = 36
BOOTSTRAP_ITERATIONS = 500
N_GPU_LAYERS = -1
LIVE_BATCH_PAIR_COUNT = 9
INFERENCE_SUBSTRATE = "live_local_sota_gguf_plus_exact_validator"
SPEC_REFS = ("REQ-VERIFY-5567", "SCENARIO-VERIFY-5567", "REQ-VERIFY-5566")

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_IDS = (
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_HEADLINE_IDS = (QWEN_ID, *GEMMA_IDS)

ARMS = (
    "discrete_verdict",
    "criteria_decomposition",
    "granular_score",
    "repeated_verdict_3x",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "gate_receipt",
    "MODEL_SPECS",
    "model_cache_paths",
    "live_model_invoked",
    "gpu_offload_authenticated",
    "device_receipt",
    "corpus_path",
    "n_independent_instances",
    "family_counts",
    "arms",
    "solve_accuracy_by_model",
    "verifier_metrics_by_model_and_arm",
    "solve_verify_asymmetry",
    "confidence_intervals",
    "mcnemar_results",
    "exact_validator_is_oracle",
    "verifier_is_oracle",
    "parser_failure_count",
    "raw_response_hash",
    "inference_duration_s",
    "inference_substrate",
    "honest_verdict",
    "panel_complete",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Keeps every headline and gate field annotated by the evidence boundary it protects.",
    "gate_receipt": "Records cache, CUDA, offload, corpus, and no-legacy-model gates before interpreting model quality.",
    "MODEL_SPECS": "Names the exact headline GGUF models and local paths so Qwen plus Gemma cannot be silently replaced.",
    "model_cache_paths": "Pins resolved cache files without causing downloads.",
    "live_model_invoked": "Separates real local inference from blocked preflight artifacts.",
    "gpu_offload_authenticated": "Prevents CPU-only llama.cpp receipts from unlocking headline claims.",
    "device_receipt": "Preserves CUDA device identity and offload evidence for replay.",
    "corpus_path": "Pins the Exp 5566 exact-labeled source corpus.",
    "n_independent_instances": "Defines the paired statistical denominator.",
    "family_counts": "Confirms the sample did not collapse onto one ASP/FSM family.",
    "arms": "Names the four verifier prompt strategies being compared.",
    "solve_accuracy_by_model": "Measures direct generation success under the exact validator.",
    "verifier_metrics_by_model_and_arm": "Measures verifier balanced accuracy, FPR, and FNR against the exact oracle labels.",
    "solve_verify_asymmetry": "Reports solve accuracy minus verifier balanced accuracy without implying a moat claim.",
    "confidence_intervals": "Stores paired bootstrap intervals over independent instances.",
    "mcnemar_results": "Stores paired solve-versus-verify discordance tests.",
    "exact_validator_is_oracle": "Bare boolean disclosing the exact ASP/FSM validators are the correctness oracle.",
    "verifier_is_oracle": "Bare boolean disclosing the LLM verifier is oracle-distinct and not trusted as authority.",
    "parser_failure_count": "Keeps malformed structured responses visible.",
    "raw_response_hash": "Preserves response provenance without relying on prose summaries.",
    "inference_duration_s": "Records actual model inference wall time rather than artifact formatting time.",
    "inference_substrate": "Declares live local SOTA GGUF inference plus exact validator scoring.",
    "honest_verdict": "Provides a terminal status that distinguishes complete, blocked cache, blocked CUDA/offload, and failed parsing states.",
    "panel_complete": "Only an authenticated statistically valid panel may unlock the co-evolution audit.",
}

WORKER_CODE = r"""
import argparse
import json
import time


def _extract_text(raw):
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first.get("text") or "")
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
    return ""


def _usage(raw, prompt, response):
    if isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
        return raw["usage"]
    return {
        "prompt_tokens": len(str(prompt).split()),
        "completion_tokens": len(str(response).split()),
        "total_tokens": len(str(prompt).split()) + len(str(response).split()),
        "source": "whitespace_estimate",
    }


parser = argparse.ArgumentParser()
parser.add_argument("--workload", required=True)
args = parser.parse_args()
payload = json.loads(open(args.workload, "r", encoding="utf-8").read())
started = time.perf_counter()
llm = None
responses = []
try:
    import torch
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp import llama_cpp as low

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append({"index": index, "name": torch.cuda.get_device_name(index)})
    llm = Llama(
        model_path=payload["model_path"],
        n_ctx=int(payload.get("n_ctx", 8192)),
        n_batch=int(payload.get("n_batch", 256)),
        n_gpu_layers=int(payload.get("n_gpu_layers", -1)),
        seed=int(payload.get("seed", 5567)),
        verbose=True,
    )
    for task in payload["tasks"]:
        task_started = time.perf_counter()
        try:
            raw = llm(
                task["prompt"],
                max_tokens=int(task["max_tokens"]),
                temperature=float(task.get("temperature", 0.0)),
                top_p=1.0,
                repeat_penalty=1.0,
                seed=int(task.get("seed", payload.get("seed", 5567))),
            )
            text = _extract_text(raw)
            responses.append(
                {
                    "task_id": task["task_id"],
                    "ok": bool(text),
                    "text": text,
                    "duration_s": round(time.perf_counter() - task_started, 6),
                    "usage": _usage(raw, task["prompt"], text),
                    "error": "",
                }
            )
        except Exception as exc:
            responses.append(
                {
                    "task_id": task["task_id"],
                    "ok": False,
                    "text": "",
                    "duration_s": round(time.perf_counter() - task_started, 6),
                    "usage": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print(
        json.dumps(
            {
                "ok": True,
                "model_hf_id": payload["model_hf_id"],
                "model_path": payload["model_path"],
                "llama_cpp_version": getattr(llama_cpp, "__version__", None),
                "llama_cpp_supports_gpu_offload": bool(low.llama_supports_gpu_offload()),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_device_count": int(torch.cuda.device_count()),
                "devices": devices,
                "load_and_inference_duration_s": round(time.perf_counter() - started, 6),
                "responses": responses,
            },
            sort_keys=True,
        )
    )
except Exception as exc:
    print(
        json.dumps(
            {
                "ok": False,
                "model_hf_id": payload.get("model_hf_id", ""),
                "model_path": payload.get("model_path", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "load_and_inference_duration_s": round(time.perf_counter() - started, 6),
                "responses": responses,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1)
finally:
    close = getattr(llm, "close", None)
    if callable(close):
        close()
"""


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used for checksums and hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for a text response or prompt."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible content."""

    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def resolve_headline_model_specs(
    *,
    pair_resolver: PairResolver = cached_sota_pair,
) -> tuple[list[JsonDict], JsonDict]:
    """Resolve Qwen plus one Gemma through the repository cached SOTA path."""

    gate: JsonDict = {
        "cached_sota_pair_called": True,
        "cache_gate_passed": False,
        "blocked_reason": "blocked_missing_sota_cache",
        "cached_pair_hf_ids": [],
        "selected_headline_model_ids": [],
        "legacy_cpu_model_substituted": False,
        "corpus_gate_passed": False,
        "offload_gate_passed": False,
    }
    try:
        resolved = list(pair_resolver() or [])
    except Exception as exc:  # noqa: BLE001
        gate["resolver_error"] = f"{type(exc).__name__}: {exc}"
        return [], gate

    gate["cached_pair_hf_ids"] = [str(row.get("hf_id", "")) for row in resolved]
    enriched = [_enrich_model_spec(row) for row in resolved]
    qwen = next((row for row in enriched if row.get("hf_id") == QWEN_ID), None)
    gemma = next((row for row in enriched if row.get("hf_id") in GEMMA_IDS), None)
    selected = [row for row in (qwen, gemma) if row is not None]
    paths_present = all(Path(str(row.get("model_path", ""))).is_file() for row in selected)
    if len(selected) != 2 or not paths_present:
        gate["missing_headline_ids"] = [
            model_id
            for model_id in (QWEN_ID, "one_of:" + "|".join(GEMMA_IDS))
            if (model_id == QWEN_ID and qwen is None)
            or (model_id.startswith("one_of:") and gemma is None)
        ]
        return [], gate

    gate["cache_gate_passed"] = True
    gate["blocked_reason"] = ""
    gate["selected_headline_model_ids"] = [str(row["hf_id"]) for row in selected]
    return selected, gate


def _enrich_model_spec(row: Mapping[str, Any]) -> JsonDict:
    spec = dict(row)
    hf_id = str(spec.get("hf_id", ""))
    registry = {item["hf_id"]: item for item in SOTA_GGUF_MODELS}
    if hf_id in registry:
        spec.setdefault("role", registry[hf_id]["role"])
        spec.setdefault("active_params_b", registry[hf_id]["active_params_b"])
        spec.setdefault("total_params_b", registry[hf_id]["total_params_b"])
        spec.setdefault("quantization", registry[hf_id]["quantization"])
    spec["family"] = model_family(hf_id)
    spec["headline_eligible"] = hf_id == QWEN_ID or hf_id in GEMMA_IDS
    spec["local_model_present"] = Path(str(spec.get("model_path", ""))).is_file()
    return spec


def model_family(hf_id: str) -> str:
    """Return the coarse family used for Qwen-vs-Gemma deltas."""

    lower = hf_id.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    return "other"


def load_corpus_rows(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Load the checked-in Exp5566 corpus rows without regenerating labels."""

    try:
        artifact = json.loads((repo_root / CORPUS_RELATIVE_PATH).read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = artifact.get("corpus_rows")
    if artifact.get("corpus_ready") is not True or not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def sample_independent_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    n: int = MIN_INDEPENDENT_INSTANCES,
) -> list[JsonDict]:
    """Sample balanced valid/near-miss pairs with instance ID as the unit."""

    valid_by_id = {
        str(row.get("row_id")): dict(row)
        for row in rows
        if row.get("label") == "valid" and row.get("accepted_by_exact_validator") is True
    }
    by_family: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        if row.get("label") != "invalid" or row.get("accepted_by_exact_validator") is not False:
            continue
        parent = str(row.get("parent_row_id") or "")
        valid = valid_by_id.get(parent)
        if valid is None:
            continue
        family = str(row.get("family"))
        instance_id = parent.removeprefix("exp5566_").removesuffix("_valid")
        by_family[family].append(
            {
                "instance_id": instance_id,
                "family": family,
                "candidate_kind": valid.get("candidate_kind"),
                "valid_row": valid,
                "invalid_row": dict(row),
            }
        )

    families = [family for family in corpus5566.REQUIRED_FAMILIES if by_family.get(family)]
    if not families:
        return []
    base = n // len(families)
    remainder = n % len(families)
    selected: list[JsonDict] = []
    for index, family in enumerate(families):
        quota = base + (1 if index < remainder else 0)
        selected.extend(sorted(by_family[family], key=lambda row: str(row["instance_id"]))[:quota])
    return selected[:n]


def family_counts_from_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count independent sampled instances by corpus family."""

    counts = Counter(str(pair.get("family")) for pair in pairs)
    return {family: int(counts[family]) for family in sorted(counts)}


def extract_json_object(text: str) -> tuple[JsonDict | None, str]:
    """Extract the first JSON object from a short model response."""

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value, ""
        return None, "json_not_object"
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return None, "json_parse_failure"
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(cleaned[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(cleaned[start : offset + 1])
                except json.JSONDecodeError:
                    return None, "json_parse_failure"
                if isinstance(value, dict):
                    return value, ""
                return (
                    None,
                    "json_not_object",
                )  # pragma: no cover - balanced braces parse only as dict
    return None, "json_parse_failure"


def normalize_label(value: Any) -> str | None:
    """Normalize model verdict words to valid or invalid labels."""

    text = str(value).strip().lower()
    valid = {"valid", "accept", "accepted", "true", "yes", "pass", "passed"}
    invalid = {"invalid", "reject", "rejected", "false", "no", "fail", "failed", "near_miss"}
    if text in valid:
        return "valid"
    if text in invalid:
        return "invalid"
    return None


def parse_verifier_response(text: str, arm: str) -> tuple[str | None, str]:
    """Parse one verifier-arm response into a binary label."""

    payload, error = extract_json_object(text)
    if payload is None:
        return None, f"verifier_{error}"
    if arm == "granular_score":
        try:
            score = float(payload.get("score"))
        except (TypeError, ValueError):
            return None, "verifier_missing_score"
        if not 0.0 <= score <= 100.0:
            return None, "verifier_score_out_of_range"
        return ("valid" if score >= 50.0 else "invalid"), ""
    verdict = normalize_label(payload.get("verdict", payload.get("label")))
    if verdict is not None:
        return verdict, ""
    if arm == "criteria_decomposition":
        criteria = payload.get("criteria")
        if isinstance(criteria, Mapping) and criteria:
            return ("valid" if all(bool(value) for value in criteria.values()) else "invalid"), ""
    return None, "verifier_missing_label"


def parse_and_score_solve_response(text: str, pair: Mapping[str, Any]) -> JsonDict:
    """Parse one direct solve and score it with the exact Exp5566 validator."""

    response_hash = sha256_text(text)
    payload, error = extract_json_object(text)
    if payload is None:
        return {
            "parser_ok": False,
            "exact_accepted": False,
            "response_hash": response_hash,
            "error_type": "solve_" + error,
        }
    valid_row = dict(pair["valid_row"])
    candidate_kind = str(payload.get("candidate_kind", valid_row.get("candidate_kind", "")))
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        return {
            "parser_ok": False,
            "exact_accepted": False,
            "response_hash": response_hash,
            "error_type": "solve_missing_candidate",
        }
    row = {
        "candidate_kind": candidate_kind,
        "candidate": dict(candidate),
        "expected_signature_sha256": valid_row["expected_signature_sha256"],
    }
    try:
        validation = corpus5566.exact_validate_corpus_row(row)
    except Exception as exc:  # noqa: BLE001
        return {
            "parser_ok": True,
            "exact_accepted": False,
            "response_hash": response_hash,
            "candidate_kind": candidate_kind,
            "error_type": f"solve_exact_validation_error:{type(exc).__name__}",
        }
    return {
        "parser_ok": True,
        "exact_accepted": bool(validation["accepted"]),
        "response_hash": response_hash,
        "candidate_kind": candidate_kind,
        "actual_signature_sha256": validation["actual_signature_sha256"],
        "error_type": "" if validation["accepted"] else "solve_exact_rejected",
    }


def compute_solve_accuracy(
    solve_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
) -> dict[str, JsonDict]:
    """Compute direct-solve exact acceptance rate per model."""

    out: dict[str, JsonDict] = {}
    for model_id in model_ids:
        rows = [row for row in solve_records if row.get("model_hf_id") == model_id]
        correct = sum(1 for row in rows if row.get("exact_accepted") is True)
        out[model_id] = {
            "accuracy": _rate(correct, len(rows)),
            "correct": correct,
            "n": len(rows),
            "parser_failures": sum(1 for row in rows if row.get("parser_ok") is not True),
            "independent_unit": "instance_id",
        }
    return out


def compute_verifier_metrics(
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    arms: Sequence[str],
) -> dict[str, dict[str, JsonDict]]:
    """Compute verifier balanced accuracy, FPR, and FNR per model and arm."""

    out: dict[str, dict[str, JsonDict]] = {}
    for model_id in model_ids:
        out[model_id] = {}
        for arm in arms:
            rows = [
                row
                for row in verifier_records
                if row.get("model_hf_id") == model_id and row.get("arm") == arm
            ]
            counts = _confusion_counts(rows)
            positive = counts["tp"] + counts["fn"]
            negative = counts["tn"] + counts["fp"]
            tpr = _rate(counts["tp"], positive)
            tnr = _rate(counts["tn"], negative)
            out[model_id][arm] = {
                "balanced_accuracy": round((tpr + tnr) / 2.0, 6) if rows else 0.0,
                "accuracy": _rate(counts["tp"] + counts["tn"], len(rows)),
                "false_positive_rate": _rate(counts["fp"], negative),
                "false_negative_rate": _rate(counts["fn"], positive),
                "tp": counts["tp"],
                "tn": counts["tn"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "n_candidates": len(rows),
                "n_independent_instances": len({str(row.get("instance_id")) for row in rows}),
                "parser_failures": counts["parser_failures"],
                "n_repeated_calls": sum(len(row.get("response_hashes", []) or []) for row in rows),
                "repeat_calls_per_candidate": 3 if arm == "repeated_verdict_3x" else 1,
                "independent_unit": "instance_id",
            }
    return out


def _confusion_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter({"tp": 0, "tn": 0, "fp": 0, "fn": 0, "parser_failures": 0})
    for row in rows:
        true_label = normalize_label(row.get("true_label"))
        predicted = normalize_label(row.get("predicted_label"))
        true_valid = true_label == "valid"
        if predicted is None:
            counts["parser_failures"] += 1
            counts["fn" if true_valid else "fp"] += 1
        elif true_valid and predicted == "valid":
            counts["tp"] += 1
        elif true_valid:
            counts["fn"] += 1
        elif predicted == "valid":
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return counts


def compute_solve_verify_asymmetry(
    solve_accuracy: Mapping[str, Mapping[str, Any]],
    verifier_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, JsonDict]]:
    """Report solve accuracy minus verifier balanced accuracy."""

    out: dict[str, dict[str, JsonDict]] = {}
    for model_id, solve in solve_accuracy.items():
        out[model_id] = {}
        solve_acc = float(solve.get("accuracy", 0.0))
        for arm, metrics in verifier_metrics.get(model_id, {}).items():
            verify_bacc = float(metrics.get("balanced_accuracy", 0.0))
            out[model_id][arm] = {
                "solve_accuracy": solve_acc,
                "verifier_balanced_accuracy": verify_bacc,
                "solve_minus_verify_balanced_accuracy": round(solve_acc - verify_bacc, 6),
                "negative_means_verification_easier": True,
            }
    return out


def compute_confidence_intervals(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    arms: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, JsonDict]:
    """Build paired bootstrap intervals over independent instance IDs."""

    units = sorted({str(row.get("instance_id")) for row in solve_records})
    rng = random.Random(seed)
    out: dict[str, JsonDict] = {}
    solve_by_model_unit = {
        model_id: {
            str(row.get("instance_id")): bool(row.get("exact_accepted"))
            for row in solve_records
            if row.get("model_hf_id") == model_id
        }
        for model_id in model_ids
    }
    verifier_by_model_arm_unit: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = {}
    for model_id in model_ids:
        for arm in arms:
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in verifier_records:
                if row.get("model_hf_id") == model_id and row.get("arm") == arm:
                    grouped[str(row.get("instance_id"))].append(row)
            verifier_by_model_arm_unit[(model_id, arm)] = grouped

    for model_id in model_ids:
        out[model_id] = {}
        solve_values: list[float] = []
        asymmetry_values: dict[str, list[float]] = {arm: [] for arm in arms}
        for _ in range(max(1, iterations)):
            sample = [rng.choice(units) for _ in units] if units else []
            solve_acc = _mean([solve_by_model_unit[model_id].get(unit, False) for unit in sample])
            solve_values.append(solve_acc)
            for arm in arms:
                sampled_records: list[Mapping[str, Any]] = []
                by_unit = verifier_by_model_arm_unit[(model_id, arm)]
                for unit in sample:
                    sampled_records.extend(by_unit.get(unit, []))
                metrics = compute_verifier_metrics(sampled_records, [model_id], [arm])[model_id][
                    arm
                ]
                asymmetry_values[arm].append(solve_acc - float(metrics["balanced_accuracy"]))
        out[model_id]["solve_accuracy"] = _interval(solve_values, iterations)
        for arm, values in asymmetry_values.items():
            out[model_id][f"asymmetry_{arm}"] = _interval(values, iterations)
    return out


def compute_mcnemar_results(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    arms: Sequence[str],
) -> dict[str, dict[str, JsonDict]]:
    """Run paired McNemar tests on instance-level solve versus verify correctness."""

    out: dict[str, dict[str, JsonDict]] = {}
    for model_id in model_ids:
        solve_by_unit = {
            str(row.get("instance_id")): bool(row.get("exact_accepted"))
            for row in solve_records
            if row.get("model_hf_id") == model_id
        }
        out[model_id] = {}
        for arm in arms:
            verifier_by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in verifier_records:
                if row.get("model_hf_id") == model_id and row.get("arm") == arm:
                    verifier_by_unit[str(row.get("instance_id"))].append(row)
            units = sorted(set(solve_by_unit) & set(verifier_by_unit))
            solve_correct = [solve_by_unit[unit] for unit in units]
            verify_correct = [
                _all_verifier_records_correct(verifier_by_unit[unit]) for unit in units
            ]
            result = mcnemar_exact(solve_correct, verify_correct)
            result["paired_unit"] = "instance_id"
            result["n_pairs"] = len(units)
            out[model_id][arm] = result
    return out


def mcnemar_exact(solve_correct: Sequence[bool], verify_correct: Sequence[bool]) -> JsonDict:
    """Return the exact two-sided McNemar binomial p-value."""

    b = sum(1 for s, v in zip(solve_correct, verify_correct, strict=False) if s and not v)
    c = sum(1 for s, v in zip(solve_correct, verify_correct, strict=False) if (not s) and v)
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(0, min(b, c) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "b_solve_correct_verify_wrong": b,
        "c_solve_wrong_verify_correct": c,
        "discordant_pairs": discordant,
        "p_value_exact": round(p_value, 6),
    }


def build_artifact(
    *,
    corpus_rows: Sequence[Mapping[str, Any]] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    gate_receipt: Mapping[str, Any] | None = None,
    device_receipt: Mapping[str, Any] | None = None,
    sampled_pairs: Sequence[Mapping[str, Any]] | None = None,
    panel_result: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    """Build a terminal complete or blocked Exp5567 artifact."""

    rows = [dict(row) for row in (corpus_rows or [])]
    specs = [dict(row) for row in (model_specs or [])]
    gate = _default_gate_receipt() | dict(gate_receipt or {})
    device = dict(device_receipt or {})
    pairs = [dict(pair) for pair in (sampled_pairs or sample_independent_pairs(rows))]
    panel = dict(panel_result or {})
    solve_records = [dict(row) for row in panel.get("solve_records", [])]
    verifier_records = [dict(row) for row in panel.get("verifier_records", [])]
    model_ids = [str(row.get("hf_id")) for row in specs]
    cache_gate = _model_specs_have_qwen_and_gemma(specs) and gate.get("cache_gate_passed") is True
    corpus_gate = len(pairs) >= MIN_INDEPENDENT_INSTANCES
    offload_gate = bool(
        device.get("gpu_offload_authenticated") or gate.get("offload_gate_passed") is True
    )
    gate["corpus_gate_passed"] = corpus_gate
    gate["offload_gate_passed"] = offload_gate
    blocked_reason = _blocked_reason(
        cache_gate=cache_gate, corpus_gate=corpus_gate, offload_gate=offload_gate
    )
    live_model_invoked = bool(solve_records or verifier_records) and not blocked_reason
    panel_complete = bool(
        not blocked_reason
        and live_model_invoked
        and len(model_ids) >= 2
        and len(pairs) >= MIN_INDEPENDENT_INSTANCES
    )

    solve_accuracy = compute_solve_accuracy(solve_records, model_ids) if solve_records else {}
    verifier_metrics = (
        compute_verifier_metrics(verifier_records, model_ids, ARMS) if verifier_records else {}
    )
    asymmetry = (
        compute_solve_verify_asymmetry(solve_accuracy, verifier_metrics)
        if solve_accuracy and verifier_metrics
        else {}
    )
    confidence = (
        compute_confidence_intervals(
            solve_records,
            verifier_records,
            model_ids,
            ARMS,
            iterations=bootstrap_iterations,
        )
        if solve_records and verifier_records
        else {}
    )
    mcnemar = (
        compute_mcnemar_results(solve_records, verifier_records, model_ids, ARMS)
        if solve_records and verifier_records
        else {}
    )
    error_taxonomy = _error_taxonomy(solve_records, verifier_records)
    parser_failure_count = int(error_taxonomy.get("parser_failure", 0))
    raw_hash = dict(panel.get("raw_response_hash", {}))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "gate_receipt": gate,
        "MODEL_SPECS": specs,
        "model_specs": specs,
        "model_cache_paths": {
            str(row.get("hf_id")): str(row.get("model_path"))
            for row in specs
            if row.get("hf_id") and row.get("model_path")
        },
        "live_model_invoked": live_model_invoked,
        "gpu_offload_authenticated": offload_gate,
        "device_receipt": device,
        "corpus_path": CORPUS_RELATIVE_PATH.as_posix(),
        "n_independent_instances": len(pairs) if corpus_gate else 0,
        "family_counts": family_counts_from_pairs(pairs) if corpus_gate else {},
        "arms": list(ARMS),
        "solve_accuracy_by_model": solve_accuracy,
        "verifier_metrics_by_model_and_arm": verifier_metrics,
        "solve_verify_asymmetry": asymmetry,
        "confidence_intervals": confidence,
        "mcnemar_results": mcnemar,
        "exact_validator_is_oracle": True,
        "verifier_is_oracle": False,
        "parser_failure_count": parser_failure_count,
        "raw_response_hash": raw_hash,
        "inference_duration_s": round(float(panel.get("inference_duration_s", 0.0) or 0.0), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(panel_complete, blocked_reason),
        "panel_complete": panel_complete,
        "model_family_deltas": _model_family_deltas(solve_accuracy, verifier_metrics, model_ids),
        "latency_by_model_and_phase": _latency_by_model_and_phase(
            solve_records, verifier_records, model_ids
        ),
        "token_usage_by_model": _token_usage_by_model(solve_records, verifier_records, model_ids),
        "error_taxonomy": dict(error_taxonomy),
        "sampled_instance_ids": [str(pair.get("instance_id")) for pair in pairs],
        "n_candidate_labels": len(verifier_records),
        "repeat_pooling_policy": "repeated calls are aggregated per candidate before metrics; independent_unit=instance_id",
        "legacy_smoke_models_used": [],
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate terminal artifact fields and fail closed on overclaim."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(artifact.get("exact_validator_is_oracle") is True, "exact_validator_is_oracle")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("panel_complete"), bool), "panel_complete")
    _require(artifact.get("legacy_smoke_models_used") == [], "legacy_smoke_models_used")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("corpus_path") == CORPUS_RELATIVE_PATH.as_posix(), "corpus_path")
    _require(artifact.get("arms") == list(ARMS), "arms")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    if artifact.get("panel_complete") is True:
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
        _require(artifact.get("live_model_invoked") is True, "live_model_invoked")
        _require(artifact.get("gpu_offload_authenticated") is True, "gpu_offload_authenticated")
        _require(
            int(artifact.get("n_independent_instances", 0)) >= MIN_INDEPENDENT_INSTANCES,
            "n_independent_instances",
        )
        _require(_model_specs_have_qwen_and_gemma(artifact.get("MODEL_SPECS", [])), "MODEL_SPECS")
        _require(bool(artifact.get("raw_response_hash")), "raw_response_hash")
        _require(bool(artifact.get("solve_accuracy_by_model")), "solve_accuracy_by_model")
        _require(
            bool(artifact.get("verifier_metrics_by_model_and_arm")),
            "verifier_metrics_by_model_and_arm",
        )
        _require(bool(artifact.get("confidence_intervals")), "confidence_intervals")
        _require(bool(artifact.get("mcnemar_results")), "mcnemar_results")
    else:
        _require(str(artifact.get("honest_verdict", "")).startswith("blocked_"), "honest_verdict")
        _require(artifact.get("live_model_invoked") is False, "live_model_invoked")


def honest_verdict(panel_complete: bool, blocked_reason: str) -> str:
    """Return the terminal verdict without sub-percent effect claims."""

    if panel_complete:
        return "complete: authenticated paired local SOTA solve-versus-verify panel; no sub-percent effect claims"
    if blocked_reason:
        return blocked_reason
    return "blocked_no_live_panel"


def probe_cuda_device_receipt() -> JsonDict:  # pragma: no cover
    """Probe CUDA and llama.cpp support without loading headline weights."""

    receipt: JsonDict = {
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "devices": [],
        "llama_cpp_supports_gpu_offload": False,
        "gpu_offload_authenticated": False,
    }
    try:
        import torch  # noqa: PLC0415

        receipt["torch_cuda_available"] = bool(torch.cuda.is_available())
        receipt["torch_device_count"] = int(torch.cuda.device_count())
        receipt["devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # noqa: BLE001
        receipt["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from llama_cpp import llama_cpp as low  # noqa: PLC0415

        receipt["llama_cpp_supports_gpu_offload"] = bool(low.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001
        receipt["llama_cpp_error"] = f"{type(exc).__name__}: {exc}"
    receipt["gpu_offload_authenticated"] = bool(
        receipt["torch_cuda_available"]
        and int(receipt["torch_device_count"]) > 0
        and receipt["llama_cpp_supports_gpu_offload"]
    )
    return receipt


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    pair_resolver: PairResolver = cached_sota_pair,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:  # pragma: no cover
    """Run the live panel or write a precise blocked artifact."""

    started = time.perf_counter()
    model_specs, gate = resolve_headline_model_specs(pair_resolver=pair_resolver)
    corpus_rows = load_corpus_rows(repo_root)
    pairs = sample_independent_pairs(corpus_rows)
    device = probe_cuda_device_receipt()
    if model_specs and device.get("gpu_offload_authenticated") is True:
        device = _merge_device_with_model_receipts(
            device,
            _authenticate_model_offload_receipts(model_specs),
        )
    if device.get("gpu_offload_authenticated") is not True:
        gate["offload_gate_passed"] = False
        artifact = build_artifact(
            corpus_rows=corpus_rows,
            model_specs=model_specs,
            gate_receipt=gate,
            device_receipt=device,
            sampled_pairs=pairs,
            panel_result=None,
            tests_run=tests_run,
        )
    elif not model_specs:
        artifact = build_artifact(
            corpus_rows=corpus_rows,
            model_specs=model_specs,
            gate_receipt=gate,
            device_receipt=device,
            sampled_pairs=pairs,
            panel_result=None,
            tests_run=tests_run,
        )
    else:
        panel = run_live_local_sota_panel(model_specs=model_specs, pairs=pairs)
        device = _merge_device_with_model_receipts(device, panel.get("model_receipts", []))
        gate["offload_gate_passed"] = bool(device.get("gpu_offload_authenticated"))
        if "inference_duration_s" not in panel:
            panel["inference_duration_s"] = round(time.perf_counter() - started, 6)
        artifact = build_artifact(
            corpus_rows=corpus_rows,
            model_specs=model_specs,
            gate_receipt=gate,
            device_receipt=device,
            sampled_pairs=pairs,
            panel_result=panel,
            tests_run=tests_run,
        )
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return artifact


def run_live_local_sota_panel(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover
    """Invoke each headline model once with a full solve/verify workload."""

    all_solve: list[JsonDict] = []
    all_verify: list[JsonDict] = []
    raw_hashes: dict[str, str] = {}
    model_receipts: list[JsonDict] = []
    started = time.perf_counter()
    for spec in model_specs:
        workload = _workload_for_model(spec, pairs)
        worker = _run_model_workload_subprocess(spec, workload)
        model_receipts.append(worker["model_receipt"])
        response_by_id = {
            str(row.get("task_id")): dict(row)
            for row in worker.get("responses", [])
            if isinstance(row, Mapping)
        }
        solve_records, verifier_records, hashes = _records_from_worker_responses(
            spec=spec,
            pairs=pairs,
            response_by_id=response_by_id,
        )
        all_solve.extend(solve_records)
        all_verify.extend(verifier_records)
        raw_hashes.update(hashes)
        gc.collect()
    return {
        "solve_records": all_solve,
        "verifier_records": all_verify,
        "raw_response_hash": raw_hashes,
        "model_receipts": model_receipts,
        "inference_duration_s": round(time.perf_counter() - started, 6),
    }


def _workload_for_model(
    spec: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover
    tasks: list[JsonDict] = []
    batches = _chunk_pairs(pairs, LIVE_BATCH_PAIR_COUNT)
    for batch_index, batch in enumerate(batches):
        batch_id = f"batch{batch_index:02d}"
        tasks.append(
            {
                "task_id": f"solve_batch::{batch_id}",
                "prompt": build_solve_batch_prompt(batch),
                "max_tokens": 2048,
                "temperature": 0.0,
                "seed": RANDOM_SEED + batch_index,
            }
        )
        for arm in ARMS:
            repeats = 3 if arm == "repeated_verdict_3x" else 1
            for repeat in range(repeats):
                tasks.append(
                    {
                        "task_id": f"verify_batch::{batch_id}::{arm}::{repeat}",
                        "prompt": build_verifier_batch_prompt(batch, arm=arm, repeat=repeat),
                        "max_tokens": 1536 if arm == "criteria_decomposition" else 1024,
                        "temperature": 0.2 if arm == "repeated_verdict_3x" else 0.0,
                        "seed": RANDOM_SEED + batch_index * 100 + repeat,
                    }
                )
    return tasks


def _chunk_pairs(
    pairs: Sequence[Mapping[str, Any]],
    size: int,
) -> list[list[Mapping[str, Any]]]:  # pragma: no cover
    return [list(pairs[index : index + size]) for index in range(0, len(pairs), size)]


def build_solve_batch_prompt(pairs: Sequence[Mapping[str, Any]]) -> str:  # pragma: no cover
    targets = []
    for pair in pairs:
        valid = dict(pair["valid_row"])
        targets.append(
            {
                "instance_id": pair["instance_id"],
                "candidate_kind": valid["candidate_kind"],
                "family": pair["family"],
                "target_signature": valid["expected_signature"],
            }
        )
    return (
        "Return exactly one JSON object and no Markdown. The object must contain "
        "solves, an array with one item per requested instance. Each solve item "
        "must have instance_id, candidate_kind, and candidate. Synthesize each "
        "candidate so the exact ASP/FSM validator matches the target_signature.\n"
        f"requested_solves: {canonical_json(targets)}\n"
        'answer_shape: {"solves":[{"instance_id":"...","candidate_kind":"asp_row","candidate":{}}]}\n'
    )


def build_verifier_batch_prompt(
    pairs: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    repeat: int,
) -> str:  # pragma: no cover
    candidates = []
    for pair in pairs:
        for key in ("valid_row", "invalid_row"):
            row = dict(pair[key])
            candidates.append(
                {
                    "candidate_id": row["row_id"],
                    "instance_id": pair["instance_id"],
                    "candidate_kind": row["candidate_kind"],
                    "candidate": row["candidate"],
                    "target_signature": row["expected_signature"],
                }
            )
    if arm == "criteria_decomposition":
        item_shape = '{"candidate_id":"...","criteria":{"schema":true,"constraints":true,"target_signature_match":true},"verdict":"valid"}'
    elif arm == "granular_score":
        item_shape = '{"candidate_id":"...","score":0}'
    else:
        item_shape = '{"candidate_id":"...","verdict":"valid"}'
    return (
        "Return exactly one JSON object and no Markdown. The object must contain "
        "labels, an array with one item per candidate_id. Use valid only if the "
        "candidate matches the target exact signature; otherwise use invalid.\n"
        f"arm: {arm}\nrepeat_index: {repeat}\n"
        f"candidate_packets: {canonical_json(candidates)}\n"
        f"label_item_shape: {item_shape}\n"
    )


def build_solve_prompt(pair: Mapping[str, Any]) -> str:  # pragma: no cover
    valid = dict(pair["valid_row"])
    return (
        "Return exactly one JSON object with keys candidate_kind and candidate. "
        "Synthesize a candidate that matches the target exact ASP/FSM signature. "
        "Do not include Markdown.\n"
        f"candidate_kind: {valid['candidate_kind']}\n"
        f"family: {pair['family']}\n"
        f"target_signature: {canonical_json(valid['expected_signature'])}\n"
    )


def build_verifier_prompt(
    row: Mapping[str, Any], *, arm: str, repeat: int
) -> str:  # pragma: no cover
    candidate_payload = {
        "candidate_kind": row["candidate_kind"],
        "candidate": row["candidate"],
        "target_signature": row["expected_signature"],
    }
    if arm == "criteria_decomposition":
        answer_shape = '{"criteria":{"schema":true,"constraints":true,"target_signature_match":true},"verdict":"valid"}'
    elif arm == "granular_score":
        answer_shape = '{"score": 0}'
    else:
        answer_shape = '{"verdict":"valid"}'
    return (
        "You are checking one exact ASP/FSM candidate. Return exactly one JSON object "
        "and no Markdown. Use valid only if the candidate matches the target exact "
        "signature; otherwise use invalid.\n"
        f"arm: {arm}\nrepeat_index: {repeat}\n"
        f"candidate_packet: {canonical_json(candidate_payload)}\n"
        f"answer_shape: {answer_shape}\n"
    )


def _run_model_workload_subprocess(
    spec: Mapping[str, Any],
    workload: Sequence[Mapping[str, Any]],
    *,
    timeout_s: int | None = None,
) -> JsonDict:  # pragma: no cover
    payload = {
        "model_hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "seed": RANDOM_SEED,
        "n_gpu_layers": N_GPU_LAYERS,
        "tasks": list(workload),
    }
    worker_timeout_s = int(timeout_s or os.environ.get("CARNOT_5567_WORKER_TIMEOUT_S", "7200"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        workload_path = handle.name
    env = dict(os.environ)
    if "gpu" in spec:
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    command = [selected_python(), "-c", WORKER_CODE, "--workload", workload_path]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=worker_timeout_s,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        model_receipt = {
            "model_hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "returncode": None,
            "worker_ok": False,
            "timeout_s": worker_timeout_s,
            "stderr_tail": _tail(str(exc.stderr or "")),
            "offloaded_layer_count_from_backend_log": _parse_offloaded_layers(
                str(exc.stderr or "")
            ),
            "llama_cpp_supports_gpu_offload": False,
            "torch_cuda_available": False,
            "torch_device_count": 0,
            "devices": [],
            "duration_s": float(worker_timeout_s),
            "gpu_offload_authenticated": False,
        }
        return {"responses": [], "model_receipt": model_receipt}
    finally:
        Path(workload_path).unlink(missing_ok=True)
    payload_out = _first_json_line(completed.stdout)
    responses = (
        payload_out.get("responses", []) if isinstance(payload_out.get("responses"), list) else []
    )
    backend_text = completed.stderr + "\n" + str(payload_out.get("backend_log_tail", ""))
    model_receipt = {
        "model_hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "returncode": completed.returncode,
        "worker_ok": completed.returncode == 0 and payload_out.get("ok") is True,
        "stderr_tail": _tail(backend_text),
        "offloaded_layer_count_from_backend_log": _parse_offloaded_layers(backend_text),
        "llama_cpp_supports_gpu_offload": payload_out.get("llama_cpp_supports_gpu_offload") is True,
        "torch_cuda_available": payload_out.get("torch_cuda_available") is True,
        "torch_device_count": int(payload_out.get("torch_device_count", 0) or 0),
        "devices": payload_out.get("devices", []),
        "duration_s": float(payload_out.get("load_and_inference_duration_s", 0.0) or 0.0),
    }
    model_receipt["gpu_offload_authenticated"] = bool(
        model_receipt["worker_ok"]
        and model_receipt["llama_cpp_supports_gpu_offload"]
        and int(model_receipt["torch_device_count"]) > 0
        and (
            int(model_receipt.get("offloaded_layer_count_from_backend_log") or 0) > 0
            or "cuda" in str(model_receipt.get("stderr_tail", "")).lower()
        )
    )
    return {"responses": responses, "model_receipt": model_receipt}


def _authenticate_model_offload_receipts(
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover
    receipts: list[JsonDict] = []
    for spec in model_specs:
        smoke = [
            {
                "task_id": "offload_smoke",
                "prompt": 'Return exactly this JSON object: {"ok":true}',
                "max_tokens": 4,
                "temperature": 0.0,
                "seed": RANDOM_SEED,
            }
        ]
        receipts.append(
            _run_model_workload_subprocess(
                spec,
                smoke,
                timeout_s=int(os.environ.get("CARNOT_5567_SMOKE_TIMEOUT_S", "600")),
            )["model_receipt"]
        )
    return receipts


def _records_from_worker_responses(
    *,
    spec: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    response_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], dict[str, str]]:  # pragma: no cover
    hf_id = str(spec["hf_id"])
    solve_records: list[JsonDict] = []
    verifier_records: list[JsonDict] = []
    hashes: dict[str, str] = {}
    batches = _chunk_pairs(pairs, LIVE_BATCH_PAIR_COUNT)
    for batch_index, batch in enumerate(batches):
        batch_id = f"batch{batch_index:02d}"
        solve_task = f"solve_batch::{batch_id}"
        solve_response = response_by_id.get(solve_task, {})
        solve_text = str(solve_response.get("text", ""))
        solve_hash = sha256_text(solve_text)
        hashes[f"{hf_id}:{solve_task}"] = solve_hash
        solve_payload, solve_error = extract_json_object(solve_text)
        solves_by_instance = _items_by_key(solve_payload, "solves", "instance_id")
        for pair in batch:
            instance_id = str(pair["instance_id"])
            item = solves_by_instance.get(instance_id)
            item_text = json.dumps(item, sort_keys=True) if item else ""
            solve_score = (
                parse_and_score_solve_response(item_text, pair)
                if item
                else {
                    "parser_ok": False,
                    "exact_accepted": False,
                    "response_hash": solve_hash,
                    "error_type": "solve_batch_missing_item"
                    if solve_payload is not None
                    else f"solve_{solve_error}",
                }
            )
            solve_records.append(
                {
                    "model_hf_id": hf_id,
                    "instance_id": instance_id,
                    "family": pair["family"],
                    "parser_ok": solve_score["parser_ok"],
                    "exact_accepted": solve_score["exact_accepted"],
                    "latency_s": _apportioned_float(solve_response, "duration_s", len(batch)),
                    "prompt_tokens": _apportioned_usage(
                        solve_response, "prompt_tokens", len(batch)
                    ),
                    "completion_tokens": _apportioned_usage(
                        solve_response,
                        "completion_tokens",
                        len(batch),
                    ),
                    "response_hash": str(solve_score.get("response_hash", solve_hash)),
                    "error_type": solve_score.get("error_type", ""),
                }
            )
        for pair in batch:
            instance_id = str(pair["instance_id"])
            for arm in ARMS:
                repeats = 3 if arm == "repeated_verdict_3x" else 1
                repeat_maps: list[dict[str, Mapping[str, Any]]] = []
                repeat_errors: list[str] = []
                repeat_hashes: list[str] = []
                latency = 0.0
                prompt_tokens = 0
                completion_tokens = 0
                for repeat in range(repeats):
                    task_id = f"verify_batch::{batch_id}::{arm}::{repeat}"
                    response = response_by_id.get(task_id, {})
                    text = str(response.get("text", ""))
                    response_hash = sha256_text(text)
                    hashes[f"{hf_id}:{task_id}"] = response_hash
                    payload, error = extract_json_object(text)
                    repeat_maps.append(_items_by_key(payload, "labels", "candidate_id"))
                    repeat_errors.append(f"verifier_{error}" if error else "")
                    repeat_hashes.append(response_hash)
                    latency += _apportioned_float(response, "duration_s", max(1, len(batch) * 2))
                    prompt_tokens += _apportioned_usage(
                        response,
                        "prompt_tokens",
                        max(1, len(batch) * 2),
                    )
                    completion_tokens += _apportioned_usage(
                        response,
                        "completion_tokens",
                        max(1, len(batch) * 2),
                    )
                for candidate_key in ("valid_row", "invalid_row"):
                    row = dict(pair[candidate_key])
                    labels: list[str | None] = []
                    errors: list[str] = []
                    response_hashes: list[str] = []
                    for repeat_index, label_map in enumerate(repeat_maps):
                        item = label_map.get(str(row["row_id"]))
                        if item is None:
                            labels.append(None)
                            errors.append(
                                repeat_errors[repeat_index] or "verifier_batch_missing_item"
                            )
                            response_hashes.append(repeat_hashes[repeat_index])
                            continue
                        item_text = json.dumps(item, sort_keys=True)
                        label, error = parse_verifier_response(item_text, arm)
                        labels.append(label)
                        errors.append(error)
                        response_hashes.append(sha256_text(item_text))
                    predicted = (
                        _majority_label(labels) if arm == "repeated_verdict_3x" else labels[0]
                    )
                    parser_ok = predicted is not None
                    verifier_records.append(
                        {
                            "model_hf_id": hf_id,
                            "instance_id": instance_id,
                            "candidate_id": row["row_id"],
                            "family": pair["family"],
                            "arm": arm,
                            "true_label": row["label"],
                            "predicted_label": predicted,
                            "parser_ok": parser_ok,
                            "latency_s": latency,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "response_hashes": response_hashes,
                            "repeat_labels": [label for label in labels if label is not None],
                            "error_type": ""
                            if parser_ok
                            else next((err for err in errors if err), "verifier_missing_label"),
                        }
                    )
    return solve_records, verifier_records, hashes


def _items_by_key(
    payload: Mapping[str, Any] | None,
    list_key: str,
    item_key: str,
) -> dict[str, Mapping[str, Any]]:  # pragma: no cover
    if not isinstance(payload, Mapping) or not isinstance(payload.get(list_key), list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for item in payload[list_key]:
        if isinstance(item, Mapping) and item.get(item_key) is not None:
            out[str(item[item_key])] = item
    return out


def _apportioned_float(
    response: Mapping[str, Any],
    key: str,
    denominator: int,
) -> float:  # pragma: no cover
    return float(response.get(key, 0.0) or 0.0) / max(1, denominator)


def _apportioned_usage(
    response: Mapping[str, Any],
    key: str,
    denominator: int,
) -> int:  # pragma: no cover
    return int(round(_usage_count(response, key) / max(1, denominator)))


def selected_python() -> str:  # pragma: no cover
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _default_gate_receipt() -> JsonDict:
    return {
        "cached_sota_pair_called": False,
        "cache_gate_passed": False,
        "blocked_reason": "",
        "cached_pair_hf_ids": [],
        "selected_headline_model_ids": [],
        "legacy_cpu_model_substituted": False,
        "corpus_gate_passed": False,
        "offload_gate_passed": False,
    }


def _blocked_reason(*, cache_gate: bool, corpus_gate: bool, offload_gate: bool) -> str:
    if not cache_gate:
        return "blocked_missing_sota_cache"
    if not offload_gate:
        return "blocked_no_cuda_offload"
    if not corpus_gate:
        return "blocked_corpus_unready"
    return ""


def _model_specs_have_qwen_and_gemma(rows: Any) -> bool:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    ids = [str(row.get("hf_id", "")) for row in rows if isinstance(row, Mapping)]
    return QWEN_ID in ids and any(model_id in ids for model_id in GEMMA_IDS)


def _error_taxonomy(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in solve_records:
        error = str(row.get("error_type", "") or "")
        if error:
            counts[error] += 1
        if row.get("parser_ok") is not True:
            counts["parser_failure"] += 1
    for row in verifier_records:
        error = str(row.get("error_type", "") or "")
        if error:
            counts[error] += 1
        if row.get("parser_ok") is not True:
            counts["parser_failure"] += 1
    return counts


def _model_family_deltas(
    solve_accuracy: Mapping[str, Mapping[str, Any]],
    verifier_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_ids: Sequence[str],
) -> JsonDict:
    qwen = next((model_id for model_id in model_ids if model_family(model_id) == "qwen"), None)
    gemma = next((model_id for model_id in model_ids if model_family(model_id) == "gemma"), None)
    if not qwen or not gemma:
        return {}
    return {
        "qwen_model": qwen,
        "gemma_model": gemma,
        "solve_accuracy_qwen_minus_gemma": round(
            float(solve_accuracy.get(qwen, {}).get("accuracy", 0.0))
            - float(solve_accuracy.get(gemma, {}).get("accuracy", 0.0)),
            6,
        ),
        "verifier_balanced_accuracy_qwen_minus_gemma_by_arm": {
            arm: round(
                float(verifier_metrics.get(qwen, {}).get(arm, {}).get("balanced_accuracy", 0.0))
                - float(verifier_metrics.get(gemma, {}).get(arm, {}).get("balanced_accuracy", 0.0)),
                6,
            )
            for arm in ARMS
        },
    }


def _latency_by_model_and_phase(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
) -> JsonDict:
    out: JsonDict = {}
    for model_id in model_ids:
        solve_latency = sum(
            float(row.get("latency_s", 0.0) or 0.0)
            for row in solve_records
            if row.get("model_hf_id") == model_id
        )
        verify_latency = sum(
            float(row.get("latency_s", 0.0) or 0.0)
            for row in verifier_records
            if row.get("model_hf_id") == model_id
        )
        out[model_id] = {
            "solve_latency_s": round(solve_latency, 6),
            "verify_latency_s": round(verify_latency, 6),
            "total_latency_s": round(solve_latency + verify_latency, 6),
        }
    return out


def _token_usage_by_model(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
) -> JsonDict:
    out: JsonDict = {}
    for model_id in model_ids:
        rows = [
            row for row in (*solve_records, *verifier_records) if row.get("model_hf_id") == model_id
        ]
        prompt = sum(int(row.get("prompt_tokens", 0) or 0) for row in rows)
        completion = sum(int(row.get("completion_tokens", 0) or 0) for row in rows)
        out[model_id] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return out


def _all_verifier_records_correct(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        normalize_label(row.get("true_label")) == normalize_label(row.get("predicted_label"))
        and row.get("parser_ok") is True
        for row in rows
    )


def _mean(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return sum(1.0 if bool(value) else 0.0 for value in values) / len(values)


def _interval(values: Sequence[float], iterations: int) -> JsonDict:
    if not values:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n_bootstrap": iterations}
    ordered = sorted(values)
    low_index = max(0, int(math.floor(0.025 * (len(ordered) - 1))))
    high_index = min(len(ordered) - 1, int(math.ceil(0.975 * (len(ordered) - 1))))
    return {
        "low": round(ordered[low_index], 6),
        "mid": round(sum(ordered) / len(ordered), 6),
        "high": round(ordered[high_index], 6),
        "n_bootstrap": iterations,
        "paired_unit": "instance_id",
    }


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _majority_label(labels: Sequence[str | None]) -> str | None:
    normalized = [label for label in (normalize_label(value) for value in labels) if label]
    if not normalized:
        return None
    counts = Counter(normalized)
    if counts["valid"] == counts["invalid"]:
        return None
    return "valid" if counts["valid"] > counts["invalid"] else "invalid"


def _usage_count(response: Mapping[str, Any], key: str) -> int:  # pragma: no cover
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        try:
            return int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _merge_device_with_model_receipts(
    device: Mapping[str, Any],
    model_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover
    merged = dict(device)
    receipts = [dict(row) for row in model_receipts]
    merged["model_receipts"] = receipts
    merged["gpu_offload_authenticated"] = bool(receipts) and all(
        row.get("gpu_offload_authenticated") is True for row in receipts
    )
    if receipts:
        merged["devices"] = receipts[0].get("devices", merged.get("devices", []))
        merged["offloaded_layer_count_from_backend_log"] = max(
            int(row.get("offloaded_layer_count_from_backend_log") or 0) for row in receipts
        )
    return merged


def _first_json_line(text: str) -> JsonDict:  # pragma: no cover
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _parse_offloaded_layers(text: str) -> int | None:  # pragma: no cover
    patterns = (
        r"offloaded\s+(\d+)\s*/\s*\d+\s+layers?\s+to\s+GPU",
        r"offloading\s+(\d+)\s+repeating\s+layers?\s+to\s+GPU",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _tail(text: str, *, limit: int = 4000) -> str:  # pragma: no cover
    return text[-limit:]


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "panel_complete": artifact["panel_complete"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
