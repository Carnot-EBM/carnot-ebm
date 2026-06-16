"""Exp 4284 ARC verifier efficiency versus LLM-as-judge.

Spec refs: REQ-VERIFY-4284, SCENARIO-VERIFY-4284.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4284
OUTPUT_REL = Path("results/experiment_4284_verifier_efficiency_vs_llm_judge.json")
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
MANIFEST_REL = Path("results/experiment_4270_arc_family_manifest.json")
CROSS_FAMILY_REL = Path("results/experiment_4271_arc_cross_family_transfer_existing_pool.json")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SPEC_REFS = ["REQ-VERIFY-4284", "SCENARIO-VERIFY-4284"]
INFERENCE_SUBSTRATE = "live_llm_judge_vs_oracle_distinct_arc_energy_verifier"
PROMPT_VERSION = "exp4284-arc-selector-finalists-v1"
QWEN_JUDGE_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
JUDGE_ID_PREFERENCE = (
    QWEN_JUDGE_ID,
    "unsloth/gemma-4-12B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
MIN_TASKS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
MAX_FINALISTS_PER_TASK = 4
ENERGY_DOLLARS_PER_TFLOP = 0.0000007
LLM_DOLLARS_PER_1K_TOKENS = 0.00018

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Parity-at-lower-cost AND an honest 'judge is more accurate' "
        "or 'no cost advantage' are ALL COMPLETE -- the §5 efficiency axis is "
        "decision-grade either way."
    ),
    "efficiency_parity_at_lower_cost": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff the "
        "energy verifier matches the LLM-judge's accuracy (within CI) at <=0.1x the "
        "cost -- the north-star §5 win condition."
    ),
    "accuracy_energy_verifier": (
        "BARE float: the learned energy verifier's selection accuracy on held-out "
        "families -- the cheap forward-pass arm."
    ),
    "accuracy_llm_judge": (
        "BARE float: the LLM-as-judge's selection accuracy on the SAME candidates -- "
        "the expensive generative-judge baseline."
    ),
    "accuracy_delta_ci95": (
        "Bootstrap CI95 of (energy-verifier - LLM-judge) accuracy -- 'parity' means "
        "this CI includes 0 / is not significantly negative; a positive lower bound "
        "would be a Pareto win."
    ),
    "cost_ratio": (
        "BARE float: energy-verifier cost / LLM-judge cost (wall-clock + FLOPs-proxy + "
        "$/1k) -- the efficiency multiplier; target <=0.1 (10x cheaper), ideally "
        "<=0.01 (100x)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the energy verifier on the cross-family selection is "
        "oracle-distinct (NOT the executable oracle); keeps the efficiency measurement "
        "non-circular (unlike retired code-efficiency)."
    ),
    "preconditions_checked": (
        "Records the judge GGUF cache + candidate load verified; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the selection + bootstrap.",
    "reproducibility_checksum": (
        "Hash of the candidates + the two selectors' outputs + the cost accounting; "
        "lets a third party re-run."
    ),
    "model_specs": (
        "The LLM-judge GGUF id + the energy verifier + the cost-accounting method "
        "(wall-clock/FLOPs/$); required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "efficiency_parity_at_lower_cost",
    "accuracy_energy_verifier",
    "accuracy_llm_judge",
    "accuracy_delta_ci95",
    "cost_ratio",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "inference_substrate",
    "duration_s",
    "adversarial_verify",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str, preconditions: list[dict[str, Any]] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.preconditions = preconditions or []


@dataclass(frozen=True)
class ArcCandidate:
    task_id: str
    candidate_id: str
    candidate_index: int
    correct: bool
    grid: Any
    features: dict[str, float]
    vote_weight: float


@dataclass(frozen=True)
class SelectionCase:
    task_id: str
    family_id: str
    fold: int
    energy_candidate_id: str
    energy_correct: bool
    finalists: list[ArcCandidate]
    all_candidates: list[ArcCandidate]


class CostMeteredLlmJudge:
    """llama.cpp judge wrapper that records per-call latency and token counts."""

    def __init__(
        self,
        model_spec: dict[str, Any],
        *,
        llama_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        max_tokens: int = 16,
    ) -> None:
        self.model_spec = dict(model_spec)
        self.clock = clock
        self.max_tokens = int(max_tokens)
        self.records: list[dict[str, Any]] = []
        factory = llama_factory or self._default_llama_factory
        self.llm = factory(
            model_path=self.model_spec["model_path"],
            n_ctx=8192,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False,
        )

    @staticmethod
    def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - live dependency.
        from llama_cpp import Llama

        return Llama(**kwargs)

    def _count_tokens(self, text: str) -> int:
        if hasattr(self.llm, "tokenize"):
            return len(self.llm.tokenize(text.encode("utf-8")))
        return len(text.split())

    def judge(self, problem: str, candidates: list[str]) -> int:
        prompt = _build_prompt(problem, candidates)
        prompt_tokens = self._count_tokens(prompt)
        start = self.clock()
        result = self.llm(prompt, max_tokens=self.max_tokens, temperature=0.0)
        latency_s = round(self.clock() - start, 6)
        output = str(result["choices"][0]["text"]).strip()
        if not output:
            raise RuntimeError("empty LLM judge output")
        completion_tokens = self._count_tokens(output)
        chosen = _parse_choice(output, len(candidates))
        self.records.append(
            {
                "chosen_index": chosen,
                "latency_s": latency_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "raw_output": output,
            }
        )
        return chosen


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _bare_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_json_artifact")
    return payload


def _read_gzip_json_object(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_candidate_pool")
    return payload


def _resolve_existing(root: Path, value: Any, fallback: Path | None = None) -> Path:
    raw = value if isinstance(value, str) and value else str(fallback or "")
    if not raw:
        raise BlockedRun("blocked_energy_verifier_load")
    path = Path(raw)
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        raise BlockedRun("blocked_energy_verifier_load")
    return resolved


def _model_active_params_b(spec: dict[str, Any]) -> float:
    if isinstance(spec.get("active_params_b"), (int, float)):
        return float(spec["active_params_b"])
    hf_id = str(spec.get("hf_id", ""))
    if "Qwen3.6-35B-A3B" in hf_id:
        return 3.0
    if "26B-A4B" in hf_id:
        return 4.0
    if "12B" in hf_id:
        return 12.0
    if "31B" in hf_id:
        return 31.0
    return 3.0


def default_judge_spec_provider() -> dict[str, Any] | None:  # pragma: no cover - local cache dependent.
    pair = cached_sota_pair(gpu_indices=(0, 1))
    cached_by_id = {str(item.get("hf_id")): dict(item) for item in pair or [] if isinstance(item, dict)}
    for hf_id in JUDGE_ID_PREFERENCE:
        spec = cached_by_id.get(hf_id)
        model_path = spec.get("model_path") if spec else resolve_cached_gguf(hf_id)
        if isinstance(model_path, str) and Path(model_path).exists() and Path(model_path).stat().st_size > 0:
            base = spec or {"name": hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF"), "hf_id": hf_id}
            base["model_path"] = model_path
            base["active_params_b"] = _model_active_params_b(base)
            return base
    return None


def default_llama_import_checker() -> bool:  # pragma: no cover - environment dependent.
    try:
        __import__("llama_cpp")
    except Exception:
        return False
    return True


def _candidate_feature_map(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    return {name: _safe_float(raw.get(name)) for name in exp4244.FEATURE_NAMES}


def _candidate_from_payload(task_id: str, fallback_index: int, payload: dict[str, Any]) -> ArcCandidate:
    features = _candidate_feature_map(payload)
    candidate_index = int(_safe_float(payload.get("candidate_index"), float(fallback_index)))
    return ArcCandidate(
        task_id=task_id,
        candidate_id=str(payload.get("candidate_id") or f"{task_id}::candidate{candidate_index}"),
        candidate_index=candidate_index,
        correct=payload.get("is_correct") is True,
        grid=payload.get("grid", []),
        features=features,
        vote_weight=features.get("vote_weight", 0.0),
    )


def _load_pool_candidates(root: Path) -> tuple[dict[str, list[ArcCandidate]], str]:
    path = root / POOL_REL
    if not path.exists():
        raise BlockedRun("blocked_cross_family_candidates")
    payload = _read_gzip_json_object(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise BlockedRun("blocked_cross_family_candidates")
    by_task: dict[str, list[ArcCandidate]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        raw_candidates = task.get("candidates")
        if not task_id or not isinstance(raw_candidates, list):
            continue
        candidates = [
            _candidate_from_payload(task_id, index, raw)
            for index, raw in enumerate(raw_candidates)
            if isinstance(raw, dict)
        ]
        if candidates:
            by_task[task_id] = sorted(candidates, key=lambda item: item.candidate_index)
    if not by_task:
        raise BlockedRun("blocked_cross_family_candidates")
    return by_task, _sha256_file(path)


def _load_manifest(root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = root / MANIFEST_REL
    if not path.exists():
        raise BlockedRun("blocked_cross_family_manifest")
    payload = _read_json_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise BlockedRun("blocked_cross_family_manifest")
    by_task = {str(row.get("task_id")): row for row in rows if isinstance(row, dict) and row.get("task_id")}
    if not by_task:
        raise BlockedRun("blocked_cross_family_manifest")
    return by_task, _sha256_file(path)


def _load_cross_family_rows(root: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    path = root / CROSS_FAMILY_REL
    if not path.exists():
        raise BlockedRun("blocked_cross_family_candidates")
    payload = _read_json_object(path)
    if payload.get("verifier_is_oracle") is not False:
        raise BlockedRun("blocked_cross_family_candidates")
    rows = payload.get("task_rows")
    if not isinstance(rows, list) or not rows:
        raise BlockedRun("blocked_cross_family_candidates")
    return [row for row in rows if isinstance(row, dict)], _sha256_file(path), payload


def _load_energy_verifier(root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    build_path = root / SET_ENCODER_BUILD_REL
    if not build_path.exists():
        raise BlockedRun("blocked_energy_verifier_load")
    build = _read_json_object(build_path)
    if build.get("aggregator_trained") is not True or build.get("verifier_is_oracle") is not False:
        raise BlockedRun("blocked_energy_verifier_load")
    model_path = _resolve_existing(root, build.get("learned_verifier_path"))
    model = exp4244.load_set_encoder(model_path)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun("blocked_energy_verifier_load")
    return model, model_path, build


def _by_candidate_id(candidates: list[ArcCandidate]) -> dict[str, ArcCandidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _candidate_by_id(candidates: list[ArcCandidate], candidate_id: Any) -> ArcCandidate | None:
    return _by_candidate_id(candidates).get(str(candidate_id))


def _append_unique(items: list[ArcCandidate], candidate: ArcCandidate | None) -> None:
    if candidate is not None and all(item.candidate_id != candidate.candidate_id for item in items):
        items.append(candidate)


def _build_finalists(task_row: dict[str, Any], candidates: list[ArcCandidate]) -> list[ArcCandidate]:
    finalists: list[ArcCandidate] = []
    by_id = _by_candidate_id(candidates)
    for key in (
        "set_encoder_candidate_id",
        "vote_candidate_id",
        "matched_control_candidate_id",
        "online_adapt_candidate_id",
    ):
        _append_unique(finalists, by_id.get(str(task_row.get(key))))
    for candidate in sorted(candidates, key=lambda item: (-item.vote_weight, item.candidate_index)):
        _append_unique(finalists, candidate)
        if len(finalists) >= MAX_FINALISTS_PER_TASK:
            break
    return sorted(finalists, key=lambda item: item.candidate_index)


def load_selection_cases(
    root: Path,
    *,
    min_tasks: int = MIN_TASKS,
    max_tasks: int | None = None,
) -> tuple[list[SelectionCase], dict[str, str], dict[str, Any], Path, dict[str, Any]]:
    pool_by_task, pool_sha = _load_pool_candidates(root)
    manifest_by_task, manifest_sha = _load_manifest(root)
    task_rows, cross_sha, cross_payload = _load_cross_family_rows(root)
    model, model_path, build = _load_energy_verifier(root)

    cases: list[SelectionCase] = []
    for task_row in task_rows:
        task_id = str(task_row.get("task_id") or "")
        candidates = pool_by_task.get(task_id)
        manifest_row = manifest_by_task.get(task_id)
        if not task_id or not candidates or manifest_row is None:
            continue
        energy_id = str(task_row.get("set_encoder_candidate_id") or "")
        energy_candidate = _candidate_by_id(candidates, energy_id)
        finalists = _build_finalists(task_row, candidates)
        if energy_candidate is None or len(finalists) < 2:
            continue
        cases.append(
            SelectionCase(
                task_id=task_id,
                family_id=str(task_row.get("family_id") or manifest_row.get("family_id") or ""),
                fold=int(_safe_float(task_row.get("fold"), _safe_float(manifest_row.get("fold")))),
                energy_candidate_id=energy_id,
                energy_correct=bool(task_row.get("set_encoder_correct")),
                finalists=finalists,
                all_candidates=candidates,
            )
        )
        if max_tasks is not None and len(cases) >= max_tasks:
            break
    if len(cases) < int(min_tasks):
        raise BlockedRun("blocked_insufficient_cross_family_tasks")
    checksums = {"pool_sha256": pool_sha, "manifest_sha256": manifest_sha, "cross_family_sha256": cross_sha}
    return cases, checksums, model, model_path, build


def _grown_rows(candidates: list[ArcCandidate]) -> list[exp4244.GrownPoolRow]:
    return [
        exp4244.GrownPoolRow(
            task_id=candidate.task_id,
            candidate_id=candidate.candidate_id,
            candidate_index=candidate.candidate_index,
            correct=candidate.correct,
            features=candidate.features,
            vote_weight=candidate.vote_weight,
        )
        for candidate in candidates
    ]


def measure_energy_forward_cost(cases: list[SelectionCase], model_payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    score_sum = 0.0
    candidate_count = 0
    model = model_payload.get("model", model_payload)
    for case in cases:
        rows = _grown_rows(case.all_candidates)
        scores = exp4244._score_with_payload(model, rows)
        score_sum += sum(float(score) for score in scores.values())
        candidate_count += len(rows)
    wall = time.perf_counter() - start
    hidden_dim = int(_safe_float(model.get("hidden_dim"), 32.0)) if isinstance(model, dict) else 32
    feature_count = len(exp4244.FEATURE_NAMES)
    flops = float(candidate_count * max(1, feature_count) * max(1, hidden_dim) * 8)
    dollars_per_1k = (flops / 1_000_000_000_000.0) * ENERGY_DOLLARS_PER_TFLOP
    return {
        "total_wall_clock_s": round(wall, 9),
        "candidate_forward_passes": int(candidate_count),
        "flops_proxy": flops,
        "score_checksum_component": round(score_sum, 12),
        "estimated_dollars_per_1k_selections": dollars_per_1k,
    }


def _grid_text(grid: Any) -> str:
    text = json.dumps(grid, separators=(",", ":"), ensure_ascii=True)
    if len(text) > 300:
        return text[:300] + "...[truncated]"
    return text


def _candidate_prompt_text(candidate: ArcCandidate, local_index: int) -> str:
    feature_bits = {
        "vote_weight": _round_metric(candidate.vote_weight),
        "self_consistency_margin": _round_metric(candidate.features.get("self_consistency_margin", 0.0)),
        "cell_confidence_mean": _round_metric(candidate.features.get("cell_confidence_mean", 0.0)),
        "program_demo_fit": _round_metric(candidate.features.get("program_demo_fit", 0.0)),
    }
    return (
        f"Candidate {local_index}: id={candidate.candidate_id}; "
        f"metadata={json.dumps(feature_bits, sort_keys=True, separators=(',', ':'))}; "
        f"output_grid={_grid_text(candidate.grid)}"
    )


def _problem_text(case: SelectionCase) -> str:
    return (
        "ARC held-out-family selection. Choose the candidate output grid most likely "
        f"to be correct for task {case.task_id} in family {case.family_id}. "
        "Return only the zero-based candidate index."
    )


def _build_prompt(problem: str, candidates: list[str]) -> str:
    joined = "\n\n".join(candidates)
    return (
        "You are an LLM-as-judge for ARC candidate selection. Use the candidate grids "
        "and non-label metadata only; no executable oracle is available.\n"
        "Return only the zero-based candidate index.\n\n"
        f"Problem:\n{problem}\n\nCandidates:\n{joined}\n\nChosen index:"
    )


def _parse_choice(text: str, n_candidates: int) -> int:
    for match in re.findall(r"-?\d+", text):
        choice = int(match)
        if 0 <= choice < n_candidates:
            return choice
    return 0


def run_llm_judge(cases: list[SelectionCase], judge_client: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for case in cases:
        candidate_texts = [
            _candidate_prompt_text(candidate, index) for index, candidate in enumerate(case.finalists)
        ]
        chosen_index = int(judge_client.judge(_problem_text(case), candidate_texts))
        if chosen_index < 0 or chosen_index >= len(case.finalists):
            chosen_index = 0
        record = dict(judge_client.records[-1])
        chosen = case.finalists[chosen_index]
        energy_index = next(
            index
            for index, candidate in enumerate(case.finalists)
            if candidate.candidate_id == case.energy_candidate_id
        )
        selections.append(
            {
                "task_id": case.task_id,
                "family_id": case.family_id,
                "fold": case.fold,
                "candidate_count": len(case.finalists),
                "all_candidate_count": len(case.all_candidates),
                "energy_candidate_id": case.energy_candidate_id,
                "energy_finalist_index": energy_index,
                "energy_correct": bool(case.energy_correct),
                "llm_chosen_index": chosen_index,
                "llm_candidate_id": chosen.candidate_id,
                "llm_correct": bool(chosen.correct),
                "judge_cost": record,
            }
        )
    costs = [row["judge_cost"] for row in selections]
    total_tokens = sum(int(_safe_float(cost.get("total_tokens"))) for cost in costs)
    total_prompt_tokens = sum(int(_safe_float(cost.get("prompt_tokens"))) for cost in costs)
    total_completion_tokens = sum(int(_safe_float(cost.get("completion_tokens"))) for cost in costs)
    total_wall = sum(_safe_float(cost.get("latency_s")) for cost in costs)
    return selections, {
        "total_wall_clock_s": round(total_wall, 6),
        "total_tokens": int(total_tokens),
        "prompt_tokens": int(total_prompt_tokens),
        "completion_tokens": int(total_completion_tokens),
    }


def _rate(values: list[bool]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _bootstrap_ci95(deltas: list[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(int(resamples))
    ]
    samples.sort()
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _cost_accounting(
    energy_cost: dict[str, Any],
    judge_cost: dict[str, Any],
    judge_spec: dict[str, Any],
    *,
    n_tasks: int,
) -> tuple[dict[str, Any], float]:
    active_params_b = _model_active_params_b(judge_spec)
    llm_flops = float(2.0 * active_params_b * 1_000_000_000.0 * int(judge_cost["total_tokens"]))
    llm_dollars = max(float(judge_cost["total_tokens"]) / 1000.0 * LLM_DOLLARS_PER_1K_TOKENS, 1e-12)
    energy_dollars = max(float(energy_cost["estimated_dollars_per_1k_selections"]), 1e-18)
    ratio = _round_metric(energy_dollars / llm_dollars)
    accounting = {
        "method": (
            "cost_ratio is energy estimated dollars per 1k selections divided by "
            "LLM estimated dollars per 1k selections; wall-clock and FLOPs/token "
            "ratios are reported alongside."
        ),
        "selection_task_n": int(n_tasks),
        "energy_verifier": dict(energy_cost),
        "llm_judge": {
            **judge_cost,
            "active_params_b": active_params_b,
            "flops_proxy": llm_flops,
            "estimated_dollars_per_1k_selections": llm_dollars,
        },
        "ratios": {
            "wall_clock": _round_metric(
                float(energy_cost["total_wall_clock_s"]) / max(float(judge_cost["total_wall_clock_s"]), 1e-12)
            ),
            "flops_proxy": _round_metric(float(energy_cost["flops_proxy"]) / max(llm_flops, 1.0)),
            "estimated_dollars_per_1k": ratio,
        },
        "constants": {
            "energy_dollars_per_tflop": ENERGY_DOLLARS_PER_TFLOP,
            "llm_dollars_per_1k_tokens": LLM_DOLLARS_PER_1K_TOKENS,
        },
    }
    return accounting, ratio


def _checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    reason: str,
    preconditions: list[dict[str, Any]],
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "efficiency_parity_at_lower_cost": False,
        "accuracy_energy_verifier": 0.0,
        "accuracy_llm_judge": 0.0,
        "accuracy_delta_ci95": [0.0, 0.0],
        "cost_ratio": 0.0,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _checksum({"blocked": reason, "preconditions": preconditions}),
        "model_specs": {"status": "blocked", "blocked_reason": reason},
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": "precondition_block",
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "not_run", "reason": reason},
        "acceptance_gate": reason == "blocked_judge_model_not_cached",
        "selection_task_n": 0,
        "cost_accounting": {},
        "per_task": [],
    }


def _complete_artifact(
    *,
    selections: list[dict[str, Any]],
    energy_cost: dict[str, Any],
    judge_cost: dict[str, Any],
    judge_spec: dict[str, Any],
    checksums: dict[str, str],
    model_path: Path,
    build: dict[str, Any],
    preconditions: list[dict[str, Any]],
    random_seed: int,
    bootstrap_resamples: int,
    duration_s: float,
) -> dict[str, Any]:
    energy_hits = [bool(row["energy_correct"]) for row in selections]
    judge_hits = [bool(row["llm_correct"]) for row in selections]
    deltas = [float(e) - float(j) for e, j in zip(energy_hits, judge_hits, strict=True)]
    accuracy_energy = _round_metric(_rate(energy_hits))
    accuracy_judge = _round_metric(_rate(judge_hits))
    ci95 = _bootstrap_ci95(deltas, random_seed=random_seed, resamples=bootstrap_resamples)
    cost_accounting, cost_ratio = _cost_accounting(
        energy_cost,
        judge_cost,
        judge_spec,
        n_tasks=len(selections),
    )
    not_significantly_worse = bool(ci95[1] >= 0.0)
    parity = bool(not_significantly_worse and cost_ratio <= 0.1)
    if parity:
        verdict = "complete: efficiency_parity_at_lower_cost_true"
    elif ci95[1] < 0.0:
        verdict = "complete: judge_is_more_accurate"
    elif cost_ratio > 0.1:
        verdict = "complete: no_cost_advantage"
    else:  # pragma: no cover - exhaustive guard for future verdict extensions.
        verdict = "complete: efficiency_parity_at_lower_cost_false"
    checksum_payload = {
        "checksums": checksums,
        "cost_accounting": cost_accounting,
        "judge_spec": judge_spec,
        "random_seed": int(random_seed),
        "selections": selections,
    }
    return {
        "honest_verdict": f"{verdict}_delta_{_round_metric(accuracy_energy - accuracy_judge):.4f}",
        "efficiency_parity_at_lower_cost": parity,
        "accuracy_energy_verifier": accuracy_energy,
        "accuracy_llm_judge": accuracy_judge,
        "accuracy_delta_ci95": ci95,
        "cost_ratio": cost_ratio,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _checksum(checksum_payload),
        "model_specs": {
            "judge_gguf": dict(judge_spec),
            "energy_verifier": {
                "path": str(model_path),
                "verifier_is_oracle": False,
                "model_specs": build.get("model_specs", {}),
            },
            "prompt_version": PROMPT_VERSION,
            "cost_accounting_method": cost_accounting["method"],
            "candidate_policy": (
                "LLM judge sees deduplicated selector finalists: Exp 4271 energy pick, "
                "SC-vote pick, matched control, online-adapt pick, then top vote-weight "
                f"candidates up to {MAX_FINALISTS_PER_TASK}."
            ),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), float(judge_cost["total_wall_clock_s"])), 6),
        "adversarial_verify": {"status": "pending"},
        "acceptance_gate": True,
        "selection_task_n": len(selections),
        "bootstrap_resamples": int(bootstrap_resamples),
        "accuracy_delta": _round_metric(accuracy_energy - accuracy_judge),
        "cost_accounting": cost_accounting,
        "per_task": selections,
        "precondition_checksums": checksums,
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:
    script = repo_root / "scripts" / "adversarial_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must have a terminal prefix")
    if type(artifact["efficiency_parity_at_lower_cost"]) is not bool:
        raise ValueError("efficiency_parity_at_lower_cost must be a bare bool")
    for field in ("accuracy_energy_verifier", "accuracy_llm_judge", "cost_ratio"):
        if not _bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["accuracy_delta_ci95"], list) or len(artifact["accuracy_delta_ci95"]) != 2:
        raise ValueError("accuracy_delta_ci95 must be a two-float list")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    for field in ("model_specs", "adversarial_verify"):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4284")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4284")


def _write_artifact(repo_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    validate_artifact(artifact)
    output = repo_root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run(
    repo_root: Path | str = Path("."),
    *,
    judge_spec_provider: Callable[[], dict[str, Any] | None] = default_judge_spec_provider,
    llama_import_checker: Callable[[], bool] = default_llama_import_checker,
    judge_factory: Callable[[dict[str, Any]], Any] = CostMeteredLlmJudge,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    min_tasks: int = MIN_TASKS,
    max_tasks: int | None = None,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    preconditions: list[dict[str, Any]] = []

    try:
        judge_spec = judge_spec_provider()
        if judge_spec is None:
            preconditions.append(
                {
                    "resource": "cached_judge_gguf",
                    "available": False,
                    "detail": "No non-empty Qwen3.6 or Gemma-4 GGUF path resolved.",
                }
            )
            raise BlockedRun("blocked_judge_model_not_cached", preconditions)
        model_path = Path(str(judge_spec.get("model_path") or ""))
        if not model_path.exists() or model_path.stat().st_size <= 0:
            preconditions.append(
                {
                    "resource": "cached_judge_gguf",
                    "available": False,
                    "detail": str(model_path),
                }
            )
            raise BlockedRun("blocked_judge_model_not_cached", preconditions)
        preconditions.append(
            {"resource": "cached_judge_gguf", "available": True, "detail": str(model_path)}
        )
        if not llama_import_checker():
            preconditions.append(
                {"resource": "llama_cpp", "available": False, "detail": "import llama_cpp failed"}
            )
            raise BlockedRun("blocked_llama_cpp_unavailable", preconditions)
        preconditions.append({"resource": "llama_cpp", "available": True, "detail": "import OK"})

        cases, checksums, energy_model, energy_model_path, build = load_selection_cases(
            root,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
        )
        preconditions.extend(
            [
                {
                    "resource": "cross_family_candidates",
                    "available": True,
                    "detail": f"{len(cases)} tasks loaded",
                },
                {
                    "resource": "energy_verifier",
                    "available": True,
                    "detail": str(energy_model_path),
                },
            ]
        )
        energy_cost = measure_energy_forward_cost(cases, energy_model)
        try:
            judge_client = judge_factory(judge_spec)
            selections, judge_cost = run_llm_judge(cases, judge_client)
        except Exception as exc:
            preconditions.append(
                {"resource": "llm_judge_runtime", "available": False, "detail": repr(exc)}
            )
            raise BlockedRun("blocked_llm_judge_runtime", preconditions) from exc
        artifact = _complete_artifact(
            selections=selections,
            energy_cost=energy_cost,
            judge_cost=judge_cost,
            judge_spec=judge_spec,
            checksums=checksums,
            model_path=energy_model_path,
            build=build,
            preconditions=preconditions,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as blocked:
        artifact = _blocked_artifact(
            blocked.reason,
            blocked.preconditions or preconditions,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )

    output_path = root / OUTPUT_REL
    _write_artifact(root, artifact)
    if artifact["honest_verdict"].startswith("complete:"):
        report = _run_adversarial_verify(root, output_path) if adversarial_runner is None else adversarial_runner(output_path)
        artifact["adversarial_verify"] = report
        _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by required script command.
    artifact = run(Path("."), max_tasks=MIN_TASKS)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
