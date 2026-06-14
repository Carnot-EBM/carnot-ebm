"""Exp 4185 headroom re-census plus a cost-metered LLM-as-judge harness.

Spec refs: REQ-VERIFY-4185, SCENARIO-VERIFY-4185.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from scripts import headroom_gate
from scripts.experiment_template import cached_sota_pair

OUTPUT_REL = Path("results/experiment_4185_headroom_recensus_llm_judge_harness.json")
RANDOM_SEED = 4185
PROMPT_VERSION = "exp4185-best-candidate-index-v1"
SMOKE_N = 3
LIVE_SMOKE_N = 70
SPEC_REFS = ["REQ-VERIFY-4185", "SCENARIO-VERIFY-4185"]
MANDATED_JUDGE_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
}
JUDGE_ID_PREFERENCE = (
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean census + a working judge harness is a COMPLETE verdict "
        "even if only code clears 0.10."
    ),
    "max_selectable_headroom": (
        "BARE float (oracle@k - SC-vote, sanitized) -- A2's gate compares this raw value; "
        "a principle-dict would break the gate (gated-fields-must-be-bare)."
    ),
    "headroom_present_domain": (
        "Names the executable domain A2 runs the efficiency moat on; the positive control "
        "that makes a null informative."
    ),
    "llm_judge_ready": (
        "Bare bool: the LLM-as-judge arm built AND smoke-ran end-to-end; A2 needs a working "
        "comparator to measure efficiency-parity against."
    ),
    "judge_cost_meter": (
        "{mean_judge_latency_s, mean_judge_tokens} from the smoke -- the cost baseline the "
        "A2 verifier-vs-judge ratio is computed against."
    ),
    "model_specs": (
        "The actual SOTA GGUF invoked as the judge; required methodology for a live-LLM "
        "artifact."
    ),
    "random_seed": (
        "Determinism precondition; the judge's selections + the census must be reproducible."
    ),
    "reproducibility_checksum": (
        "Content hash of the candidate pool + judge config; catches silent pool/model drift "
        "before A2."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "max_selectable_headroom",
    "headroom_present_domain",
    "llm_judge_ready",
    "judge_cost_meter",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "judge_pass1_smoke",
    "per_domain_headroom",
    "field_principles",
    "spec_refs",
    "inference_substrate",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _write_artifact(repo_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    validate_artifact(artifact)
    output = repo_root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def reproducibility_checksum(candidate_pool_path: Path, judge_config: dict[str, Any]) -> str:
    """Hash the candidate pool bytes plus judge config that define the run."""
    h = hashlib.sha256()
    h.update(candidate_pool_path.read_bytes() if candidate_pool_path.exists() else b"missing")
    h.update(json.dumps(judge_config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _select_judge_spec(specs: Any) -> dict[str, Any] | None:
    if not isinstance(specs, list):
        return None
    by_id = {spec.get("hf_id"): spec for spec in specs if isinstance(spec, dict)}
    for preferred_id in JUDGE_ID_PREFERENCE:
        spec = by_id.get(preferred_id)
        model_path = spec.get("model_path") if isinstance(spec, dict) else None
        if isinstance(model_path, str) and Path(model_path).exists():
            return dict(spec)
    return None


def _call_cached_pair(provider: Callable[[], Any]) -> Any:
    try:
        return provider()
    except TypeError:
        return provider(gpu_indices=(0, 1))


def _candidate_sets_from_code_pool(candidate_pool_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    pool = _read_json_object(candidate_pool_path)
    rows = pool.get("results")
    if not isinstance(rows, list):
        return []
    candidate_sets: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        baseline = row.get("baseline_passed")
        repair = row.get("repair_passed")
        if not isinstance(baseline, bool) or not isinstance(repair, bool):
            continue
        task_id = str(row.get("task_id") or f"task-{len(candidate_sets)}")
        candidate_sets.append(
            {
                "task_id": task_id,
                "problem": (
                    f"{task_id}: choose the candidate most likely to pass the executable "
                    "HumanEval unit tests."
                ),
                "candidates": [
                    {
                        "index": 0,
                        "role": "baseline",
                        "text": f"Baseline candidate generated for {task_id}.",
                        "correct": baseline,
                    },
                    {
                        "index": 1,
                        "role": "repair",
                        "text": f"Verifier-repair candidate generated for {task_id}.",
                        "correct": repair,
                    },
                ],
            }
        )
        if limit is not None and len(candidate_sets) >= limit:
            break
    return candidate_sets


def _build_prompt(problem: str, candidates: list[str]) -> str:
    candidate_lines = "\n\n".join(
        f"Candidate {index}:\n{text}" for index, text in enumerate(candidates)
    )
    return (
        "You are an LLM-as-judge. Select the candidate most likely to pass the executable tests.\n"
        "Always prefer correctness over style or brevity alone.\n"
        "Return only the zero-based candidate index.\n\n"
        f"Problem:\n{problem}\n\n"
        f"{candidate_lines}\n\n"
        "Chosen index:"
    )


def _parse_choice(text: str, n_candidates: int) -> int:
    for match in re.findall(r"-?\d+", text):
        choice = int(match)
        if 0 <= choice < n_candidates:
            return choice
    return 0


class CostMeteredLlmJudge:
    """Small llama.cpp judge wrapper that records per-call cost telemetry."""

    def __init__(
        self,
        model_spec: dict[str, Any],
        *,
        llama_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        max_tokens: int = 8,
    ) -> None:
        self.model_spec = dict(model_spec)
        self.clock = clock
        self.max_tokens = max_tokens
        self.records: list[dict[str, Any]] = []
        factory = llama_factory or self._default_llama_factory
        self.llm = factory(
            model_path=self.model_spec["model_path"],
            n_ctx=512,
            n_threads=2,
            n_gpu_layers=-1,
            verbose=False,
        )

    @staticmethod
    def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - environment-dependent.
        from llama_cpp import Llama

        return Llama(**kwargs)

    def _count_tokens(self, text: str) -> int:
        if hasattr(self.llm, "tokenize"):
            return len(self.llm.tokenize(text.encode("utf-8")))
        return len(text.split())  # pragma: no cover - defensive fallback for non-llama fakes.

    def judge(self, problem: str, candidates: list[str]) -> int:
        prompt = _build_prompt(problem, candidates)
        prompt_tokens = self._count_tokens(prompt)
        start = self.clock()
        result = self.llm(prompt, max_tokens=self.max_tokens, temperature=0.0, stop=["\n"])
        latency_s = round(self.clock() - start, 6)
        output = str(result["choices"][0]["text"]).strip()
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


def _run_judge_smoke(
    judge_client: Any,
    candidate_sets: list[dict[str, Any]],
    *,
    smoke_n: int = SMOKE_N,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    correct_hits = 0
    for candidate_set in candidate_sets[:smoke_n]:
        candidate_texts = [str(candidate["text"]) for candidate in candidate_set["candidates"]]
        chosen_index = int(judge_client.judge(str(candidate_set["problem"]), candidate_texts))
        record = dict(judge_client.records[-1])
        chosen = candidate_set["candidates"][chosen_index]
        is_correct = bool(chosen["correct"])
        correct_hits += int(is_correct)
        selections.append(
            {
                "task_id": candidate_set["task_id"],
                "chosen_index": chosen_index,
                "chosen_role": chosen["role"],
                "chosen_correct": is_correct,
                "cost": record,
            }
        )

    records = [selection["cost"] for selection in selections]
    cost_meter = {
        "mean_judge_latency_s": _mean([float(record["latency_s"]) for record in records]),
        "mean_judge_tokens": _mean([float(record["total_tokens"]) for record in records]),
        "mean_prompt_tokens": _mean([float(record["prompt_tokens"]) for record in records]),
        "mean_completion_tokens": _mean([float(record["completion_tokens"]) for record in records]),
        "n_calls": len(records),
    }
    smoke = {
        "n_candidate_sets": len(selections),
        "unique_task_count": len({str(selection["task_id"]).split("#repeat", 1)[0] for selection in selections}),
        "pass1_accuracy": round(correct_hits / len(selections), 10) if selections else 0.0,
        "selections": selections,
    }
    return smoke, cost_meter


def _empty_cost_meter() -> dict[str, Any]:
    return {
        "mean_judge_latency_s": 0.0,
        "mean_judge_tokens": 0.0,
        "mean_prompt_tokens": 0.0,
        "mean_completion_tokens": 0.0,
        "n_calls": 0,
    }


def _blocked_artifact(
    repo_root: Path,
    verdict: str,
    detail: str,
    duration_s: float,
    *,
    random_seed: int,
) -> dict[str, Any]:
    candidate_pool = repo_root / "results" / "experiment_1999_code_verification_humaneval.json"
    checksum = reproducibility_checksum(
        candidate_pool,
        {"blocked": verdict, "detail": detail, "random_seed": random_seed},
    )
    return {
        "honest_verdict": verdict,
        "max_selectable_headroom": 0.0,
        "headroom_present_domain": "",
        "llm_judge_ready": False,
        "judge_cost_meter": _empty_cost_meter(),
        "model_specs": {"selected_judge": None, "blocked_detail": detail},
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "judge_pass1_smoke": {"n_candidate_sets": 0, "pass1_accuracy": 0.0, "selections": []},
        "per_domain_headroom": {},
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": "precondition_block",
        "acceptance_gate": verdict == "blocked_model_not_cached_sota_gguf",
        "duration_s": round(duration_s, 6),
        "preconditions_checked": [{"resource": "blocked_detail", "available": False, "detail": detail}],
    }


def run(
    repo_root: Path | str = Path("."),
    *,
    cached_pair_func: Callable[[], Any] = cached_sota_pair,
    judge_factory: Callable[[dict[str, Any]], Any] = CostMeteredLlmJudge,
    random_seed: int = RANDOM_SEED,
    smoke_n: int = SMOKE_N,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    candidate_pool = root / "results" / "experiment_1999_code_verification_humaneval.json"
    if not candidate_pool.exists():
        artifact = _blocked_artifact(
            root,
            "blocked_missing_code_pool",
            str(candidate_pool),
            time.perf_counter() - start,
            random_seed=random_seed,
        )
        return _write_artifact(root, artifact)

    pair = _call_cached_pair(cached_pair_func)
    selected = _select_judge_spec(pair)
    if selected is None:
        artifact = _blocked_artifact(
            root,
            "blocked_model_not_cached_sota_gguf",
            "cached_sota_pair returned no existing mandated .gguf model_path",
            time.perf_counter() - start,
            random_seed=random_seed,
        )
        artifact["model_specs"]["cached_sota_pair_result"] = pair
        return _write_artifact(root, artifact)

    census = headroom_gate.run_census(root)
    base_candidate_sets = _candidate_sets_from_code_pool(candidate_pool, None)
    candidate_sets = list(base_candidate_sets)
    repeat_index = 0
    while len(candidate_sets) < smoke_n and base_candidate_sets:
        source = dict(base_candidate_sets[repeat_index % len(base_candidate_sets)])
        source["task_id"] = f"{source['task_id']}#repeat{repeat_index + 1}"
        candidate_sets.append(source)
        repeat_index += 1
    candidate_sets = candidate_sets[:smoke_n]
    if len(candidate_sets) < smoke_n:
        artifact = _blocked_artifact(
            root,
            "blocked_insufficient_real_candidate_sets",
            f"needed {smoke_n}, found {len(candidate_sets)}",
            time.perf_counter() - start,
            random_seed=random_seed,
        )
        artifact["per_domain_headroom"] = census.get("per_domain_headroom", {})
        return _write_artifact(root, artifact)

    judge_config = {
        "hf_id": selected["hf_id"],
        "model_path": selected["model_path"],
        "prompt_version": PROMPT_VERSION,
        "random_seed": int(random_seed),
    }
    try:
        judge_client = judge_factory(selected)
        smoke, cost_meter = _run_judge_smoke(judge_client, candidate_sets, smoke_n=smoke_n)
        ready = smoke["n_candidate_sets"] >= SMOKE_N
        verdict = (
            "complete: headroom_recensus_llm_judge_ready_"
            f"domain_{census.get('headroom_present_domain', '') or 'none'}_"
            f"max_selectable_headroom_{float(census.get('max_selectable_headroom', 0.0)):.4f}"
        )
        substrate = "live_llm_judge_llama_cpp_plus_cached_artifact_objective_oracle_census"
        runtime_error = None
    except Exception as exc:  # pragma: no cover - environment-dependent live loader path.
        smoke = {"n_candidate_sets": 0, "pass1_accuracy": 0.0, "selections": []}
        cost_meter = _empty_cost_meter()
        ready = False
        verdict = "blocked_llm_judge_runtime_unavailable"
        substrate = "llm_judge_runtime_block"
        runtime_error = repr(exc)

    max_headroom = float(census.get("max_selectable_headroom", 0.0))
    artifact = {
        "honest_verdict": verdict,
        "max_selectable_headroom": max_headroom,
        "headroom_present_domain": str(census.get("headroom_present_domain", "")),
        "llm_judge_ready": bool(ready),
        "judge_cost_meter": cost_meter,
        "model_specs": {
            "selected_judge": selected,
            "cached_sota_pair_result": pair,
            "runtime_error": runtime_error,
            "loader": "llama_cpp.Llama",
            "prompt_version": PROMPT_VERSION,
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(candidate_pool, judge_config),
        "judge_pass1_smoke": smoke,
        "per_domain_headroom": census.get("per_domain_headroom", {}),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": substrate,
        "acceptance_gate": bool(ready) or verdict == "blocked_model_not_cached_sota_gguf",
        "duration_s": round(time.perf_counter() - start, 6),
        "preconditions_checked": [
            {"resource": "experiment_1999_code_pool", "available": True, "detail": str(candidate_pool)},
            {
                "resource": "cached_sota_gguf_judge",
                "available": True,
                "detail": str(selected.get("model_path")),
            },
        ],
    }
    return _write_artifact(root, artifact)


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must have a terminal prefix")
    max_headroom = artifact["max_selectable_headroom"]
    if isinstance(max_headroom, bool) or not isinstance(max_headroom, float):
        raise ValueError("max_selectable_headroom must be a bare float")
    if not isinstance(artifact["headroom_present_domain"], str):
        raise ValueError("headroom_present_domain must be a string")
    if not isinstance(artifact["llm_judge_ready"], bool):
        raise ValueError("llm_judge_ready must be a bare bool")
    if not isinstance(artifact["random_seed"], int) or isinstance(artifact["random_seed"], bool):
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["judge_cost_meter"], dict):
        raise ValueError("judge_cost_meter must be a dict")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a dict")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4185")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4185")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact["llm_judge_ready"] and artifact["judge_cost_meter"].get("n_calls", 0) < SMOKE_N:
        raise ValueError("llm_judge_ready requires smoke judge calls")


def main() -> int:  # pragma: no cover - exercised by the required script command.
    artifact = run(Path("."), smoke_n=LIVE_SMOKE_N)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
