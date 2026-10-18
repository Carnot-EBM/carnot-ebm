"""Exp 3057 local SOTA solution-verifier gain panel.

Spec refs: REQ-VERIFY-3057,
           SCENARIO-VERIFY-3057,
           SCENARIO-VERIFY-3057-BLOCKED.

This module measures a deliberately small claim: when exact SAT/SMT-style
ground truth is available, does a local mandated GGUF verifier improve selection
over a one-shot local GGUF solver candidate? The fixture set is tiny by design
so the artifact can record live model provenance, exact labels, false accepts,
and false rejects without turning a calibration check into a benchmark claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.experiment_3043_verified_speculation_transcript_fingerprint import (
    _cuda_probe,
    _extract_text,
    _file_evidence,
    _gpu_inventory,
    _normalize_output,
    _python_environment,
    _repo_commit,
)
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf

try:  # pragma: no cover - tests exercise the normal dependency-present path.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
ResolveGgufFn = Callable[[str, str], str | None]
LlamaFactory = Callable[..., Any]
ClockFn = Callable[[], float]
RepoCommitFn = Callable[[Path], str]

ARTIFACT = "experiment_3057_local_sota_solution_verifier_gain_panel_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
SCHEMA = "carnot.local_sota_solution_verifier_gain_panel.v1"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
PANEL_ROWS_REL_PATH = (
    Path("results") / "local_sota_solution_verifier_gain_panel_3057" / ("panel_rows.jsonl")
)
DEFAULT_SEED = 305700
DEFAULT_DECODE_CONFIG: JsonDict = {
    "max_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
}
DEFAULT_LOAD_CONFIG: JsonDict = {
    "n_ctx": 1024,
    "n_batch": 64,
    "n_ubatch": 64,
    "n_gpu_layers": -1,
    "main_gpu": 0,
    "verbose": False,
}
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
REQUIRED_ARTIFACT_FIELDS = (
    "solution_verifier_calibration_ready",
    "verifier_gain_delta",
    "false_positive_rate",
    "false_negative_rate",
    "exact_ground_truth_count",
    "models_used",
    "model_specs",
    "legacy_smoke_only_used",
    "cross_family_used",
    "prompt_hashes",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 3057 calibration panel."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_dir: Path | None = None
    seed: int = DEFAULT_SEED
    preferred_quant: str = "Q4_K_M"
    decode_config: Mapping[str, Any] | None = None
    load_config: Mapping[str, Any] | None = None
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def panel_rows_path(self) -> Path:
        return self.raw_dir or self.repo_root / PANEL_ROWS_REL_PATH

    def effective_decode_config(self) -> JsonDict:
        config = dict(DEFAULT_DECODE_CONFIG)
        if self.decode_config:
            config.update(dict(self.decode_config))
        return config

    def effective_load_config(self, gpu: int = 0) -> JsonDict:
        config = dict(DEFAULT_LOAD_CONFIG)
        if self.load_config:
            config.update(dict(self.load_config))
        config["main_gpu"] = int(gpu)
        return config


def build_sat_smt_fixtures() -> list[JsonDict]:
    """Return eight deterministic integer fixtures for exact Z3 checking."""

    return [
        _fixture("lin-01", ("x", "y"), [_eq({"x": 1, "y": 1}, 5), _eq({"x": 1, "y": -1}, 1)]),
        _fixture("lin-02", ("a", "b"), [_eq({"a": 2, "b": 1}, 7), _eq({"a": 1, "b": -1}, 2)]),
        _fixture(
            "lin-03", ("p", "q"), [_eq({"p": 1, "q": 1}, 4), _eq({"p": 1}, 1), _ge({"q": 1}, 0)]
        ),
        _fixture("lin-04", ("m", "n"), [_eq({"m": 1, "n": 1}, 10), _eq({"m": 1}, 4)]),
        _fixture("lin-05", ("u", "v"), [_eq({"u": 1, "v": 1}, 3), _eq({"u": 1, "v": 1}, 4)]),
        _fixture("lin-06", ("r",), [_ge({"r": 1}, 2), _le({"r": 1}, 1)]),
        _fixture(
            "lin-07", ("s", "t"), [_eq({"s": 1, "t": 1}, 8), _eq({"s": 1}, 5), _ge({"t": 1}, 0)]
        ),
        _fixture("lin-08", ("k", "l"), [_eq({"k": 1, "l": -1}, 2), _eq({"k": 1, "l": 1}, 6)]),
    ]


def compute_exact_ground_truth(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    z3_module: Any = _z3,
) -> list[JsonDict]:
    """Evaluate every fixture with Z3 and return exact row labels."""

    if z3_module is None:
        raise RuntimeError("z3_solver_unavailable")
    return [_exact_row(fixture, z3_module) for fixture in fixtures]


def evaluate_candidate(truth_row: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Return whether a candidate agrees with exact solver authority."""

    status = str(candidate.get("status", "")).lower()
    if truth_row["solver_status"] == "unsat":
        return status == "unsat"
    if status != "sat":
        return False
    assignment = _int_assignment(candidate.get("assignment"))
    if sorted(assignment) != sorted(truth_row["variables"]):
        return False
    return _constraints_hold(truth_row["constraints"], assignment)


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn = _repo_commit,
) -> JsonDict:
    """Run Exp 3057, persist row evidence, and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = monotonic()
    cache_resolution = _resolve_cache(resolve_gguf_func, active)
    selected_models = _select_models(cache_resolution)
    if not selected_models:
        artifact = _blocked_artifact(
            config=active,
            cache_resolution=cache_resolution,
            duration_s=round(monotonic() - started, 6),
            repo_commit_func=repo_commit_func,
        )
        _write_json(active.artifact_path(), artifact)
        return artifact

    try:
        rows, prompt_hashes, raw_transcript_paths = _run_live_panel(
            config=active,
            selected_models=selected_models,
            llama_factory=llama_factory or _default_llama_factory,
        )
        runtime_blocker = None
    except Exception as exc:  # pragma: no cover - live runtime failure path.
        rows = []
        prompt_hashes = []
        raw_transcript_paths = []
        runtime_blocker = f"{type(exc).__name__}: {exc}"

    if rows:
        _write_jsonl(active.panel_rows_path(), rows)
    duration_s = round(monotonic() - started, 6)
    artifact = _build_artifact(
        config=active,
        rows=rows,
        prompt_hashes=prompt_hashes,
        raw_transcript_paths=raw_transcript_paths,
        selected_models=selected_models,
        cache_resolution=cache_resolution,
        duration_s=duration_s,
        runtime_blocker=runtime_blocker,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3057 artifact violates the calibration contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("legacy_smoke_only_used") is not False:
        raise ValueError("legacy smoke evidence cannot satisfy REQ-VERIFY-3057")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("solution_verifier_calibration_ready") is not True:
        if not verdict.startswith("blocked_sota_gguf_unavailable"):
            raise ValueError("honest_verdict must disclose blocked_sota_gguf_unavailable")
        return
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must be present when calibration is ready")
    if int(artifact.get("exact_ground_truth_count") or 0) < 6:
        raise ValueError("exact_ground_truth_count must be at least 6 when ready")
    if not artifact.get("prompt_hashes"):
        raise ValueError("prompt_hashes must be non-empty when ready")
    if not str(artifact.get("exact_solver_authority", "")).startswith("z3"):
        raise ValueError("exact_solver_authority must name the Z3 solver")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL panel rows written by this module."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_live_panel(
    *,
    config: ExperimentConfig,
    selected_models: Sequence[Mapping[str, Any]],
    llama_factory: LlamaFactory,
) -> tuple[list[JsonDict], list[str], list[str]]:
    fixtures = build_sat_smt_fixtures()
    truth_rows = compute_exact_ground_truth(fixtures)
    solver_model = selected_models[0]
    verifier_model = selected_models[1] if len(selected_models) > 1 else selected_models[0]
    decode_config = config.effective_decode_config()
    solver_outputs = _call_role_model(
        model=solver_model,
        prompts=[_solver_prompt(row) for row in truth_rows],
        config=config,
        decode_config=decode_config,
        llama_factory=llama_factory,
    )
    verifier_prompts = []
    panel_rows = []
    for truth_row, solver_output in zip(truth_rows, solver_outputs, strict=True):
        solver_candidate = _parse_candidate(solver_output["text"])
        candidate_rows = _candidate_pool(truth_row, solver_candidate)
        verifier_prompts.append(_verifier_prompt(truth_row, candidate_rows))
        panel_rows.append(
            {
                "fixture_id": truth_row["fixture_id"],
                "truth": truth_row,
                "solver_candidate": solver_candidate,
                "solver_raw_output_hash": _sha256_text(solver_output["text"]),
                "candidate_rows": candidate_rows,
            }
        )
    verifier_outputs = _call_role_model(
        model=verifier_model,
        prompts=verifier_prompts,
        config=config,
        decode_config=decode_config,
        llama_factory=llama_factory,
    )
    rows = []
    for row, verifier_output in zip(panel_rows, verifier_outputs, strict=True):
        decision = _parse_verifier_decision(verifier_output["text"])
        rows.append(_score_panel_row(row, decision, verifier_output["text"]))
    prompt_hashes = [entry["prompt_hash"] for entry in solver_outputs + verifier_outputs]
    return rows, prompt_hashes, []


def _call_role_model(
    *,
    model: Mapping[str, Any],
    prompts: Sequence[str],
    config: ExperimentConfig,
    decode_config: Mapping[str, Any],
    llama_factory: LlamaFactory,
) -> list[JsonDict]:
    load_config = config.effective_load_config(int(model.get("gpu", 0)))
    llm = llama_factory(model_path=str(model["model_path"]), **load_config)
    rows: list[JsonDict] = []
    try:
        for prompt in prompts:
            raw = llm(prompt, **dict(decode_config), seed=config.seed)
            text = _normalize_output(_extract_text(raw))
            rows.append({"prompt_hash": _sha256_text(prompt), "text": text})
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return rows


def _build_artifact(
    *,
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    prompt_hashes: Sequence[str],
    raw_transcript_paths: Sequence[str],
    selected_models: Sequence[Mapping[str, Any]],
    cache_resolution: Mapping[str, str | None],
    duration_s: float,
    runtime_blocker: str | None,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    metrics = _metrics(rows)
    model_specs = [_model_spec(row) for row in selected_models] if runtime_blocker is None else []
    models_used = [str(row["hf_id"]) for row in selected_models] if runtime_blocker is None else []
    ready = (
        runtime_blocker is None
        and metrics["exact_ground_truth_count"] >= 6
        and bool(model_specs)
        and bool(prompt_hashes)
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "solution_verifier_calibration_ready": ready,
        "verifier_gain_delta": metrics["verifier_gain_delta"],
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negative_rate": metrics["false_negative_rate"],
        "exact_ground_truth_count": metrics["exact_ground_truth_count"] if ready else 0,
        "models_used": models_used,
        "model_specs": model_specs,
        "legacy_smoke_only_used": False,
        "cross_family_used": _cross_family_used(selected_models)
        if runtime_blocker is None
        else False,
        "prompt_hashes": list(prompt_hashes) if runtime_blocker is None else [],
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=selected_models if runtime_blocker is None else [],
            duration_s=duration_s,
            repo_commit_func=repo_commit_func,
        ),
        "honest_verdict": _honest_verdict(ready, metrics, runtime_blocker),
        "one_shot_solver_accuracy": metrics["one_shot_solver_accuracy"],
        "verifier_selected_accuracy": metrics["verifier_selected_accuracy"],
        "exact_solver_agreement": metrics["exact_solver_agreement"],
        "exact_solver_authority": "z3_solver" if ready else "z3_solver_not_promoted",
        "panel_rows_path": str(_relative_to(config.repo_root, config.panel_rows_path())),
        "panel_row_count": len(rows),
        "panel_rows_sha256": _sha256_file(config.panel_rows_path()) if rows else "",
        "raw_transcript_paths": list(raw_transcript_paths),
        "tests_or_checks_run": list(config.tests_run),
        "decode_config": config.effective_decode_config(),
        "seed": config.seed,
        "duration_s": duration_s,
        "runtime_blocker": runtime_blocker,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "models_used": artifact["models_used"],
            "prompt_hashes": artifact["prompt_hashes"],
            "metrics": {
                name: artifact[name]
                for name in (
                    "verifier_gain_delta",
                    "false_positive_rate",
                    "false_negative_rate",
                    "exact_ground_truth_count",
                )
            },
        }
    )
    return artifact


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    duration_s: float,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "solution_verifier_calibration_ready": False,
        "verifier_gain_delta": 0.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "exact_ground_truth_count": 0,
        "models_used": [],
        "model_specs": [],
        "legacy_smoke_only_used": False,
        "cross_family_used": False,
        "prompt_hashes": [],
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=[],
            duration_s=duration_s,
            repo_commit_func=repo_commit_func,
        ),
        "honest_verdict": "blocked_sota_gguf_unavailable: no mandated local SOTA GGUF resolved",
        "one_shot_solver_accuracy": 0.0,
        "verifier_selected_accuracy": 0.0,
        "exact_solver_agreement": 0.0,
        "exact_solver_authority": "z3_solver_not_promoted",
        "panel_rows_path": str(_relative_to(config.repo_root, config.panel_rows_path())),
        "panel_row_count": 0,
        "panel_rows_sha256": "",
        "raw_transcript_paths": [],
        "tests_or_checks_run": list(config.tests_run),
        "decode_config": config.effective_decode_config(),
        "seed": config.seed,
        "duration_s": duration_s,
        "runtime_blocker": "no_mandated_gguf_resolved",
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {"models_used": [], "prompt_hashes": [], "metrics": "blocked"}
    )
    validate_artifact(artifact)
    return artifact


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    n = len(rows)
    if n == 0:
        return {
            "exact_ground_truth_count": 0,
            "one_shot_solver_accuracy": 0.0,
            "verifier_selected_accuracy": 0.0,
            "verifier_gain_delta": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "exact_solver_agreement": 0.0,
        }
    solver_correct = sum(1 for row in rows if row["solver_exact_correct"])
    selected_correct = sum(1 for row in rows if row["verifier_selected_exact_correct"])
    candidate_rows = [candidate for row in rows for candidate in row["candidate_rows"]]
    invalid = [candidate for candidate in candidate_rows if not candidate["exact_correct"]]
    valid = [candidate for candidate in candidate_rows if candidate["exact_correct"]]
    false_positives = sum(1 for candidate in invalid if candidate["verifier_accepted"])
    false_negatives = sum(1 for candidate in valid if not candidate["verifier_accepted"])
    one_shot = solver_correct / n
    selected = selected_correct / n
    return {
        "exact_ground_truth_count": n,
        "one_shot_solver_accuracy": round(one_shot, 6),
        "verifier_selected_accuracy": round(selected, 6),
        "verifier_gain_delta": round(selected - one_shot, 6),
        "false_positive_rate": round(false_positives / len(invalid), 6) if invalid else 0.0,
        "false_negative_rate": round(false_negatives / len(valid), 6) if valid else 0.0,
        "exact_solver_agreement": 1.0,
    }


def _score_panel_row(
    row: Mapping[str, Any],
    decision: Mapping[str, Any],
    verifier_text: str,
) -> JsonDict:
    accepted = set(_string_list(decision.get("accepted")))
    selected = str(decision.get("selected") or "")
    candidates = []
    selected_correct = False
    for candidate in row["candidate_rows"]:
        candidate_row = dict(candidate)
        candidate_row["verifier_accepted"] = candidate["candidate_id"] in accepted
        candidates.append(candidate_row)
        if candidate["candidate_id"] == selected:
            selected_correct = bool(candidate["exact_correct"])
    return {
        "fixture_id": row["fixture_id"],
        "exact_authority": "z3_solver",
        "truth": row["truth"],
        "solver_candidate": row["solver_candidate"],
        "solver_exact_correct": bool(candidates[0]["exact_correct"]),
        "solver_raw_output_hash": row["solver_raw_output_hash"],
        "verifier_decision": dict(decision),
        "verifier_raw_output_hash": _sha256_text(verifier_text),
        "verifier_selected_candidate_id": selected,
        "verifier_selected_exact_correct": selected_correct,
        "candidate_rows": candidates,
    }


def _candidate_pool(
    truth_row: Mapping[str, Any],
    solver_candidate: Mapping[str, Any],
) -> list[JsonDict]:
    rows = [
        {
            "candidate_id": "candidate_a",
            "source": "live_solver",
            "candidate": dict(solver_candidate),
        },
        {
            "candidate_id": "candidate_b",
            "source": "deterministic_control",
            "candidate": truth_row["ground_truth_candidate"],
        },
        {
            "candidate_id": "candidate_c",
            "source": "deterministic_distractor",
            "candidate": _distractor(truth_row),
        },
    ]
    return [
        row | {"exact_correct": evaluate_candidate(truth_row, row["candidate"])} for row in rows
    ]


def _solver_prompt(truth_row: Mapping[str, Any]) -> str:
    return (
        "Role: solver\n"
        "Solve this tiny integer SAT/SMT fixture. Return only JSON with either "
        '{"status":"sat","assignment":{...}} or {"status":"unsat"}.\n'
        f"Fixture ID: {truth_row['fixture_id']}\n"
        f"Variables: {', '.join(truth_row['variables'])}\n"
        f"Constraints: {_constraints_text(truth_row['constraints'])}\n"
    )


def _verifier_prompt(
    truth_row: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> str:
    compact_candidates = [
        {"candidate_id": row["candidate_id"], "candidate": row["candidate"]}
        for row in candidate_rows
    ]
    return (
        "Role: verifier\n"
        "Judge candidates for this tiny integer SAT/SMT fixture. Return only JSON "
        'as {"accepted":["candidate_id"],"selected":"candidate_id"}.\n'
        f"Fixture ID: {truth_row['fixture_id']}\n"
        f"Variables: {', '.join(truth_row['variables'])}\n"
        f"Constraints: {_constraints_text(truth_row['constraints'])}\n"
        f"Candidates: {json.dumps(compact_candidates, sort_keys=True)}\n"
    )


def _parse_candidate(text: str) -> JsonDict:
    parsed = _parse_json_object(text)
    status = str(parsed.get("status", "")).lower()
    if status == "unsat":
        return {"status": "unsat"}
    if status == "sat":
        return {"status": "sat", "assignment": _int_assignment(parsed.get("assignment"))}
    return {"status": "unparseable"}


def _parse_verifier_decision(text: str) -> JsonDict:
    parsed = _parse_json_object(text)
    return {
        "accepted": _string_list(parsed.get("accepted")),
        "selected": str(parsed.get("selected") or ""),
    }


def _parse_json_object(text: str) -> JsonDict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _exact_row(fixture: Mapping[str, Any], z3_module: Any) -> JsonDict:
    variables = {name: z3_module.Int(name) for name in fixture["variables"]}
    solver = z3_module.Solver()
    solver.add(
        *[_z3_assertion(constraint, variables, z3_module) for constraint in fixture["constraints"]]
    )
    status = solver.check()
    if status == z3_module.unsat:
        candidate = {"status": "unsat"}
        assignment: JsonDict = {}
        solver_status = "unsat"
    else:
        model = solver.model()
        assignment = {
            name: model.eval(variables[name], model_completion=True).as_long()
            for name in fixture["variables"]
        }
        candidate = {"status": "sat", "assignment": assignment}
        solver_status = "sat"
    return {
        "fixture_id": fixture["fixture_id"],
        "variables": list(fixture["variables"]),
        "constraints": [dict(row) for row in fixture["constraints"]],
        "solver_status": solver_status,
        "ground_truth_assignment": assignment,
        "ground_truth_candidate": candidate,
        "exact_checked": True,
        "exact_authority": "z3_solver",
    }


def _z3_assertion(
    constraint: Mapping[str, Any],
    variables: Mapping[str, Any],
    z3_module: Any,
) -> Any:
    lhs = z3_module.IntVal(0)
    for name, coefficient in sorted(dict(constraint["terms"]).items()):
        lhs += int(coefficient) * variables[str(name)]
    rhs = z3_module.IntVal(int(constraint["rhs"]))
    op = constraint["op"]
    if op == "eq":
        return lhs == rhs
    if op == "ge":
        return lhs >= rhs
    return lhs <= rhs


def _constraints_hold(
    constraints: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]
) -> bool:
    for constraint in constraints:
        lhs = sum(
            int(coefficient) * assignment[str(name)]
            for name, coefficient in constraint["terms"].items()
        )
        rhs = int(constraint["rhs"])
        op = constraint["op"]
        if op == "eq" and lhs != rhs:
            return False
        if op == "ge" and lhs < rhs:
            return False
        if op == "le" and lhs > rhs:
            return False
    return True


def _distractor(truth_row: Mapping[str, Any]) -> JsonDict:
    if truth_row["solver_status"] == "unsat":
        return {"status": "sat", "assignment": {name: 0 for name in truth_row["variables"]}}
    assignment = dict(truth_row["ground_truth_assignment"])
    first = truth_row["variables"][0]
    assignment[first] = int(assignment[first]) + 1
    return {"status": "sat", "assignment": assignment}


def _select_models(cache_resolution: Mapping[str, str | None]) -> list[JsonDict]:
    models = []
    for index, model in enumerate(SOTA_GGUF_MODELS):
        path = cache_resolution.get(model["hf_id"])
        if path:
            models.append(
                {
                    "name": model["name"],
                    "hf_id": model["hf_id"],
                    "model_path": path,
                    "gpu": min(index, 1),
                    "role": model["role"],
                    "family": _model_family(model["hf_id"]),
                }
            )
    if len(models) < 2:
        return models[:1]
    first = models[0]
    for model in models[1:]:
        if model["family"] != first["family"]:
            return [first, model]
    return models[:2]


def _resolve_cache(
    resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig
) -> dict[str, str | None]:
    return {hf_id: resolve_gguf_func(hf_id, config.preferred_quant) for hf_id in MANDATED_MODEL_IDS}


def _model_spec(model: Mapping[str, Any]) -> JsonDict:
    evidence = _file_evidence(model["model_path"], full_limit_bytes=512 * 1024 * 1024)
    return {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "model_path": model["model_path"],
        "gpu": model["gpu"],
        "role": model["role"],
        "family": model["family"],
        "model_hash_or_cache_path": evidence.get("model_hash_or_cache_path"),
        "checksum_feasibility": {
            "method": evidence.get("method"),
            "full_sha256_feasible": bool(evidence.get("full_sha256_feasible")),
            "size_bytes": evidence.get("size_bytes"),
        },
    }


def _substrate(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_models: Sequence[Mapping[str, Any]],
    duration_s: float,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    return {
        "cuda_probe": _cuda_probe(),
        "gpu_inventory": _gpu_inventory(),
        "python_environment": _python_environment(),
        "repo_commit": repo_commit_func(config.repo_root),
        "gguf_cache_resolution": dict(cache_resolution),
        "selected_model_paths": [model["model_path"] for model in selected_models],
        "seed": config.seed,
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "wall_clock_duration_s": duration_s,
        "runtime": "llama_cpp",
        "exact_solver": "z3",
    }


def _honest_verdict(
    ready: bool,
    metrics: Mapping[str, Any],
    runtime_blocker: str | None,
) -> str:
    if ready:
        return (
            "complete: solution_verifier_calibration_ready=true; "
            f"verifier_gain_delta={metrics['verifier_gain_delta']}; "
            f"false_positive_rate={metrics['false_positive_rate']}; "
            f"false_negative_rate={metrics['false_negative_rate']}"
        )
    if runtime_blocker:
        return f"blocked_sota_gguf_unavailable: live GGUF runtime failed: {runtime_blocker}"
    return "blocked_sota_gguf_unavailable: no mandated local SOTA GGUF resolved"


def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - live hardware path.
    from llama_cpp import Llama  # noqa: PLC0415

    return Llama(**kwargs)


def _constraints_text(constraints: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"{constraint['terms']} {constraint['op']} {constraint['rhs']}"
        for constraint in constraints
    )


def _fixture(
    fixture_id: str, variables: Sequence[str], constraints: Sequence[Mapping[str, Any]]
) -> JsonDict:
    return {
        "fixture_id": fixture_id,
        "variables": list(variables),
        "constraints": [dict(constraint) for constraint in constraints],
    }


def _eq(terms: Mapping[str, int], rhs: int) -> JsonDict:
    return {"op": "eq", "terms": dict(terms), "rhs": rhs}


def _ge(terms: Mapping[str, int], rhs: int) -> JsonDict:
    return {"op": "ge", "terms": dict(terms), "rhs": rhs}


def _le(terms: Mapping[str, int], rhs: int) -> JsonDict:
    return {"op": "le", "terms": dict(terms), "rhs": rhs}


def _int_assignment(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    output = {}
    for key, raw in value.items():
        try:
            output[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return output


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _model_family(hf_id: str) -> str:
    lowered = hf_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return hf_id.split("/", 1)[0].lower()


def _cross_family_used(selected_models: Sequence[Mapping[str, Any]]) -> bool:
    families = {model["family"] for model in selected_models}
    return len(families) > 1


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path
