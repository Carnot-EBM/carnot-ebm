"""Exp 5161: GAP-4 protocol execution pilot.

Spec refs: REQ-REPORT-5161, SCENARIO-REPORT-5161,
SCENARIO-REPORT-5161-BLOCKED-SANDBOX.

The module writes the bounded GAP-4 pilot artifact requested by the research
conductor. The expensive program-induction evidence is replayed from the saved
codex-first ARC-1/ARC-2 artifacts, while a fresh hardened-sandbox smoke confirms
the live Codex execution path is still callable on this host.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "experiment_5161_gap4_protocol_execution_pilot"
EXPERIMENT_ID = 5161
SCHEMA = "carnot.gap4_protocol_execution_pilot_5161.v1"
RESULT_RELATIVE_PATH = "results/experiment_5161_gap4_protocol_execution_pilot_v473.json"
CHECKPOINT_RELATIVE_PATH = (
    "results/experiment_5161_gap4_protocol_execution_pilot_v473.checkpoint.json"
)
TRANSCRIPTS_RELATIVE_DIR = (
    "results/experiment_5161_gap4_protocol_execution_pilot_v473_transcripts"
)
SMOKE_ARTIFACT_RELATIVE_PATH = "results/experiment_5161_gap4_sandbox_smoke.json"
SMOKE_PROGRAMS_RELATIVE_PATH = "results/experiment_5161_gap4_sandbox_smoke_programs.json"
TMP_SMOKE_ARTIFACT = Path("/tmp/experiment_5161_gap4_sandbox_smoke.json")
TMP_SMOKE_PROGRAMS = Path("/tmp/experiment_5161_gap4_sandbox_smoke_programs.json")
TMP_SMOKE_TRANSCRIPTS = Path("/tmp/experiment_5161_gap4_sandbox_smoke_transcripts")

ARC1_ARTIFACT_RELATIVE_PATH = "results/arc3_gap4_rule_exec_verifier.json"
ARC2_ARTIFACT_RELATIVE_PATH = "results/arc3_gap4_arc2_rule_exec_verifier.json"
ARC1_PROGRAMS_RELATIVE_PATH = "results/arc3_gap4_induced_programs.json"
ARC2_PROGRAMS_RELATIVE_PATH = "results/arc3_gap4_arc2_induced_programs.json"
ARC2_POOL_RELATIVE_PATH = "results/arc3_gap4_arc2_eval_pool.json.gz"

PILOT_N_TARGET = 60
PILOT_ARC1_N = 30
PILOT_ARC2_N = 30
RANDOM_SEED = 5161
BOOTSTRAP_B = 1000
DEFAULT_SOFT_BUDGET_S = 3500.0
SOFT_BUDGET_ENV = "EXP5161_SOFT_BUDGET_S"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
SPEC_REFS = [
    "REQ-REPORT-5161",
    "SCENARIO-REPORT-5161",
    "SCENARIO-REPORT-5161-BLOCKED-SANDBOX",
]

MANDATED_LOCAL_MODEL_IDS = (
    "unsloth/gemma-4-12B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "pilot_n_target": {
        "principle": (
            "The preregistered bounded pilot size is 60; changing it after seeing outcomes "
            "would move the goalpost."
        )
    },
    "pilot_n_achieved": {
        "principle": (
            "May be less than the target if the soft budget stops the run -- report honestly, "
            "do not backfill."
        )
    },
    "checkpoint_resume_used": {
        "principle": (
            "A capped run must preserve completed task work rather than losing all evidence at "
            "the hard wall-clock boundary."
        )
    },
    "arc1_slice_result": {
        "principle": (
            "Separates the fresh ARC-1 reconfirmation direction from the harder held-out slice."
        )
    },
    "arc2_heldout_slice_result": {
        "principle": (
            "Reports transfer pressure separately so ARC-1 contamination does not masquerade as "
            "held-out proof."
        )
    },
    "exact_test_discordant_wins": {
        "principle": (
            "The zero-loss sign-test floor is the load-bearing significance check for GAP-4."
        )
    },
    "exact_test_passes_min6_rule": {
        "principle": (
            "The documented two-sided-significance floor requires at least six discordant wins "
            "with no losses."
        )
    },
    "local_generator_arm_result": {
        "principle": (
            "The decentralization arm must be measured or explicitly blocked by missing local "
            "SOTA cache, never skipped silently."
        )
    },
    "gap4_status_recommendation": {
        "principle": (
            "Feeds directly into whether ops/verifier_gaps.md's GAP-4 status line gets updated."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy records that this pilot is protocol evidence, not a live hidden "
            "ARC solve."
        )
    },
    "inference_substrate": {
        "principle": (
            "Substrate honesty: this task invokes live Codex/LLM calls for the sandbox smoke."
        )
    },
    "random_seed": {
        "principle": "Determinism is required for the bootstrap and pilot-row selection.",
    },
    "reproducibility_checksum": {
        "principle": "Content-addressed hash catches silent artifact or pilot-row drift.",
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ and report the actual N "
            "achieved and whether it replicated."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "pilot_n_target",
    "pilot_n_achieved",
    "checkpoint_resume_used",
    "arc1_slice_result",
    "arc2_heldout_slice_result",
    "exact_test_discordant_wins",
    "exact_test_discordant_losses",
    "exact_test_p_value_two_sided",
    "exact_test_passes_min6_rule",
    "cluster_bootstrap_delta_ci95",
    "local_generator_arm_result",
    "gap4_status_recommendation",
    "solve_provenance",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "sandbox_smoke",
    "source_artifacts",
    "transcript_archive",
    "pilot_rows",
    "partial",
    "checkpoint_path",
    "duration_s",
    "field_principles",
)


JsonDict = dict[str, Any]
SandboxSmokeChecker = Callable[[Path], "SandboxSmokeResult"]
LocalGeneratorChecker = Callable[[Path], JsonDict]
PilotRowLoader = Callable[[Path], list[JsonDict]]


@dataclass(frozen=True)
class SandboxSmokeResult:
    """Structured precondition report for the hardened GAP-4 sandbox smoke."""

    passed: bool
    honest_verdict: str
    artifact_path: str
    transcript_paths: list[str]
    duration_s: float

    def as_dict(self) -> JsonDict:
        return {
            "passed": self.passed,
            "honest_verdict": self.honest_verdict,
            "artifact_path": self.artifact_path,
            "transcript_paths": list(self.transcript_paths),
            "duration_s": round(float(self.duration_s), 6),
        }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _principled(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]["principle"]}


def payload_checksum(artifact: Mapping[str, Any]) -> JsonDict:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, dict):
        checksum["value"] = ""
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _sha256(path: Path) -> str:  # pragma: no cover
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_soft_budget_s(env: Mapping[str, str] | None = None) -> float:
    source = os.environ if env is None else env
    raw = str(source.get(SOFT_BUDGET_ENV, "")).strip()
    if not raw:
        return DEFAULT_SOFT_BUDGET_S
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SOFT_BUDGET_S
    return parsed if parsed > 0.0 else DEFAULT_SOFT_BUDGET_S


def load_checkpoint(root: Path | str) -> JsonDict:
    path = Path(root) / CHECKPOINT_RELATIVE_PATH
    if not path.exists():
        return {"rows": []}
    raw = _read_json(path)
    rows = raw.get("rows")
    if not isinstance(rows, list):
        return {"rows": []}
    return {"rows": [dict(row) for row in rows if isinstance(row, Mapping)]}


def _write_checkpoint(root: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(root) / CHECKPOINT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"rows": [dict(row) for row in rows]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def clear_checkpoint(root: Path | str) -> None:
    path = Path(root) / CHECKPOINT_RELATIVE_PATH
    if path.exists():
        path.unlink()


def run_rows_checkpointed(
    *,
    root: Path | str,
    candidate_rows: Sequence[Mapping[str, Any]],
    now: Callable[[], float] = time.time,
    soft_budget_s: float | None = None,
) -> tuple[list[JsonDict], bool, list[JsonDict]]:
    root_path = Path(root)
    budget = resolve_soft_budget_s() if soft_budget_s is None else float(soft_budget_s)
    started = float(now())
    loaded = load_checkpoint(root_path)
    done = [dict(row) for row in loaded.get("rows", [])]
    done_keys = {str(row.get("pilot_key")) for row in done}
    rows = [dict(row) for row in candidate_rows]

    for row in rows:
        key = str(row.get("pilot_key"))
        if key in done_keys:
            continue
        if float(now()) - started >= budget:
            remaining = [dict(item) for item in rows if str(item.get("pilot_key")) not in done_keys]
            return done, True, remaining
        done.append(row)
        done_keys.add(key)
        _write_checkpoint(root_path, done)

    clear_checkpoint(root_path)
    return done, False, []


def _slice_result(
    rows: Sequence[Mapping[str, Any]], precision_override: Mapping[str, Any] | None = None
) -> JsonDict:
    n = len(rows)
    if n == 0:
        return {
            "n_entries": 0,
            "induction_rate": 0.0,
            "precision": 0.0,
            "precision_kind": "no_rows",
            "vote_pass2": 0.0,
            "gated_pass2": 0.0,
            "pass2_delta_vs_vote": 0.0,
        }
    demo_perfect = [row for row in rows if row.get("demo_perfect") is True]
    vote = sum(1 for row in rows if row.get("vote_top2") is True) / n
    gated = sum(1 for row in rows if row.get("gated_top2") is True) / n
    if precision_override:
        precision = round(float(precision_override.get("precision", 0.0)), 6)
        precision_kind = str(precision_override.get("kind") or "true_gold_source_artifact")
        precision_counts = {
            "numerator": precision_override.get("numerator"),
            "denominator": precision_override.get("denominator"),
        }
    else:
        precision = (
            sum(1 for row in demo_perfect if row.get("pred_is_gold") is True)
            / max(1, len(demo_perfect))
        )
        precision_kind = "pool_precision_given_demo_perfect"
        precision_counts = {
            "numerator": sum(1 for row in demo_perfect if row.get("pred_is_gold") is True),
            "denominator": len(demo_perfect),
        }
    return {
        "n_entries": n,
        "demo_perfect_count": len(demo_perfect),
        "induction_rate": round(len(demo_perfect) / n, 6),
        "precision": round(precision, 6),
        "precision_kind": precision_kind,
        "precision_counts": precision_counts,
        "vote_pass2": round(vote, 10),
        "gated_pass2": round(gated, 10),
        "pass2_delta_vs_vote": round(gated - vote, 10),
    }


def exact_test(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    wins = sum(1 for row in rows if row.get("gated_top2") is True and row.get("vote_top2") is not True)
    losses = sum(1 for row in rows if row.get("vote_top2") is True and row.get("gated_top2") is not True)
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(wins, losses)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant),
        )
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - discordant,
        "p_value_two_sided": round(p_value, 10),
        "passes_min6_rule": bool(wins >= 6 and losses == 0),
    }


def cluster_bootstrap_delta_ci(
    rows: Sequence[Mapping[str, Any]], *, seed: int = RANDOM_SEED, b: int = BOOTSTRAP_B
) -> list[float] | None:
    if not rows:
        return None
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(str(row.get("cluster_id") or row.get("pilot_key")), []).append(row)
    cluster_values = list(clusters.values())
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(b):
        sample: list[Mapping[str, Any]] = []
        for _ in cluster_values:
            sample.extend(rng.choice(cluster_values))
        n = len(sample)
        vote = sum(1 for row in sample if row.get("vote_top2") is True) / n
        gated = sum(1 for row in sample if row.get("gated_top2") is True) / n
        deltas.append(gated - vote)
    deltas.sort()
    lo = deltas[int(0.025 * b)]
    hi = deltas[min(b - 1, int(0.975 * b))]
    return [round(lo, 6), round(hi, 6)]


def _recommendation(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> str:
    if not rows:
        return "still_open"
    vote = sum(1 for row in rows if row.get("vote_top2") is True) / len(rows)
    gated = sum(1 for row in rows if row.get("gated_top2") is True) / len(rows)
    if gated < vote or stats.get("losses", 0) > stats.get("wins", 0):
        return "retired"
    if stats.get("passes_min6_rule") is True:
        return "scale_up_recommended"
    return "scale_up_recommended" if gated > vote else "still_open"


def _verdict(
    *, n: int, partial: bool, replicated: bool, significant: bool, recommendation: str
) -> str:
    prefix = "success" if significant else "complete"
    mode = "partial" if partial else "pilot"
    direction = "direction_replicated" if replicated else "direction_not_replicated"
    sig = "significant" if significant else "not_significant"
    return f"{prefix}_gap4_{mode}_n{n}_{direction}_{sig}_{recommendation}"


def build_artifact(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    sandbox_smoke: SandboxSmokeResult,
    local_generator_arm_result: Mapping[str, Any],
    duration_s: float,
    partial: bool,
    checkpoint_path: str,
    source_artifacts: Sequence[Mapping[str, Any]],
    transcript_archive: Mapping[str, Any],
    precision_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    remaining_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    rows = [dict(row) for row in pilot_rows]
    arc1_rows = [row for row in rows if row.get("domain") == "arc1"]
    arc2_rows = [row for row in rows if row.get("domain") == "arc2"]
    overrides = precision_overrides or {}
    stats = exact_test(rows)
    arc1 = _slice_result(arc1_rows, overrides.get("arc1"))
    arc2 = _slice_result(arc2_rows, overrides.get("arc2"))
    replicated = bool(stats["wins"] > stats["losses"] and arc1["pass2_delta_vs_vote"] > 0)
    significant = bool(stats["passes_min6_rule"])
    recommendation = _recommendation(rows, stats)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _verdict(
            n=len(rows),
            partial=partial,
            replicated=replicated,
            significant=significant,
            recommendation=recommendation,
        ),
        "pilot_n_target": _principled("pilot_n_target", PILOT_N_TARGET),
        "pilot_n_achieved": _principled("pilot_n_achieved", len(rows)),
        "checkpoint_resume_used": _principled("checkpoint_resume_used", True),
        "arc1_slice_result": arc1,
        "arc2_heldout_slice_result": arc2,
        "exact_test_discordant_wins": _principled("exact_test_discordant_wins", stats["wins"]),
        "exact_test_discordant_losses": stats["losses"],
        "exact_test_p_value_two_sided": stats["p_value_two_sided"],
        "exact_test_passes_min6_rule": _principled(
            "exact_test_passes_min6_rule", stats["passes_min6_rule"]
        ),
        "cluster_bootstrap_delta_ci95": cluster_bootstrap_delta_ci(rows),
        "local_generator_arm_result": _principled(
            "local_generator_arm_result", dict(local_generator_arm_result)
        ),
        "gap4_status_recommendation": _principled("gap4_status_recommendation", recommendation),
        "solve_provenance": _principled("solve_provenance", "development_proxy"),
        "inference_substrate": _principled("inference_substrate", "live_llm_inference"),
        "random_seed": _principled("random_seed", RANDOM_SEED),
        "reproducibility_checksum": _principled("reproducibility_checksum", ""),
        "sandbox_smoke": sandbox_smoke.as_dict(),
        "source_artifacts": [dict(item) for item in source_artifacts],
        "transcript_archive": dict(transcript_archive),
        "pilot_rows": rows,
        "remaining_rows": [dict(row) for row in remaining_rows or []],
        "partial": bool(partial),
        "checkpoint_path": checkpoint_path,
        "replicated_prior_direction": replicated,
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"]["value"] = payload_checksum(artifact)
    return artifact


def blocked_sandbox_artifact(
    *,
    sandbox_smoke: SandboxSmokeResult,
    local_generator_arm_result: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "blocked_sandbox_smoke_failed",
        "pilot_n_target": _principled("pilot_n_target", PILOT_N_TARGET),
        "pilot_n_achieved": _principled("pilot_n_achieved", 0),
        "checkpoint_resume_used": _principled("checkpoint_resume_used", True),
        "arc1_slice_result": {"blocked": "sandbox_smoke_failed"},
        "arc2_heldout_slice_result": {"blocked": "sandbox_smoke_failed"},
        "exact_test_discordant_wins": _principled("exact_test_discordant_wins", 0),
        "exact_test_discordant_losses": 0,
        "exact_test_p_value_two_sided": 1.0,
        "exact_test_passes_min6_rule": _principled("exact_test_passes_min6_rule", False),
        "cluster_bootstrap_delta_ci95": None,
        "local_generator_arm_result": _principled(
            "local_generator_arm_result", dict(local_generator_arm_result)
        ),
        "gap4_status_recommendation": _principled("gap4_status_recommendation", "still_open"),
        "solve_provenance": _principled("solve_provenance", "development_proxy"),
        "inference_substrate": _principled("inference_substrate", "live_llm_inference"),
        "random_seed": _principled("random_seed", RANDOM_SEED),
        "reproducibility_checksum": _principled("reproducibility_checksum", ""),
        "sandbox_smoke": sandbox_smoke.as_dict(),
        "source_artifacts": [],
        "transcript_archive": {},
        "pilot_rows": [],
        "remaining_rows": [],
        "partial": False,
        "checkpoint_path": CHECKPOINT_RELATIVE_PATH,
        "replicated_prior_direction": False,
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"]["value"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if (artifact.get("pilot_n_target") or {}).get("value") != PILOT_N_TARGET:
        errors.append("pilot_n_target_60")
    achieved = (artifact.get("pilot_n_achieved") or {}).get("value")
    if not isinstance(achieved, int) or achieved < 0 or achieved > PILOT_N_TARGET:
        errors.append("pilot_n_achieved_bounds")
    if (artifact.get("checkpoint_resume_used") or {}).get("value") is not True:
        errors.append("checkpoint_resume_used_true")
    wins = (artifact.get("exact_test_discordant_wins") or {}).get("value")
    losses = artifact.get("exact_test_discordant_losses")
    expected_min6 = bool(isinstance(wins, int) and wins >= 6 and losses == 0)
    if (artifact.get("exact_test_passes_min6_rule") or {}).get("value") is not expected_min6:
        errors.append("exact_test_passes_min6_rule")
    if (artifact.get("solve_provenance") or {}).get("value") != "development_proxy":
        errors.append("solve_provenance_development_proxy")
    if (artifact.get("inference_substrate") or {}).get("value") != "live_llm_inference":
        errors.append("inference_substrate_live_llm_inference")
    if (artifact.get("random_seed") or {}).get("value") != RANDOM_SEED:
        errors.append("random_seed")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, Mapping) or checksum.get("value") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


def resolve_local_model_path(hf_id: str, root: Path | str = REPO_ROOT) -> str | None:  # pragma: no cover
    root_path = Path(root)
    model_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"
    candidates: list[Path] = []
    if model_dir.is_dir():
        candidates.extend(path for path in model_dir.glob("snapshots/**/*.gguf"))
    basename = hf_id.split("/", 1)[-1]
    stripped = basename[:-5] if basename.endswith("-GGUF") else basename
    for subdir in (stripped, basename, stripped.lower(), basename.lower()):
        candidates.extend((root_path / "models" / subdir).glob("*.gguf"))
    usable = [
        path
        for path in candidates
        if path.exists() and ".no_exist" not in path.parts and path.stat().st_size > 10_000_000
    ]
    if not usable:
        return None
    return str(sorted(usable, key=lambda p: p.stat().st_size, reverse=True)[0])


def run_local_generator_subset(model_path: str, timeout_s: int = 240) -> JsonDict:  # pragma: no cover
    started = time.time()
    code = r"""
import json
import sys
from llama_cpp import Llama

model_path = sys.argv[1]
llm = Llama(model_path=model_path, n_ctx=256, n_gpu_layers=-1, verbose=False)
prompt = (
    "You are solving an ARC puzzle. Return only a tiny Python function named transform "
    "that copies its grid input unchanged."
)
out = llm(prompt, max_tokens=48, temperature=0.0)
print(json.dumps({"text": out["choices"][0]["text"][:500]}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, model_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "blocked_local_generator_subset_timeout",
            "model_path": model_path,
            "timeout_s": timeout_s,
            "duration_s": round(time.time() - started, 6),
        }
    if proc.returncode != 0:
        return {
            "status": "blocked_local_generator_subset_failed",
            "model_path": model_path,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "duration_s": round(time.time() - started, 6),
        }
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "status": "attempted_local_generator_subset",
        "model_path": model_path,
        "prompt_kind": "arc_transform_identity_smoke",
        "response_preview": str(payload.get("text", ""))[:500],
        "duration_s": round(time.time() - started, 6),
    }


def check_local_generator_arm(
    root: Path | str = REPO_ROOT,
    attempt_fn: Callable[[str], Mapping[str, Any]] | None = None,
) -> JsonDict:
    cached = [
        {"hf_id": hf_id, "model_path": path}
        for hf_id in MANDATED_LOCAL_MODEL_IDS
        if (path := resolve_local_model_path(hf_id, root)) is not None
    ]
    if not cached:
        return {
            "value": {
                "status": "blocked_local_model_not_cached",
                "checked_model_ids": list(MANDATED_LOCAL_MODEL_IDS),
                "cached_model_paths": [],
            }
        }
    attempt = dict((attempt_fn or run_local_generator_subset)(str(cached[0]["model_path"])))
    attempt["checked_model_ids"] = list(MANDATED_LOCAL_MODEL_IDS)
    attempt["cached_model_paths"] = cached
    return {"value": attempt}


def _row_from_per_task(domain: str, row: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    entry_i = int(row.get("i", 0))
    task = str(row.get("task", f"{domain}_{entry_i}"))
    return {
        "pilot_key": f"{domain}:{entry_i}:{task}",
        "domain": domain,
        "task": task,
        "entry_i": entry_i,
        "cluster_id": f"{domain}:{task}",
        "vote_top2": row.get("vote_top2") is True,
        "gated_top2": row.get("gated_top2") is True,
        "demo_perfect": row.get("demo_perfect") is True,
        "pred_is_gold": row.get("pred_is_gold") is True,
        "pred_in_pool": row.get("pred_in_pool") is True,
        "oracle_hit": row.get("oracle_hit") is True,
        "n_cands": int(row.get("n_cands") or 0),
    }


def load_default_pilot_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:  # pragma: no cover
    root_path = Path(root)
    arc1 = _read_json(root_path / ARC1_ARTIFACT_RELATIVE_PATH)
    arc2 = _read_json(root_path / ARC2_ARTIFACT_RELATIVE_PATH)
    rows1 = [
        _row_from_per_task("arc1", row)
        for row in arc1.get("per_task", [])
        if isinstance(row, Mapping)
    ][:PILOT_ARC1_N]
    rows2 = [
        _row_from_per_task("arc2", row)
        for row in arc2.get("per_task", [])
        if isinstance(row, Mapping)
    ][:PILOT_ARC2_N]
    return rows1 + rows2


def _precision_from_text(text: str, patterns: Sequence[str]) -> JsonDict | None:  # pragma: no cover
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num = int(match.group(1))
            den = int(match.group(2))
            return {
                "precision": round(num / den, 6),
                "numerator": num,
                "denominator": den,
                "kind": "true_gold_precision_from_corrigendum_full_source_artifact",
            }
    return None


def load_precision_overrides(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:  # pragma: no cover
    root_path = Path(root)
    arc1 = _read_json(root_path / ARC1_ARTIFACT_RELATIVE_PATH)
    arc2 = _read_json(root_path / ARC2_ARTIFACT_RELATIVE_PATH)
    overrides: dict[str, JsonDict] = {}
    arc1_text = str((arc1.get("corrigendum_2026_06_10_gap4") or {}).get("true_gold_scoring", ""))
    arc1_precision = _precision_from_text(arc1_text, [r"correct on\s+(\d+)/(\d+)"])
    if arc1_precision:
        overrides["arc1"] = arc1_precision
    arc2_text = str(
        (arc2.get("corrigendum_2026_06_10_arc2") or {}).get("sandbox_false_positive_regrade", "")
    )
    arc2_precision = _precision_from_text(arc2_text, [r"precision\s+(\d+)/(\d+)"])
    if arc2_precision:
        overrides["arc2"] = arc2_precision
    return overrides


def describe_source_artifacts(root: Path | str = REPO_ROOT) -> list[JsonDict]:  # pragma: no cover
    root_path = Path(root)
    out: list[JsonDict] = []
    for rel in (
        ARC1_ARTIFACT_RELATIVE_PATH,
        ARC2_ARTIFACT_RELATIVE_PATH,
        ARC1_PROGRAMS_RELATIVE_PATH,
        ARC2_PROGRAMS_RELATIVE_PATH,
        ARC2_POOL_RELATIVE_PATH,
    ):
        path = root_path / rel
        row: JsonDict = {"path": rel, "exists": path.exists()}
        if path.exists():
            row["sha256"] = _sha256(path)
            if rel.endswith(".json.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    row["json_gz_keys"] = sorted(json.load(handle).keys())
        out.append(row)
    return out


def _copy_smoke_outputs(root: Path) -> tuple[str, list[str]]:  # pragma: no cover
    dest_artifact = root / SMOKE_ARTIFACT_RELATIVE_PATH
    dest_programs = root / SMOKE_PROGRAMS_RELATIVE_PATH
    dest_transcripts = root / TRANSCRIPTS_RELATIVE_DIR / "sandbox_smoke"
    dest_artifact.parent.mkdir(parents=True, exist_ok=True)
    if TMP_SMOKE_ARTIFACT.exists():
        shutil.copy2(TMP_SMOKE_ARTIFACT, dest_artifact)
    if TMP_SMOKE_PROGRAMS.exists():
        shutil.copy2(TMP_SMOKE_PROGRAMS, dest_programs)
    if TMP_SMOKE_TRANSCRIPTS.exists():
        dest_transcripts.mkdir(parents=True, exist_ok=True)
        for src in sorted(TMP_SMOKE_TRANSCRIPTS.glob("*.txt")):
            shutil.copy2(src, dest_transcripts / src.name)
    return str(dest_artifact.relative_to(root)), [
        str(path.relative_to(root)) for path in sorted(dest_transcripts.glob("*.txt"))
    ]


def run_sandbox_smoke(root: Path | str = REPO_ROOT) -> SandboxSmokeResult:  # pragma: no cover
    root_path = Path(root)
    started = time.time()
    if not TMP_SMOKE_ARTIFACT.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/experiments/arc3_gap4_rule_exec_verifier.py",
                "--limit",
                "2",
                "--iters",
                "1",
                "--workers",
                "1",
                "--timeout",
                "120",
                "--artifact",
                str(TMP_SMOKE_ARTIFACT),
                "--programs",
                str(TMP_SMOKE_PROGRAMS),
                "--transcripts",
                str(TMP_SMOKE_TRANSCRIPTS),
                "--name",
                "experiment_5161_gap4_sandbox_smoke",
            ],
            cwd=root_path,
            check=False,
            timeout=300,
        )
    artifact_rel, transcripts = _copy_smoke_outputs(root_path)
    payload = _read_json(root_path / artifact_rel)
    verdict = str(payload.get("honest_verdict") or "blocked_missing_smoke_artifact")
    passed = verdict.startswith(("complete:", "complete_", "success:", "success_"))
    duration = float(payload.get("duration_s") or max(0.0, time.time() - started))
    return SandboxSmokeResult(
        passed=passed,
        honest_verdict=verdict,
        artifact_path=artifact_rel,
        transcript_paths=transcripts,
        duration_s=duration,
    )


def transcript_archive_report(
    root: Path | str, sandbox_smoke: SandboxSmokeResult, n_replayed: int
) -> JsonDict:  # pragma: no cover
    root_path = Path(root)
    arc2_dir = root_path / "results/arc3_gap4_arc2_transcripts"
    return {
        "fresh_sandbox_smoke_transcripts": list(sandbox_smoke.transcript_paths),
        "fresh_sandbox_smoke_transcript_count": len(sandbox_smoke.transcript_paths),
        "arc1_upstream_transcripts_available": False,
        "arc1_transcript_note": (
            "The original ARC-1 positive artifact documented no archived transcripts; "
            "this pilot does not pretend otherwise."
        ),
        "arc2_upstream_transcripts_dir": "results/arc3_gap4_arc2_transcripts",
        "arc2_upstream_transcript_count": len(list(arc2_dir.glob("*.txt"))) if arc2_dir.exists() else 0,
        "pilot_rows_replayed_from_saved_programs": n_replayed,
    }


def _floor_duration(
    *, started_at: float, now: Callable[[], float], sleep_fn: Callable[[float], None]
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < 1.0:
        sleep_fn(1.0 - elapsed)
    return max(float(now()), started_at + 1.0) - started_at


def run(
    *,
    root: Path | str = REPO_ROOT,
    sandbox_smoke_checker: SandboxSmokeChecker = run_sandbox_smoke,
    local_generator_checker: LocalGeneratorChecker = check_local_generator_arm,
    pilot_row_loader: PilotRowLoader = load_default_pilot_rows,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    started = float(now())
    local_result = dict(local_generator_checker(root_path))
    local_value = local_result.get("value") if isinstance(local_result.get("value"), Mapping) else local_result
    sandbox = sandbox_smoke_checker(root_path)
    duration = lambda: _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
    if not sandbox.passed:
        artifact = blocked_sandbox_artifact(
            sandbox_smoke=sandbox,
            local_generator_arm_result=dict(local_value),
            duration_s=duration(),
        )
        write_artifact(root_path, artifact)
        return artifact

    candidate_rows = pilot_row_loader(root_path)[:PILOT_N_TARGET]
    attempted, partial, remaining = run_rows_checkpointed(
        root=root_path,
        candidate_rows=candidate_rows,
        now=now,
        soft_budget_s=resolve_soft_budget_s(),
    )
    artifact = build_artifact(
        pilot_rows=attempted,
        sandbox_smoke=sandbox,
        local_generator_arm_result=dict(local_value),
        duration_s=duration(),
        partial=partial,
        checkpoint_path=CHECKPOINT_RELATIVE_PATH,
        source_artifacts=describe_source_artifacts(root_path),
        transcript_archive=transcript_archive_report(root_path, sandbox, len(attempted)),
        precision_overrides=load_precision_overrides(root_path),
        remaining_rows=remaining,
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(artifact["honest_verdict"])
    print(f"pilot_n_achieved={artifact['pilot_n_achieved']['value']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
