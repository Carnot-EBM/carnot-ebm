"""Exp 4425: Qwen config-rule vocabulary transfer retest.

Spec refs: REQ-LEARN-4425, SCENARIO-LEARN-4425,
SCENARIO-LEARN-4425-NULL.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4425_config_rule_vocabulary_transfer.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
EXP4414_RELATIVE_PATH = "results/experiment_4414_config_rule_induction_solve.json"
EXP4421_RELATIVE_PATH = "results/experiment_4421_config_rule_solve_unseen.json"
COLD_REPEAT_BENCH_RELATIVE_PATH = "results/arc3_layerb_repeat_bench_qwen3_5-9b-mtp_mtp.json"
SEEDED_REPEAT_BENCH_RELATIVE_PATH = (
    "results/arc3_layerb_repeat_bench_qwen3_5-9b-mtp_vocabseed_mtp.json"
)
RANDOM_SEED = 4425
BOOTSTRAP_RESAMPLES = 2000
SEED_COUNT = 4

DEFAULT_SCAFFOLDED_TASK = (
    "You are inducing the WIN RULE of an ARC-AGI-3 configuration puzzle. "
    "Use the extracted editable region, non-wins, and reference digest to "
    "write a relational Python is_win(grid) predicate."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "config_rule_vocabulary_transfers",
    "config_rule_vocabulary",
    "vocabulary_seeded_prompt",
    "transfer_learning_curve",
    "verifier_is_oracle",
    "random_seed",
    "logged_gaps",
    "model_specs",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "config_rule_vocabulary_transfers": (
        "BARE bool := held-out grounding-rate lift CI-excl-0; principle: a "
        "transferable vocabulary is the self-learning compounding signal."
    ),
    "verifier_is_oracle": (
        "false; the verifier grounds proposed predicates, it does not define "
        "correctness for the transfer claim."
    ),
    "random_seed": "Bare integer; fixes split order and bootstrap reproducibility.",
    "honest_verdict": "Terminal-prefixed honest outcome.",
}

PRIMITIVE_SPECS = (
    {
        "name": "editable_count==reference_count",
        "description": "Editable colour/object count must equal a reference count.",
        "prompt_hint": "Look for count equality between editable cells and reference features.",
        "cues": ("editable count", "reference count", "count_4_equals_reference", "count equals"),
    },
    {
        "name": "match-reference",
        "description": "Editable state directly matches a reference value, object, or relation.",
        "prompt_hint": "Use the reference digest as the source of constants and relations.",
        "cues": ("match reference", "equals reference", "reference region", "reference"),
    },
    {
        "name": "progress-fill",
        "description": "A controlled marker/region advances until it reaches a target extent.",
        "prompt_hint": "Detect monotone progress toward a target marker, edge, or fill boundary.",
        "cues": ("progress fill", "extend", "extends", "fill", "controlled marker"),
    },
    {
        "name": "glyph-rewrite",
        "description": "Editable glyphs or sequences are rewritten through an LHS/RHS map.",
        "prompt_hint": "Infer symbolic glyph rewrites instead of literal editable arrays.",
        "cues": ("glyph", "rewrite", "lhs", "rhs", "sequence"),
    },
    {
        "name": "marker-coverage",
        "description": "A win fires when controlled markers cover all target markers.",
        "prompt_hint": "Check target marker coverage, not incidental path length.",
        "cues": ("marker coverage", "target marker", "controlled marker", "occupied by controlled"),
    },
    {
        "name": "shape-pattern-match",
        "description": "Editable shape, cross, or cast-grid pattern must match a reference pattern.",
        "prompt_hint": "Compare shape topology and small pattern layout.",
        "cues": ("shape pattern", "3x3", "cross", "cast grid", "cast-grid", "alignment"),
    },
    {
        "name": "symmetry",
        "description": "Mirror or rotational symmetry constrains the winning editable region.",
        "prompt_hint": "Check reflection/rotation invariants before literal constants.",
        "cues": ("symmetry", "mirror", "reflect", "rotation"),
    },
    {
        "name": "program-command-map",
        "description": "Program commands or action labels map objects to target attributes.",
        "prompt_hint": "Ground command semantics against target object attributes.",
        "cues": ("program", "command", "action label", "scale rotation", "property"),
    },
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_registry(root: Path) -> dict[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {"games": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {"games": []}


def _source_record(game: str, source: str, rule_text: str) -> dict[str, str]:
    return {"game": game, "source": source, "rule_text": rule_text}


def extract_grounded_rule_sources(root: Path) -> list[dict[str, str]]:
    registry = _load_registry(root)
    sources: list[dict[str, str]] = []
    for entry in registry.get("games", []) if isinstance(registry.get("games"), list) else []:
        if not isinstance(entry, Mapping):  # pragma: no cover - malformed registry guard
            continue
        reproduced = entry.get("reproducibility") == "reproduced" or int(
            entry.get("levels_reproduced") or 0
        ) > 0
        if not reproduced:
            continue
        text = " ".join(
            str(entry.get(key) or "") for key in ("win_condition", "solver", "reproduce", "action_model")
        ).strip()
        if text:
            sources.append(_source_record(str(entry.get("game") or "unknown"), REGISTRY_RELATIVE_PATH, text))

    exp4414 = _load_json(root / EXP4414_RELATIVE_PATH)
    for rule in exp4414.get("config_win_rules_grounded", []):
        if not isinstance(rule, Mapping):  # pragma: no cover - malformed artifact guard
            continue
        false_positive_rate = rule.get("false_positive_rate")
        grounded = (
            int(rule.get("tier") or 0) >= 1
            and false_positive_rate is not None
            and float(false_positive_rate) == 0.0
        )
        if grounded:
            sources.append(
                _source_record(
                    str(rule.get("game") or "unknown"),
                    EXP4414_RELATIVE_PATH,
                    str(rule.get("predicate") or ""),
                )
            )

    exp4421 = _load_json(root / EXP4421_RELATIVE_PATH)
    grounded_win = exp4421.get("grounded_win_condition")
    if exp4421.get("offline_reproduced") is True and isinstance(grounded_win, Mapping):
        sources.append(
            _source_record(
                str(exp4421.get("target_game") or "unknown"),
                EXP4421_RELATIVE_PATH,
                str(grounded_win.get("predicate") or ""),
            )
        )
    return sources


def _primitive_matches(spec: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    text = f"{source.get('game', '')} {source.get('rule_text', '')}".lower().replace("_", " ")
    return any(str(cue).lower() in text for cue in spec["cues"])


def build_rule_vocabulary(rule_sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    vocabulary: list[dict[str, Any]] = []
    for spec in PRIMITIVE_SPECS:
        matching_sources = [source for source in rule_sources if _primitive_matches(spec, source)]
        if matching_sources:
            vocabulary.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "prompt_hint": spec["prompt_hint"],
                    "source_games": sorted({str(source.get("game")) for source in matching_sources}),
                    "source_count": len(matching_sources),
                }
            )
    return vocabulary


def build_vocabulary_seeded_prompt(
    vocabulary: Sequence[Mapping[str, Any]],
    base_prompt: str = DEFAULT_SCAFFOLDED_TASK,
) -> str:
    lines = [
        "/no_think",
        "RELATIONAL WIN-RULE VOCABULARY",
        "Generator: Qwen3.5-9B-MTP, iGPU, /no_think, MTP, four seeds",
    ]
    for primitive in vocabulary:
        games = ", ".join(str(game) for game in primitive.get("source_games", []))
        lines.append(
            f"- {primitive['name']}: {primitive['description']} "
            f"Hint: {primitive['prompt_hint']} Source games: {games or 'unknown'}."
        )
    lines.extend(["", base_prompt])
    return "\n".join(lines)


def _repeat_observations(payload: Mapping[str, Any]) -> dict[str, list[bool]]:
    observations: dict[str, list[bool]] = {}
    per_game = payload.get("per_game")
    if not isinstance(per_game, Mapping):
        return observations
    for game, row in per_game.items():
        if not isinstance(row, Mapping):  # pragma: no cover - malformed repeat-bench guard
            continue
        runs = row.get("runs")
        if not isinstance(runs, list):  # pragma: no cover - malformed repeat-bench guard
            continue
        observations[str(game)] = [bool(run.get("grounded")) for run in runs if isinstance(run, Mapping)]
    return observations


def _rate(values: Sequence[bool]) -> float:
    return round(sum(1 for value in values if value) / len(values), 6)


def _bootstrap_ci95(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(set(values)) == 1:
        only = round(float(values[0]), 6)
        return [only, only]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(0.975 * (resamples - 1))]
    return [round(lo, 6), round(hi, 6)]


def _transfer_curve(
    cold_observations: Mapping[str, Sequence[bool]],
    seeded_observations: Mapping[str, Sequence[bool]],
    *,
    random_seed: int,
) -> tuple[list[dict[str, Any]], float | None, list[float] | None, bool, list[str]]:
    rows: list[dict[str, Any]] = []
    paired_lifts: list[float] = []
    logged_gaps: list[str] = []
    for index, game in enumerate(sorted(set(cold_observations) | set(seeded_observations))):
        cold = list(cold_observations.get(game, ()))
        seeded = list(seeded_observations.get(game, ()))
        if not cold:
            logged_gaps.append(f"missing_cold_start_observations:{game}")
            continue
        if not seeded:
            logged_gaps.append(f"missing_vocabulary_seeded_observations:{game}")
            rows.append(
                {
                    "held_out_game": game,
                    "seed_count": len(cold),
                    "cold_grounded_count": sum(1 for value in cold if value),
                    "seeded_grounded_count": None,
                    "cold_start_grounding_rate": _rate(cold),
                    "seeded_grounding_rate": None,
                    "lift": None,
                    "lift_ci95": None,
                }
            )
            continue
        n = min(len(cold), len(seeded))
        lifts = [float(seeded[i]) - float(cold[i]) for i in range(n)]
        paired_lifts.extend(lifts)
        lift = round(_rate(seeded[:n]) - _rate(cold[:n]), 6)
        rows.append(
            {
                "held_out_game": game,
                "seed_count": n,
                "cold_grounded_count": sum(1 for value in cold[:n] if value),
                "seeded_grounded_count": sum(1 for value in seeded[:n] if value),
                "cold_start_grounding_rate": _rate(cold[:n]),
                "seeded_grounding_rate": _rate(seeded[:n]),
                "lift": lift,
                "lift_ci95": _bootstrap_ci95(lifts, seed=random_seed + index),
            }
        )
    if not paired_lifts:
        return rows, None, None, False, logged_gaps
    overall_lift = round(sum(paired_lifts) / len(paired_lifts), 6)
    overall_ci = _bootstrap_ci95(paired_lifts, seed=random_seed)
    transfers = overall_lift > 0.0 and overall_ci[0] > 0.0
    return rows, overall_lift, overall_ci, transfers, logged_gaps


def _honest_verdict(transfers: bool, seeded_available: bool, overall_lift: float | None) -> str:
    if transfers:
        return "complete: config_rule_vocabulary_transfer_grounding_lift_ci_excludes_zero"
    if not seeded_available:
        return "complete: null_config_rule_vocabulary_transfer_seeded_arm_missing"
    if overall_lift is None:
        return "complete: null_config_rule_vocabulary_transfer_no_paired_heldout"
    return "complete: null_config_rule_vocabulary_transfer_lift_ci_includes_zero"


def build_artifact(
    *,
    root: Path,
    cold_repeat_bench: Mapping[str, Any],
    seeded_repeat_bench: Mapping[str, Any] | None,
    started_at: float,
    ended_at: float,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    rule_sources = extract_grounded_rule_sources(root)
    vocabulary = build_rule_vocabulary(rule_sources)
    seeded_available = bool(seeded_repeat_bench)
    cold_observations = _repeat_observations(cold_repeat_bench)
    seeded_observations = _repeat_observations(seeded_repeat_bench or {})
    curve, overall_lift, overall_ci, transfers, curve_gaps = _transfer_curve(
        cold_observations,
        seeded_observations,
        random_seed=random_seed,
    )
    logged_gaps = list(curve_gaps)
    if not seeded_available:
        logged_gaps.append("missing_vocabulary_seeded_repeat_bench")
    if not vocabulary:
        logged_gaps.append("missing_config_rule_vocabulary")
    checksum_payload = {
        "vocabulary": vocabulary,
        "curve": curve,
        "overall_lift": overall_lift,
        "overall_ci": overall_ci,
        "logged_gaps": logged_gaps,
        "random_seed": random_seed,
    }
    return {
        "experiment": "experiment_4425_config_rule_vocabulary_transfer",
        "schema": "carnot.exp4425.config_rule_vocabulary_transfer.v1",
        "honest_verdict": _honest_verdict(transfers, seeded_available, overall_lift),
        "config_rule_vocabulary_transfers": bool(transfers),
        "config_rule_vocabulary": vocabulary,
        "vocabulary_seeded_prompt": build_vocabulary_seeded_prompt(vocabulary),
        "transfer_learning_curve": curve,
        "overall_grounding_rate_lift": overall_lift,
        "overall_lift_ci95": overall_ci,
        "verifier_is_oracle": False,
        "random_seed": random_seed,
        "logged_gaps": logged_gaps,
        "preconditions_checked": {
            "grounded_rule_source_count": len(rule_sources),
            "grounded_rule_source_games": sorted({source["game"] for source in rule_sources}),
            "cold_repeat_bench_available": bool(cold_repeat_bench),
            "seeded_repeat_bench_available": seeded_available,
        },
        "model_specs": {
            "generator": "Qwen3.5-9B-MTP",
            "device": "iGPU",
            "no_think": True,
            "mtp": True,
            "seed_count": SEED_COUNT,
            "repeat_bench": "arc3_layerb_repeat_bench.py",
            "cold_repeat_bench": COLD_REPEAT_BENCH_RELATIVE_PATH,
            "seeded_repeat_bench": SEEDED_REPEAT_BENCH_RELATIVE_PATH,
        },
        "field_principles": FIELD_PRINCIPLES,
        "reproducibility_checksum": _sha256(checksum_payload),
        "duration_s": max(0.001, round(float(ended_at - started_at), 6)),
        "spec_refs": ["REQ-LEARN-4425", "SCENARIO-LEARN-4425"],
    }


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("complete:", "blocked:", "success:"))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("config_rule_vocabulary_transfers"), bool):
        errors.append("config_rule_vocabulary_transfers must be bare bool")
    if not isinstance(artifact.get("config_rule_vocabulary"), list):
        errors.append("config_rule_vocabulary must be list")
    if not isinstance(artifact.get("vocabulary_seeded_prompt"), str):
        errors.append("vocabulary_seeded_prompt must be str")
    if not isinstance(artifact.get("transfer_learning_curve"), list):
        errors.append("transfer_learning_curve must be list")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not isinstance(artifact.get("random_seed"), int):
        errors.append("random_seed must be bare int")
    if not isinstance(artifact.get("logged_gaps"), list):
        errors.append("logged_gaps must be list")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be dict")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    now: Any = time.perf_counter,
) -> Path:
    started_at = now()
    cold_repeat_bench = _load_json(root / COLD_REPEAT_BENCH_RELATIVE_PATH)
    seeded_path = root / SEEDED_REPEAT_BENCH_RELATIVE_PATH
    seeded_repeat_bench = _load_json(seeded_path) if seeded_path.exists() else None
    artifact = build_artifact(
        root=root,
        cold_repeat_bench=cold_repeat_bench,
        seeded_repeat_bench=seeded_repeat_bench,
        started_at=started_at,
        ended_at=now(),
    )
    return write_artifact(root, artifact)


def main() -> int:  # pragma: no cover - exercised through results wrapper
    path = run(REPO_ROOT)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
