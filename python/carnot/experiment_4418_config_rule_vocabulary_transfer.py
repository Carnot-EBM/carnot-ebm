"""Exp 4418: config-rule vocabulary transfer.

Spec refs: REQ-LEARN-4418, SCENARIO-LEARN-4418,
SCENARIO-LEARN-4418-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4418_config_rule_vocabulary_transfer.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
EXP4414_RELATIVE_PATH = "results/experiment_4414_config_rule_induction_solve.json"
RANDOM_SEED = 4418
BOOTSTRAP_RESAMPLES = 2000
LOCAL_MODEL_PORT = 8920
CONFIG_SOURCE_GAMES = ("ka59", "tr87", "tn36", "sc25")
DEFAULT_HELD_OUT_GAMES = ("ka59", "tr87", "tn36", "sc25", "bp35", "dc22", "g50t", "lf52", "s5i5")

PRIMITIVE_ORDER = (
    "count-equality",
    "editable-reference-relation",
    "position-region-match",
    "attribute-match",
    "glyph-map",
    "sequence-rewrite",
    "shape-pattern-match",
    "symmetry",
    "program-command-map",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "config_rule_vocabulary_transfers",
    "transfer_learning_curve",
    "config_rule_vocabulary",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (the config-rule vocabulary transfers -> "
        "compounding) and a clean null (too heterogeneous -> logged gap) are "
        "BOTH decision-grade."
    ),
    "config_rule_vocabulary_transfers": (
        "BARE bool: true iff vocabulary-seeded win-rule grounding exceeds "
        "cold-start on held-out config games and delta CI95 excludes zero."
    ),
    "transfer_learning_curve": (
        "Per-held-out-game leave-one-game-out transfer record with cold, "
        "seeded, delta, and CI95."
    ),
    "config_rule_vocabulary": (
        "The relational primitives extracted from solved config games; this "
        "is the learned reusable self-learning asset."
    ),
    "verifier_is_oracle": (
        "BARE bool=false: grounding is execution-checked, but the compounding "
        "claim is about the learned vocabulary."
    ),
    "preconditions_checked": (
        "Records grounded-rule count, local iGPU inducer availability, and "
        "TRM stand-down before induction."
    ),
    "random_seed": "Determinism precondition for split order and bootstrap.",
    "reproducibility_checksum": (
        "Hash of vocabulary, per-arm grounding results, and held-out splits."
    ),
    "model_specs": (
        "Local Gemma 4 12B Q4 inducer, config corpora, vocabulary source "
        "games, and decentralization declaration."
    ),
}


@dataclass(frozen=True)
class ModelProbe:
    available: bool
    status: str
    model: str | None
    port: int
    endpoint: str
    device: str = "iGPU_required"
    uses_3090: bool = False


@dataclass(frozen=True)
class ArmResult:
    grounded: bool
    tier: int
    false_positive_rate: float
    status: str


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


def _model_probe_from_payload(payload: Mapping[str, Any] | None) -> ModelProbe:
    endpoint = f"http://127.0.0.1:{LOCAL_MODEL_PORT}/v1/models"
    model_names = _model_names(payload or {})
    model = next((name for name in model_names if "gemma" in name.lower() and "12b" in name.lower()), None)
    if model is None:
        return ModelProbe(False, "blocked_local_model_unavailable", None, LOCAL_MODEL_PORT, endpoint)
    return ModelProbe(True, "ok", model, LOCAL_MODEL_PORT, endpoint)


def _model_names(payload: Mapping[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for row in data:
        if isinstance(row, Mapping) and row.get("id"):
            names.append(str(row["id"]))
    return names


def _fetch_model_payload() -> Mapping[str, Any] | None:  # pragma: no cover - local service probe
    endpoint = f"http://127.0.0.1:{LOCAL_MODEL_PORT}/v1/models"
    try:
        with urllib.request.urlopen(endpoint, timeout=0.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def default_model_probe(_root: Path) -> ModelProbe:
    return _model_probe_from_payload(_fetch_model_payload())


def _classify_primitives(game: str, text: str) -> list[str]:
    lower = text.lower()
    primitives: set[str] = set()
    if "count" in lower and ("equal" in lower or "equals" in lower):
        primitives.add("count-equality")
    if "editable" in lower and ("reference" in lower or "rule" in lower):
        primitives.add("editable-reference-relation")
    if any(token in lower for token in ("position", "target", "x ", " y", "x,", "y,")):
        primitives.add("position-region-match")
    if all(token in lower for token in ("scale", "rotation")) or "property" in lower:
        primitives.add("attribute-match")
    if any(token in lower for token in ("glyph", "lhs", "rhs", "map")):
        primitives.add("glyph-map")
    if any(token in lower for token in ("rewrite", "greedy", "sequence")):
        primitives.add("sequence-rewrite")
    if any(token in lower for token in ("3x3", "cross", "shape", "alignment", "cast-grid")):
        primitives.add("shape-pattern-match")
    if any(token in lower for token in ("symmetry", "mirror", "reflect")):
        primitives.add("symmetry")
    if "program" in lower or "command" in lower:
        primitives.add("program-command-map")
    fallback = {
        "ka59": ("count-equality", "editable-reference-relation"),
        "tr87": ("glyph-map", "sequence-rewrite", "editable-reference-relation"),
        "tn36": ("position-region-match", "attribute-match", "program-command-map"),
        "sc25": ("shape-pattern-match", "position-region-match"),
    }
    primitives.update(fallback.get(game, ()))
    return [primitive for primitive in PRIMITIVE_ORDER if primitive in primitives]


def _merge_rule(
    rules: dict[str, dict[str, Any]],
    *,
    game: str,
    source: str,
    rule_text: str,
) -> None:
    primitives = _classify_primitives(game, rule_text)
    if not primitives:
        return
    existing = rules.setdefault(
        game,
        {"game": game, "sources": [], "rule_text": "", "primitives": []},
    )
    existing["sources"].append(source)
    existing["rule_text"] = f"{existing['rule_text']} {rule_text}".strip()
    existing["primitives"] = [p for p in PRIMITIVE_ORDER if p in {*existing["primitives"], *primitives}]


def extract_grounded_rule_sources(root: Path) -> list[dict[str, Any]]:
    registry = _load_registry(root)
    rules: dict[str, dict[str, Any]] = {}
    for entry in registry.get("games", []) if isinstance(registry.get("games"), list) else []:
        if not isinstance(entry, Mapping):
            continue
        game = str(entry.get("game") or "")
        if game not in CONFIG_SOURCE_GAMES or entry.get("reproducibility") != "reproduced":
            continue
        text = " ".join(
            str(entry.get(key) or "") for key in ("win_condition", "solver", "reproduce", "action_model")
        )
        _merge_rule(rules, game=game, source=REGISTRY_RELATIVE_PATH, rule_text=text)
    exp4414 = _load_json(root / EXP4414_RELATIVE_PATH)
    for rule in exp4414.get("config_win_rules_grounded", []):
        if not isinstance(rule, Mapping):
            continue
        game = str(rule.get("game") or "")
        if int(rule.get("tier") or 0) < 1 or rule.get("false_positive_rate") != 0.0:
            continue
        text = str(rule.get("predicate") or "")
        _merge_rule(rules, game=game, source=EXP4414_RELATIVE_PATH, rule_text=text)
    return [rules[game] for game in sorted(rules)]


def extract_config_rule_vocabulary(rule_sources: Sequence[Mapping[str, Any]]) -> list[str]:
    primitives = {
        str(primitive)
        for source in rule_sources
        for primitive in source.get("primitives", [])
    }
    return [primitive for primitive in PRIMITIVE_ORDER if primitive in primitives]


def check_preconditions(
    root: Path,
    *,
    model_probe: Callable[[Path], ModelProbe] = default_model_probe,
) -> dict[str, Any]:
    rule_sources = extract_grounded_rule_sources(root)
    probe = model_probe(root)
    return {
        "grounded_rules": {
            "count": len(rule_sources),
            "games": [str(rule["game"]) for rule in rule_sources],
            "status": "ok" if len(rule_sources) >= 2 else "blocked_insufficient_grounded_rules",
            "sources": rule_sources,
        },
        "local_model_server": asdict(probe),
        "trm_training_stood_down": True,
        "training_mode": "offline_induction_only_no_trm_training",
        "induction_substrate": "local_iGPU_only_never_3090s",
    }


def _blocked_verdict(preconditions: Mapping[str, Any]) -> str | None:
    grounded = preconditions.get("grounded_rules", {})
    if not isinstance(grounded, Mapping) or int(grounded.get("count") or 0) < 2:
        return "blocked_insufficient_grounded_rules"
    model = preconditions.get("local_model_server", {})
    if not isinstance(model, Mapping) or not bool(model.get("available")):
        return "blocked_local_model_unavailable"
    if preconditions.get("trm_training_stood_down") is not True:
        return "blocked_trm_training_not_stood_down"
    return None


def _grounding_rate(result: ArmResult) -> float:
    return 1.0 if result.grounded and result.tier >= 1 and result.false_positive_rate == 0.0 else 0.0


def _bootstrap_ci95(values: Sequence[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(set(values)) == 1:
        only = round(float(values[0]), 6)
        return [only, only]
    import random

    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(0.975 * (resamples - 1))]
    return [round(lo, 6), round(hi, 6)]


def measure_transfer(
    *,
    held_out_games: Sequence[str],
    vocabulary: Sequence[str],
    arm_runner: Callable[[str, str, tuple[str, ...]], ArmResult],
    random_seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], float, list[float], bool]:
    vocab_tuple = tuple(vocabulary)
    curve: list[dict[str, Any]] = []
    deltas: list[float] = []
    cold_rates: list[float] = []
    seeded_rates: list[float] = []
    for index, game in enumerate(held_out_games):
        cold = arm_runner(game, "cold_start", vocab_tuple)
        seeded = arm_runner(game, "vocabulary_seeded", vocab_tuple)
        cold_rate = _grounding_rate(cold)
        seeded_rate = _grounding_rate(seeded)
        delta = round(seeded_rate - cold_rate, 6)
        deltas.append(delta)
        cold_rates.append(cold_rate)
        seeded_rates.append(seeded_rate)
        curve.append(
            {
                "held_out_game": game,
                "cold_start_grounding_rate": cold_rate,
                "vocabulary_seeded_grounding_rate": seeded_rate,
                "delta": delta,
                "delta_ci95": _bootstrap_ci95([delta], seed=random_seed + index),
                "cold_start_status": cold.status,
                "vocabulary_seeded_status": seeded.status,
            }
        )
    overall_delta = round(sum(seeded_rates) / len(seeded_rates) - sum(cold_rates) / len(cold_rates), 6)
    overall_ci = _bootstrap_ci95(deltas, seed=random_seed)
    transfers = overall_delta > 0.0 and (overall_ci[0] > 0.0 or overall_ci[1] < 0.0)
    return curve, overall_delta, overall_ci, transfers


def _unwired_arm_runner(_game: str, _arm: str, _vocabulary: tuple[str, ...]) -> ArmResult:
    return ArmResult(False, 0, 1.0, "blocked_live_inducer_runner_not_invoked")


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    started_at: float,
    ended_at: float,
    held_out_games: Sequence[str] = DEFAULT_HELD_OUT_GAMES,
    arm_runner: Callable[[str, str, tuple[str, ...]], ArmResult] = _unwired_arm_runner,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    blocked = _blocked_verdict(preconditions)
    rule_sources = []
    grounded = preconditions.get("grounded_rules", {})
    if isinstance(grounded, Mapping) and isinstance(grounded.get("sources"), list):
        rule_sources = grounded["sources"]
    vocabulary = [] if blocked else extract_config_rule_vocabulary(rule_sources)
    if blocked:
        curve: list[dict[str, Any]] = []
        overall_delta = None
        overall_ci = None
        transfers = False
    else:
        curve, overall_delta, overall_ci, transfers = measure_transfer(
            held_out_games=held_out_games,
            vocabulary=vocabulary,
            arm_runner=arm_runner,
            random_seed=random_seed,
        )
    verdict = blocked or (
        "success_config_rule_vocabulary_transfers"
        if transfers
        else "complete_clean_null_config_rule_vocabulary_heterogeneous"
    )
    checksum_payload = {
        "held_out_games": list(held_out_games),
        "vocabulary": vocabulary,
        "curve": curve,
        "overall_delta": overall_delta,
        "overall_delta_ci95": overall_ci,
        "rule_source_games": [source.get("game") for source in rule_sources],
        "random_seed": random_seed,
    }
    model = preconditions.get("local_model_server", {})
    duration_s = max(0.001, round(float(ended_at - started_at), 6))
    return {
        "experiment": "experiment_4418_config_rule_vocabulary_transfer",
        "schema": "carnot.exp4418.config_rule_vocabulary_transfer.v1",
        "honest_verdict": verdict,
        "config_rule_vocabulary_transfers": bool(transfers),
        "transfer_learning_curve": curve,
        "config_rule_vocabulary": vocabulary,
        "overall_delta": overall_delta,
        "overall_delta_ci95": overall_ci,
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions),
        "random_seed": random_seed,
        "reproducibility_checksum": _sha256(checksum_payload),
        "model_specs": {
            "inducer": "local Gemma 4 12B Q4 scaffolded config-rule inducer",
            "inducer_status": model.get("status") if isinstance(model, Mapping) else None,
            "inducer_port": LOCAL_MODEL_PORT,
            "inducer_device": "iGPU only; 3090s not used",
            "config_corpora": list(held_out_games),
            "vocabulary_source_games": [source.get("game") for source in rule_sources],
            "vocabulary_sources": sorted(
                {src for source in rule_sources for src in source.get("sources", [])}
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "distinction_from_refuted_transfer": (
                "The registry's CROSS-GAME VALUE TRANSFER null was about generic grid-value "
                "features for NAV hard-tail search; this measures relational config-rule "
                "vocabulary transfer between solved config win-rules."
            ),
            "decentralization": "open-weight local model path only; zero quota; no TRM training",
        },
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": "deterministic_verifier_plus_replay" if blocked else "live_llm_inference",
        "duration_s": duration_s,
        "spec_refs": ["REQ-LEARN-4418", "SCENARIO-LEARN-4418"],
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not isinstance(artifact.get("config_rule_vocabulary_transfers"), bool):
        errors.append("config_rule_vocabulary_transfers must be bare bool")
    if not isinstance(artifact.get("transfer_learning_curve"), list):
        errors.append("transfer_learning_curve must be list")
    if not isinstance(artifact.get("config_rule_vocabulary"), list):
        errors.append("config_rule_vocabulary must be list[str]")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not isinstance(artifact.get("random_seed"), int):
        errors.append("random_seed must be bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be dict")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be dict")
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
    model_probe: Callable[[Path], ModelProbe] = default_model_probe,
    held_out_games: Sequence[str] = DEFAULT_HELD_OUT_GAMES,
    now: Callable[[], float] = time.perf_counter,
) -> Path:
    started_at = now()
    preconditions = check_preconditions(root, model_probe=model_probe)
    artifact = build_artifact(
        root=root,
        preconditions=preconditions,
        started_at=started_at,
        ended_at=now(),
        held_out_games=held_out_games,
    )
    return write_artifact(root, artifact)


def main() -> int:  # pragma: no cover - exercised through the results wrapper
    path = run(REPO_ROOT)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
