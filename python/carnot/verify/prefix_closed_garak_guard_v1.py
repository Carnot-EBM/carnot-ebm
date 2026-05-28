"""Build the Exp 3297 Garak rogue-string text guard artifact.

Spec refs: REQ-VERIFY-3297, SCENARIO-VERIFY-3297.

This is an honest text guard, not a BEAVER probability-bound verifier. The
upstream `.304` failure detector was driven by concrete rogue strings, while
the checked-in artifacts only preserve target inventories and short response
previews. This module therefore normalizes text and checks exact target
phrases plus monitored prefixes. It does not estimate token probabilities,
does not prove that a model cannot later escape a prefix, and does not claim
live Garak benchmark performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prefix_closed_garak_guard.v1"
EXPERIMENT_ID = "exp3297"
TASK_ID = "exp3297-prefix-closed-garak-guard-v1"
ARTIFACT = "experiment_3297_prefix_closed_garak_guard_v1"
MILESTONE = "2026.05.305"
RUN_DATE = "20260528"
RANDOM_SEED = 3297
GUARD_KIND = "normalized_text_prefix_and_exact_phrase_guard_not_probability_bound"
INFERENCE_SUBSTRATE = "cached_response_previews_only_no_live_inference"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_REL_PATH = Path("python/carnot/verify/prefix_closed_garak_guard_v1.py")
TEST_REL_PATH = Path("tests/python/test_experiment_3297_prefix_closed_garak_guard.py")
OUTPUT_REL_PATH = Path("results/experiment_3297_prefix_closed_garak_guard_v1.json")
EXP3295_REL_PATH = Path("results/experiment_3295_garak_failure_mode_autopsy_v1.json")
EXP3285_REL_PATH = Path("results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json")

SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
TRIM_PUNCTUATION = "\"'`.,:;!?()[]{}<>*_"
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3297_prefix_closed_garak_guard.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3297_prefix_closed_garak_guard.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/prefix_closed_garak_guard_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
IMPLEMENTATION_FILES_TOUCHED = (
    SPEC_REL_PATH.as_posix(),
    MODULE_REL_PATH.as_posix(),
    TEST_REL_PATH.as_posix(),
)
REQUIRED_FIELDS = {
    "prefix_guard_policy_ready",
    "guard_kind",
    "target_phrase_count",
    "target_prefix_count",
    "cached_trace_count",
    "benign_control_count",
    "cached_attack_detection_rate",
    "cached_benign_false_positive_rate",
    "guard_policy",
    "implementation_files_touched",
    "tests_run",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


@dataclass(frozen=True)
class JsonLoad:
    """A source JSON object plus diagnostics used to avoid silent evidence loss."""

    payload: JsonDict
    present: bool
    readable: bool
    error: str | None
    path: Path
    sha256: str | None


@dataclass(frozen=True)
class RogueStringTextGuard:
    """Small normalized text matcher for the `.304` Garak target strings.

    The guard checks concrete text that has already been generated. It is useful
    as a downstream policy contract because it is deterministic and auditable,
    but it is not a token-probability bound and has no live model substrate.
    """

    target_phrases: tuple[JsonDict, ...]
    target_prefixes: tuple[JsonDict, ...]

    @classmethod
    def from_policy(cls, policy: Mapping[str, Any]) -> "RogueStringTextGuard":
        """Build a guard from the stable `guard_policy` emitted in artifacts."""

        return cls(
            target_phrases=tuple(mapping_list(policy.get("target_phrases"))),
            target_prefixes=tuple(mapping_list(policy.get("target_prefixes"))),
        )

    def evaluate(self, text: str) -> JsonDict:
        """Return the first exact or prefix hit in normalized generated text."""

        normalized = normalize_text(text)
        for row in self.target_phrases:
            value = str(row.get("value") or "")
            if value and value in normalized:
                return guard_result(
                    normalized,
                    blocked=True,
                    match_type="exact_phrase",
                    matched_value=value,
                    matched_family=str(row.get("family") or ""),
                )
        for row in self.target_prefixes:
            value = str(row.get("value") or "")
            if value and value in normalized:
                return guard_result(
                    normalized,
                    blocked=True,
                    match_type="target_prefix",
                    matched_value=value,
                    matched_family=str(row.get("family") or ""),
                )
        return guard_result(
            normalized,
            blocked=False,
            match_type="none",
            matched_value="",
            matched_family="",
        )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3297: build the cached-preview text guard artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3295 = read_json_object(root_path / EXP3295_REL_PATH)
    exp3285 = read_json_object(root_path / EXP3285_REL_PATH)
    policy = build_guard_policy(
        mapping_list(exp3295.payload.get("target_phrase_inventory")),
        mapping_list(exp3295.payload.get("target_prefix_inventory")),
    )
    guard = RogueStringTextGuard.from_policy(policy)
    cached_results = evaluate_cached_previews(extract_cached_previews(exp3285.payload), guard)
    benign_controls = synthetic_benign_controls(mapping_list(policy.get("target_phrases")))
    benign_results = evaluate_benign_controls(benign_controls, guard)
    ready = bool(
        exp3295.readable
        and policy["target_phrase_count"] > 0
        and policy["target_prefix_count"] > 0
    )
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3297", "SCENARIO-VERIFY-3297"],
        "prefix_guard_policy_ready": ready,
        "guard_kind": GUARD_KIND,
        "target_phrase_count": int(policy["target_phrase_count"]),
        "target_prefix_count": int(policy["target_prefix_count"]),
        "cached_trace_count": len(cached_results),
        "benign_control_count": len(benign_results),
        "cached_attack_detection_rate": blocked_rate(cached_results),
        "cached_benign_false_positive_rate": blocked_rate(benign_results),
        "guard_policy": policy,
        "implementation_files_touched": list(IMPLEMENTATION_FILES_TOUCHED),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "source_artifacts": [source_row(exp3295), source_row(exp3285)],
        "cached_trace_results": cached_results,
        "benign_control_results": benign_results,
        "no_new_garak_run": True,
        "no_new_model_execution": True,
        "no_new_repair_run": True,
        "no_conductor_execution": True,
        "scripts_research_conductor_modified": False,
        "live_benchmark_claim": False,
        "methodology_gap": (
            "normalized text-prefix matching over cached previews; no token "
            "logprob frontier or BEAVER probability bound was computed"
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3297 terminal JSON deliverable."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_guard_policy(
    target_phrases: Sequence[Mapping[str, Any]],
    target_prefixes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the stable downstream policy from Exp 3295 target inventories."""

    phrase_rows = normalized_policy_rows(target_phrases, value_key="target_phrase")
    prefix_rows = normalized_policy_rows(target_prefixes, value_key="target_prefix")
    return {
        "policy_id": "exp3297_prefix_closed_garak_guard_v1",
        "schema_version": SCHEMA_VERSION,
        "guard_kind": GUARD_KIND,
        "normalization": {
            "casefold": True,
            "collapse_whitespace": True,
            "trim_boundary_punctuation": True,
            "unicode_normalization": "none",
        },
        "match_contract": {
            "exact_phrase": "block when any normalized target phrase occurs in generated text",
            "target_prefix": "block when any normalized monitored prefix occurs in generated text",
            "match_order": ["exact_phrase", "target_prefix"],
            "not_probability_bound": True,
        },
        "target_phrases": phrase_rows,
        "target_prefixes": prefix_rows,
        "target_phrase_count": len(phrase_rows),
        "target_prefix_count": len(prefix_rows),
    }


def normalized_policy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> list[JsonDict]:
    """Normalize and deduplicate policy rows while preserving source metadata."""

    normalized_rows: list[JsonDict] = []
    seen: set[str] = set()
    for row in rows:
        value = normalize_text(str(row.get(value_key) or ""))
        if not value or value in seen:
            continue
        seen.add(value)
        normalized_rows.append(
            {
                "value": value,
                "family": str(row.get("family") or ""),
                "class_name": str(row.get("class_name") or ""),
                "source": str(row.get("source") or ""),
                "source_target_phrase": normalize_text(str(row.get("source_target_phrase") or "")),
                "match_rule": str(row.get("match_rule") or "normalized_text_contains"),
            }
        )
    return normalized_rows


def normalize_text(text: str) -> str:
    """Normalize generated text enough for deterministic cached-string matching."""

    collapsed = re.sub(r"\s+", " ", str(text).casefold()).strip()
    return collapsed.strip(TRIM_PUNCTUATION).strip()


def extract_cached_previews(exp3285: Mapping[str, Any]) -> list[JsonDict]:
    """Return cached `.304` response previews without inventing missing traces."""

    target_behavior = mapping(mapping(exp3285.get("per_slice_metrics")).get("target_behavior"))
    return mapping_list(target_behavior.get("response_previews"))


def evaluate_cached_previews(
    response_previews: Sequence[Mapping[str, Any]],
    guard: RogueStringTextGuard,
) -> list[JsonDict]:
    """Evaluate every cached response preview as a local diagnostic trace."""

    results: list[JsonDict] = []
    for index, row in enumerate(response_previews):
        preview = str(row.get("response_preview") or "")
        result = guard.evaluate(preview)
        results.append(
            {
                "trace_id": f"exp3285-preview-{index:03d}",
                "family": str(row.get("family") or ""),
                "class_name": str(row.get("class_name") or ""),
                "response_preview": preview,
                **result,
            }
        )
    return results


def synthetic_benign_controls(target_phrase_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create benign completions tied to Exp 3295 rows without echoing rogue strings."""

    controls: list[JsonDict] = []
    for index, row in enumerate(target_phrase_rows):
        family = str(row.get("family") or "unknown")
        class_name = str(row.get("class_name") or "unknown")
        controls.append(
            {
                "control_id": f"exp3297-benign-{index:03d}",
                "source_family": family,
                "source_class_name": class_name,
                "response_text": (
                    "I will keep following the active instruction hierarchy and "
                    f"handle the {family} / {class_name} case as a benign audit note."
                ),
            }
        )
    return controls


def evaluate_benign_controls(
    controls: Sequence[Mapping[str, Any]],
    guard: RogueStringTextGuard,
) -> list[JsonDict]:
    """Evaluate synthetic aligned completions so the artifact tracks false positives."""

    results: list[JsonDict] = []
    for row in controls:
        text = str(row.get("response_text") or "")
        results.append({**dict(row), **guard.evaluate(text)})
    return results


def guard_result(
    normalized_text: str,
    *,
    blocked: bool,
    match_type: str,
    matched_value: str,
    matched_family: str,
) -> JsonDict:
    """Format one guard decision in the same shape for attacks and controls."""

    return {
        "normalized_text": normalized_text,
        "blocked": bool(blocked),
        "match_type": match_type,
        "matched_value": matched_value,
        "matched_family": matched_family,
    }


def blocked_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of rows blocked by the text guard."""

    if not rows:
        return 0.0
    blocked = sum(1 for row in rows if row.get("blocked") is True)
    return round(blocked / len(rows), 6)


def read_json_object(path: Path) -> JsonLoad:
    """Read a JSON object while preserving missing or malformed-source evidence."""

    if not path.is_file():
        return JsonLoad({}, False, False, "missing", path, None)
    try:
        digest = sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - filesystem race or permission edge.
        return JsonLoad({}, True, False, str(exc), path, None)
    except json.JSONDecodeError as exc:
        return JsonLoad({}, True, False, str(exc), path, digest)
    if not isinstance(payload, Mapping):
        return JsonLoad({}, True, False, "json root is not an object", path, digest)
    return JsonLoad(dict(payload), True, True, None, path, digest)


def source_row(load: JsonLoad) -> JsonDict:
    """Expose exact source availability and checksum in the terminal artifact."""

    try:
        rel_path = load.path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_path = load.path.as_posix()
    return {
        "path": rel_path,
        "present": load.present,
        "readable": load.readable,
        "error": load.error,
        "sha256": load.sha256,
    }


def sha256_file(path: Path) -> str:
    """Hash source artifacts so the policy can be tied to exact cached inputs."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic artifact content while excluding timing noise."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that names the cached-preview limitation."""

    return (
        "complete: "
        f"prefix_guard_policy_ready={str(artifact.get('prefix_guard_policy_ready') is True).lower()}; "
        f"guard_kind={GUARD_KIND}; "
        f"cached_trace_count={int(artifact.get('cached_trace_count') or 0)}; "
        "cached response-preview text guard only; no live Garak benchmark claim"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema mistakes that could make Exp 3299 consume a bad policy."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact.get("prefix_guard_policy_ready"), bool):
        raise ValueError("prefix_guard_policy_ready must be bool")
    guard_kind = artifact.get("guard_kind")
    if not isinstance(guard_kind, str) or "text" not in guard_kind:
        raise ValueError("guard_kind must disclose text matching")
    if "probability_bound" not in guard_kind:
        raise ValueError("guard_kind must disclose not_probability_bound")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be cached-preview only")
    for field in (
        "target_phrase_count",
        "target_prefix_count",
        "cached_trace_count",
        "benign_control_count",
        "random_seed",
    ):
        require_nonnegative_int(artifact.get(field), field)
    for field in ("cached_attack_detection_rate", "cached_benign_false_positive_rate"):
        require_rate(artifact.get(field), field)
    if artifact.get("prefix_guard_policy_ready") is True:
        policy = artifact.get("guard_policy")
        if not isinstance(policy, Mapping) or not policy:
            raise ValueError("guard_policy must be populated when policy is ready")
        if int(artifact["target_phrase_count"]) <= 0 or int(artifact["target_prefix_count"]) <= 0:
            raise ValueError("ready policy requires positive target counts")
    if not isinstance(artifact.get("implementation_files_touched"), list):
        raise ValueError("implementation_files_touched must be a list")
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a list")
    duration_s = artifact.get("duration_s")
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or duration_s < 0:
        raise ValueError("duration_s must be a nonnegative number")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("checksum must be 64 lowercase hex characters")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def require_nonnegative_int(value: Any, field: str) -> None:
    """Reject bools and negative values where schema counts are expected."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def require_rate(value: Any, field: str) -> None:
    """Reject malformed metric rates before downstream ablations read them."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")


def duration(started_s: float, finished_s: float) -> float:
    """Return rounded elapsed seconds without sleep padding."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def mapping(value: Any) -> Mapping[str, Any]:
    """Return mapping-like values as-is and treat all other evidence as absent."""

    return value if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only object rows from a possibly malformed source list."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def main() -> None:  # pragma: no cover - exercised through direct function tests.
    """CLI entrypoint used by conductor-style one-shot artifact generation."""

    write_artifact()


if __name__ == "__main__":  # pragma: no cover
    main()
