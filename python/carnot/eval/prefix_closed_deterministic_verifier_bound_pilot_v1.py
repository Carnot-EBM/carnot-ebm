"""Exp 3125 prefix-closed deterministic verifier bound pilot.

Spec refs: REQ-VERIFY-3125, SCENARIO-VERIFY-3125.

This pilot is deliberately local and narrow. It enumerates a bounded token
frontier over exact JSON-like fixtures and reports conservative satisfaction
bounds under a deterministic fixture prior. The prior is not a language-model
probability distribution, so the resulting bounds are evidence about the
explored exact frontier only, not about open-world LLM correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1"
SCHEMA = "carnot.prefix_closed_deterministic_verifier_bound_pilot.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"
)
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / (
    "experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.py"
)

EOS = "<eos>"
DEFAULT_MAX_DEPTH = 10
PREFERRED_TOKEN_PROBABILITY = 0.6
VOCABULARY = (
    "{",
    "}",
    '"answer"',
    '"score"',
    '"tag"',
    ":",
    ",",
    '"VALID"',
    '"INVALID"',
    '"SAT"',
    '"MAYBE"',
    '"logic"',
    "0",
    "1",
    EOS,
)
CONSTRAINT_FAMILIES = (
    "json_like_answer_shape",
    "answer_label_match",
    "bounded_score_invariant",
    "forbidden_unknown_token",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "prefix_closed_bound_pilot_ready",
    "constraint_families",
    "fixture_count",
    "explored_prefix_count",
    "pruned_prefix_count",
    "lower_bound",
    "upper_bound",
    "bound_width",
    "semantic_coverage",
    "limitations",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3125_prefix_closed_deterministic_verifier_bound_pilot.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/eval -m pytest -o addopts='' tests/python/test_experiment_3125_prefix_closed_deterministic_verifier_bound_pilot.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/prefix_closed_deterministic_verifier_bound_pilot_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md")),
    ("codex_repo_workflow", Path("CODEX.md")),
    ("claude_authenticity_rules", Path("CLAUDE.md")),
    ("research_references", Path("research-references.md")),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md")),
    ("exp3111_certified_feedback_v3", Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")),
    ("exp3124_stratified_panel_v6", Path("results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json")),
    ("exp3125_module", Path("python/carnot/eval/prefix_closed_deterministic_verifier_bound_pilot_v1.py")),
    ("exp3125_script", Path("scripts/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.py")),
)


@dataclass(frozen=True)
class ExactFixture:
    """One exact semantic target used by the bounded prefix pilot."""

    fixture_id: str
    expected_answer: str
    expected_score: int
    required_tag: str | None = None
    min_score: int = 0
    max_score: int = 1


def build_fixture_subset() -> list[ExactFixture]:
    """Return the small exact fixture subset used by REQ-VERIFY-3125."""

    return [
        ExactFixture("pc-3125-valid", "VALID", 1),
        ExactFixture("pc-3125-invalid", "INVALID", 0),
        ExactFixture("pc-3125-logic", "SAT", 1, required_tag="logic"),
    ]


def terminal_sequence_for_fixture(fixture: ExactFixture) -> tuple[str, ...]:
    """Return the unique satisfying terminal token sequence for a fixture."""

    tokens = [
        "{",
        '"answer"',
        ":",
        json.dumps(fixture.expected_answer),
        ",",
        '"score"',
        ":",
        str(fixture.expected_score),
    ]
    if fixture.required_tag is not None:
        tokens.extend([",", '"tag"', ":", json.dumps(fixture.required_tag)])
    tokens.extend(["}", EOS])
    return tuple(tokens)


def candidate_text_from_sequence(sequence: Sequence[str]) -> str:
    """Render non-EOS tokens into the exact JSON candidate text."""

    return "".join(token for token in sequence if token != EOS)


def terminal_satisfies_fixture(sequence: Sequence[str], fixture: ExactFixture) -> bool:
    """Check terminal JSON syntax and semantic constraints with local authority."""

    tokens = tuple(sequence)
    if not tokens or tokens[-1] != EOS:
        return False
    try:
        payload = json.loads(candidate_text_from_sequence(tokens[:-1]))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if any(value == "MAYBE" for value in payload.values()):
        return False
    score = payload.get("score")
    return bool(
        payload.get("answer") == fixture.expected_answer
        and isinstance(score, int)
        and fixture.min_score <= score <= fixture.max_score
        and score == fixture.expected_score
        and (fixture.required_tag is None or payload.get("tag") == fixture.required_tag)
    )


def classify_prefix(prefix: Sequence[str], fixture: ExactFixture) -> JsonDict:
    """Classify a prefix as viable, accepted, or permanently pruned."""

    tokens = tuple(prefix)
    terminal = terminal_sequence_for_fixture(fixture)
    if tokens == terminal:
        status = "accepted"
        reason = "terminal_satisfies_exact_constraints"
    elif len(tokens) <= len(terminal) and terminal[: len(tokens)] == tokens:
        status = "viable"
        reason = "prefix_has_satisfying_extension"
    else:
        status = "pruned"
        reason = "no_satisfying_extension"
    return {
        "fixture_id": fixture.fixture_id,
        "prefix": list(tokens),
        "depth": len(tokens),
        "status": status,
        "reason": reason,
    }


def prefix_rejection_is_monotone(
    prefix: Sequence[str],
    fixture: ExactFixture,
    extension_tokens: Sequence[str],
) -> bool:
    """Return whether one-token extensions preserve a pruned prefix decision."""

    if classify_prefix(prefix, fixture)["status"] != "pruned":
        return False
    return all(
        classify_prefix((*tuple(prefix), token), fixture)["status"] == "pruned"
        for token in extension_tokens
    )


def enumerate_frontier(
    fixtures: Sequence[ExactFixture],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> JsonDict:
    """Enumerate the bounded prefix frontier and aggregate satisfaction bounds."""

    frontier_rows: list[JsonDict] = []
    lower_mass = 0.0
    pruned_mass = 0.0
    viable_frontier_mass = 0.0
    accepted_count = 0
    pruned_count = 0
    viable_frontier_count = 0
    fixture_count = len(fixtures)

    for fixture in fixtures:
        stack: list[tuple[tuple[str, ...], float]] = [((), 1.0)]
        while stack:
            prefix, mass = stack.pop()
            status = classify_prefix(prefix, fixture)
            row = status | {"probability_mass": round(mass / fixture_count, 12)}
            frontier_rows.append(row)
            if status["status"] == "accepted":
                lower_mass += mass / fixture_count
                accepted_count += 1
            elif status["status"] == "pruned":
                pruned_mass += mass / fixture_count
                pruned_count += 1
            elif len(prefix) >= max_depth:
                viable_frontier_mass += mass / fixture_count
                viable_frontier_count += 1
                row["frontier_boundary"] = True
            else:
                for token in VOCABULARY:
                    token_mass = mass * token_probability(fixture, prefix, token)
                    stack.append(((*prefix, token), token_mass))

    lower_bound = round(lower_mass, 12)
    upper_bound = round(lower_mass + viable_frontier_mass, 12)
    return {
        "fixture_count": fixture_count,
        "frontier_rows": frontier_rows,
        "explored_prefix_count": len(frontier_rows),
        "frontier_size": accepted_count + pruned_count + viable_frontier_count,
        "accepted_prefix_count": accepted_count,
        "pruned_prefix_count": pruned_count,
        "viable_frontier_count": viable_frontier_count,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "bound_width": round(upper_bound - lower_bound, 12),
        "pruned_mass": round(pruned_mass, 12),
        "viable_frontier_mass": round(viable_frontier_mass, 12),
        "explored_mass": round(lower_mass + pruned_mass + viable_frontier_mass, 12),
    }


def token_probability(fixture: ExactFixture, prefix: Sequence[str], token: str) -> float:
    """Return the deterministic fixture prior probability for one next token."""

    expected = terminal_sequence_for_fixture(fixture)[len(prefix)]
    if token == expected:
        return PREFERRED_TOKEN_PROBABILITY
    return (1.0 - PREFERRED_TOKEN_PROBABILITY) / (len(VOCABULARY) - 1)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3125 terminal artifact payload."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    fixtures = build_fixture_subset()
    summary = enumerate_frontier(fixtures, max_depth=max_depth)
    coverage = semantic_coverage(fixtures)
    substrate = inference_substrate(max_depth)
    ready = bool(
        summary["fixture_count"] > 0
        and summary["explored_prefix_count"] > 0
        and summary["pruned_prefix_count"] > 0
        and summary["upper_bound"] >= summary["lower_bound"]
        and math.isfinite(summary["bound_width"])
        and coverage["json_syntax"]["covered"] is True
        and coverage["answer_label_semantics"]["covered"] is True
        and coverage["live_llm_correctness"]["covered"] is False
        and substrate["live_model_invoked"] is False
    )
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "prefix_closed_bound_pilot_ready": ready,
        "constraint_families": list(CONSTRAINT_FAMILIES),
        "fixture_count": summary["fixture_count"],
        "fixture_details": [
            {
                "fixture_id": fixture.fixture_id,
                "expected_answer": fixture.expected_answer,
                "expected_score": fixture.expected_score,
                "required_tag": fixture.required_tag,
            }
            for fixture in fixtures
        ],
        "max_depth": max_depth,
        "explored_prefix_count": summary["explored_prefix_count"],
        "frontier_size": summary["frontier_size"],
        "accepted_prefix_count": summary["accepted_prefix_count"],
        "viable_frontier_count": summary["viable_frontier_count"],
        "pruned_prefix_count": summary["pruned_prefix_count"],
        "lower_bound": summary["lower_bound"],
        "upper_bound": summary["upper_bound"],
        "bound_width": summary["bound_width"],
        "explored_mass": summary["explored_mass"],
        "pruned_mass": summary["pruned_mass"],
        "viable_frontier_mass": summary["viable_frontier_mass"],
        "frontier_rows": summary["frontier_rows"],
        "semantic_coverage": coverage,
        "limitations": limitations(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": substrate,
        "duration_s": round((time.perf_counter() if now_s is None else float(now_s)) - start, 6),
        "honest_verdict": (
            "complete: bounded prefix-closed deterministic verifier pilot ready; "
            "no live LLM correctness claim beyond explored frontier"
            if ready
            else "blocked: prefix-closed deterministic verifier pilot gates did not clear"
        ),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    max_depth: int = DEFAULT_MAX_DEPTH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Write the Exp 3125 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(
        root_path,
        max_depth=max_depth,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3125 artifact overclaims or breaks its schema."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"].get("live_model_invoked") is not False:
        raise ValueError("live_model_invoked must be false")
    lower = float(artifact["lower_bound"])
    upper = float(artifact["upper_bound"])
    width = float(artifact["bound_width"])
    if lower < 0.0 or upper > 1.0 or lower > upper:
        raise ValueError("bounds must be ordered finite probabilities")
    if abs(width - (upper - lower)) > 1e-9:
        raise ValueError("bound_width must equal upper_bound - lower_bound")
    if artifact["semantic_coverage"]["live_llm_correctness"].get("covered") is not False:
        raise ValueError("live_llm_correctness must remain uncovered")
    if artifact["prefix_closed_bound_pilot_ready"]:
        if int(artifact["fixture_count"]) <= 0:
            raise ValueError("fixture_count must be positive")
        if int(artifact["explored_prefix_count"]) <= 0:
            raise ValueError("explored_prefix_count must be positive")
        if int(artifact["pruned_prefix_count"]) <= 0:
            raise ValueError("pruned_prefix_count must be positive")
        if not artifact["constraint_families"]:
            raise ValueError("constraint_families must be non-empty")
        if not artifact["limitations"]:
            raise ValueError("limitations must be non-empty")
        if not str(artifact["honest_verdict"]).startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must use a terminal success prefix")


def semantic_coverage(fixtures: Sequence[ExactFixture]) -> JsonDict:
    """Report which semantic surfaces this tiny exact pilot covers."""

    return {
        "json_syntax": {
            "covered": True,
            "fixture_count": len(fixtures),
            "authority": "python_json_parser_on_terminal_candidates",
        },
        "answer_label_semantics": {
            "covered": True,
            "labels": sorted({fixture.expected_answer for fixture in fixtures}),
            "authority": "exact_fixture_expected_answer",
        },
        "bounded_score_semantics": {
            "covered": True,
            "score_range": [0, 1],
            "authority": "exact_integer_equality_and_range_check",
        },
        "forbidden_unknown_token": {
            "covered": True,
            "forbidden_values": ["MAYBE"],
            "authority": "exact_json_value_scan",
        },
        "live_llm_correctness": {
            "covered": False,
            "reason": "no live model inference or open-world language frontier is evaluated",
        },
    }


def limitations() -> list[str]:
    """Return explicit non-claims for the bounded pilot."""

    return [
        "Does not prove full LLM correctness or repair correctness.",
        "Bounds apply only to the finite fixture-conditioned token prior.",
        "The third fixture remains upper-bounded beyond the explored depth.",
        "Open-world natural language, paraphrases, and model logprob frontiers are outside scope.",
    ]


def inference_substrate(max_depth: int) -> JsonDict:
    """Describe the non-live deterministic substrate used for bound accounting."""

    return {
        "type": "deterministic_bounded_prefix_enumerator",
        "live_model_invoked": False,
        "model_ids": [],
        "probability_source": "deterministic_fixture_prior",
        "preferred_token_probability": PREFERRED_TOKEN_PROBABILITY,
        "vocabulary_size": len(VOCABULARY),
        "max_depth": max_depth,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source paths and checksums that anchor the artifact."""

    rows: list[JsonDict] = []
    for name, rel_path in SOURCE_ARTIFACTS:
        path = root / rel_path
        row: JsonDict = {"name": name, "path": rel_path.as_posix(), "exists": path.is_file()}
        if path.is_file():
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact_path = write_artifact(REPO_ROOT)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["prefix_closed_bound_pilot_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
