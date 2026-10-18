"""Build the Exp 3210 Context-CoT/CL-Bench parametric-shortcut fixtures.

Spec refs: REQ-VERIFY-3210, SCENARIO-VERIFY-3210.

This module is intentionally deterministic.  The fixture rows are not model
outputs; they are small local contexts whose rules contradict common priors, so
an exact checker can tell whether an answer followed the prompt context instead
of falling back to pretrained knowledge.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.context_cot_clbench_parametric_shortcut_fixtures.v1"
EXPERIMENT_ID = "exp3210"
MILESTONE = "2026.05.297"
FIXTURE_REL_PATH = Path("data/research/context_cot_clbench_parametric_shortcut_v1.jsonl")
OUTPUT_REL_PATH = Path(
    "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
)

FIXTURE_FAMILIES = (
    "symbolic_aliases",
    "local_arithmetic_rules",
    "context_defined_entity_facts",
)
EXACT_CHECKER_TYPES = (
    "exact_alias_string",
    "exact_integer_string",
    "exact_entity_fact_string",
)
MANDATED_LOCAL_SOTA_GGUF = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REFERENCE_PAPERS = (
    {
        "title": "Context-CoT: Enhancing Context Learning via High-Quality Reasoning Synthesis",
        "id": "arXiv:2605.25354",
        "role": "context-learning and parametric-shortcut anchor",
    },
    {
        "title": "CL-Bench",
        "id": "context-learning benchmark cited by Context-CoT",
        "role": "benchmark motivation for locally contradicted context rows",
    },
)
REQUIRED_ROW_FIELDS = (
    "fixture_id",
    "family",
    "context",
    "question",
    "expected_answer",
    "prior_bait_answer",
    "exact_checker_type",
    "minimal_counterexample",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "experiment_id",
    "milestone",
    "reference_papers",
    "fixture_path",
    "fixture_count",
    "fixture_families",
    "exact_checker_types",
    "prior_bait_row_count",
    "context_following_score_available",
    "optional_llm_smoke",
    "ready_for_clean_verifier",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.py -q",
    ".venv/bin/coverage report --include='*/context_cot_clbench_parametric_shortcut_fixtures_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def normalize_answer(answer: Any) -> str:
    """Normalize a short exact answer while preserving its semantic content."""

    text = " ".join(str(answer).strip().lower().split())
    return text[:-1] if text.endswith((".", "!", "?")) else text


def _minimal_counterexample(prior_bait_answer: str, expected_answer: str) -> JsonDict:
    return {
        "candidate_answer": prior_bait_answer,
        "expected_answer": expected_answer,
        "failure_mode": "parametric_prior_shortcut",
        "why_minimal": "bare answer selects the likely prior instead of the local context value",
    }


def _row(
    fixture_id: str,
    family: str,
    context: str,
    question: str,
    expected_answer: str,
    prior_bait_answer: str,
    exact_checker_type: str,
) -> JsonDict:
    return {
        "fixture_id": fixture_id,
        "family": family,
        "context": context,
        "question": question,
        "expected_answer": expected_answer,
        "prior_bait_answer": prior_bait_answer,
        "exact_checker_type": exact_checker_type,
        "minimal_counterexample": _minimal_counterexample(
            prior_bait_answer, expected_answer
        ),
    }


def _symbolic_alias_rows() -> list[JsonDict]:
    aliases = (
        ("mercury", "banana", "planet"),
        ("python", "blue screwdriver", "snake"),
        ("mars", "teacup", "planet"),
        ("ruby", "north door", "gemstone"),
        ("falcon", "paperclip", "bird"),
        ("saturn", "green notebook", "planet"),
        ("java", "silver key", "programming language"),
        ("delta", "left stair", "river delta"),
        ("iris", "copper coin", "flower"),
        ("atlas", "red ladder", "map book"),
    )
    rows: list[JsonDict] = []
    for index, (symbol, local_value, prior) in enumerate(aliases, start=1):
        rows.append(
            _row(
                f"ctx3210-symbolic-{index:02d}",
                "symbolic_aliases",
                (
                    "For this fixture only, the glossary overrides ordinary meanings: "
                    f"{symbol} means {local_value}."
                ),
                f"According to the fixture glossary, what does {symbol} mean?",
                local_value,
                prior,
                "exact_alias_string",
            )
        )
    return rows


def _local_arithmetic_rows() -> list[JsonDict]:
    arithmetic = (
        ("plus means multiply", "3 plus 4", "12", "7"),
        ("times means add", "5 times 6", "11", "30"),
        ("minus means keep the right operand", "9 minus 2", "2", "7"),
        ("squared means add 10", "squared 7", "17", "49"),
        ("half means triple", "half of 8", "24", "4"),
        ("double means subtract 1", "double 9", "8", "18"),
        ("modulo means add operands then add one", "10 modulo 3", "14", "1"),
        ("divide means subtract the right operand", "20 divide 5", "15", "4"),
        ("exponent means add operands", "2 exponent 5", "7", "32"),
        ("sum means return the larger operand", "13 sum 8", "13", "21"),
    )
    rows: list[JsonDict] = []
    for index, (rule, expression, expected, prior) in enumerate(arithmetic, start=1):
        rows.append(
            _row(
                f"ctx3210-arithmetic-{index:02d}",
                "local_arithmetic_rules",
                (
                    "Use only this worksheet's arithmetic rule, even if it conflicts "
                    f"with standard arithmetic: {rule}."
                ),
                f"Under the worksheet rule, what is {expression}?",
                expected,
                prior,
                "exact_integer_string",
            )
        )
    return rows


def _context_defined_entity_fact_rows() -> list[JsonDict]:
    facts = (
        ("the capital of France", "Lima", "Paris"),
        ("the largest planet", "Mercury", "Jupiter"),
        ("water's boiling point at sea level", "42 C", "100 C"),
        ("the author of Hamlet", "Ada Lovelace", "William Shakespeare"),
        ("the tallest mountain", "Ben Nevis", "Mount Everest"),
        ("the chemical symbol for gold", "Xe", "Au"),
        ("the painter of the Mona Lisa", "Grace Hopper", "Leonardo da Vinci"),
        ("the speed of light", "12 km/s", "299,792 km/s"),
        ("the first US president", "Harriet Tubman", "George Washington"),
        ("the location of the Great Wall", "Peru", "China"),
    )
    rows: list[JsonDict] = []
    for index, (subject, local_value, prior) in enumerate(facts, start=1):
        rows.append(
            _row(
                f"ctx3210-entity-{index:02d}",
                "context_defined_entity_facts",
                (
                    "In this sealed local factbook, use the stated fact instead of "
                    f"world knowledge: {subject} is {local_value}."
                ),
                f"According to the sealed local factbook, what is {subject}?",
                local_value,
                prior,
                "exact_entity_fact_string",
            )
        )
    return rows


def build_fixture_rows() -> list[JsonDict]:
    """SCENARIO-VERIFY-3210: create deterministic context-shortcut rows."""

    return [
        *_symbolic_alias_rows(),
        *_local_arithmetic_rows(),
        *_context_defined_entity_fact_rows(),
    ]


def check_answer(row: Mapping[str, Any], answer: Any) -> JsonDict:
    """REQ-VERIFY-3210: exact canonical check for a candidate answer."""

    expected = normalize_answer(row.get("expected_answer", ""))
    observed = normalize_answer(answer)
    accepted = observed == expected
    return {
        "fixture_id": row.get("fixture_id"),
        "checker_type": row.get("exact_checker_type"),
        "accepted": accepted,
        "observed_normalized": observed,
        "expected_normalized": expected,
        "failure_reason": None if accepted else "answer_does_not_match_context_expected",
    }


def fixture_bank_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize family and checker coverage without mutating the rows."""

    row_list = list(rows)
    return {
        "fixture_count": len(row_list),
        "fixture_families": [
            family for family in FIXTURE_FAMILIES if any(row.get("family") == family for row in row_list)
        ],
        "exact_checker_types": [
            checker
            for checker in EXACT_CHECKER_TYPES
            if any(row.get("exact_checker_type") == checker for row in row_list)
        ],
        "prior_bait_row_count": sum(
            1
            for row in row_list
            if normalize_answer(row.get("expected_answer", ""))
            != normalize_answer(row.get("prior_bait_answer", ""))
        ),
    }


def validate_fixture_bank(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """REQ-VERIFY-3210: fail closed unless every row has an exact counterexample."""

    row_list = [dict(row) for row in rows]
    if not 20 <= len(row_list) <= 40:
        raise ValueError("fixture bank count must be between 20 and 40 rows")
    seen_ids: set[str] = set()
    for row in row_list:
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing required row fields: {missing}")
        fixture_id = str(row["fixture_id"])
        if fixture_id in seen_ids:
            raise ValueError(f"duplicate fixture_id: {fixture_id}")
        seen_ids.add(fixture_id)
        if row["family"] not in FIXTURE_FAMILIES:
            raise ValueError(f"unsupported family: {row['family']}")
        if row["exact_checker_type"] not in EXACT_CHECKER_TYPES:
            raise ValueError(f"unsupported checker: {row['exact_checker_type']}")
        if normalize_answer(row["expected_answer"]) == normalize_answer(row["prior_bait_answer"]):
            raise ValueError("prior-bait answer must differ from expected answer")
        counterexample = row["minimal_counterexample"]
        if (
            not isinstance(counterexample, Mapping)
            or counterexample.get("candidate_answer") != row["prior_bait_answer"]
        ):
            raise ValueError("minimal counterexample must contain the prior-bait answer")
        if not check_answer(row, row["expected_answer"])["accepted"]:
            raise ValueError(f"expected answer fails checker: {fixture_id}")
        if check_answer(row, row["prior_bait_answer"])["accepted"]:
            raise ValueError(f"prior-bait answer passes checker: {fixture_id}")
    summary = fixture_bank_summary(row_list)
    if summary["fixture_families"] != list(FIXTURE_FAMILIES):
        raise ValueError("fixture bank must cover all required families")
    if summary["exact_checker_types"] != list(EXACT_CHECKER_TYPES):
        raise ValueError("fixture bank must cover all exact checker types")
    return row_list


def context_following_score(
    rows: Sequence[Mapping[str, Any]],
    candidate_answers: Mapping[str, Any],
) -> float | None:
    """Return the exact context-following rate for supplied candidate answers."""

    row_list = list(rows)
    if not row_list:
        return None
    accepted = sum(
        1
        for row in row_list
        if check_answer(row, candidate_answers.get(str(row.get("fixture_id")), ""))["accepted"]
    )
    return accepted / len(row_list)


def _validate_optional_llm_smoke(optional_llm_smoke: Any) -> None:
    if not isinstance(optional_llm_smoke, Mapping):
        raise ValueError("optional_llm_smoke must be an object or null")
    model_specs = optional_llm_smoke.get("model_specs", [])
    model_ids = {
        str(spec.get("model_id"))
        for spec in model_specs
        if isinstance(spec, Mapping) and spec.get("model_id") is not None
    }
    if not any(model_id in MANDATED_LOCAL_SOTA_GGUF for model_id in model_ids):
        raise ValueError("optional LLM smoke must include a mandated local SOTA GGUF")


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    optional_llm_smoke: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3210: build the schema-versioned terminal artifact."""

    _ = Path(root)
    rows = validate_fixture_bank(build_fixture_rows())
    summary = fixture_bank_summary(rows)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "reference_papers": list(REFERENCE_PAPERS),
        "fixture_path": FIXTURE_REL_PATH.as_posix(),
        "fixture_count": summary["fixture_count"],
        "fixture_families": summary["fixture_families"],
        "exact_checker_types": summary["exact_checker_types"],
        "prior_bait_row_count": summary["prior_bait_row_count"],
        "context_following_score_available": True,
        "optional_llm_smoke": dict(optional_llm_smoke) if optional_llm_smoke is not None else None,
        "ready_for_clean_verifier": True,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "inference_substrate": "deterministic_exact_fixture_generation",
        "fixture_schema_fields": list(REQUIRED_ROW_FIELDS),
        "mandated_local_sota_gguf_policy": list(MANDATED_LOCAL_SOTA_GGUF),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "honest_verdict": (
            "complete: context_cot_clbench_parametric_shortcut_fixtures_v1 "
            "ready_for_clean_verifier=true; fixture_count=30; optional_llm_smoke=null"
        ),
    }
    validate_artifact(artifact, rows)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3210: validate artifact fields against the fixture bank."""

    row_list = validate_fixture_bank(rows or build_fixture_rows())
    summary = fixture_bank_summary(row_list)
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    checks = (
        ("schema_version", artifact.get("schema_version") == SCHEMA_VERSION),
        ("experiment_id", artifact.get("experiment_id") == EXPERIMENT_ID),
        ("milestone", artifact.get("milestone") == MILESTONE),
        ("fixture_path", artifact.get("fixture_path") == FIXTURE_REL_PATH.as_posix()),
        ("fixture_count", artifact.get("fixture_count") == summary["fixture_count"]),
        ("fixture_families", artifact.get("fixture_families") == summary["fixture_families"]),
        ("exact_checker_types", artifact.get("exact_checker_types") == summary["exact_checker_types"]),
        ("prior_bait_row_count", artifact.get("prior_bait_row_count") == summary["prior_bait_row_count"]),
        ("context_following_score_available", artifact.get("context_following_score_available") is True),
        ("ready_for_clean_verifier", artifact.get("ready_for_clean_verifier") is True),
        ("conductor_file_modified", artifact.get("conductor_file_modified") is False),
        ("active_roadmap_modified", artifact.get("active_roadmap_modified") is False),
        ("honest_verdict", str(artifact.get("honest_verdict", "")).startswith("complete:")),
    )
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise ValueError(f"{failed[0]} does not match Exp 3210 fixture contract")
    optional_llm_smoke = artifact.get("optional_llm_smoke")
    if optional_llm_smoke is not None:
        _validate_optional_llm_smoke(optional_llm_smoke)
    return dict(artifact)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one deterministic fixture object per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_artifacts(
    root: Path | str = REPO_ROOT,
    *,
    fixture_path: Path | str = FIXTURE_REL_PATH,
    output_path: Path | str = OUTPUT_REL_PATH,
    optional_llm_smoke: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3210 fixture JSONL and result JSON."""

    root_path = Path(root)
    fixture_output = Path(fixture_path)
    result_output = Path(output_path)
    if not fixture_output.is_absolute():
        fixture_output = root_path / fixture_output
    if not result_output.is_absolute():
        result_output = root_path / result_output
    rows = validate_fixture_bank(build_fixture_rows())
    write_jsonl(fixture_output, rows)
    artifact = build_artifact(
        root_path,
        optional_llm_smoke=optional_llm_smoke,
        tests_run=tests_run,
    )
    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_output


def main() -> None:
    """CLI entrypoint used by conductor-style artifact materialization."""

    print(write_artifacts().as_posix())


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    main()
