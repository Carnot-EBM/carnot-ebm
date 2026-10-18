"""Build the Exp 3301 stratified exact repair panel manifest.

Spec refs: REQ-VERIFY-3301, SCENARIO-VERIFY-3301.

This module creates the fixed input panel for a later live repair rerun. It is
deliberately boring: every row has a deterministic exact checker, a known
wrong starting answer, and local repair feedback. That gives the future live
repair experiment a stable denominator without letting an LLM judge decide
whether a repair should count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.exact_repair_panel_manifest.v11"
EXPERIMENT_ID = "exp3301"
TASK_ID = "exp3301-exact-repair-panel-manifest-v11"
ARTIFACT = "experiment_3301_exact_repair_panel_manifest_v11"
MILESTONE = "2026.05.305"
RUN_DATE = "20260529"
RANDOM_SEED = 3301
MIN_PANEL_CASES = 30
MIN_FAMILY_COUNT = 5
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3301_exact_repair_panel_manifest_v11.json")
PANEL_CASES_REL_PATH = Path("data/research/exact_repair_panel_v11.jsonl")

REQUIRED_ARTIFACT_FIELDS = {
    "repair_panel_manifest_ready",
    "panel_case_count",
    "case_family_counts",
    "exact_checker_types",
    "llm_judge_required_count",
    "panel_cases_path",
    "case_hashes",
    "localized_feedback_coverage",
    "known_failing_candidate_count",
    "validation_commands",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
REQUIRED_CASE_FIELDS = {
    "case_id",
    "family",
    "context",
    "question",
    "failing_candidate",
    "expected_answer",
    "exact_checker_type",
    "localized_repair_feedback",
    "case_hash",
}
EXACT_CHECKER_TYPES = (
    "exact_alias_string",
    "exact_bool_string",
    "exact_context_string",
    "exact_integer_string",
    "exact_stdout_string",
)
DEFAULT_VALIDATION_COMMANDS = (
    ".venv/bin/python -m carnot.verify.exact_repair_panel_manifest_v11",
    ".venv/bin/pytest tests/python/test_experiment_3301_exact_repair_panel_manifest_v11.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3301_exact_repair_panel_manifest_v11.py -q",
    ".venv/bin/coverage report --include='*/exact_repair_panel_manifest_v11.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python -m json.tool results/experiment_3301_exact_repair_panel_manifest_v11.json >/tmp/exp3301.json.pretty",
    ".venv/bin/python -c \"import json, pathlib; rows=[json.loads(line) for line in pathlib.Path('data/research/exact_repair_panel_v11.jsonl').read_text().splitlines()]; assert len(rows) == 30\"",
    ".venv/bin/pytest tests/python -q",
)


def build_panel_cases(random_seed: int = RANDOM_SEED) -> list[JsonDict]:
    """REQ-VERIFY-3301: create a fixed, stratified exact-checkable case bank."""

    del random_seed
    cases: list[JsonDict] = []
    cases.extend(symbolic_alias_cases())
    cases.extend(arithmetic_exact_cases())
    cases.extend(context_shortcut_cases())
    cases.extend(code_output_cases())
    cases.extend(bounded_logic_cases())
    validate_cases(cases)
    return cases


def symbolic_alias_cases() -> list[JsonDict]:
    """Return local glossary rows where common meanings are the known trap."""

    rows = (
        ("mercury", "banana", "planet"),
        ("python", "blue screwdriver", "snake"),
        ("mars", "teacup", "planet"),
        ("ruby", "north door", "gemstone"),
        ("java", "silver lantern", "coffee"),
        ("saturn", "paper kite", "planet"),
    )
    return [
        make_case(
            case_id=f"exp3301-symbolic-{idx:02d}",
            family="symbolic_aliases",
            context=(
                "For this panel row only, the glossary overrides ordinary meanings: "
                f"{term} means {answer}."
            ),
            question=f"According to the row glossary, what does {term} mean?",
            failing_candidate=wrong,
            expected_answer=answer,
            exact_checker_type="exact_alias_string",
        )
        for idx, (term, answer, wrong) in enumerate(rows, start=1)
    ]


def arithmetic_exact_cases() -> list[JsonDict]:
    """Return small integer rows whose answers are checked by parsing integers."""

    rows = (
        ("Local rule: triangle(a,b) = a*b + a.", "What is triangle(3,4)?", "15", "7"),
        ("Local rule: delta(a,b) = a*b - b.", "What is delta(5,2)?", "8", "7"),
        ("Local rule: box(a,b) = (a+b)*2.", "What is box(6,1)?", "14", "7"),
        ("Local rule: star(a,b) = a*a + b.", "What is star(4,3)?", "19", "7"),
        ("Local rule: hook(a,b) = a - b + 10.", "What is hook(9,4)?", "15", "5"),
        ("Local rule: wave(a,b) = a*b + b*b.", "What is wave(2,5)?", "35", "10"),
    )
    return [
        make_case(
            case_id=f"exp3301-arithmetic-{idx:02d}",
            family="arithmetic_exact_rows",
            context=context,
            question=question,
            failing_candidate=wrong,
            expected_answer=answer,
            exact_checker_type="exact_integer_string",
        )
        for idx, (context, question, answer, wrong) in enumerate(rows, start=1)
    ]


def context_shortcut_cases() -> list[JsonDict]:
    """Return rows where the last local fact overrides a tempting shortcut."""

    rows = (
        ("A rumor says the red key opens locker A. The signed card says it opens locker C.", "Which locker does the red key open?", "locker C", "locker A"),
        ("The map legend first calls the river path short. The route note says the hill path is the approved path.", "Which path is approved?", "hill path", "river path"),
        ("The old roster lists Mira as lead. The update line says Niko is lead today.", "Who is lead today?", "Niko", "Mira"),
        ("The cover label says crate two is empty. The inspection line says crate two holds bolts.", "What does crate two hold?", "bolts", "nothing"),
        ("The memo says badge green is inactive. The final rule says badge green is active for bay 4.", "Which bay is badge green active for?", "bay 4", "inactive"),
        ("The draft says use port 7000. The release note says use port 7105.", "Which port should be used?", "port 7105", "port 7000"),
    )
    return [
        make_case(
            case_id=f"exp3301-context-{idx:02d}",
            family="context_shortcuts",
            context=context,
            question=question,
            failing_candidate=wrong,
            expected_answer=answer,
            exact_checker_type="exact_context_string",
        )
        for idx, (context, question, answer, wrong) in enumerate(rows, start=1)
    ]


def code_output_cases() -> list[JsonDict]:
    """Return tiny code/output rows checked by exact stdout text."""

    rows = (
        ("Python snippet:\nvalue = 3\nprint(value * value)", "What is printed?", "9", "6"),
        ("Python snippet:\nitems = ['a', 'b', 'c']\nprint(len(items))", "What is printed?", "3", "2"),
        ("Python snippet:\nname = 'car'\nprint(name[::-1])", "What is printed?", "rac", "car"),
        ("Python snippet:\nprint('-'.join(['x', 'y']))", "What is printed?", "x-y", "xy"),
        ("Python snippet:\nflag = not False\nprint(flag)", "What is printed?", "True", "False"),
        ("Python snippet:\nprint(sum([2, 4, 6]))", "What is printed?", "12", "10"),
    )
    return [
        make_case(
            case_id=f"exp3301-code-{idx:02d}",
            family="code_output_checks",
            context=context,
            question=question,
            failing_candidate=wrong,
            expected_answer=answer,
            exact_checker_type="exact_stdout_string",
        )
        for idx, (context, question, answer, wrong) in enumerate(rows, start=1)
    ]


def bounded_logic_cases() -> list[JsonDict]:
    """Return propositional rows with a boolean exact authority."""

    rows = (
        ("Facts: If A then B. A is true.", "Is B true?", "true", "false"),
        ("Facts: Exactly one of left/right is active. Left is active.", "Is right active?", "false", "true"),
        ("Facts: All silver tokens are valid. Token q is silver.", "Is token q valid?", "true", "false"),
        ("Facts: No closed ticket can be pending. Ticket 8 is closed.", "Is ticket 8 pending?", "false", "true"),
        ("Facts: A switch is safe only if guard is on. Guard is off.", "Is the switch safe?", "false", "true"),
        ("Facts: Every row in set R is audited. Row 12 is in set R.", "Is row 12 audited?", "true", "false"),
    )
    return [
        make_case(
            case_id=f"exp3301-logic-{idx:02d}",
            family="bounded_logical_consistency",
            context=context,
            question=question,
            failing_candidate=wrong,
            expected_answer=answer,
            exact_checker_type="exact_bool_string",
        )
        for idx, (context, question, answer, wrong) in enumerate(rows, start=1)
    ]


def make_case(
    *,
    case_id: str,
    family: str,
    context: str,
    question: str,
    failing_candidate: str,
    expected_answer: str,
    exact_checker_type: str,
) -> JsonDict:
    """Create one case and hash every identity field except the hash itself."""

    case: JsonDict = {
        "case_id": case_id,
        "family": family,
        "context": context,
        "question": question,
        "failing_candidate": failing_candidate,
        "expected_answer": expected_answer,
        "exact_checker_type": exact_checker_type,
        "llm_judge_required": False,
        "localized_repair_feedback": (
            f"Exact local check failed for {case_id}: replace {failing_candidate!r} "
            f"with {expected_answer!r}; use only this row's {family} constraints."
        ),
    }
    case["case_hash"] = case_hash(case)
    return case


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    random_seed: int = RANDOM_SEED,
    panel_cases_path: Path | str = PANEL_CASES_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-3301: summarize the fixed manifest without inference."""

    del root
    started = time.perf_counter() if started_s is None else float(started_s)
    cases = build_panel_cases(random_seed=random_seed)
    family_counts = case_family_counts(cases)
    exact_types = exact_checker_types(cases)
    case_hashes = [str(case["case_hash"]) for case in cases]
    llm_judge_count = sum(case.get("llm_judge_required") is True for case in cases)
    feedback_coverage = rate(
        sum(bool(str(case.get("localized_repair_feedback") or "")) for case in cases),
        len(cases),
    )
    known_failing_count = sum(
        not exact_check(case, str(case.get("failing_candidate") or "")) for case in cases
    )
    manifest_text = manifest_jsonl_text(cases)
    finished = time.perf_counter() if now_s is None else float(now_s)
    ready = (
        len(cases) >= MIN_PANEL_CASES
        and len(family_counts) >= MIN_FAMILY_COUNT
        and llm_judge_count == 0
        and feedback_coverage == 1.0
        and known_failing_count == len(cases)
        and len(set(case_hashes)) == len(case_hashes)
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3301", "SCENARIO-VERIFY-3301"],
        "repair_panel_manifest_ready": ready,
        "panel_case_count": len(cases),
        "case_family_counts": family_counts,
        "exact_checker_types": exact_types,
        "llm_judge_required_count": llm_judge_count,
        "panel_cases_path": Path(panel_cases_path).as_posix(),
        "panel_cases_sha256": sha256_text(manifest_text),
        "case_hashes": case_hashes,
        "localized_feedback_coverage": feedback_coverage,
        "known_failing_candidate_count": known_failing_count,
        "validation_commands": list(tests_run or DEFAULT_VALIDATION_COMMANDS),
        "inference_substrate": "deterministic_exact_manifest_no_live_inference",
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration(started, finished),
        "honest_verdict": "",
        "repair_gate_contract_preserved": {
            "source_experiment_id": "exp3289",
            "repair_gate_open_required_before_live_rerun": True,
            "live_repair_panel_must_be_gated_on_fixed_manifest": True,
        },
        "clean_verifier_contract_preserved": {
            "source_experiment_id": "exp3287",
            "accepted_repairs_require_exact_pass": True,
            "llm_judge_allowed": False,
            "decision_contract": "exact_checker_only_for_manifest",
        },
        "panel_cases": cases,
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    panel_cases_path: Path | str = PANEL_CASES_REL_PATH,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist both the result JSON and case JSONL file."""

    root_path = Path(root)
    output = resolve_path(root_path, output_path)
    panel_path = resolve_path(root_path, panel_cases_path)
    artifact = build_artifact(
        root_path,
        random_seed=random_seed,
        panel_cases_path=panel_cases_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_text(manifest_jsonl_text(artifact["panel_cases"]), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on any row that would make the future repair denominator fuzzy."""

    if len(cases) < MIN_PANEL_CASES:
        raise ValueError("repair panel manifest requires at least 30 cases")
    hashes = [str(case.get("case_hash") or "") for case in cases]
    if len(set(hashes)) != len(hashes) or any(len(value) != 64 for value in hashes):
        raise ValueError("case hashes must be unique 64-character checksums")
    if len(case_family_counts(cases)) < MIN_FAMILY_COUNT:
        raise ValueError("repair panel manifest must include at least five case families")
    for case in cases:
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            raise ValueError(f"case {case.get('case_id')} missing required fields: {sorted(missing)}")
        if case.get("llm_judge_required") is True:
            raise ValueError(f"case {case.get('case_id')} requires an LLM judge")
        if not str(case.get("localized_repair_feedback") or "").strip():
            raise ValueError(f"case {case.get('case_id')} lacks localized repair feedback")
        if str(case.get("exact_checker_type") or "") not in EXACT_CHECKER_TYPES:
            raise ValueError(f"case {case.get('case_id')} has an unsupported exact checker")
        if case_hash(case) != case.get("case_hash"):
            raise ValueError(f"case {case.get('case_id')} has a stale case_hash")
        if not exact_check(case, str(case.get("expected_answer") or "")):
            raise ValueError(f"case {case.get('case_id')} expected answer fails exact checker")
        if exact_check(case, str(case.get("failing_candidate") or "")):
            raise ValueError(f"case {case.get('case_id')} lacks a known failing candidate")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact and block headline-unsafe overclaiming."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("repair_panel_manifest_ready"), bool):
        raise ValueError("repair_panel_manifest_ready must be a bool")
    panel_count = artifact.get("panel_case_count")
    if not isinstance(panel_count, int) or isinstance(panel_count, bool) or panel_count < MIN_PANEL_CASES:
        raise ValueError("panel_case_count must be an integer >= 30")
    if artifact.get("llm_judge_required_count") != 0:
        raise ValueError("LLM judge required count must be zero")
    if artifact.get("known_failing_candidate_count") != panel_count:
        raise ValueError("known_failing_candidate_count must equal panel_case_count")
    if artifact.get("localized_feedback_coverage") != 1.0:
        raise ValueError("localized_feedback_coverage must be 1.0")
    if not isinstance(artifact.get("case_family_counts"), Mapping):
        raise ValueError("case_family_counts must be a dict")
    if len(artifact["case_family_counts"]) < MIN_FAMILY_COUNT:
        raise ValueError("case_family_counts must include at least five families")
    if not isinstance(artifact.get("exact_checker_types"), list):
        raise ValueError("exact_checker_types must be a list")
    if not isinstance(artifact.get("case_hashes"), list) or len(artifact["case_hashes"]) != panel_count:
        raise ValueError("case_hashes must list one hash per case")
    if len(set(str(value) for value in artifact["case_hashes"])) != panel_count:
        raise ValueError("case_hashes must be unique")
    if not isinstance(artifact.get("validation_commands"), list):
        raise ValueError("validation_commands must be a list")
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or duration_s < 0:
        raise ValueError("duration_s must be a non-negative number")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a 64-character checksum")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")
    if artifact.get("repair_panel_manifest_ready") is True:
        validate_cases(mapping_list(artifact.get("panel_cases")))


def exact_check(case: Mapping[str, Any], candidate_answer: str) -> bool:
    """Apply the case's deterministic checker to one candidate string."""

    checker = str(case.get("exact_checker_type") or "")
    expected = str(case.get("expected_answer") or "")
    candidate = str(candidate_answer or "")
    if checker == "exact_integer_string":
        parsed_candidate = parse_int_string(candidate)
        parsed_expected = parse_int_string(expected)
        return parsed_candidate is not None and parsed_candidate == parsed_expected
    if checker == "exact_bool_string":
        parsed_candidate = normalize_bool_string(candidate)
        parsed_expected = normalize_bool_string(expected)
        return parsed_expected in {"true", "false"} and parsed_candidate == parsed_expected
    if checker == "exact_stdout_string":
        return candidate.strip() == expected.strip()
    return normalize_text(candidate) == normalize_text(expected)


def case_family_counts(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count rows per family in sorted-key order for stable JSON output."""

    counts = Counter(str(case.get("family") or "") for case in cases)
    return {key: counts[key] for key in sorted(counts) if key}


def exact_checker_types(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the exact checker inventory used by the manifest."""

    return sorted({str(case.get("exact_checker_type") or "") for case in cases if case.get("exact_checker_type")})


def case_hash(case: Mapping[str, Any]) -> str:
    """Hash the stable case identity while ignoring the stored hash field."""

    payload = {key: value for key, value in dict(case).items() if key != "case_hash"}
    return stable_hash(payload)


def manifest_jsonl_text(cases: Sequence[Mapping[str, Any]]) -> str:
    """Serialize panel cases as deterministic JSONL with a trailing newline."""

    return "".join(json.dumps(dict(case), sort_keys=True) + "\n" for case in cases)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable manifest content while excluding timing and command noise."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum", "validation_commands"}
    }
    return stable_hash(stable)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that names the exact manifest gate result."""

    return (
        "complete: "
        f"repair_panel_manifest_ready={str(artifact['repair_panel_manifest_ready']).lower()}; "
        f"panel_case_count={artifact['panel_case_count']}; "
        f"llm_judge_required_count={artifact['llm_judge_required_count']}; "
        f"known_failing_candidate_count={artifact['known_failing_candidate_count']}"
    )


def normalize_text(value: str) -> str:
    """Normalize exact textual answers without granting semantic equivalence."""

    return " ".join(str(value).strip().casefold().split())


def parse_int_string(value: str) -> int | None:
    """Parse an integer-only answer and reject decimals or extra prose."""

    text = str(value).strip()
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-"):
        return int(text) if text[1:].isdigit() else None
    return int(text) if text.isdigit() else None


def normalize_bool_string(value: str) -> str:
    """Map common boolean literals to true/false while preserving unknown text."""

    normalized = normalize_text(value)
    if normalized in {"true", "yes", "1"}:
        return "true"
    if normalized in {"false", "no", "0"}:
        return "false"
    return normalized


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with explicit zero-denominator behavior."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def duration(started: float, finished: float) -> float:
    """Return non-negative elapsed seconds rounded for stable artifacts."""

    return round(max(0.0, float(finished) - float(started)), 6)


def resolve_path(root: Path, value: Path | str) -> Path:
    """Resolve repository-relative output paths used by the artifact writer."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from arbitrary JSON-like input."""

    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def stable_hash(payload: Any) -> str:
    """Return a SHA-256 checksum for JSON-compatible payloads."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(text: str) -> str:
    """Hash serialized manifest content for artifact-to-file integrity checks."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:  # pragma: no cover - CLI wrapper.
    """Write the default Exp 3301 artifact and JSONL case manifest."""

    output = write_artifact()
    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
