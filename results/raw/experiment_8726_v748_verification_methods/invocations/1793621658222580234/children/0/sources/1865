"""Exp5566 deterministic exact ASP/FSM near-miss corpus.

Spec refs: REQ-VERIFY-5566, SCENARIO-VERIFY-5566.

This experiment builds a corpus for later solve/verify separation work. The
generator proposes valid rows and controlled one- or two-edit corruptions, but
the label authority is only the exact fixture validator: Exp5555's
stable-model evaluator for ASP rows and Exp5541's exact FSM parser/solver for
transition rows. No model or heuristic is allowed to decide labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod
from carnot import experiment_5555_asp_fsm_nonmonotonic_fixture as asp_mod


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json")
CORPUS_RELATIVE_PATH = Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.jsonl")
SOURCE_FIXTURE_PATH = asp_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5566.exact_asp_fsm_near_miss_corpus.v504"
CORPUS_ROW_SCHEMA = "carnot.corpus.exact_asp_fsm_near_miss_5566.row.v1"
EXPERIMENT = 5566
EXPERIMENT_ID = "exp5566-exact-asp-fsm-near-miss-corpus"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
RANDOM_SEED = 5566
INFERENCE_SUBSTRATE = "deterministic_exact_fixture_no_llm"
EXACT_VALIDATOR_BACKEND = "exp5555_stable_model_evaluator+exp5541_fsm_exact_validator"
SPEC_REFS = ("REQ-VERIFY-5566", "SCENARIO-VERIFY-5566", "REQ-VERIFY-5555", "REQ-VERIFY-5541")
TERMINAL_PREFIXES = ("complete:", "blocked:")
REQUIRED_FAMILIES = (
    "defaults_exceptions",
    "contradictions",
    "soft_preference_optimality",
    "fsm_transition_consistency",
)
PARTITIONS = ("train", "dev", "test")
ROWS_PER_FAMILY = 30
VALID_ROWS_PER_FAMILY = 15
MIN_ROWS = 120
MIN_FAMILY_ROWS = 30

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Keeps every headline and gate field annotated by its evidence boundary.",
    "corpus_path": "Locates the reusable exact-labeled corpus artifact.",
    "corpus_sha256": "Pins the reusable corpus bytes to a stable digest.",
    "source_fixture_path": "Pins generation to the exact ASP/FSM fixture source.",
    "exact_validator_backend": "Names the deterministic exact validators used as authority.",
    "exact_validator_is_oracle": "Bare boolean disclosing that labels come from an exact fixture oracle, not a learned verifier.",
    "llm_invoked": "Prevents the corpus from being mistaken for live model inference.",
    "n_instances": "Counts exact-labeled independent rows available for comparisons.",
    "family_counts": "Confirms the four required families meet their row floors.",
    "label_counts": "Confirms valid and invalid classes are balanced.",
    "mutation_operator_counts": "Audits which controlled corruptions generated the near misses.",
    "mutation_distance_counts": "Audits one-edit and two-edit invalid controls separately from valid identity rows.",
    "partition_counts": "Confirms train, dev, and test partitions are populated.",
    "duplicate_leakage_count": "Gates readiness on no duplicate candidate leakage across partitions.",
    "valid_acceptance_rate": "Positive control rate for exact acceptance of valid rows.",
    "invalid_rejection_rate": "Positive control rate for exact rejection of invalid near misses.",
    "positive_control_passed": "Bare boolean requiring valid acceptance and invalid rejection controls to pass.",
    "tests_run": "Records the focused verification commands used for this artifact.",
    "inference_substrate": "Declares deterministic exact fixture validation with no LLM.",
    "honest_verdict": "Provides a terminal evidence boundary without model quality or speedup claims.",
    "n_rows": "The exact corpus must meet the preregistered scale floor.",
    "corpus_ready": "Only a complete leak-free corpus with passing exact controls may unlock live inference.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable row and artifact hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of already serialized bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_source_artifact(repo_root: Path = REPO_ROOT) -> JsonDict:
    """Load the Exp5555 source fixture artifact."""

    return _load_json(repo_root / SOURCE_FIXTURE_PATH)


def build_corpus_rows(upstream_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Build all exact-labeled valid and near-miss corpus rows."""

    if not source_fixture_ready(upstream_artifact):
        return []
    rows: list[JsonDict] = []
    for family in REQUIRED_FAMILIES:
        for index in range(VALID_ROWS_PER_FAMILY):
            partition = partition_for_index(index)
            valid_candidate = valid_candidate_for_family(family, index)
            expected_signature = exact_signature(
                valid_candidate["candidate_kind"], valid_candidate["candidate"]
            )
            valid_row_id = f"exp5566_{family}_{index:02d}_valid"
            rows.append(
                corpus_row(
                    row_id=valid_row_id,
                    family=family,
                    partition=partition,
                    label="valid",
                    candidate_kind=valid_candidate["candidate_kind"],
                    candidate=valid_candidate["candidate"],
                    expected_signature=expected_signature,
                    mutation_operators=["identity_valid_candidate"],
                    mutation_distance=0,
                    parent_row_id=None,
                )
            )
            distance = 1 if index % 2 == 0 else 2
            mutated = invalid_candidate_for_family(family, valid_candidate, index, distance)
            rows.append(
                corpus_row(
                    row_id=f"exp5566_{family}_{index:02d}_near_miss",
                    family=family,
                    partition=partition,
                    label="invalid",
                    candidate_kind=mutated["candidate_kind"],
                    candidate=mutated["candidate"],
                    expected_signature=expected_signature,
                    mutation_operators=mutated["mutation_operators"],
                    mutation_distance=distance,
                    parent_row_id=valid_row_id,
                )
            )
    return rows


def valid_candidate_for_family(family: str, index: int) -> JsonDict:
    """Return one deterministic valid candidate for a required corpus family."""

    if family == "defaults_exceptions":
        return {"candidate_kind": "asp_row", "candidate": default_exception_row(index)}
    if family == "contradictions":
        return {"candidate_kind": "asp_row", "candidate": contradiction_row(index)}
    if family == "soft_preference_optimality":
        return {"candidate_kind": "asp_row", "candidate": soft_preference_row(index)}
    if family == "fsm_transition_consistency":
        return {"candidate_kind": "fsm_machine", "candidate": fsm_machine_candidate(index)}
    raise ValueError(f"unknown_family:{family}")


def invalid_candidate_for_family(
    family: str,
    valid_candidate: Mapping[str, Any],
    index: int,
    distance: int,
) -> JsonDict:
    """Apply deterministic one- or two-edit corruption operators."""

    candidate = deepcopy(valid_candidate["candidate"])
    if family == "defaults_exceptions":
        operators = ["add_exception_fact"]
        _add_fact(candidate, _name("def", index, "exception"))
        if distance == 2:
            operators.append("remove_default_derivation_rule")
            _remove_rule(candidate, _rule_id("DEF", index, "01"))
    elif family == "contradictions":
        operators = ["remove_contradiction_constraint"]
        _remove_rule(candidate, _rule_id("CONTRA", index, "01"))
        if distance == 2:
            operators.append("add_unblocked_escape_fact")
            _add_fact(candidate, _name("contra", index, "escape"))
    elif family == "soft_preference_optimality":
        if distance == 1:
            operators = ["remove_preference_dominance_constraint"]
            _remove_rule(candidate, _rule_id("PREF", index, "04"))
        else:
            operators = ["remove_preferred_b_fact", "add_preferred_a_fact"]
            _remove_fact(candidate, _name("pref", index, "prefer_b"))
            _add_fact(candidate, _name("pref", index, "prefer_a"))
    elif family == "fsm_transition_consistency":
        machine = fsm_mod.parse_machine_description_text(str(candidate["machine_description_yaml"]))
        if distance == 1:
            operators = ["add_conflicting_required_transition"]
            machine["transition_constraints"].append(
                {
                    "constraint_id": _rule_id("TC_MUT", index, "00"),
                    "kind": "require",
                    "source": _name("fsm", index, "s0"),
                    "symbol": _name("sym", index, "x"),
                    "target": _name("fsm", index, "s2"),
                }
            )
        else:
            operators = ["set_accepting_to_error_state", "set_error_to_accepting_state"]
            machine["accepting_states"] = [_name("fsm", index, "s2")]
            machine["error_states"] = [_name("fsm", index, "s1")]
        candidate = {"machine_description_yaml": fsm_mod.machine_description_text(machine)}
    else:
        raise ValueError(f"unknown_family:{family}")
    return {
        "candidate_kind": str(valid_candidate["candidate_kind"]),
        "candidate": candidate,
        "mutation_operators": operators,
    }


def corpus_row(
    *,
    row_id: str,
    family: str,
    partition: str,
    label: str,
    candidate_kind: str,
    candidate: Mapping[str, Any],
    expected_signature: Mapping[str, Any],
    mutation_operators: Sequence[str],
    mutation_distance: int,
    parent_row_id: str | None,
) -> JsonDict:
    """Build one corpus row and stamp its exact validation decision."""

    row: JsonDict = {
        "schema": CORPUS_ROW_SCHEMA,
        "row_id": row_id,
        "family": family,
        "partition": partition,
        "label": label,
        "parent_row_id": parent_row_id,
        "candidate_kind": candidate_kind,
        "candidate": deepcopy(candidate),
        "candidate_sha256": sha256_json({"candidate_kind": candidate_kind, "candidate": candidate}),
        "expected_signature": deepcopy(expected_signature),
        "expected_signature_sha256": sha256_json(expected_signature),
        "mutation_operators": list(mutation_operators),
        "mutation_distance": mutation_distance,
        "source_fixture_path": SOURCE_FIXTURE_PATH.as_posix(),
        "exact_validator_backend": EXACT_VALIDATOR_BACKEND,
        "exact_validator_is_oracle": True,
    }
    validation = exact_validate_corpus_row(row)
    row["actual_signature"] = validation["actual_signature"]
    row["actual_signature_sha256"] = validation["actual_signature_sha256"]
    row["exact_validator_decision"] = validation["exact_validator_decision"]
    row["accepted_by_exact_validator"] = validation["accepted"]
    return row


def exact_validate_corpus_row(row: Mapping[str, Any]) -> JsonDict:
    """Recompute one row's exact signature and compare it to the stored target."""

    actual = exact_signature(str(row["candidate_kind"]), row["candidate"])
    actual_hash = sha256_json(actual)
    expected_hash = str(row["expected_signature_sha256"])
    accepted = actual_hash == expected_hash
    return {
        "accepted": accepted,
        "exact_validator_decision": "accepted" if accepted else "rejected",
        "actual_signature": actual,
        "actual_signature_sha256": actual_hash,
        "expected_signature_sha256": expected_hash,
    }


def exact_signature(candidate_kind: str, candidate: Mapping[str, Any]) -> JsonDict:
    """Return the exact validator signature used as the label authority."""

    if candidate_kind == "asp_row":
        report = asp_mod.evaluate_asp_row(candidate)
        return {
            "validator": "exp5555_stable_model_evaluator",
            "solver_status": report["solver_status"],
            "stable_model_count": report["stable_model_count"],
            "stable_model_samples": report["stable_model_samples"],
            "status_matches_expected": report["status_matches_expected"],
        }
    if candidate_kind == "fsm_machine":
        machine = fsm_mod.parse_machine_description_text(str(candidate["machine_description_yaml"]))
        schema_errors = fsm_mod.validate_machine_description(machine)
        report = fsm_mod.solve_instance(machine)
        return {
            "validator": "exp5541_fsm_exact_validator",
            "schema_errors": schema_errors,
            "solver_status": report["solver_status"],
            "completion_count": report["completion_count"],
            "transition_consistency_passed": report["transition_consistency_passed"],
            "contradictions": report["contradictions"],
            "trace_checks_passed": report["trace_checks_passed"],
            "trace_labels": [
                {
                    "trace_id": trace["trace_id"],
                    "expected_label": trace["expected_label"],
                    "actual_label": trace["actual_label"],
                    "passed": trace["passed"],
                }
                for trace in report["trace_checks"]
            ],
        }
    raise ValueError(f"unknown_candidate_kind:{candidate_kind}")


def default_exception_row(index: int) -> JsonDict:
    """Build a default/exception ASP row with no exception present."""

    prefix = "def"
    base = _name(prefix, index, "base")
    policy = _name(prefix, index, "policy")
    exception = _name(prefix, index, "exception")
    default_accept = _name(prefix, index, "default_accept")
    verified = _name(prefix, index, "verified")
    return {
        "row_id": f"asp_defaults_exceptions_{index:02d}",
        "description": "Default acceptance derives only when the exception fact is absent.",
        "fsm_instance_id": "fsm_sat_accept_error",
        "facts": [base, policy],
        "rules": [
            asp_mod.asp_rule(
                _rule_id("DEF", index, "00"),
                default_accept,
                positive=(base, policy),
                default_negated=(exception,),
            ),
            asp_mod.asp_rule(_rule_id("DEF", index, "01"), verified, positive=(default_accept,)),
            asp_mod.asp_rule(_rule_id("DEF", index, "02"), None, positive=(verified, exception)),
        ],
        "expected_status": "satisfiable",
        "contradiction_row": False,
    }


def contradiction_row(index: int) -> JsonDict:
    """Build an ASP contradiction control whose stable-model class is unsat."""

    claim = _name("contra", index, "claim")
    conflict = _name("contra", index, "conflict")
    impossible = _name("contra", index, "impossible")
    return {
        "row_id": f"asp_contradictions_{index:02d}",
        "description": "A hard constraint blocks the explicitly contradictory witness.",
        "fsm_instance_id": "fsm_unsat_conflicting_transition",
        "facts": [claim, conflict],
        "rules": [
            asp_mod.asp_rule(_rule_id("CONTRA", index, "00"), impossible, positive=(claim, conflict)),
            asp_mod.asp_rule(_rule_id("CONTRA", index, "01"), None, positive=(impossible,)),
        ],
        "expected_status": "unsatisfiable",
        "contradiction_row": True,
    }


def soft_preference_row(index: int) -> JsonDict:
    """Build a normal-rule encoding of soft preference optimality."""

    option_a = _name("pref", index, "option_a")
    option_b = _name("pref", index, "option_b")
    prefer_a = _name("pref", index, "prefer_a")
    prefer_b = _name("pref", index, "prefer_b")
    choose_a = _name("pref", index, "choose_a")
    choose_b = _name("pref", index, "choose_b")
    better_a = _name("pref", index, "better_a")
    better_b = _name("pref", index, "better_b")
    return {
        "row_id": f"asp_soft_preference_{index:02d}",
        "description": "A preferred option is selected by finite dominance constraints, not ASP optimization syntax.",
        "fsm_instance_id": "fsm_ambiguous_sparse_branch",
        "facts": [option_a, option_b, prefer_b],
        "rules": [
            asp_mod.asp_rule(
                _rule_id("PREF", index, "00"),
                choose_a,
                positive=(option_a,),
                default_negated=(choose_b,),
            ),
            asp_mod.asp_rule(
                _rule_id("PREF", index, "01"),
                choose_b,
                positive=(option_b,),
                default_negated=(choose_a,),
            ),
            asp_mod.asp_rule(_rule_id("PREF", index, "02"), better_b, positive=(prefer_b,)),
            asp_mod.asp_rule(_rule_id("PREF", index, "03"), better_a, positive=(prefer_a,)),
            asp_mod.asp_rule(_rule_id("PREF", index, "04"), None, positive=(choose_a, better_b)),
            asp_mod.asp_rule(_rule_id("PREF", index, "05"), None, positive=(choose_b, better_a)),
        ],
        "expected_status": "satisfiable",
        "contradiction_row": False,
    }


def fsm_machine_candidate(index: int) -> JsonDict:
    """Build an exact FSM machine row whose transition labels are consistent."""

    s0 = _name("fsm", index, "s0")
    s1 = _name("fsm", index, "s1")
    s2 = _name("fsm", index, "s2")
    symbol = _name("sym", index, "x")
    machine = fsm_mod.build_fixture_instance(
        instance_id=f"fsm_transition_consistency_{index:02d}",
        states=[s0, s1, s2],
        alphabet=[symbol],
        start_state=s0,
        accepting_states=[s1],
        error_states=[s2],
        required_transitions=[
            (_rule_id("TC", index, "00"), s0, symbol, s1),
            (_rule_id("TC", index, "01"), s1, symbol, s1),
            (_rule_id("TC", index, "02"), s2, symbol, s2),
        ],
        forbidden_transitions=[],
        trace_specs=[
            (_name("trace", index, "empty"), []),
            (_name("trace", index, "one"), [symbol]),
            (_name("trace", index, "two"), [symbol, symbol]),
        ],
        expected_status="satisfiable",
    )
    return {"machine_description_yaml": fsm_mod.machine_description_text(machine)}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate corpus counts and exact positive-control rates."""

    family_counts = _ordered_counts(Counter(str(row["family"]) for row in rows), REQUIRED_FAMILIES)
    label_counts = _ordered_counts(Counter(str(row["label"]) for row in rows), ("invalid", "valid"))
    partition_counts = _ordered_counts(Counter(str(row["partition"]) for row in rows), ("dev", "test", "train"))
    operator_counter: Counter[str] = Counter()
    for row in rows:
        operator_counter.update(str(op) for op in row["mutation_operators"])
    distance_counts = _ordered_counts(
        Counter(str(row["mutation_distance"]) for row in rows),
        ("0", "1", "2"),
    )
    valid_rows = [row for row in rows if row["label"] == "valid"]
    invalid_rows = [row for row in rows if row["label"] == "invalid"]
    valid_acceptance_rate = _rate(
        sum(int(row["accepted_by_exact_validator"] is True) for row in valid_rows),
        len(valid_rows),
    )
    invalid_rejection_rate = _rate(
        sum(int(row["accepted_by_exact_validator"] is False) for row in invalid_rows),
        len(invalid_rows),
    )
    leak_count = duplicate_leakage_count(rows)
    positive_control_passed = bool(
        valid_rows
        and invalid_rows
        and valid_acceptance_rate == 1.0
        and invalid_rejection_rate == 1.0
    )
    return {
        "n_rows": len(rows),
        "n_instances": len(rows),
        "family_counts": family_counts,
        "label_counts": label_counts,
        "mutation_operator_counts": dict(sorted(operator_counter.items())),
        "mutation_distance_counts": distance_counts,
        "partition_counts": partition_counts,
        "duplicate_leakage_count": leak_count,
        "valid_acceptance_rate": valid_acceptance_rate,
        "invalid_rejection_rate": invalid_rejection_rate,
        "positive_control_passed": positive_control_passed,
    }


def duplicate_leakage_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count duplicate candidate hashes that appear in more than one partition."""

    partitions_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        partitions_by_hash[str(row["candidate_sha256"])].add(str(row["partition"]))
    return sum(1 for partitions in partitions_by_hash.values() if len(partitions) > 1)


def readiness_blockers(source_ready: bool, summary: Mapping[str, Any]) -> list[str]:
    """Return readiness blockers for the corpus gate."""

    blockers: list[str] = []
    if not source_ready:
        blockers.append("source_fixture_ready")
    if int(summary.get("n_rows", 0)) < MIN_ROWS:
        blockers.append("n_rows")
    for family in REQUIRED_FAMILIES:
        if int(summary.get("family_counts", {}).get(family, 0)) < MIN_FAMILY_ROWS:
            blockers.append(f"family_counts:{family}")
    label_counts = summary.get("label_counts", {})
    if label_counts.get("valid") != label_counts.get("invalid") or label_counts.get("valid", 0) == 0:
        blockers.append("label_counts")
    if int(summary.get("duplicate_leakage_count", 0)) != 0:
        blockers.append("duplicate_leakage_count")
    if float(summary.get("valid_acceptance_rate", 0.0)) != 1.0:
        blockers.append("valid_acceptance_rate")
    if float(summary.get("invalid_rejection_rate", 0.0)) != 1.0:
        blockers.append("invalid_rejection_rate")
    if summary.get("positive_control_passed") is not True:
        blockers.append("positive_control_passed")
    return blockers


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    upstream_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5566 artifact and in-memory corpus rows."""

    upstream = load_source_artifact(repo_root) if upstream_artifact is None else dict(upstream_artifact)
    source_ready = source_fixture_ready(upstream)
    rows = build_corpus_rows(upstream) if source_ready else []
    summary = summarize_rows(rows)
    corpus_digest = sha256_bytes(corpus_bytes(rows)) if rows else ""
    blockers = readiness_blockers(source_ready, summary)
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "corpus_path": CORPUS_RELATIVE_PATH.as_posix(),
        "corpus_sha256": corpus_digest,
        "source_fixture_path": SOURCE_FIXTURE_PATH.as_posix(),
        "source_fixture_ready": source_ready,
        "exact_validator_backend": EXACT_VALIDATOR_BACKEND,
        "exact_validator_is_oracle": True,
        "llm_invoked": False,
        "tests_run": [dict(row) for row in tests_run],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "corpus_ready": ready,
        "readiness_blockers": blockers,
        "research_conductor_modified": False,
        "corpus_rows": rows,
        "reproducibility_checksum": "",
    }
    artifact.update(summary)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    repo_root: Path = REPO_ROOT,
    upstream_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5566 result JSON and reusable JSONL corpus."""

    artifact = build_artifact(
        repo_root=repo_root,
        upstream_artifact=upstream_artifact,
        tests_run=tests_run,
    )
    rows = list(artifact["corpus_rows"])
    corpus_path = repo_root / CORPUS_RELATIVE_PATH
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_bytes(corpus_bytes(rows))
    result_path = repo_root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    validate_artifact(artifact, repo_root=repo_root)
    return artifact


def validate_artifact(artifact: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    """Validate the terminal artifact and fail closed on overclaim."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(artifact.get("source_fixture_path") == SOURCE_FIXTURE_PATH.as_posix(), "source_fixture_path")
    _require(artifact.get("exact_validator_backend") == EXACT_VALIDATOR_BACKEND, "exact_validator_backend")
    _require(artifact.get("exact_validator_is_oracle") is True, "exact_validator_is_oracle")
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require("model_specs" not in artifact, "model_specs")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(int(artifact.get("n_rows", -1)) >= MIN_ROWS, "n_rows")
    _require(artifact.get("n_instances") == artifact.get("n_rows"), "n_instances")
    _require(artifact.get("family_counts") == {family: ROWS_PER_FAMILY for family in REQUIRED_FAMILIES}, "family_counts")
    _require(artifact.get("label_counts") == {"invalid": 60, "valid": 60}, "label_counts")
    _require(artifact.get("partition_counts") == {"dev": 24, "test": 24, "train": 72}, "partition_counts")
    _require(int(artifact.get("duplicate_leakage_count", -1)) == 0, "duplicate_leakage_count")
    _require(float(artifact.get("valid_acceptance_rate", 0.0)) == 1.0, "valid_acceptance_rate")
    _require(float(artifact.get("invalid_rejection_rate", 0.0)) == 1.0, "invalid_rejection_rate")
    _require(artifact.get("positive_control_passed") is True, "positive_control_passed")
    _require(artifact.get("corpus_ready") is True, "corpus_ready")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(str(artifact.get("corpus_path")) == CORPUS_RELATIVE_PATH.as_posix(), "corpus_path")
    corpus_path = repo_root / CORPUS_RELATIVE_PATH
    _require(corpus_path.exists(), "corpus_path")
    _require(artifact.get("corpus_sha256") == sha256_file(corpus_path), "corpus_sha256")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def corpus_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize corpus rows as stable JSONL bytes."""

    if not rows:
        return b""
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows) + "\n"
    return text.encode("utf-8")


def source_fixture_ready(artifact: Mapping[str, Any]) -> bool:
    """Return whether the source ASP/FSM fixture exposes the required exact gate."""

    return bool(
        artifact.get("exact_fsm_fixture_extended_ready") is True
        and artifact.get("exact_asp_validator_ready") is True
        and isinstance(artifact.get("asp_fixture_rows"), list)
        and isinstance(artifact.get("stable_model_reports"), list)
    )


def partition_for_index(index: int) -> str:
    """Assign paired valid/invalid rows to stable train/dev/test partitions."""

    if index < 9:
        return "train"
    if index < 12:
        return "dev"
    return "test"


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict without implying model quality."""

    if ready:
        return "complete: exact ASP/FSM near-miss corpus ready with deterministic oracle labels"
    return "blocked: exact ASP/FSM near-miss corpus not ready_" + "_".join(blockers)


def _ordered_counts(counter: Counter[str], order: Sequence[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in order if int(counter.get(key, 0)) > 0}


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _name(prefix: str, index: int, suffix: str) -> str:
    return f"{prefix}_{index:02d}_{suffix}"


def _rule_id(prefix: str, index: int, suffix: str) -> str:
    return f"{prefix}_{index:02d}_{suffix}"


def _add_fact(row: JsonDict, atom: str) -> None:
    facts = list(row.get("facts", []))
    if atom not in facts:
        facts.append(atom)
    row["facts"] = sorted(facts)


def _remove_fact(row: JsonDict, atom: str) -> None:
    row["facts"] = [fact for fact in row.get("facts", []) if fact != atom]


def _remove_rule(row: JsonDict, rule_id: str) -> None:
    row["rules"] = [rule for rule in row.get("rules", []) if str(rule.get("rule_id")) != rule_id]


def _load_json(path: Path) -> JsonDict:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"load_error": "missing", "path": path.as_posix()}
    except json.JSONDecodeError as exc:
        return {"load_error": "json_decode", "path": path.as_posix(), "detail": str(exc)}
    if not isinstance(decoded, dict):
        return {"load_error": "json_not_object", "path": path.as_posix()}
    return decoded


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "corpus_path": artifact["corpus_path"],
                "corpus_ready": artifact["corpus_ready"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
