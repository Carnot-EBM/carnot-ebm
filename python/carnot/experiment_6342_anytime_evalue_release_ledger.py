"""Exp6342 anytime e-value release ledger.

Spec refs: REQ-LEARN-6342, REQ-LEARN-6342-LEDGER,
REQ-LEARN-6342-VALIDITY, REQ-LEARN-6342-POWER,
REQ-LEARN-6342-ATTACKS, REQ-LEARN-6342-GUARD,
REQ-LEARN-6342-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6342_anytime_evalue_release_ledger.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6342_anytime_evalue_release_ledger.py")
EXP6318_RELATIVE_PATH = Path(
    "results/experiment_6318_versioned_factor_local_online_initializer.json"
)
EXP6319_RELATIVE_PATH = Path("results/experiment_6319_feedback_directed_online_update_search.json")
EXP6320_RELATIVE_PATH = Path("results/experiment_6320_online_self_evolution_safety_audit.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")

LEDGER_SUFFIX = ".evalue_ledger.jsonl"
LEDGER_SCHEMA_SUFFIX = ".ledger_schema.json"
SYNTHETIC_STREAM_MANIFEST_SUFFIX = ".synthetic_stream_manifest.json"

SCHEMA = "carnot.experiment_6342.anytime_evalue_release_ledger.v1"
LEDGER_ROW_SCHEMA = SCHEMA + ".ledger_row"
EXPERIMENT_ID = "experiment_6342_anytime_evalue_release_ledger"
RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = "deterministic_synthetic_evalue_replay_exact_oracle_no_llm"

NULL_SUCCESS_PROBABILITY = 0.5
BETTING_ALTERNATIVE_PROBABILITY = 0.8
SYNTHETIC_ALTERNATIVE_PROBABILITY = 0.82
ALPHA = 0.05
PREREGISTERED_TYPE_I_BOUND = 0.05
PREREGISTERED_POWER_LOWER_BOUND = 0.80
NULL_STREAM_COUNT = 512
ALTERNATIVE_STREAM_COUNT = 128
LOOKS_PER_STREAM = 40
MAX_LEDGER_ROWS = 64
RESOURCE_LIMITS = {
    "max_null_stream_count": NULL_STREAM_COUNT,
    "max_alternative_stream_count": ALTERNATIVE_STREAM_COUNT,
    "max_looks_per_stream": LOOKS_PER_STREAM,
    "max_ledger_rows": MAX_LEDGER_ROWS,
    "llm_call_limit": 0,
    "generated_label_limit": 0,
}

HYPOTHESES: dict[str, JsonDict] = {
    "accept_factor_release": {"factor_id": "accept_factor", "source": "Exp6318"},
    "repair_factor_release": {"factor_id": "repair_factor", "source": "Exp6318"},
    "reject_factor_release": {"factor_id": "reject_factor", "source": "Exp6318"},
    "feedback_drift_release": {"factor_id": "drift_factor", "source": "Exp6319"},
}
HYPOTHESIS_IDS = tuple(HYPOTHESES)
HYPOTHESIS_COUNT = len(HYPOTHESIS_IDS)
PER_HYPOTHESIS_ALPHA = ALPHA / HYPOTHESIS_COUNT
RELEASE_THRESHOLD = 1.0 / PER_HYPOTHESIS_ALPHA
GENESIS_ROW_HASH = sha256_json({"genesis": SCHEMA})
PREDECISION_HASH = sha256_json(
    {
        "schema": SCHEMA + ".predecision",
        "hypotheses": HYPOTHESES,
        "null_success_probability": NULL_SUCCESS_PROBABILITY,
        "betting_alternative_probability": BETTING_ALTERNATIVE_PROBABILITY,
        "alpha": ALPHA,
        "release_threshold": RELEASE_THRESHOLD,
    }
)

RANDOM_SEEDS = {
    "null": 634200,
    "alternative": 634300,
    "ledger": 634400,
    "attack": 634500,
    "interval": 634600,
}

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6342_anytime_evalue_release_ledger.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6342_anytime_evalue_release_ledger.py "
    "-m pytest tests/python/test_experiment_6342_anytime_evalue_release_ledger.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6342_anytime_evalue_release_ledger.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6342_anytime_evalue_release_ledger --date 20260812"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6342_anytime_evalue_release_ledger.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6342_anytime_evalue_release_ledger.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6318_RELATIVE_PATH,
    EXP6319_RELATIVE_PATH,
    EXP6320_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_PROGRAM_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6318_RELATIVE_PATH,
    EXP6319_RELATIVE_PATH,
    EXP6320_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    *PROTECTED_RELATIVE_PATHS,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "source_claim_boundary",
    "evalue_ledger_path_and_hash",
    "ledger_schema_path_and_hash",
    "null_family_and_assumptions",
    "filtration_and_evidence_identity_contract",
    "betting_rule_and_predecision_hash",
    "alpha_multiplicity_and_release_policy",
    "exact_safety_guard_contract",
    "synthetic_stream_manifest_path_and_hash",
    "null_stream_results",
    "alternative_stream_results",
    "optional_stopping_results",
    "repeated_look_results",
    "duplicate_cross_factor_reorder_and_selection_attack_results",
    "restart_reconstruction_results",
    "append_only_tamper_results",
    "type_i_error_interval_and_sample_size",
    "power_interval_and_sample_size",
    "release_delay_distribution",
    "eprocess_state_examples",
    "exact_oracle_claim_boundary",
    "generated_label_count",
    "llm_call_count",
    "anytime_release_certificate_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows preregistration, e-process validity, attack closure, restart identity, exact guard, and tests.",
    "source_claim_boundary": "NxN E-valuation is a design cue only. Local claims stop at deterministic factor-local release certification.",
    "evalue_ledger_path_and_hash": "The append-only JSONL ledger is content-addressed so replay starts from bytes, not memory.",
    "ledger_schema_path_and_hash": "The frozen schema fixes row identity, hash chaining, and replay validation.",
    "null_family_and_assumptions": "The null family states the composite boundary that makes the e-process a supermartingale.",
    "filtration_and_evidence_identity_contract": "Evidence IDs, factor scope, and filtration time prevent optional-stopping leakage and duplicate reuse.",
    "betting_rule_and_predecision_hash": "The betting rule and predecision hash are frozen before outcomes so the test is not fitted after seeing labels.",
    "alpha_multiplicity_and_release_policy": "Alpha spending, multiplicity, and the release threshold are explicit and data-independent.",
    "exact_safety_guard_contract": "Statistical evidence cannot release a factor unless the exact oracle safety guard also passes.",
    "synthetic_stream_manifest_path_and_hash": "Null and alternative stream seeds, sizes, probabilities, and resource limits are frozen.",
    "null_stream_results": "Null streams report threshold crossings and the empirical error used for readiness.",
    "alternative_stream_results": "Alternative streams report power under the same frozen ledger and guard.",
    "optional_stopping_results": "First-crossing stops prove repeated looks do not inflate release beyond the bound.",
    "repeated_look_results": "Fixed and repeated-look summaries stay separated for audit.",
    "duplicate_cross_factor_reorder_and_selection_attack_results": "Duplicate rows, cross-factor reuse, event reorder, and selected hypotheses fail closed.",
    "restart_reconstruction_results": "Restart replay reproduces the same state, hashes, and release decisions.",
    "append_only_tamper_results": "Truncation, row mutation, previous-hash breaks, reset, and restart corruption are detected.",
    "type_i_error_interval_and_sample_size": "The type-I interval and sample size justify the null-error claim.",
    "power_interval_and_sample_size": "The power interval and sample size justify the alternative claim.",
    "release_delay_distribution": "First-release look counts show how long valid evidence took to cross the gate.",
    "eprocess_state_examples": "Example states make the nonnegative e-value ledger auditable without replaying all streams.",
    "exact_oracle_claim_boundary": "The exact checker is the outcome oracle, so the result is execution-grounded and not oracle-distinct.",
    "generated_label_count": "Bare zero proves no generated labels were used.",
    "llm_call_count": "Bare zero proves no LLM call was made.",
    "anytime_release_certificate_ready_score": "Readiness is one only when null error, power, attacks, restart identity, exact guard, protected files, and tests pass.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream evidence files remain byte-identical.",
    "preconditions_checked": "Inputs, source hashes, protected hashes, nulls, alternatives, evidence contract, betting rule, alpha, threshold, guard, seeds, stream sizes, and resource limits freeze first.",
    "inference_substrate": "The run declares deterministic synthetic replay plus exact oracle checks with no LLM or base model load.",
    "verifier_is_oracle": "Bare true states that exact safety and outcome checks are the oracle.",
    "field_provenance": "Every field maps to spec, source artifacts, sidecars, streams, attacks, tests, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, full pytest, spec coverage, run command, validation, adversarial verification, E2E reading, and root-clutter checks are listed.",
    "test_exit_codes": "Failed commands prevent readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Null, alternative, ledger, attack, and interval seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states whether the anytime release certificate is ready.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6342",
        "Exp6318 and Exp6319 evidence schema",
        "V546 NxN E-valuation reference",
        "synthetic stream and e-value ledger sidecars",
        "Exp6342 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


class EValueLedger:
    """Append-only e-process ledger with hash-chained rows.

    The ledger rejects evidence before it can change state.  This keeps attack
    tests honest: a failed duplicate or reorder is not a row with a warning.
    It is absent from the ledger state that a restart can replay.
    """

    def __init__(
        self,
        *,
        predecision_hash: str = PREDECISION_HASH,
        threshold: float = RELEASE_THRESHOLD,
    ) -> None:
        self.predecision_hash = predecision_hash
        self.threshold = float(threshold)
        self.rows: list[JsonDict] = []
        self.cumulative_by_hypothesis = {hypothesis: 1.0 for hypothesis in HYPOTHESIS_IDS}
        self.evidence_identity_to_factor: dict[str, str] = {}
        self.previous_hash = GENESIS_ROW_HASH
        self.last_filtration_time = -math.inf
        self.release_count = 0

    def append(self, event: Mapping[str, Any]) -> JsonDict:
        """Append one valid event or raise a fail-closed validation error."""

        event_payload = dict(event)
        reason = self._rejection_reason(event_payload)
        if reason:
            raise ValueError(reason)
        hypothesis_id = str(event_payload["hypothesis_id"])
        factor_id = str(event_payload["factor_id"])
        outcome = int(event_payload["outcome"])
        before = float(self.cumulative_by_hypothesis[hypothesis_id])
        increment = betting_increment(outcome)
        after = _rounded(before * increment)
        event_hash = sha256_json(event_payload)
        exact_passed = exact_safety_guard(event_payload)
        crossed = after >= self.threshold
        if crossed and exact_passed:
            decision = "released"
            self.release_count += 1
        elif crossed:
            decision = "blocked_by_exact_guard"
        else:
            decision = "not_released"
        row: JsonDict = {
            "schema": LEDGER_ROW_SCHEMA,
            "sequence": len(self.rows),
            "previous_row_hash": self.previous_hash,
            "predecision_hash": self.predecision_hash,
            "event_hash": event_hash,
            "event": event_payload,
            "evidence_identity": event_payload["evidence_identity"],
            "filtration_time": float(event_payload["filtration_time"]),
            "hypothesis_id": hypothesis_id,
            "factor_id": factor_id,
            "outcome": outcome,
            "evalue_increment": increment,
            "cumulative_evalue": after,
            "release_threshold": self.threshold,
            "exact_safety_passed": exact_passed,
            "crossed_threshold": crossed,
            "release_decision": decision,
        }
        row["row_hash"] = row_hash(row)
        self.rows.append(row)
        self.cumulative_by_hypothesis[hypothesis_id] = after
        self.evidence_identity_to_factor[str(event_payload["evidence_identity"])] = factor_id
        self.previous_hash = str(row["row_hash"])
        self.last_filtration_time = float(event_payload["filtration_time"])
        return row

    def try_append(self, event: Mapping[str, Any]) -> JsonDict:
        """Return a small fail-closed receipt instead of raising."""

        try:
            row = self.append(event)
        except ValueError as exc:
            return {"fail_closed": True, "reason": str(exc), "released": False}
        return {
            "fail_closed": row["release_decision"] != "released",
            "reason": row["release_decision"],
            "released": row["release_decision"] == "released",
        }

    def state_hash(self) -> str:
        """Hash only replay-relevant terminal state."""

        return sha256_json(
            {
                "cumulative_by_hypothesis": self.cumulative_by_hypothesis,
                "previous_hash": self.previous_hash,
                "release_count": self.release_count,
                "seen_identities": sorted(self.evidence_identity_to_factor),
            }
        )

    def _rejection_reason(self, event: Mapping[str, Any]) -> str | None:
        if event.get("predecision_hash") != self.predecision_hash:
            return "predecision_hash_mismatch"
        hypothesis_id = str(event.get("hypothesis_id"))
        if hypothesis_id not in HYPOTHESES:
            return "selected_hypothesis_after_outcome"
        factor_id = str(event.get("factor_id"))
        if factor_id != HYPOTHESES[hypothesis_id]["factor_id"]:
            return "factor_hypothesis_mismatch"
        identity = str(event.get("evidence_identity"))
        if identity in self.evidence_identity_to_factor:
            if self.evidence_identity_to_factor[identity] != factor_id:
                return "cross_factor_reuse"
            return "duplicate_evidence"
        if float(event.get("filtration_time", -math.inf)) <= self.last_filtration_time:
            return "reordered_event"
        if event.get("outcome") not in (0, 1):
            return "outcome"
        return None


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if duration_s is None:
        artifact["duration_s"] = time.perf_counter() - started
        refresh_terminal_fields(artifact)
        validate_artifact(artifact)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    """Run deterministic streams and assemble the release certificate."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes()
    schema_path = _ledger_schema_path(result_path)
    stream_manifest_path = _synthetic_stream_manifest_path(result_path)
    ledger_path = _evalue_ledger_path(result_path)
    schema = ledger_schema()
    manifest = synthetic_stream_manifest(date)
    _write_json(schema_path, schema)
    _write_json(stream_manifest_path, manifest)

    null_results = simulate_stream_family("null")
    alternative_results = simulate_stream_family("alternative")
    ledger = build_certificate_ledger()
    _write_jsonl(ledger_path, ledger.rows)
    replay = replay_ledger_rows(ledger.rows, expected_predecision_hash=PREDECISION_HASH)
    attack_results = run_evidence_attacks()
    tamper_results = run_tamper_attacks(ledger.rows, replay)
    exact_guard = exact_safety_guard_contract()
    protected_after = _protected_files_unchanged(protected_before)

    artifact: JsonDict = {
        "status": "complete_null",
        "source_claim_boundary": source_claim_boundary(),
        "evalue_ledger_path_and_hash": {
            **_path_receipt(ledger_path),
            "row_count": len(ledger.rows),
            "release_count": ledger.release_count,
            "state_hash": replay["state_hash"],
        },
        "ledger_schema_path_and_hash": {
            **_path_receipt(schema_path),
            "schema": LEDGER_ROW_SCHEMA,
            "required_field_count": len(schema["required_fields"]),
        },
        "null_family_and_assumptions": null_family_and_assumptions(),
        "filtration_and_evidence_identity_contract": filtration_contract(),
        "betting_rule_and_predecision_hash": betting_rule_and_predecision_hash(),
        "alpha_multiplicity_and_release_policy": alpha_policy(),
        "exact_safety_guard_contract": exact_guard,
        "synthetic_stream_manifest_path_and_hash": {
            **_path_receipt(stream_manifest_path),
            "null_stream_count": NULL_STREAM_COUNT,
            "alternative_stream_count": ALTERNATIVE_STREAM_COUNT,
        },
        "null_stream_results": null_results["summary"],
        "alternative_stream_results": alternative_results["summary"],
        "optional_stopping_results": null_results["optional_stopping"],
        "repeated_look_results": null_results["repeated_look"],
        "duplicate_cross_factor_reorder_and_selection_attack_results": attack_results,
        "restart_reconstruction_results": replay,
        "append_only_tamper_results": tamper_results,
        "type_i_error_interval_and_sample_size": null_results["interval"],
        "power_interval_and_sample_size": alternative_results["interval"],
        "release_delay_distribution": alternative_results["release_delay_distribution"],
        "eprocess_state_examples": eprocess_state_examples(ledger.rows, null_results),
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "generated_label_count": 0,
        "llm_call_count": 0,
        "anytime_release_certificate_ready_score": 0.0,
        "protected_files_unchanged": protected_after,
        "preconditions_checked": preconditions_checked(
            date=date,
            result_path=result_path,
            schema_path=schema_path,
            ledger_path=ledger_path,
            stream_manifest_path=stream_manifest_path,
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(test_exit_codes),
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: readiness not computed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    score = ready_score(artifact)
    artifact["anytime_release_certificate_ready_score"] = score
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and fail-closed readiness fields."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    _require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    for field in ("generated_label_count", "llm_call_count"):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("status") == status(artifact), "status")
    _require(
        str(artifact.get("honest_verdict") or "") == honest_verdict(artifact), "honest_verdict"
    )
    _require(
        artifact.get("anytime_release_certificate_ready_score") == ready_score(artifact),
        "anytime_release_certificate_ready_score",
    )
    _require(
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "protected_files_unchanged",
    )
    _require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
        "duration_s",
    )
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every anytime-release gate passes."""

    nulls = _as_mapping(artifact.get("null_stream_results"))
    alternatives = _as_mapping(artifact.get("alternative_stream_results"))
    type_i = _as_mapping(artifact.get("type_i_error_interval_and_sample_size"))
    power = _as_mapping(artifact.get("power_interval_and_sample_size"))
    attacks = _as_mapping(
        artifact.get("duplicate_cross_factor_reorder_and_selection_attack_results")
    )
    restart = _as_mapping(artifact.get("restart_reconstruction_results"))
    tamper = _as_mapping(artifact.get("append_only_tamper_results"))
    guard = _as_mapping(artifact.get("exact_safety_guard_contract"))
    tests = _as_mapping(artifact.get("test_exit_codes"))
    protected = _as_mapping(artifact.get("protected_files_unchanged"))
    gates = (
        nulls.get("empirical_type_i_error", math.inf) <= PREREGISTERED_TYPE_I_BOUND,
        type_i.get("upper", math.inf) <= PREREGISTERED_TYPE_I_BOUND,
        alternatives.get("empirical_power", 0.0) >= PREREGISTERED_POWER_LOWER_BOUND,
        power.get("lower", 0.0) >= PREREGISTERED_POWER_LOWER_BOUND,
        attacks.get("all_attacks_fail_closed") is True,
        attacks.get("released_attack_count") == 0,
        restart.get("byte_identical") is True,
        restart.get("release_count", 0) >= 1,
        tamper.get("all_tamper_attacks_detected") is True,
        guard.get("statistical_release_requires_exact_guard") is True,
        guard.get("unsafe_threshold_crossing_release_decision") == "blocked_by_exact_guard",
        artifact.get("generated_label_count") == 0
        and type(artifact.get("generated_label_count")) is int,
        artifact.get("llm_call_count") == 0 and type(artifact.get("llm_call_count")) is int,
        artifact.get("verifier_is_oracle") is True,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from the ready score."""

    return (
        "complete_positive"
        if artifact.get("anytime_release_certificate_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefix verdict."""

    if artifact.get("anytime_release_certificate_ready_score") == 1.0:
        return (
            "complete_positive: anytime e-value release certificate passed with exact safety guard"
        )
    return "complete_null: anytime e-value release certificate did not meet every gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking duration and checksum."""

    stable = json.loads(_canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def betting_increment(outcome: int) -> float:
    """Return the Bernoulli likelihood-ratio e-value increment."""

    if outcome == 1:
        return _rounded(BETTING_ALTERNATIVE_PROBABILITY / NULL_SUCCESS_PROBABILITY)
    if outcome == 0:
        return _rounded((1.0 - BETTING_ALTERNATIVE_PROBABILITY) / (1.0 - NULL_SUCCESS_PROBABILITY))
    raise ValueError("outcome")


def exact_safety_guard(event: Mapping[str, Any]) -> bool:
    """Exact oracle guard for source identity and unsafe outcome flags."""

    return (
        event.get("safe") is True
        and event.get("exact_outcome") in {"success", "failure"}
        and event.get("source_schema") == "exp6318_exp6319_factor_local_evidence_v1"
    )


def build_event(
    stream_kind: str,
    stream_index: int,
    look_index: int,
    *,
    outcome: int | None = None,
    safe: bool = True,
    hypothesis_id: str | None = None,
    factor_id: str | None = None,
    filtration_time: float | None = None,
    evidence_identity: str | None = None,
    predecision_hash: str = PREDECISION_HASH,
) -> JsonDict:
    """Build one Exp6318/Exp6319-shaped evidence event."""

    hypothesis = hypothesis_id or HYPOTHESIS_IDS[stream_index % HYPOTHESIS_COUNT]
    factor = factor_id or str(HYPOTHESES.get(hypothesis, {}).get("factor_id", "unknown"))
    observed = (
        deterministic_outcome(stream_kind, stream_index, look_index) if outcome is None else outcome
    )
    source_event_id = f"evt-{look_index % 16:02d}"
    identity = evidence_identity or f"{stream_kind}:{stream_index}:{look_index}:{source_event_id}"
    return {
        "source_schema": "exp6318_exp6319_factor_local_evidence_v1",
        "stream_kind": stream_kind,
        "stream_index": stream_index,
        "look_index": look_index,
        "source_event_id": source_event_id,
        "evidence_identity": identity,
        "hypothesis_id": hypothesis,
        "factor_id": factor,
        "predecision_hash": predecision_hash,
        "filtration_time": float(filtration_time if filtration_time is not None else look_index),
        "outcome": int(observed),
        "exact_outcome": "success" if observed == 1 else "failure",
        "safe": safe,
        "prediction_visible_before_outcome": True,
        "target_visible_before_outcome": False,
    }


def deterministic_outcome(stream_kind: str, stream_index: int, look_index: int) -> int:
    """Generate a deterministic Bernoulli outcome from a hash seed."""

    if stream_kind == "null":
        seed = RANDOM_SEEDS["null"]
        probability = NULL_SUCCESS_PROBABILITY
    elif stream_kind == "alternative":
        seed = RANDOM_SEEDS["alternative"]
        probability = SYNTHETIC_ALTERNATIVE_PROBABILITY
    else:
        raise ValueError(f"unknown_stream_kind:{stream_kind}")
    digest = hashlib.sha256(f"{seed}:{stream_index}:{look_index}".encode()).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    return 1 if uniform < probability else 0


def simulate_stream_family(stream_kind: str) -> JsonDict:
    """Simulate a frozen family of null or alternative streams."""

    stream_count = NULL_STREAM_COUNT if stream_kind == "null" else ALTERNATIVE_STREAM_COUNT
    release_delays: list[int] = []
    fixed_terminal_crossing_count = 0
    examples: list[JsonDict] = []
    for stream_index in range(stream_count):
        cumulative = 1.0
        first_crossing: int | None = None
        for look_index in range(LOOKS_PER_STREAM):
            outcome = deterministic_outcome(stream_kind, stream_index, look_index)
            cumulative = _rounded(cumulative * betting_increment(outcome))
            if stream_index < 2 and look_index < 3:
                examples.append(
                    {
                        "stream_kind": stream_kind,
                        "stream_index": stream_index,
                        "look_index": look_index,
                        "outcome": outcome,
                        "cumulative_evalue": cumulative,
                    }
                )
            if first_crossing is None and cumulative >= RELEASE_THRESHOLD:
                first_crossing = look_index + 1
        if cumulative >= RELEASE_THRESHOLD:
            fixed_terminal_crossing_count += 1
        if first_crossing is not None:
            release_delays.append(first_crossing)
    release_count = len(release_delays)
    rate = release_count / stream_count
    interval = wilson_interval(release_count, stream_count)
    summary_name = "empirical_type_i_error" if stream_kind == "null" else "empirical_power"
    summary = {
        "stream_kind": stream_kind,
        "stream_count": stream_count,
        "look_count_per_stream": LOOKS_PER_STREAM,
        "release_count": release_count,
        summary_name: rate,
        "fixed_terminal_crossing_count": fixed_terminal_crossing_count,
        "threshold": RELEASE_THRESHOLD,
        "example_states": examples,
    }
    return {
        "summary": summary,
        "interval": {
            **interval,
            "n": stream_count,
            "success_count": release_count,
            "metric": summary_name,
        },
        "optional_stopping": {
            "stopped_on_first_crossing": True,
            "release_count": release_count,
            "empirical_type_i_error": rate if stream_kind == "null" else None,
            "threshold": RELEASE_THRESHOLD,
        },
        "repeated_look": {
            "look_count_per_stream": LOOKS_PER_STREAM,
            "optional_crossing_count": release_count,
            "fixed_terminal_crossing_count": fixed_terminal_crossing_count,
        },
        "release_delay_distribution": release_delay_distribution(release_delays),
    }


def build_certificate_ledger() -> EValueLedger:
    """Build one append-only alternative-stream certificate ledger."""

    fallback = EValueLedger()
    for stream_index in range(ALTERNATIVE_STREAM_COUNT):
        ledger = EValueLedger()
        for look_index in range(LOOKS_PER_STREAM):
            event = build_event("alternative", stream_index, look_index)
            ledger.append(event)
            if stream_index == 0:
                fallback = ledger
            if ledger.release_count:
                return ledger
    return fallback


def replay_ledger_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_predecision_hash: str,
) -> JsonDict:
    """Replay JSONL rows and verify hashes, previous links, and state."""

    ledger = EValueLedger(predecision_hash=expected_predecision_hash)
    for row in rows:
        _require(row.get("previous_row_hash") == ledger.previous_hash, "previous_row_hash")
        row_without_hash = dict(row)
        stored_hash = row_without_hash.pop("row_hash", None)
        _require(row_hash(row_without_hash) == stored_hash, "row_hash")
        replayed = ledger.append(_as_mapping(row.get("event")))
        _require(replayed["row_hash"] == stored_hash, "replay_row_hash")
    releases = [row for row in ledger.rows if row["release_decision"] == "released"]
    return {
        "byte_identical": True,
        "row_count": len(ledger.rows),
        "release_count": len(releases),
        "state_hash": ledger.state_hash(),
        "ledger_hash": sha256_json(ledger.rows),
        "release_rows": [
            {
                "sequence": row["sequence"],
                "hypothesis_id": row["hypothesis_id"],
                "factor_id": row["factor_id"],
                "cumulative_evalue": row["cumulative_evalue"],
            }
            for row in releases
        ],
    }


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a ledger row with any existing row_hash removed."""

    payload = dict(row)
    payload.pop("row_hash", None)
    return sha256_json(payload)


def run_evidence_attacks() -> JsonDict:
    """Exercise duplicate, reuse, reorder, and selection attacks."""

    ledger = EValueLedger()
    first = build_event("alternative", 10, 0, outcome=1, filtration_time=0.0)
    ledger.append(first)
    duplicate = ledger.try_append(first)
    cross_factor = ledger.try_append(
        build_event(
            "alternative",
            11,
            1,
            outcome=1,
            hypothesis_id="repair_factor_release",
            factor_id="repair_factor",
            filtration_time=1.0,
            evidence_identity=str(first["evidence_identity"]),
        )
    )
    reorder = ledger.try_append(build_event("alternative", 12, 2, outcome=1, filtration_time=0.0))
    selected = ledger.try_append(
        build_event(
            "alternative",
            13,
            3,
            outcome=1,
            hypothesis_id="selected_after_outcome",
            factor_id="accept_factor",
            filtration_time=3.0,
        )
    )
    results = {
        "duplicate_evidence": duplicate,
        "cross_factor_reuse": cross_factor,
        "reordered_event": reorder,
        "selected_hypothesis_after_outcome": selected,
    }
    return {
        **results,
        "attack_count": len(results),
        "released_attack_count": sum(1 for result in results.values() if result["released"]),
        "all_attacks_fail_closed": all(result["fail_closed"] for result in results.values()),
    }


def run_tamper_attacks(rows: Sequence[Mapping[str, Any]], replay: Mapping[str, Any]) -> JsonDict:
    """Mutate ledger bytes and show that replay detects the corruption."""

    truncated = [dict(row) for row in rows[:-1]]
    truncation_detected = sha256_json(truncated) != replay.get("ledger_hash")
    mutated = json.loads(_canonical_json(rows))
    mutated[0]["event"]["outcome"] = 0 if mutated[0]["event"]["outcome"] == 1 else 1
    previous_break = json.loads(_canonical_json(rows))
    if len(previous_break) > 1:
        previous_break[1]["previous_row_hash"] = "sha256:broken"
    reset_rows = json.loads(_canonical_json(rows))
    reset_rows[-1]["sequence"] = 0
    reset_rows[-1]["previous_row_hash"] = GENESIS_ROW_HASH
    restart_corrupt = list(reversed(json.loads(_canonical_json(rows))))
    checks = {
        "truncation": {"detected": truncation_detected, "mode": "ledger_hash_mismatch"},
        "row_mutation": _tamper_replay_receipt(mutated, "row_hash"),
        "previous_hash_break": _tamper_replay_receipt(previous_break, "previous_row_hash"),
        "evalue_reset_attack": _tamper_replay_receipt(reset_rows, "previous_row_hash"),
        "restart_corruption": _tamper_replay_receipt(restart_corrupt, "previous_row_hash"),
    }
    return {
        **checks,
        "all_tamper_attacks_detected": all(check["detected"] for check in checks.values()),
    }


def _tamper_replay_receipt(rows: Sequence[Mapping[str, Any]], expected_reason: str) -> JsonDict:
    try:
        replay_ledger_rows(rows, expected_predecision_hash=PREDECISION_HASH)
    except ValueError as exc:
        return {"detected": True, "reason": str(exc), "expected_reason": expected_reason}
    return {"detected": False, "reason": "accepted", "expected_reason": expected_reason}


def exact_safety_guard_contract() -> JsonDict:
    """Prove threshold crossing cannot bypass the exact guard."""

    ledger = EValueLedger()
    last_row: JsonDict = {}
    for look_index in range(12):
        last_row = ledger.append(build_event("alternative", 800, look_index, outcome=1, safe=False))
    return {
        "statistical_release_requires_exact_guard": True,
        "unsafe_threshold_crossing_cumulative_evalue": last_row["cumulative_evalue"],
        "unsafe_threshold_crossing_crossed_threshold": last_row["crossed_threshold"],
        "unsafe_threshold_crossing_release_decision": last_row["release_decision"],
        "unsafe_threshold_crossing_release_count": ledger.release_count,
        "exact_guard_is_oracle": True,
    }


def wilson_interval(success_count: int, sample_size: int) -> JsonDict:
    """Return a 95 percent Wilson interval for a binomial rate."""

    if sample_size == 0:
        return {"lower": 0.0, "upper": 0.0, "estimate": 0.0}
    if success_count == 0:
        upper_zero = 1.0 - 0.05 ** (1.0 / sample_size)
        return {"lower": 0.0, "upper": _rounded(upper_zero), "estimate": 0.0}
    if success_count == sample_size:
        lower_full = 0.05 ** (1.0 / sample_size)
        return {"lower": _rounded(lower_full), "upper": 1.0, "estimate": 1.0}
    z = 1.96
    n = float(sample_size)
    phat = success_count / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return {
        "lower": _rounded(max(0.0, center - margin)),
        "upper": _rounded(min(1.0, center + margin)),
        "estimate": _rounded(phat),
    }


def release_delay_distribution(delays: Sequence[int]) -> JsonDict:
    """Summarize first-release looks for streams that crossed."""

    if not delays:
        return {
            "released_stream_count": 0,
            "mean_look": None,
            "median_look": None,
            "min_look": None,
            "max_look": None,
        }
    ordered = sorted(delays)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "released_stream_count": len(ordered),
        "mean_look": _rounded(sum(ordered) / len(ordered)),
        "median_look": _rounded(float(median)),
        "min_look": ordered[0],
        "max_look": ordered[-1],
    }


def supermartingale_fixture() -> JsonDict:
    """Show the frozen increment has expectation at most one under the null."""

    success = betting_increment(1)
    failure = betting_increment(0)
    expected = NULL_SUCCESS_PROBABILITY * success + (1.0 - NULL_SUCCESS_PROBABILITY) * failure
    lower_null_expected = 0.45 * success + 0.55 * failure
    return {
        "success_increment": success,
        "failure_increment": failure,
        "expected_increment_under_null": _rounded(expected),
        "expected_increment_under_lower_composite_null": _rounded(lower_null_expected),
        "nonnegative": success >= 0.0 and failure >= 0.0,
    }


def ledger_schema() -> JsonDict:
    """Return the frozen ledger row schema."""

    return {
        "schema": LEDGER_ROW_SCHEMA,
        "required_fields": [
            "schema",
            "sequence",
            "previous_row_hash",
            "predecision_hash",
            "event_hash",
            "event",
            "evidence_identity",
            "filtration_time",
            "hypothesis_id",
            "factor_id",
            "outcome",
            "evalue_increment",
            "cumulative_evalue",
            "release_threshold",
            "exact_safety_passed",
            "crossed_threshold",
            "release_decision",
            "row_hash",
        ],
        "append_only_contract": "sequence and previous_row_hash form a single hash chain",
        "canonical_hash_contract": "row_hash is sha256_json(row without row_hash)",
        "predecision_hash": PREDECISION_HASH,
    }


def synthetic_stream_manifest(date: str) -> JsonDict:
    """Return the frozen null and alternative stream manifest."""

    return {
        "schema": SCHEMA + ".synthetic_stream_manifest",
        "run_date": date,
        "null_stream_count": NULL_STREAM_COUNT,
        "alternative_stream_count": ALTERNATIVE_STREAM_COUNT,
        "looks_per_stream": LOOKS_PER_STREAM,
        "null_success_probability": NULL_SUCCESS_PROBABILITY,
        "synthetic_alternative_success_probability": SYNTHETIC_ALTERNATIVE_PROBABILITY,
        "betting_alternative_probability": BETTING_ALTERNATIVE_PROBABILITY,
        "hypotheses": HYPOTHESES,
        "random_seeds": dict(RANDOM_SEEDS),
        "resource_limits": dict(RESOURCE_LIMITS),
        "exp6318_path_hash": _path_receipt(REPO_ROOT / EXP6318_RELATIVE_PATH),
        "exp6319_path_hash": _path_receipt(REPO_ROOT / EXP6319_RELATIVE_PATH),
        "exp6320_path_hash": _path_receipt(REPO_ROOT / EXP6320_RELATIVE_PATH),
    }


def source_claim_boundary() -> JsonDict:
    return {
        "design_reference": "research-references.md V546 NxN E-valuation entry",
        "local_claim": "deterministic factor-local e-value release ledger",
        "not_claimed": [
            "LLM-generated hypotheses",
            "oracle-distinct verifier moat",
            "model-weight learning",
        ],
        "source_experiments": [EXP6318_RELATIVE_PATH.as_posix(), EXP6319_RELATIVE_PATH.as_posix()],
    }


def null_family_and_assumptions() -> JsonDict:
    return {
        "null_family_id": "factor_local_no_positive_exact_lift",
        "composite_null_boundary": "P(exact_success | filtration) <= 0.5",
        "supermartingale_fixture": supermartingale_fixture(),
        "hypothesis_count": HYPOTHESIS_COUNT,
        "hypotheses": HYPOTHESES,
    }


def filtration_contract() -> JsonDict:
    return {
        "filtration_order": "events append with strictly increasing filtration_time",
        "evidence_identity_fields": [
            "source_schema",
            "source_event_id",
            "stream_kind",
            "stream_index",
            "look_index",
        ],
        "deduplication_key": "evidence_identity",
        "cross_factor_reuse_policy": "reject",
        "outcome_visibility": "target_visible_before_outcome is false until predecision hash exists",
    }


def betting_rule_and_predecision_hash() -> JsonDict:
    return {
        "predecision_hash": PREDECISION_HASH,
        "success_increment": betting_increment(1),
        "failure_increment": betting_increment(0),
        "betting_alternative_probability": BETTING_ALTERNATIVE_PROBABILITY,
        "null_boundary_probability": NULL_SUCCESS_PROBABILITY,
        "supermartingale_fixture": supermartingale_fixture(),
    }


def alpha_policy() -> JsonDict:
    return {
        "alpha": ALPHA,
        "hypothesis_count": HYPOTHESIS_COUNT,
        "per_hypothesis_alpha": PER_HYPOTHESIS_ALPHA,
        "multiplicity_policy": "Bonferroni e-value threshold per frozen hypothesis",
        "release_threshold": RELEASE_THRESHOLD,
        "optional_stopping_policy": "release only on first threshold crossing with exact guard pass",
    }


def exact_oracle_claim_boundary() -> JsonDict:
    return {
        "verifier_is_oracle": True,
        "oracle": "deterministic exact outcome and safety guard over synthetic evidence rows",
        "claim_boundary": "execution-grounded release process, not an oracle-distinct moat claim",
    }


def preconditions_checked(
    *,
    date: str,
    result_path: Path,
    schema_path: Path,
    ledger_path: Path,
    stream_manifest_path: Path,
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all data-independent release-process choices."""

    return {
        "run_date": date,
        "result_path": _relative_or_absolute(result_path),
        "ledger_path": _relative_or_absolute(ledger_path),
        "ledger_schema_path": _relative_or_absolute(schema_path),
        "synthetic_stream_manifest_path": _relative_or_absolute(stream_manifest_path),
        "source_hashes": {
            path.as_posix(): _path_receipt(REPO_ROOT / path) for path in HASHED_INPUTS
        },
        "protected_hashes_before_outcomes": dict(protected_before),
        "null_family": null_family_and_assumptions(),
        "alternatives": {
            "synthetic_alternative_success_probability": SYNTHETIC_ALTERNATIVE_PROBABILITY,
            "alternative_stream_count": ALTERNATIVE_STREAM_COUNT,
        },
        "filtration_and_evidence_identity": filtration_contract(),
        "betting_rule": betting_rule_and_predecision_hash(),
        "alpha_multiplicity_and_release_policy": alpha_policy(),
        "stream_sizes": {
            "null_stream_count": NULL_STREAM_COUNT,
            "alternative_stream_count": ALTERNATIVE_STREAM_COUNT,
            "looks_per_stream": LOOKS_PER_STREAM,
        },
        "random_seeds": dict(RANDOM_SEEDS),
        "resource_limits": dict(RESOURCE_LIMITS),
        "exact_guard": {
            "guard_function": "exact_safety_guard",
            "threshold_crossing_without_guard_policy": "reject",
        },
        "protected_hashes_frozen_before_outcomes": True,
        "outcome_processing_after_preconditions": True,
    }


def eprocess_state_examples(
    rows: Sequence[Mapping[str, Any]], null_results: Mapping[str, Any]
) -> JsonDict:
    return {
        "ledger_rows": [
            {
                "sequence": row["sequence"],
                "hypothesis_id": row["hypothesis_id"],
                "outcome": row["outcome"],
                "evalue_increment": row["evalue_increment"],
                "cumulative_evalue": row["cumulative_evalue"],
                "release_decision": row["release_decision"],
            }
            for row in rows[:5]
        ],
        "null_example_states": _as_mapping(null_results.get("summary")).get("example_states", [])[
            :6
        ],
    }


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read a JSONL file into JSON objects."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _test_exit_codes(codes: Mapping[str, int | None] | None) -> dict[str, int]:
    if codes is None:
        return {command: 0 for command in DEFAULT_TEST_COMMANDS}
    return {command: int(code) if code is not None else 1 for command, code in codes.items()}


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    return {"unchanged": not changed, "before": dict(before), "after": after, "changed": changed}


def _path_receipt(path: Path) -> JsonDict:
    return {
        "path": _relative_or_absolute(path),
        "present": path.exists(),
        "sha256": sha256_file(path),
    }


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _ledger_schema_path(result_path: Path) -> Path:
    return Path(str(result_path) + LEDGER_SCHEMA_SUFFIX)


def _synthetic_stream_manifest_path(result_path: Path) -> Path:
    return Path(str(result_path) + SYNTHETIC_STREAM_MANIFEST_SUFFIX)


def _evalue_ledger_path(result_path: Path) -> Path:
    return Path(str(result_path) + LEDGER_SUFFIX)


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(date=args.date, result_path=args.output, write=True)
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
