"""Build the Exp6836 typed obligation program fixture.

Spec refs: REQ-CONSTRAINT-6836, SCENARIO-CONSTRAINT-6836-*,
REQ-CL-6836, SCENARIO-CL-6836-*.

This module compiles Exp6832 obligation atoms into one deterministic program.
Every view reads the same atom ledger. That keeps cost, predicates, guards, and
diagnostics aligned without asking a model to produce or judge text.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from carnot import experiment_6832_operational_obligation_saturation_fixture as exp6832


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTRAINT_SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
CONTINUOUS_LEARNING_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6836_typed_obligation_program_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6836_typed_obligation_program_fixture.py")
OUTPUT_PATH = Path("results/experiment_6836_typed_obligation_program_fixture.json")
SOURCE_PATHS = {
    "exp6811": Path("results/experiment_6811_operational_obligation_automaton_v3.json"),
    "exp6832": Path("results/experiment_6832_operational_obligation_saturation_fixture.json"),
    "exp6835": Path("results/experiment_6835_v598_terminal_evidence_freeze.json"),
}

ARTIFACT_SCHEMA = "carnot.experiment_6836.typed_obligation_program_fixture.v1"
TYPED_PROGRAM_SCHEMA = "carnot.experiment_6836.typed_obligation_program.v1"
INFERENCE_SUBSTRATE = "deterministic CPU compilation"
RANDOM_SEED = 6836
RUN_DATE = "20260901"
ATOM_VALUE_ALLOW = "allow"
ATOM_VALUE_BLOCK = "block"
OBLIGATION_FIELDS = tuple(exp6832.OBLIGATION_FIELDS)
COMPILED_VIEW_NAMES = (
    "scalar_energy",
    "satisfaction_predicate",
    "memory_admission_guard",
    "arc_shadow_action_guard",
    "per_atom_diagnostic",
)
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
EVALUATED_TOKENIZERS = {
    "utf8_byte_v1": "Counts UTF-8 bytes in the fixed candidate sequence.",
    "ascii_char_v1": "Counts ASCII characters in the fixed candidate sequence.",
}
REQUIRED_CASE_KINDS = {
    "atom_contradiction",
    "atom_omission",
    "impossible_set",
    "joint_violation",
}
REQUIRED_MUTATION_CASES = {
    "atom_contradiction",
    "atom_omission",
    "impossible_set",
    "invalid_json",
    "joint_violation",
    "unknown_action",
}

EXPECTED_SOURCE_HASHES = {
    "exp6811": "sha256:8e776a3eb887cb0565af6ba0970c84ab54e2a196f4d1b738515071f8e7b3922f",
    "exp6832": "sha256:05c88b1cc075fc789763e155df1ab67fa5ea5d3845851aea4d4d83f24027993f",
    "exp6835": "sha256:d2a6cf80b19f37aa14ab7151a8e836ad8340f8cb74c2bddca38df6fed27611a2",
}
EXPECTED_SCHEMAS = {
    "exp6811": "carnot.experiment_6811.operational_obligation_automaton_v3.v1",
    "exp6832": exp6832.ARTIFACT_SCHEMA,
    "exp6835": "carnot.experiment_6835.v598_terminal_evidence_freeze.v1",
}
EXPECTED_REPRODUCIBILITY_CHECKSUMS = {
    "exp6811": "sha256:4e77728d02365ce293fa35694186e2f10db512353a4f20867eb2bf5e33115e5b",
    "exp6832": "sha256:a5cf34be7d9041e9cfcc74009cfa765bae2875027f0b48ce3ae3eda3ad54fc99",
    "exp6835": "sha256:80cb64ba726e3bd5e96bec250d2ab31838cee7aadc46f76f975c5530a8119130",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "random_seed",
    "reproducibility_checksum",
    "typed_program_schema",
    "compiled_view_manifest",
    "atom_identity_manifest",
    "compile_parity_results",
    "rows",
    "candidate_pair_manifest",
    "exact_candidate_labels",
    "shortcut_control_manifest",
    "checker_mutation_results",
    "typed_obligation_program_ready_score",
    "obligation_pair_fixture_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "schema": "The schema fixes the artifact contract for downstream readers.",
    "experiment_id": "The identifier binds this record to Exp6836.",
    "run_date": "The date distinguishes this deterministic build from later runs.",
    "status": "The status separates completion from a precondition block.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Source gates stop drift before compilation.",
    "inference_substrate": "The substrate records deterministic CPU compilation and no LLM.",
    "duration_s": "Wall time makes skipped execution visible.",
    "source_artifact_hashes": "Hashes bind Exp6811, Exp6832, and Exp6835 inputs.",
    "implementation_hashes": "Code and spec hashes bind the producer.",
    "random_seed": "The seed fixes row ordering and mutation identities.",
    "reproducibility_checksum": "The checksum binds deterministic content except duration.",
    "typed_program_schema": "The schema names the atom and candidate interfaces.",
    "compiled_view_manifest": "The manifest proves all views share one evaluator.",
    "atom_identity_manifest": "The manifest freezes every atom identity.",
    "compile_parity_results": "Parity proves energy, predicates, guards, and diagnostics agree.",
    "rows": "Rows freeze fixed candidate pairs and controls for later scoring.",
    "candidate_pair_manifest": "The manifest records pair identities and length controls.",
    "exact_candidate_labels": "Labels come from exact checks, not model scores.",
    "shortcut_control_manifest": "Controls expose identifier, order, label, length, and surface cues.",
    "checker_mutation_results": "Mutation cases prove unsafe candidates fail closed.",
    "typed_obligation_program_ready_score": "This readiness field depends only on compile parity.",
    "obligation_pair_fixture_ready_score": "This readiness field depends only on fixture integrity.",
    "gate_check_summary": "The summary names failed or completed gates.",
    "verifier_is_oracle": "False keeps exact checkers external to later learned scores.",
    "verdict_class": "The closed class prevents readiness from becoming a positive result.",
    "honest_verdict": "The terminal verdict is prefixed with complete_.",
}


class ProgramFixtureError(ValueError):
    """Expose one stable error type for malformed deterministic inputs."""


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON once so hashes and raw candidate text stay stable."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed digest format for every artifact hash field."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value through the canonical byte encoding."""

    return sha256_bytes(canonical_bytes(value))


def _sha256_file(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _json_object(raw: bytes) -> JsonDict:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(check: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def check_by_name(checks: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    """Return a named gate row so tests inspect exact observed values."""

    for row in checks:
        if row.get("check") == name:
            return row
    raise ProgramFixtureError(f"missing_check:{name}")


def _read_source(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        return canonical_bytes({"path": str(path), "read_error": type(exc).__name__})


def _source_bytes(repo_root: Path) -> dict[str, bytes]:
    return {name: _read_source(repo_root / path) for name, path in SOURCE_PATHS.items()}


def evaluate_preconditions(source_bytes: Mapping[str, bytes]) -> list[JsonDict]:
    """Evaluate source gates without stopping at the first failure."""

    payloads = {name: _json_object(source_bytes.get(name, b"")) for name in SOURCE_PATHS}
    return (
        [
            _check(f"{name}_file_sha256", EXPECTED_SOURCE_HASHES[name], sha256_bytes(raw))
            for name, raw in source_bytes.items()
            if name in EXPECTED_SOURCE_HASHES
        ]
        + [
            _check(f"{name}_schema", EXPECTED_SCHEMAS[name], payloads[name].get("schema"))
            for name in SOURCE_PATHS
        ]
        + [
            _check(
                f"{name}_reproducibility_checksum",
                EXPECTED_REPRODUCIBILITY_CHECKSUMS[name],
                payloads[name].get("reproducibility_checksum"),
            )
            for name in SOURCE_PATHS
        ]
        + [
            _check(
                "exp6811_operational_automaton_fixture_ready",
                True,
                payloads["exp6811"].get("operational_automaton_fixture_ready"),
            ),
            _check(
                "exp6832_operational_saturation_fixture_ready",
                True,
                payloads["exp6832"].get("operational_saturation_fixture_ready"),
            ),
            _check("exp6832_scenario_count", 150, len(payloads["exp6832"].get("scenarios") or [])),
            _check(
                "v598_evidence_root_ready_score",
                1,
                payloads["exp6835"].get("v598_evidence_root_ready_score"),
            ),
        ]
    )


def _source_identities(source_bytes: Mapping[str, bytes]) -> JsonDict:
    identities: JsonDict = {}
    for name, path in SOURCE_PATHS.items():
        payload = _json_object(source_bytes.get(name, b""))
        identities[name] = {
            "file_sha256": sha256_bytes(source_bytes.get(name, b"")),
            "path": str(path),
            "reproducibility_checksum": payload.get("reproducibility_checksum"),
            "schema": payload.get("schema"),
        }
    identities["exp6811"]["operational_automaton_fixture_ready"] = _json_object(
        source_bytes.get("exp6811", b"")
    ).get("operational_automaton_fixture_ready")
    identities["exp6832"]["operational_saturation_fixture_ready"] = _json_object(
        source_bytes.get("exp6832", b"")
    ).get("operational_saturation_fixture_ready")
    identities["exp6835"]["v598_evidence_root_ready_score"] = _json_object(
        source_bytes.get("exp6835", b"")
    ).get("v598_evidence_root_ready_score")
    return identities


def _load_exp6832_scenarios(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    payload = _json_object(_read_source(repo_root / SOURCE_PATHS["exp6832"]))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ProgramFixtureError("exp6832_scenarios_missing")
    return [deepcopy(row) for row in scenarios if isinstance(row, dict)]


def selected_source_scenarios(repo_root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    """Select compact source cases that cover constructive, joint, and impossible paths."""

    scenarios = _load_exp6832_scenarios(repo_root)

    def pick(name: str, predicate: Any) -> JsonDict:
        for scenario in scenarios:
            if predicate(scenario):
                return deepcopy(scenario)
        raise ProgramFixtureError(f"missing_source_scenario:{name}")

    return {
        "constructive": pick(
            "constructive",
            lambda row: (
                row.get("obligation_count") == 2
                and row.get("semantic_class") == "constructive"
                and row.get("dependency_mode") == "independent"
                and str(row.get("scenario_id", "")).endswith("-p0")
            ),
        ),
        "interacting": pick(
            "interacting",
            lambda row: (
                row.get("obligation_count") == 2
                and row.get("semantic_class") == "constructive"
                and row.get("dependency_mode") == "interacting"
                and str(row.get("scenario_id", "")).endswith("-p0")
            ),
        ),
        "impossible": pick(
            "impossible",
            lambda row: (
                row.get("obligation_count") == 1
                and row.get("semantic_class") == "intentionally_unsatisfiable"
                and str(row.get("scenario_id", "")).endswith("-p0")
            ),
        ),
        "safe_noop": pick(
            "safe_noop",
            lambda row: (
                row.get("obligation_count") == 1
                and row.get("semantic_class") == "safe_no_op"
                and str(row.get("scenario_id", "")).endswith("-p0")
            ),
        ),
    }


def _canonical_scenario(scenario: Mapping[str, Any]) -> JsonDict:
    return {
        **deepcopy(dict(scenario)),
        "candidates": sorted(scenario["candidates"], key=lambda row: row["action_id"]),
        "legal_action_ids": sorted(exp6832.resolve_scenario(scenario)["legal_action_ids"]),
        "obligations": sorted(scenario["obligations"], key=lambda row: row["obligation_id"]),
        "observed_facts": sorted(scenario["observed_facts"]),
    }


def _atom_id(payload: Mapping[str, Any]) -> str:
    return "atom-" + sha256_json(payload).split(":", 1)[1][:24]


def _field_atoms(scenario: Mapping[str, Any]) -> list[JsonDict]:
    atoms: list[JsonDict] = []
    for obligation in sorted(scenario["obligations"], key=lambda row: row["obligation_id"]):
        for field in OBLIGATION_FIELDS:
            payload = {
                "field": field,
                "obligation_id": obligation["obligation_id"],
                "scenario_id": scenario["scenario_id"],
                "source": obligation["contract"][field],
                "target_action": obligation["action"],
            }
            atoms.append(
                {
                    "atom_id": _atom_id(payload),
                    "field": field,
                    "kind": "field",
                    "obligation_id": obligation["obligation_id"],
                    "source_hash": sha256_json(payload),
                    "weight": 1,
                }
            )
    return atoms


def _joint_atom(scenario: Mapping[str, Any]) -> JsonDict:
    payload = {
        "field": "joint_action_set",
        "legal_action_ids": sorted(exp6832.resolve_scenario(scenario)["legal_action_ids"]),
        "scenario_id": scenario["scenario_id"],
    }
    return {
        "atom_id": _atom_id(payload),
        "field": "joint_action_set",
        "kind": "joint",
        "obligation_id": "__joint__",
        "source_hash": sha256_json(payload),
        "weight": 1,
    }


def _parse_failure(code: str) -> JsonDict:
    return {"error": code, "parsed": False, "value": {}}


def parse_candidate_text(text: str | bytes) -> JsonDict:
    """Parse fixed candidate JSON without repair, extraction, or label lookup."""

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            return _parse_failure("invalid_utf8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return _parse_failure("invalid_json")
    if not isinstance(value, dict):
        return _parse_failure("invalid_candidate_fields")
    expected = {
        "atom_values",
        "candidate_id",
        "padding_control",
        "scenario_id",
        "selected_action_ids",
        "surface_form",
    }
    if set(value) != expected:
        return _parse_failure("invalid_candidate_fields")
    if canonical_bytes(value).decode("ascii") != text:
        return _parse_failure("non_canonical_json")
    action_ids = value["selected_action_ids"]
    if not isinstance(action_ids, list):
        return _parse_failure("invalid_action_list")
    if any(not isinstance(item, str) or not item for item in action_ids):
        return _parse_failure("invalid_action_id")
    if len(action_ids) != len(set(action_ids)):
        return _parse_failure("duplicate_action_id")
    if not isinstance(value["atom_values"], dict):
        return _parse_failure("invalid_atom_values")
    if not isinstance(value["candidate_id"], str) or not value["candidate_id"]:
        return _parse_failure("invalid_candidate_id")
    if not isinstance(value["padding_control"], str):
        return _parse_failure("invalid_padding_control")
    if not isinstance(value["scenario_id"], str) or not value["scenario_id"]:
        return _parse_failure("invalid_scenario_id")
    if not isinstance(value["surface_form"], str) or not value["surface_form"]:
        return _parse_failure("invalid_surface_form")
    return {"error": None, "parsed": True, "value": value}


def render_candidate_text(candidate: Mapping[str, Any]) -> str:
    """Render candidate payloads as the exact fixed sequence later scorers read."""

    return canonical_bytes(dict(candidate)).decode("ascii")


def evaluated_token_lengths(text: str) -> JsonDict:
    """Return deterministic length controls for the tokenizers this fixture evaluates."""

    return {
        "ascii_char_v1": len(text),
        "utf8_byte_v1": len(text.encode("utf-8")),
    }


class TypedObligationProgram:
    """One compiled atom ledger with views derived from one evaluator."""

    compiled_view_names = COMPILED_VIEW_NAMES

    def __init__(self, scenario: Mapping[str, Any]) -> None:
        self.scenario = _canonical_scenario(scenario)
        self.scenario_id = str(self.scenario["scenario_id"])
        self.legal_action_ids = tuple(sorted(self.scenario["legal_action_ids"]))
        self.field_atoms = _field_atoms(self.scenario)
        self.joint_atom = _joint_atom(self.scenario)
        self.atom_ledger = tuple([*self.field_atoms, self.joint_atom])
        self.atom_ids = [row["atom_id"] for row in self.atom_ledger]
        self.program_id = "program-" + sha256_json(self.program_source()).split(":", 1)[1][:16]

    @classmethod
    def compile(cls, scenario: Mapping[str, Any]) -> TypedObligationProgram:
        return cls(scenario)

    def program_source(self) -> JsonDict:
        return {
            "candidates": self.scenario["candidates"],
            "fail_closed_action_id": self.scenario["fail_closed_action_id"],
            "legal_action_ids": list(self.legal_action_ids),
            "obligations": self.scenario["obligations"],
            "observed_facts": self.scenario["observed_facts"],
            "scenario_id": self.scenario_id,
        }

    def compatible_atom_values(self) -> dict[str, str]:
        return {atom_id: ATOM_VALUE_ALLOW for atom_id in self.atom_ids}

    def atom_identity_manifest(self) -> JsonDict:
        return {
            "atom_count": len(self.atom_ids),
            "atom_ids": list(self.atom_ids),
            "field_atom_count": len(self.field_atoms),
            "joint_atom_id": self.joint_atom["atom_id"],
            "program_id": self.program_id,
            "scenario_hash": self.scenario.get("scenario_hash"),
            "scenario_id": self.scenario_id,
        }

    def _view_atom_identities(self) -> JsonDict:
        views = {name: list(self.atom_ids) for name in COMPILED_VIEW_NAMES}
        views["energy"] = list(self.atom_ids)
        views["all_views_equal"] = all(views[name] == self.atom_ids for name in COMPILED_VIEW_NAMES)
        return views

    def _parse_failure_evaluation(self, code: str) -> JsonDict:
        diagnostics = [
            {
                "actual_pass": False,
                "assertion_value": None,
                "atom_id": atom["atom_id"],
                "cause": "parse_failure",
                "field": atom["field"],
                "kind": atom["kind"],
                "obligation_id": atom["obligation_id"],
                "passed": False,
            }
            for atom in self.atom_ledger
        ]
        return self._evaluation(
            candidate_id=None,
            diagnostics=diagnostics,
            parse_error=code,
            selected_action_ids=[],
        )

    def evaluate_candidate(self, text: str | bytes) -> JsonDict:
        parsed = parse_candidate_text(text)
        if not parsed["parsed"]:
            return self._parse_failure_evaluation(str(parsed["error"]))
        value = parsed["value"]
        if value["scenario_id"] != self.scenario_id:
            return self._parse_failure_evaluation("scenario_id_mismatch")
        selected = tuple(sorted(value["selected_action_ids"]))
        atom_values = {str(key): val for key, val in value["atom_values"].items()}
        action_ids = {row["action_id"] for row in self.scenario["candidates"]}
        unknown_actions = sorted(set(selected) - action_ids)
        extra_atom_ids = sorted(set(atom_values) - set(self.atom_ids))
        diagnostics = [
            self._field_diagnostic(atom, atom_values, set(selected)) for atom in self.field_atoms
        ]
        diagnostics.append(
            self._joint_diagnostic(
                atom_values,
                selected,
                unknown_actions=unknown_actions,
                extra_atom_ids=extra_atom_ids,
            )
        )
        return self._evaluation(
            candidate_id=str(value["candidate_id"]),
            diagnostics=diagnostics,
            parse_error=None,
            selected_action_ids=list(selected),
        )

    def _field_diagnostic(
        self,
        atom: Mapping[str, Any],
        atom_values: Mapping[str, Any],
        selected: set[str],
    ) -> JsonDict:
        check = exp6832.check_obligation(
            self.scenario,
            sorted(selected),
            str(atom["obligation_id"]),
        )
        actual_pass = bool((check.get("fields") or {}).get(atom["field"]))
        return self._atom_diagnostic(atom, atom_values, actual_pass, None)

    def _joint_diagnostic(
        self,
        atom_values: Mapping[str, Any],
        selected: Sequence[str],
        *,
        unknown_actions: Sequence[str],
        extra_atom_ids: Sequence[str],
    ) -> JsonDict:
        actual_pass = tuple(sorted(selected)) == self.legal_action_ids
        cause = None
        if unknown_actions:
            actual_pass = False
            cause = "unknown_action"
        elif extra_atom_ids:
            actual_pass = False
            cause = "atom_identity_drift"
        elif not actual_pass:
            cause = "joint_action_set_mismatch"
        return self._atom_diagnostic(self.joint_atom, atom_values, actual_pass, cause)

    def _atom_diagnostic(
        self,
        atom: Mapping[str, Any],
        atom_values: Mapping[str, Any],
        actual_pass: bool,
        forced_cause: str | None,
    ) -> JsonDict:
        atom_id = str(atom["atom_id"])
        value = atom_values.get(atom_id)
        cause = forced_cause
        passed = False
        if atom_id not in atom_values:
            cause = "atom_omission"
        elif value not in {ATOM_VALUE_ALLOW, ATOM_VALUE_BLOCK}:
            cause = "invalid_atom_value"
        elif not actual_pass:
            cause = cause or "field_violation"
        elif value != ATOM_VALUE_ALLOW:
            cause = "atom_contradiction"
        else:
            passed = True
        return {
            "actual_pass": actual_pass,
            "assertion_value": value,
            "atom_id": atom_id,
            "cause": cause,
            "field": atom["field"],
            "kind": atom["kind"],
            "obligation_id": atom["obligation_id"],
            "passed": passed,
        }

    def _evaluation(
        self,
        *,
        candidate_id: str | None,
        diagnostics: Sequence[Mapping[str, Any]],
        parse_error: str | None,
        selected_action_ids: Sequence[str],
    ) -> JsonDict:
        energy = sum(1 for row in diagnostics if row.get("passed") is not True)
        satisfied = energy == 0
        return {
            "arc_shadow_action_guard": satisfied,
            "candidate_id": candidate_id,
            "diagnostics": [deepcopy(dict(row)) for row in diagnostics],
            "energy": energy,
            "memory_admission_guard": satisfied,
            "parse_error": parse_error,
            "satisfaction_predicate": satisfied,
            "selected_action_ids": list(selected_action_ids),
            "view_atom_identities": self._view_atom_identities(),
        }


def diagnostic_causes(evaluation: Mapping[str, Any]) -> set[str]:
    """Return nonempty diagnostic causes for concise failure assertions."""

    return {
        str(row["cause"])
        for row in evaluation.get("diagnostics", [])
        if isinstance(row, Mapping) and row.get("cause")
    }


def _candidate_payload(
    program: TypedObligationProgram,
    *,
    atom_values: Mapping[str, str] | None,
    candidate_id: str,
    selected_action_ids: Sequence[str] | None,
    surface_form: str,
) -> JsonDict:
    return {
        "atom_values": dict(
            atom_values if atom_values is not None else program.compatible_atom_values()
        ),
        "candidate_id": candidate_id,
        "padding_control": "",
        "scenario_id": program.scenario_id,
        "selected_action_ids": list(
            selected_action_ids if selected_action_ids is not None else program.legal_action_ids
        ),
        "surface_form": surface_form,
    }


def _balance_pair(left: JsonDict, right: JsonDict) -> tuple[JsonDict, JsonDict]:
    balanced_left = deepcopy(left)
    balanced_right = deepcopy(right)
    balanced_left["padding_control"] = ""
    balanced_right["padding_control"] = ""
    left_len = len(render_candidate_text(balanced_left))
    right_len = len(render_candidate_text(balanced_right))
    if left_len > right_len:
        balanced_right["padding_control"] = "x" * (left_len - right_len)
    elif right_len > left_len:
        balanced_left["padding_control"] = "x" * (right_len - left_len)
    if len(render_candidate_text(balanced_left)) != len(
        render_candidate_text(balanced_right)
    ):  # pragma: no cover
        raise ProgramFixtureError("candidate_length_balance_failed")
    return balanced_left, balanced_right


def _fixed_sequence_input(prompt_text: str, candidate_text: str) -> str:
    return f"{prompt_text}\nCANDIDATE_TEXT_BEGIN\n{candidate_text}\nCANDIDATE_TEXT_END"


def _candidate_record(
    program: TypedObligationProgram,
    prompt_text: str,
    payload: Mapping[str, Any],
) -> JsonDict:
    raw_text = render_candidate_text(payload)
    exact = program.evaluate_candidate(raw_text)
    return {
        "candidate_id": payload["candidate_id"],
        "exact_check": exact,
        "expected_tokenization_inputs": {
            "candidate_text": raw_text,
            "fixed_sequence_text": _fixed_sequence_input(prompt_text, raw_text),
            "prompt_text": prompt_text,
        },
        "prompt_length": len(prompt_text.encode("utf-8")),
        "prompt_token_lengths": evaluated_token_lengths(prompt_text),
        "raw_text": raw_text,
        "raw_text_sha256": sha256_bytes(raw_text.encode("utf-8")),
        "selected_action_ids": list(payload["selected_action_ids"]),
        "surface_form": payload["surface_form"],
        "token_lengths": evaluated_token_lengths(raw_text),
    }


def _nonlegal_action(scenario: Mapping[str, Any], legal: Sequence[str]) -> str:
    for action in scenario["candidates"]:
        if action["action_id"] not in legal:
            return str(action["action_id"])
    raise ProgramFixtureError("nonlegal_action_missing")


def _pair_specs(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    scenarios = selected_source_scenarios(repo_root)
    constructive = TypedObligationProgram.compile(scenarios["constructive"])
    safe = TypedObligationProgram.compile(scenarios["safe_noop"])
    interacting = TypedObligationProgram.compile(scenarios["interacting"])
    impossible = TypedObligationProgram.compile(scenarios["impossible"])

    contradiction_atoms = constructive.compatible_atom_values()
    contradiction_atoms[constructive.field_atoms[0]["atom_id"]] = ATOM_VALUE_BLOCK
    omitted_atoms = safe.compatible_atom_values()
    omitted_atoms.pop(safe.field_atoms[0]["atom_id"])
    joint_selected = [
        *interacting.legal_action_ids,
        _nonlegal_action(interacting.scenario, interacting.legal_action_ids),
    ]
    impossible_target = impossible.scenario["obligations"][0]["action"]["action_id"]

    return [
        {
            "case_kind": "atom_contradiction",
            "negative_atom_values": contradiction_atoms,
            "negative_selected_action_ids": list(constructive.legal_action_ids),
            "program": constructive,
            "surface_form": "compact_json",
        },
        {
            "case_kind": "atom_omission",
            "negative_atom_values": omitted_atoms,
            "negative_selected_action_ids": list(safe.legal_action_ids),
            "program": safe,
            "surface_form": "flat_json",
        },
        {
            "case_kind": "joint_violation",
            "negative_atom_values": interacting.compatible_atom_values(),
            "negative_selected_action_ids": joint_selected,
            "program": interacting,
            "surface_form": "compact_json",
        },
        {
            "case_kind": "impossible_set",
            "negative_atom_values": impossible.compatible_atom_values(),
            "negative_selected_action_ids": [impossible_target],
            "program": impossible,
            "surface_form": "flat_json",
        },
    ]


def build_candidate_rows(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Freeze compatible and unsafe candidate pairs with label-swap rows."""

    rows: list[JsonDict] = []
    for index, spec in enumerate(_pair_specs(repo_root)):
        program: TypedObligationProgram = spec["program"]
        prompt_key = "typed" if spec["surface_form"] == "compact_json" else "compressed"
        prompt_text = program.scenario["prompts"][prompt_key]
        compatible_id = f"cand-6836-{index:02d}-a"
        negative_id = f"cand-6836-{index:02d}-b"
        compatible = _candidate_payload(
            program,
            atom_values=program.compatible_atom_values(),
            candidate_id=compatible_id,
            selected_action_ids=program.legal_action_ids,
            surface_form=spec["surface_form"],
        )
        negative = _candidate_payload(
            program,
            atom_values=spec["negative_atom_values"],
            candidate_id=negative_id,
            selected_action_ids=spec["negative_selected_action_ids"],
            surface_form=spec["surface_form"],
        )
        compatible, negative = _balance_pair(compatible, negative)
        records = {
            compatible_id: _candidate_record(program, prompt_text, compatible),
            negative_id: _candidate_record(program, prompt_text, negative),
        }
        orders = (
            ("canonical", [compatible_id, negative_id]),
            ("swapped", [negative_id, compatible_id]),
        )
        for label_swap, candidate_order in orders:
            row = {
                "case_kind": spec["case_kind"],
                "candidate_order": candidate_order,
                "candidates": [deepcopy(records[candidate_id]) for candidate_id in candidate_order],
                "label_swap": label_swap,
                "pair_id": f"pair-6836-{index:02d}",
                "pair_token_length_equal": records[compatible_id]["token_lengths"]
                == records[negative_id]["token_lengths"],
                "program_id": program.program_id,
                "prompt_length": len(prompt_text.encode("utf-8")),
                "prompt_sha256": sha256_bytes(prompt_text.encode("utf-8")),
                "prompt_text": prompt_text,
                "row_id": f"row-6836-{len(rows):03d}",
                "row_order": len(rows),
                "scenario_id": program.scenario_id,
                "surface_form": spec["surface_form"],
            }
            row["row_hash"] = sha256_json(
                {key: value for key, value in row.items() if key != "row_hash"}
            )
            rows.append(row)
    return rows


def _compiled_view_manifest(programs: Sequence[TypedObligationProgram]) -> JsonDict:
    return {
        "compiled_program_count": len(programs),
        "compiled_views": list(COMPILED_VIEW_NAMES),
        "evaluated_tokenizers": deepcopy(EVALUATED_TOKENIZERS),
        "field_atoms": list(OBLIGATION_FIELDS),
        "joint_atom_present": True,
        "single_evaluation_method": "TypedObligationProgram.evaluate_candidate",
        "single_program_interface": "TypedObligationProgram",
        "views_share_atom_identities": True,
    }


def _atom_identity_manifest(programs: Sequence[TypedObligationProgram]) -> list[JsonDict]:
    return [program.atom_identity_manifest() for program in programs]


def _candidate_pair_manifest(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    canonical_rows = [row for row in rows if row["label_swap"] == "canonical"]
    return [
        {
            "case_kind": row["case_kind"],
            "candidate_ids": list(row["candidate_order"]),
            "label_swap_rows": [
                other["row_id"] for other in rows if other["pair_id"] == row["pair_id"]
            ],
            "pair_id": row["pair_id"],
            "prompt_length": row["prompt_length"],
            "scenario_id": row["scenario_id"],
            "surface_form": row["surface_form"],
            "token_lengths": row["candidates"][0]["token_lengths"],
        }
        for row in canonical_rows
    ]


def _exact_candidate_labels(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    labels: JsonDict = {}
    for row in rows:
        for candidate in row["candidates"]:
            exact = candidate["exact_check"]
            labels[candidate["candidate_id"]] = {
                "compatible": exact["satisfaction_predicate"],
                "energy": exact["energy"],
                "guards": {
                    "arc_shadow_action_guard": exact["arc_shadow_action_guard"],
                    "memory_admission_guard": exact["memory_admission_guard"],
                },
                "label": "compatible" if exact["satisfaction_predicate"] else "incompatible",
            }
    return labels


def _shortcut_control_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    candidate_ids = [candidate["candidate_id"] for row in rows for candidate in row["candidates"]]
    prompt_lengths = [row["prompt_length"] for row in rows]
    return {
        "candidate_identifier_control": {
            "all_ids_equal_length": len({len(item) for item in candidate_ids}) == 1,
            "candidate_id_count": len(set(candidate_ids)),
        },
        "evaluated_tokenizers": deepcopy(EVALUATED_TOKENIZERS),
        "label_swap_control": sorted({row["label_swap"] for row in rows}),
        "prompt_length_control": {
            "lengths_by_row": {row["row_id"]: row["prompt_length"] for row in rows},
            "positive_with_negative_same_prompt": all(
                len({candidate["prompt_length"] for candidate in row["candidates"]}) == 1
                for row in rows
            ),
            "unique_prompt_lengths": sorted(set(prompt_lengths)),
        },
        "row_order_control": {
            "row_orders": [row["row_order"] for row in rows],
            "strictly_increasing": [row["row_order"] for row in rows] == list(range(len(rows))),
        },
        "surface_form_control": sorted({row["surface_form"] for row in rows}),
        "token_length_control": {
            "all_pairs_equal": all(row["pair_token_length_equal"] for row in rows),
            "tokenizers": sorted(EVALUATED_TOKENIZERS),
        },
    }


def _row_fixture_integrity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    labels = _exact_candidate_labels(rows)
    no_model_scores = "model_score" not in json.dumps(rows, sort_keys=True)
    no_generated_answer = candidate_fixture_generated_answer_absent(rows)["passed"]
    return {
        "all_candidate_checks_present": all(
            "exact_check" in candidate for row in rows for candidate in row["candidates"]
        ),
        "all_pairs_equal_token_length": all(
            row.get("pair_token_length_equal") is True for row in rows
        ),
        "case_kinds_present": sorted({str(row.get("case_kind")) for row in rows}),
        "label_swap_values": sorted({str(row.get("label_swap")) for row in rows}),
        "no_generated_answer": no_generated_answer,
        "no_model_scores": no_model_scores,
        "passed": bool(rows)
        and [row.get("row_order") for row in rows] == list(range(len(rows)))
        and REQUIRED_CASE_KINDS.issubset({str(row.get("case_kind")) for row in rows})
        and {str(row.get("label_swap")) for row in rows} == {"canonical", "swapped"}
        and all(row.get("pair_token_length_equal") is True for row in rows)
        and all(value["compatible"] is (value["energy"] == 0) for value in labels.values())
        and no_generated_answer
        and no_model_scores,
        "row_count": len(rows),
    }


def _compile_parity_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_candidate = [
        {
            "arc_shadow_action_guard": candidate["exact_check"]["arc_shadow_action_guard"],
            "candidate_id": candidate["candidate_id"],
            "diagnostic_atom_ids_match": [
                row["atom_id"] for row in candidate["exact_check"]["diagnostics"]
            ]
            == candidate["exact_check"]["view_atom_identities"]["energy"],
            "energy": candidate["exact_check"]["energy"],
            "memory_admission_guard": candidate["exact_check"]["memory_admission_guard"],
            "satisfaction_predicate": candidate["exact_check"]["satisfaction_predicate"],
            "view_atom_identities_equal": candidate["exact_check"]["view_atom_identities"][
                "all_views_equal"
            ],
        }
        for row in rows
        for candidate in row["candidates"]
    ]
    return {
        "all_candidates_exactly_checked": bool(per_candidate)
        and all(item["diagnostic_atom_ids_match"] for item in per_candidate),
        "all_views_share_atom_identities": bool(per_candidate)
        and all(item["view_atom_identities_equal"] for item in per_candidate),
        "candidate_count": len({item["candidate_id"] for item in per_candidate}),
        "energy_zero_matches_satisfaction": all(
            (item["energy"] == 0) is item["satisfaction_predicate"] for item in per_candidate
        ),
        "per_candidate": per_candidate,
        "satisfaction_matches_guards": all(
            item["satisfaction_predicate"] is item["memory_admission_guard"]
            and item["satisfaction_predicate"] is item["arc_shadow_action_guard"]
            for item in per_candidate
        ),
        "view_count": len(COMPILED_VIEW_NAMES),
    }


def candidate_fixture_generated_answer_absent(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reject candidate fixtures that carry generated-answer keys or score text."""

    violations: list[JsonDict] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in {"generated_answer", "generated_text"}:
                    violations.append({"path": child_path, "reason": key_text})
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str) and '"generated_answer"' in value:
            violations.append({"path": path, "reason": "generated_answer_text"})

    visit(rows, "rows")
    return {
        "check": "candidate_fixture_generated_answer_absent",
        "expected": True,
        "observed": {"violation_count": len(violations), "violations": violations[:5]},
        "passed": not violations,
    }


def _mutation_result(case: str, evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "case": case,
        "causes": sorted(diagnostic_causes(evaluation)),
        "energy": evaluation["energy"],
        "expected_rejected": True,
        "observed_rejected": evaluation["satisfaction_predicate"] is False,
    }


def checker_mutation_results(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Run one mutation for each fail-closed class required by the fixture."""

    scenarios = selected_source_scenarios(repo_root)
    constructive = TypedObligationProgram.compile(scenarios["constructive"])
    impossible = TypedObligationProgram.compile(scenarios["impossible"])
    base = _candidate_payload(
        constructive,
        atom_values=constructive.compatible_atom_values(),
        candidate_id="mutation-constructive-a",
        selected_action_ids=constructive.legal_action_ids,
        surface_form="compact_json",
    )

    contradiction = deepcopy(base)
    contradiction["atom_values"][constructive.field_atoms[0]["atom_id"]] = ATOM_VALUE_BLOCK
    omission = deepcopy(base)
    omission["atom_values"].pop(constructive.field_atoms[0]["atom_id"])
    unknown = deepcopy(base)
    unknown["selected_action_ids"] = [*constructive.legal_action_ids, "unknown-action"]
    joint = deepcopy(base)
    joint["selected_action_ids"] = [
        *constructive.legal_action_ids,
        _nonlegal_action(constructive.scenario, constructive.legal_action_ids),
    ]
    impossible_payload = _candidate_payload(
        impossible,
        atom_values=impossible.compatible_atom_values(),
        candidate_id="mutation-impossible-a",
        selected_action_ids=[impossible.scenario["obligations"][0]["action"]["action_id"]],
        surface_form="compact_json",
    )
    return [
        _mutation_result(
            "atom_contradiction",
            constructive.evaluate_candidate(render_candidate_text(contradiction)),
        ),
        _mutation_result(
            "atom_omission",
            constructive.evaluate_candidate(render_candidate_text(omission)),
        ),
        _mutation_result(
            "impossible_set",
            impossible.evaluate_candidate(render_candidate_text(impossible_payload)),
        ),
        _mutation_result("invalid_json", constructive.evaluate_candidate("not-json")),
        _mutation_result(
            "joint_violation",
            constructive.evaluate_candidate(render_candidate_text(joint)),
        ),
        _mutation_result(
            "unknown_action",
            constructive.evaluate_candidate(render_candidate_text(unknown)),
        ),
    ]


def _implementation_hashes(root: Path) -> JsonDict:
    return {
        "constraint_spec": {
            "path": str(CONSTRAINT_SPEC_PATH),
            "sha256": _sha256_file(root / CONSTRAINT_SPEC_PATH),
        },
        "continuous_learning_spec": {
            "path": str(CONTINUOUS_LEARNING_SPEC_PATH),
            "sha256": _sha256_file(root / CONTINUOUS_LEARNING_SPEC_PATH),
        },
        "module": {"path": str(MODULE_PATH), "sha256": _sha256_file(root / MODULE_PATH)},
        "wrapper": {"path": str(WRAPPER_PATH), "sha256": _sha256_file(root / WRAPPER_PATH)},
    }


def _typed_program_schema() -> JsonDict:
    return {
        "atom_values": [ATOM_VALUE_ALLOW, ATOM_VALUE_BLOCK],
        "candidate_input_schema": {
            "atom_values": "mapping atom_id to allow or block",
            "candidate_id": "string",
            "padding_control": "string used only for length balancing",
            "scenario_id": "string",
            "selected_action_ids": ["string"],
            "surface_form": "string",
        },
        "compiled_views": list(COMPILED_VIEW_NAMES),
        "field_atoms": list(OBLIGATION_FIELDS),
        "joint_atom": "joint_action_set",
        "schema": TYPED_PROGRAM_SCHEMA,
        "source_obligation_schema": exp6832.OBLIGATION_SCHEMA,
    }


def _base_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    duration_s: float,
    implementations: Mapping[str, Any],
    run_date: str,
    sources: Mapping[str, Any],
) -> JsonDict:
    failed = next((row for row in checks if row.get("passed") is not True), None)
    gate = (
        {
            "expected": failed.get("expected"),
            "failed_check": failed.get("check"),
            "observed": failed.get("observed"),
            "passed": False,
        }
        if failed
        else {"expected": True, "failed_check": None, "observed": True, "passed": True}
    )
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "6836",
        "run_date": run_date,
        "status": "complete_blocked_typed_obligation_program_fixture",
        "field_principles": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(sources)),
        "implementation_hashes": deepcopy(dict(implementations)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "typed_program_schema": _typed_program_schema(),
        "compiled_view_manifest": {},
        "atom_identity_manifest": [],
        "compile_parity_results": {},
        "rows": [],
        "candidate_pair_manifest": [],
        "exact_candidate_labels": {},
        "shortcut_control_manifest": {},
        "checker_mutation_results": [],
        "typed_obligation_program_ready_score": 0,
        "obligation_pair_fixture_ready_score": 0,
        "gate_check_summary": gate,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_typed_obligation_program_fixture",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic artifact content while excluding measured duration."""

    payload = deepcopy(dict(artifact))
    payload["duration_s"] = 0.0
    payload["reproducibility_checksum"] = ""
    return sha256_json(payload)


def _finish(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = deepcopy(FIELD_PRINCIPLES)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    source_bytes: Mapping[str, bytes] | None = None,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the ready fixture, or stop with an exact blocked artifact."""

    sources_raw = dict(source_bytes) if source_bytes is not None else _source_bytes(repo_root)
    checks = evaluate_preconditions(sources_raw)
    sources = _source_identities(sources_raw)
    implementations = _implementation_hashes(repo_root)
    artifact = _base_artifact(
        checks=checks,
        duration_s=duration_s,
        implementations=implementations,
        run_date=run_date,
        sources=sources,
    )
    if not all(row["passed"] for row in checks):
        return _finish(artifact)

    rows = (
        [deepcopy(dict(row)) for row in candidate_rows]
        if candidate_rows is not None
        else build_candidate_rows(repo_root)
    )
    candidate_check = candidate_fixture_generated_answer_absent(rows)
    artifact["preconditions_checked"].append(candidate_check)
    if not candidate_check["passed"]:
        artifact["gate_check_summary"] = {
            "expected": candidate_check["expected"],
            "failed_check": candidate_check["check"],
            "observed": candidate_check["observed"],
            "passed": False,
        }
        return _finish(artifact)

    program_ids = sorted({row["program_id"] for row in rows})
    programs_by_id = {
        TypedObligationProgram.compile(row).program_id: TypedObligationProgram.compile(row)
        for row in selected_source_scenarios(repo_root).values()
    }
    all_programs = sorted(
        programs_by_id.values(),
        key=lambda program: program.program_id,
    )
    row_integrity = _row_fixture_integrity(rows)
    parity = _compile_parity_results(rows)
    mutations = checker_mutation_results(repo_root)
    mutation_ready = {row["case"] for row in mutations} == REQUIRED_MUTATION_CASES and all(
        row["observed_rejected"] for row in mutations
    )
    program_ready = (
        parity["all_views_share_atom_identities"]
        and parity["energy_zero_matches_satisfaction"]
        and parity["satisfaction_matches_guards"]
        and parity["all_candidates_exactly_checked"]
    )
    fixture_ready = row_integrity["passed"] and mutation_ready
    ready = program_ready and fixture_ready
    artifact.update(
        {
            "status": "complete",
            "compiled_view_manifest": _compiled_view_manifest(all_programs),
            "atom_identity_manifest": _atom_identity_manifest(all_programs),
            "compile_parity_results": parity,
            "rows": rows,
            "candidate_pair_manifest": _candidate_pair_manifest(rows),
            "exact_candidate_labels": _exact_candidate_labels(rows),
            "shortcut_control_manifest": {
                **_shortcut_control_manifest(rows),
                "fixture_integrity": row_integrity,
            },
            "checker_mutation_results": mutations,
            "typed_obligation_program_ready_score": 1 if program_ready else 0,
            "obligation_pair_fixture_ready_score": 1 if fixture_ready else 0,
            "gate_check_summary": {
                "expected": {
                    "fixture_ready": True,
                    "mutation_ready": True,
                    "program_ready": True,
                },
                "failed_check": None if ready else "typed_program_or_pair_fixture_integrity",
                "observed": {
                    "fixture_ready": fixture_ready,
                    "mutation_ready": mutation_ready,
                    "program_ready": program_ready,
                },
                "passed": ready,
            },
            "verdict_class": "null" if ready else "partial",
            "honest_verdict": (
                "complete_null_typed_obligation_program_fixture_ready_no_model_scores"
                if ready
                else "complete_partial_typed_obligation_program_fixture_incomplete_no_model_scores"
            ),
        }
    )
    return _finish(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate ready and blocked terminal artifacts."""

    errors: list[str] = []
    if set(artifact) != set(FIELD_PRINCIPLES):
        errors.append("top-level fields differ from the declared contract")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be a nonnegative number")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict must start with complete_")

    blocked = artifact.get("status") == "complete_blocked_typed_obligation_program_fixture"
    if blocked:
        if artifact.get("rows") != []:
            errors.append("blocked artifact emitted rows")
        if artifact.get("typed_obligation_program_ready_score") != 0:
            errors.append("blocked artifact has typed readiness")
        if artifact.get("obligation_pair_fixture_ready_score") != 0:
            errors.append("blocked artifact has pair readiness")
        if not (artifact.get("gate_check_summary") or {}).get("failed_check"):
            errors.append("blocked artifact lacks failed gate")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked terminal verdict mismatch")
        return errors

    if artifact.get("status") != "complete":
        errors.append("status must be complete or complete_blocked")
    if artifact.get("typed_obligation_program_ready_score") != 1:
        errors.append("typed program readiness must be 1")
    if artifact.get("obligation_pair_fixture_ready_score") != 1:
        errors.append("pair fixture readiness must be 1")
    if (artifact.get("gate_check_summary") or {}).get("passed") is not True:
        errors.append("ready artifact has failed gate")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("ready artifact row count mismatch")
    elif not _row_fixture_integrity(rows)["passed"]:
        errors.append("ready artifact pair integrity failed")
    parity = artifact.get("compile_parity_results") or {}
    if not all(
        parity.get(key) is True
        for key in (
            "all_candidates_exactly_checked",
            "all_views_share_atom_identities",
            "energy_zero_matches_satisfaction",
            "satisfaction_matches_guards",
        )
    ):
        errors.append("ready artifact compile parity failed")
    mutations = artifact.get("checker_mutation_results") or []
    if {row.get("case") for row in mutations} != REQUIRED_MUTATION_CASES or not all(
        row.get("observed_rejected") is True for row in mutations
    ):
        errors.append("checker mutation results incomplete")
    return errors


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(artifact) + b"\n")
    os.replace(temporary, path)


def execute(root: Path, run_date: str, output_path: Path) -> JsonDict:
    """Build, validate, and atomically write the Exp6836 artifact."""

    if re.fullmatch(r"\d{8}", run_date) is None:
        raise ProgramFixtureError("invalid_run_date")
    try:
        datetime.strptime(run_date, "%Y%m%d")
    except ValueError as exc:
        raise ProgramFixtureError("invalid_run_date") from exc
    started = time.perf_counter()
    artifact = build_artifact(repo_root=root, run_date=run_date, duration_s=0.0)
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ProgramFixtureError("invalid_artifact:" + "; ".join(errors))
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_path = Path(args.output)
    try:
        if args.validate and output_path.is_file():
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            return 0
        artifact = execute(REPO_ROOT, args.date, output_path)
    except (ProgramFixtureError, json.JSONDecodeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"artifact": str(output_path), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
