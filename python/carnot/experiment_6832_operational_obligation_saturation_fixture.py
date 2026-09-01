"""Build a frozen operational-obligation saturation fixture.

Spec refs: REQ-CONSTRAINT-6832 and SCENARIO-CONSTRAINT-6832-*.

The fixture varies simultaneous obligation count without calling a model. Its
exact field and joint checks provide external truth for a later experiment.
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
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
EXP6811_PATH = Path("results/experiment_6811_operational_obligation_automaton_v3.json")
EXP6831_PATH = Path("results/experiment_6831_v597_evidence_admissibility_contract.json")
SUPERVISOR_PATH = Path("python/carnot/agentic/arc_trajectory_supervisor.py")
MODULE_PATH = Path("python/carnot/experiment_6832_operational_obligation_saturation_fixture.py")
WRAPPER_PATH = Path(
    "scripts/experiments/experiment_6832_operational_obligation_saturation_fixture.py"
)
OUTPUT_PATH = Path("results/experiment_6832_operational_obligation_saturation_fixture.json")

ARTIFACT_SCHEMA = "carnot.experiment_6832.operational_obligation_saturation_fixture.v1"
OBLIGATION_SCHEMA = "carnot.arc.operational_obligation.v3"
OUTPUT_SCHEMA = {"selected_action_ids": ["string"]}
OBLIGATION_COUNTS = (1, 2, 4, 6, 8)
OBLIGATION_FIELDS = (
    "prerequisite",
    "authority",
    "fallback",
    "execution_consequence",
    "priority",
)
PRIORITY_ORDER = ("hard", "binding", "soft")
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
RANDOM_SEED = 6832
RUN_DATE = "20260901"

EXPECTED_EXP6811_FILE_SHA256 = (
    "sha256:8e776a3eb887cb0565af6ba0970c84ab54e2a196f4d1b738515071f8e7b3922f"
)
EXPECTED_EXP6831_FILE_SHA256 = (
    "sha256:52783c6bc08656c80c658c9027b9f47574024d24d0e716b464822107c153b4f9"
)
EXPECTED_EXP6811_SCHEMA = "carnot.experiment_6811.operational_obligation_automaton_v3.v1"
EXPECTED_EXP6831_SCHEMA = "carnot.experiment_6831.v597_evidence_admissibility_contract.v1"
EXPECTED_EXP6811_CHECKSUM = (
    "sha256:4e77728d02365ce293fa35694186e2f10db512353a4f20867eb2bf5e33115e5b"
)
EXPECTED_EXP6831_CHECKSUM = (
    "sha256:821da65969fec801716a4586f858493c45ff0b5035a9420eba1800be6598dff2"
)
EXPECTED_AUTOMATON_HASH = "sha256:f01bf79bb16d6f0934935dc449574c78f5e2c985839fb4097ee543481c742b00"
EXPECTED_OBLIGATION_SOURCE_HASH = (
    "sha256:9ea1d31de1f699a9955d2bac23e4f8dc3b417d34b29edb854f1b86b24947cd9c"
)
EXPECTED_EXP6811_SOURCE_HASHES = {
    "arc_solve_registry": "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
    "exp6656_frozen_trace_artifact": (
        "sha256:913052fa24153337e8716ee24609bec3466f0e945d8351ccd8cdda77da3d1a2e"
    ),
    "exp6656_module": "sha256:c74bd562ff26af633ecbfed88d6040263b662e73b9a4891155250b11d28c2c95",
    "exp6681_exact_outcome_artifact": (
        "sha256:bf61e50970530a9844103e59f702c68e32a8aba12f78b1c68dd6154685b9107a"
    ),
    "owned_req_agentic_6810_1": (
        "sha256:f9195a0abbe966be5393c7350acebe2a43bbfebad03bc70366889f34dd5caccb"
    ),
    "owned_req_constraint_6810": (
        "sha256:095a5cc146734b0fceaaee36968bab6bb7dbb7e32dee2c64aa1c8d44f53a5fa1"
    ),
    "v1_trace_automaton_supervisor": (
        "sha256:aafb4740f005de2f7b43472ec72a2534e7d8087018c128dcffea6d1444577734"
    ),
    "v595_contract_manifest": (
        "sha256:f3a6b52a526123ad59d69eb9398fbe11278fd7e6798bed06f71e33887ea19687"
    ),
}
EXPECTED_OBLIGATION_SCHEMA = {
    "contract_fields": list(OBLIGATION_FIELDS),
    "record_fields": ["obligation_id", "action", "contract"],
    "schema": OBLIGATION_SCHEMA,
}

FIELD_PRINCIPLES = {
    "schema": "The versioned name fixes the artifact contract.",
    "experiment_id": "The stable identifier binds this result to Exp6832.",
    "run_date": "The supplied date distinguishes this execution from later runs.",
    "status": "The status separates complete generation from a precondition block.",
    "field_principles": "Every top-level field explains why it exists.",
    "preconditions_checked": "Exact source and worktree checks prevent drift from entering the fixture.",
    "inference_substrate": "The substrate states that deterministic CPU code ran without an LLM.",
    "duration_s": "Measured wall time makes skipped execution visible.",
    "random_seed": "The frozen seed binds construction and prompt permutations.",
    "reproducibility_checksum": "The checksum binds code, sources, scenarios, checks, and output.",
    "source_artifact_hashes": "Exact Exp6811 and Exp6831 identities bind the fixture prerequisites.",
    "implementation_hashes": "Code hashes bind the producer and its command wrapper.",
    "obligation_schema": "The schema fixes the five typed operational fields.",
    "obligation_counts": "The count list fixes the simultaneous-constraint saturation axis.",
    "scenario_manifest": "The manifest records every immutable scenario identity and seal.",
    "prompt_arm_manifest": "The prompt manifest binds both equal-information representations.",
    "scenarios": "Every source-free unit carries prompts, obligations, candidates, and exact truth.",
    "checker_manifest": "The manifest binds each obligation field check and every joint check.",
    "checker_mutation_results": "Five candidate cases prove that each checker accepts and rejects as specified.",
    "leakage_audit": "The audit prevents fixture-only truth from entering prompts.",
    "legal_action_headroom": "The receipt proves constructive action headroom where the template requires it.",
    "operational_saturation_fixture_ready": "This exact gate is consumed by Exp6833.",
    "gate_check_summary": "The summary names the first failed gate or all successful readiness gates.",
    "verifier_is_oracle": "False keeps external fixture truth distinct from a learned verifier.",
    "verdict_class": "A closed class prevents readiness from becoming a model-result claim.",
    "honest_verdict": "The terminal complete prefix states fixture readiness only.",
}

FORBIDDEN_PROMPT_TERMS = (
    "answer key",
    "checker",
    "correct action",
    "dependency_mode",
    "expected outcome",
    "hidden outcome",
    "independent",
    "interacting",
    "legal_action_ids",
    "legal_action_set_id",
    "order cue",
    "permutation_id",
    "scenario_id",
    "semantic_class",
    "solution set",
    "template_id",
)


class FixtureError(ValueError):
    """Expose one stable code when deterministic input is malformed."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code if not detail else f"{code}: {detail}")


def canonical_bytes(value: Any) -> bytes:
    """Use one compact JSON encoding for hashes and candidate parsing."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed digest so hash fields have a single format."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value through the canonical byte encoding."""

    return sha256_bytes(canonical_bytes(value))


def _sha256_file(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _json_object(raw: bytes) -> JsonDict | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _check(check: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def evaluate_preconditions(
    exp6811_bytes: bytes,
    exp6831_bytes: bytes,
    *,
    worktree_status: str,
) -> list[JsonDict]:
    """Evaluate every source gate without stopping at the first failure."""

    exp6811 = _json_object(exp6811_bytes) or {}
    exp6831 = _json_object(exp6831_bytes) or {}
    compiler = exp6811.get("compiler_manifest") or {}
    byte_receipts = exp6811.get("canonical_byte_receipts") or {}
    return [
        _check("exp6811_file_sha256", EXPECTED_EXP6811_FILE_SHA256, sha256_bytes(exp6811_bytes)),
        _check("exp6811_schema", EXPECTED_EXP6811_SCHEMA, exp6811.get("schema")),
        _check(
            "exp6811_reproducibility_checksum",
            EXPECTED_EXP6811_CHECKSUM,
            exp6811.get("reproducibility_checksum"),
        ),
        _check("exp6811_fixture_ready", True, exp6811.get("operational_automaton_fixture_ready")),
        _check(
            "exp6811_obligation_schema",
            EXPECTED_OBLIGATION_SCHEMA,
            exp6811.get("obligation_schema"),
        ),
        _check(
            "exp6811_source_artifact_hashes",
            EXPECTED_EXP6811_SOURCE_HASHES,
            exp6811.get("source_artifact_hashes"),
        ),
        _check("exp6811_automaton_hash", EXPECTED_AUTOMATON_HASH, compiler.get("automaton_hash")),
        _check(
            "exp6811_obligation_source_hash",
            EXPECTED_OBLIGATION_SOURCE_HASH,
            byte_receipts.get("obligation_source_hash"),
        ),
        _check("exp6831_file_sha256", EXPECTED_EXP6831_FILE_SHA256, sha256_bytes(exp6831_bytes)),
        _check("exp6831_schema", EXPECTED_EXP6831_SCHEMA, exp6831.get("schema")),
        _check(
            "exp6831_reproducibility_checksum",
            EXPECTED_EXP6831_CHECKSUM,
            exp6831.get("reproducibility_checksum"),
        ),
        _check("v597_contract_ready", True, exp6831.get("v597_contract_ready")),
        _check("owned_source_tree_clean", "", worktree_status),
    ]


def _owned_worktree_status(root: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            str(EXP6811_PATH),
            str(EXP6831_PATH),
            str(SUPERVISOR_PATH),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return f"git_status_error:{completed.returncode}:{completed.stderr.strip()}"
    return completed.stdout


def _read_source_bytes(root: Path, path: Path) -> bytes:
    try:
        return (root / path).read_bytes()
    except OSError as exc:
        return canonical_bytes({"read_error": type(exc).__name__, "path": str(path)})


def _source_identities(exp6811_bytes: bytes, exp6831_bytes: bytes) -> JsonDict:
    exp6811 = _json_object(exp6811_bytes) or {}
    exp6831 = _json_object(exp6831_bytes) or {}
    compiler = exp6811.get("compiler_manifest") or {}
    receipts = exp6811.get("canonical_byte_receipts") or {}
    return {
        "exp6811": {
            "path": str(EXP6811_PATH),
            "file_sha256": sha256_bytes(exp6811_bytes),
            "schema": exp6811.get("schema"),
            "reproducibility_checksum": exp6811.get("reproducibility_checksum"),
            "source_artifact_hashes": exp6811.get("source_artifact_hashes"),
            "automaton_hash": compiler.get("automaton_hash"),
            "obligation_source_hash": receipts.get("obligation_source_hash"),
        },
        "exp6831": {
            "path": str(EXP6831_PATH),
            "file_sha256": sha256_bytes(exp6831_bytes),
            "schema": exp6831.get("schema"),
            "reproducibility_checksum": exp6831.get("reproducibility_checksum"),
            "v597_contract_ready": exp6831.get("v597_contract_ready"),
        },
    }


def _priority_class(index: int) -> str:
    return PRIORITY_ORDER[index % len(PRIORITY_ORDER)]


def _base_template(count: int, template_index: int) -> JsonDict:
    """Create one semantic template before any prompt-order permutation."""

    content_id = f"oc{count}-t{template_index:02d}"
    dependency_mode = "independent" if template_index < 5 else "interacting"
    case_index = template_index % 5
    semantic_class = (
        "safe_no_op"
        if case_index == 0
        else "intentionally_unsatisfiable"
        if case_index == 1
        else "constructive"
    )
    fail_closed_action_id = f"act-{content_id}-fail-closed"
    candidates: list[JsonDict] = [
        {
            "action_id": fail_closed_action_id,
            "authority": "fixture_safety_boundary",
            "consequence": {"add": [], "remove": []},
            "kind": "fail_closed",
            "resource": f"boundary:{content_id}",
        }
    ]
    obligations: list[JsonDict] = []
    observed_facts: list[str] = []
    for index in range(count):
        obligation_id = f"obl-{content_id}-{index:02d}"
        ready_fact = f"ready:{content_id}:{index:02d}"
        done_fact = f"done:{content_id}:{index:02d}"
        target_action_id = f"act-{content_id}-{index:02d}-target"
        fallback_action_id = f"act-{content_id}-{index:02d}-noop"
        decoy_action_id = f"act-{content_id}-{index:02d}-decoy"
        issuer = f"authority:{content_id}:{index:02d}"
        priority_class = _priority_class(index)
        resource = (
            f"resource:{content_id}:{index // 2:02d}"
            if dependency_mode == "interacting"
            else f"resource:{content_id}:{index:02d}"
        )
        consequence = {"add": [done_fact], "remove": [ready_fact]}
        candidate_authority = (
            "spoofed_authority"
            if semantic_class == "intentionally_unsatisfiable" and index == 0
            else issuer
        )
        candidates.extend(
            [
                {
                    "action_id": target_action_id,
                    "authority": candidate_authority,
                    "consequence": deepcopy(consequence),
                    "kind": "constructive",
                    "resource": resource,
                },
                {
                    "action_id": fallback_action_id,
                    "authority": "fixture_safety_boundary",
                    "consequence": {"add": [], "remove": []},
                    "kind": "no_op",
                    "resource": f"fallback:{obligation_id}",
                },
                {
                    "action_id": decoy_action_id,
                    "authority": "untrusted_candidate",
                    "consequence": {"add": [f"wrong:{obligation_id}"], "remove": []},
                    "kind": "constructive",
                    "resource": resource,
                },
            ]
        )
        obligations.append(
            {
                "action": {"action_id": target_action_id},
                "contract": {
                    "prerequisite": {"all_of": [ready_fact], "none_of": []},
                    "authority": {"issuer": issuer, "order": index},
                    "fallback": {
                        "action_id": fallback_action_id,
                        "reason": "The prerequisite or higher-priority action prevents execution.",
                    },
                    "execution_consequence": deepcopy(consequence),
                    "priority": {
                        "class": priority_class,
                        "weight": index + 1 if priority_class == "soft" else 0,
                    },
                },
                "obligation_id": obligation_id,
            }
        )
        if semantic_class != "safe_no_op":
            observed_facts.append(ready_fact)
    return {
        "candidate_actions": candidates,
        "content_id": content_id,
        "dependency_mode": dependency_mode,
        "fail_closed_action_id": fail_closed_action_id,
        "obligation_count": count,
        "obligations": obligations,
        "observed_facts": sorted(observed_facts),
        "semantic_class": semantic_class,
        "template_id": f"template-{content_id}",
    }


def _priority_key(obligation: Mapping[str, Any]) -> tuple[int, int, int, str]:
    contract = obligation["contract"]
    priority = contract["priority"]
    return (
        PRIORITY_ORDER.index(priority["class"]),
        contract["authority"]["order"],
        -priority["weight"],
        obligation["obligation_id"],
    )


def resolve_scenario(scenario: Mapping[str, Any]) -> JsonDict:
    """Derive the only action set from the five typed obligation fields."""

    candidates = {row["action_id"]: row for row in scenario["candidates"]}
    facts = set(scenario["observed_facts"])
    active: list[Mapping[str, Any]] = []
    inactive: list[Mapping[str, Any]] = []
    invalid = False
    for obligation in scenario["obligations"]:
        contract = obligation["contract"]
        prerequisite = contract["prerequisite"]
        is_active = set(prerequisite["all_of"]).issubset(facts) and not (
            set(prerequisite["none_of"]) & facts
        )
        (active if is_active else inactive).append(obligation)
        if is_active:
            action_id = obligation["action"]["action_id"]
            action = candidates.get(action_id)
            invalid = (
                invalid
                or action is None
                or action.get("authority") != contract["authority"]["issuer"]
                or action.get("consequence") != contract["execution_consequence"]
            )

    if invalid:
        fail_closed = scenario["fail_closed_action_id"]
        return {
            "legal_action_ids": [fail_closed],
            "obligations": {
                row["obligation_id"]: {
                    "action_id": fail_closed,
                    "active": row in active,
                    "disposition": "fail_closed",
                }
                for row in scenario["obligations"]
            },
        }

    resolutions: dict[str, JsonDict] = {}
    for obligation in inactive:
        resolutions[obligation["obligation_id"]] = {
            "action_id": obligation["contract"]["fallback"]["action_id"],
            "active": False,
            "disposition": "fallback",
        }
    by_resource: dict[str, list[Mapping[str, Any]]] = {}
    for obligation in active:
        action = candidates[obligation["action"]["action_id"]]
        by_resource.setdefault(action["resource"], []).append(obligation)
    for group in by_resource.values():
        ordered = sorted(group, key=_priority_key)
        winner = ordered[0]
        resolutions[winner["obligation_id"]] = {
            "action_id": winner["action"]["action_id"],
            "active": True,
            "disposition": "constructive",
        }
        for obligation in ordered[1:]:
            resolutions[obligation["obligation_id"]] = {
                "action_id": obligation["contract"]["fallback"]["action_id"],
                "active": True,
                "disposition": "preempted",
            }
    return {
        "legal_action_ids": sorted({row["action_id"] for row in resolutions.values()}),
        "obligations": resolutions,
    }


def _permuted_ids(values: Sequence[str], permutation: int) -> list[str]:
    ordered = list(values)
    if permutation == 0:
        return ordered
    if permutation == 1:
        return ordered[1:] + ordered[:1]
    return list(reversed(ordered))


def _normalize_information(value: Mapping[str, Any]) -> JsonDict:
    return {
        "candidate_actions": sorted(value["candidate_actions"], key=lambda row: row["action_id"]),
        "obligations": sorted(value["obligations"], key=lambda row: row["obligation_id"]),
        "observed_facts": sorted(value["observed_facts"]),
        "output_schema": value["output_schema"],
    }


def prompt_information(scenario: Mapping[str, Any]) -> JsonDict:
    """Return the order-free information shared by both prompt arms."""

    return _normalize_information(
        {
            "candidate_actions": scenario["candidates"],
            "obligations": scenario["obligations"],
            "observed_facts": scenario["observed_facts"],
            "output_schema": OUTPUT_SCHEMA,
        }
    )


def _render_prompts(scenario: Mapping[str, Any]) -> dict[str, str]:
    obligation_index = {row["obligation_id"]: row for row in scenario["obligations"]}
    candidate_index = {row["action_id"]: row for row in scenario["candidates"]}
    information = {
        "observed_facts": scenario["observed_facts"],
        "obligations": [obligation_index[item] for item in scenario["obligation_prompt_order"]],
        "candidate_actions": [candidate_index[item] for item in scenario["candidate_prompt_order"]],
        "output_schema": OUTPUT_SCHEMA,
    }
    instruction = (
        "Select the action identifiers that fulfill the operational handoff. "
        "Return only canonical JSON with selected_action_ids."
    )
    typed = (
        "Operational handoff.\nHANDOFF_JSON_BEGIN\n"
        + canonical_bytes(information).decode()
        + "\nHANDOFF_JSON_END\n"
        + instruction
    )
    compressed = (
        "Operational handoff.\n"
        f"OBSERVED_FACTS_JSON={canonical_bytes(information['observed_facts']).decode()}\n"
        f"OBLIGATIONS_JSON={canonical_bytes(information['obligations']).decode()}\n"
        f"CANDIDATE_ACTIONS_JSON={canonical_bytes(information['candidate_actions']).decode()}\n"
        f"OUTPUT_JSON_SCHEMA={canonical_bytes(OUTPUT_SCHEMA).decode()}\n" + instruction
    )
    return {"typed": typed, "compressed": compressed}


def extract_prompt_information(prompt: str) -> JsonDict:
    """Parse either frozen representation back to its shared information."""

    if "HANDOFF_JSON_BEGIN\n" in prompt:
        raw = prompt.split("HANDOFF_JSON_BEGIN\n", 1)[1].split("\nHANDOFF_JSON_END", 1)[0]
        value = json.loads(raw)
    elif "OBSERVED_FACTS_JSON=" in prompt:
        lines = dict(
            line.split("=", 1)
            for line in prompt.splitlines()
            if line.startswith(
                (
                    "OBSERVED_FACTS_JSON=",
                    "OBLIGATIONS_JSON=",
                    "CANDIDATE_ACTIONS_JSON=",
                    "OUTPUT_JSON_SCHEMA=",
                )
            )
        )
        value = {
            "observed_facts": json.loads(lines["OBSERVED_FACTS_JSON"]),
            "obligations": json.loads(lines["OBLIGATIONS_JSON"]),
            "candidate_actions": json.loads(lines["CANDIDATE_ACTIONS_JSON"]),
            "output_schema": json.loads(lines["OUTPUT_JSON_SCHEMA"]),
        }
    else:
        raise FixtureError("unknown_prompt_representation")
    return _normalize_information(value)


def generate_scenarios() -> list[JsonDict]:
    """Freeze 30 scenarios at each count as ten templates by three orders."""

    scenarios: list[JsonDict] = []
    for count in OBLIGATION_COUNTS:
        for template_index in range(10):
            template = _base_template(count, template_index)
            obligation_ids = [row["obligation_id"] for row in template["obligations"]]
            candidate_ids = [row["action_id"] for row in template["candidate_actions"]]
            for permutation in range(3):
                scenario: JsonDict = {
                    "scenario_id": f"scenario-{template['content_id']}-p{permutation}",
                    "template_id": template["template_id"],
                    "permutation_id": "",
                    "obligation_count": count,
                    "dependency_mode": template["dependency_mode"],
                    "semantic_class": template["semantic_class"],
                    "observed_facts": deepcopy(template["observed_facts"]),
                    "obligations": deepcopy(template["obligations"]),
                    "candidates": deepcopy(template["candidate_actions"]),
                    "fail_closed_action_id": template["fail_closed_action_id"],
                    "obligation_prompt_order": _permuted_ids(obligation_ids, permutation),
                    "candidate_prompt_order": _permuted_ids(candidate_ids, permutation),
                    "output_schema": deepcopy(OUTPUT_SCHEMA),
                }
                permutation_payload = {
                    "candidate_order": scenario["candidate_prompt_order"],
                    "obligation_order": scenario["obligation_prompt_order"],
                }
                scenario["permutation_id"] = (
                    "permutation-" + sha256_json(permutation_payload).split(":", 1)[1][:16]
                )
                resolution = resolve_scenario(scenario)
                scenario["legal_action_ids"] = resolution["legal_action_ids"]
                scenario["legal_action_set_id"] = (
                    "action-set-"
                    + sha256_json(resolution["legal_action_ids"]).split(":", 1)[1][:16]
                )
                scenario["prompts"] = _render_prompts(scenario)
                scenario["prompt_information_sha256"] = sha256_json(prompt_information(scenario))
                scenario["scenario_hash"] = sha256_json(scenario)
                scenarios.append(scenario)
    return scenarios


def parse_candidate_response(raw: bytes) -> tuple[str, ...]:
    """Parse one canonical response and normalize identifier order as a set."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError("invalid_utf8") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FixtureError("invalid_json") from exc
    if canonical_bytes(value) != raw:
        raise FixtureError("non_canonical_json")
    if not isinstance(value, dict) or set(value) != {"selected_action_ids"}:
        raise FixtureError("invalid_response_fields")
    action_ids = value["selected_action_ids"]
    if not isinstance(action_ids, list):
        raise FixtureError("invalid_action_list")
    if any(not isinstance(item, str) or not item for item in action_ids):
        raise FixtureError("invalid_action_id")
    if len(action_ids) != len(set(action_ids)):
        raise FixtureError("duplicate_action_id")
    return tuple(sorted(action_ids))


def _action_index(scenario: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["action_id"]: row for row in scenario["candidates"]}


def _expected_obligation(
    scenario: Mapping[str, Any], obligation: Mapping[str, Any]
) -> tuple[JsonDict, Mapping[str, Any] | None]:
    resolution = resolve_scenario(scenario)["obligations"][obligation["obligation_id"]]
    return resolution, _action_index(scenario).get(resolution["action_id"])


def _check_prerequisite(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> bool:
    resolution, _ = _expected_obligation(scenario, obligation)
    return resolution["action_id"] in selected


def _check_authority(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> bool:
    resolution, action = _expected_obligation(scenario, obligation)
    if resolution["disposition"] == "fail_closed":
        return selected == {scenario["fail_closed_action_id"]}
    if action is None or resolution["action_id"] not in selected:
        return False
    if resolution["disposition"] != "constructive":
        return action["authority"] == "fixture_safety_boundary"
    issuer = obligation["contract"]["authority"]["issuer"]
    resource = action["resource"]
    action_index = _action_index(scenario)
    relevant = [action_index[item] for item in selected if item in action_index]
    return action["authority"] == issuer and all(
        row["authority"] == issuer for row in relevant if row["resource"] == resource
    )


def _check_fallback(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> bool:
    resolution, _ = _expected_obligation(scenario, obligation)
    if resolution["disposition"] == "fail_closed":
        return selected == {scenario["fail_closed_action_id"]}
    target = obligation["action"]["action_id"]
    fallback = obligation["contract"]["fallback"]["action_id"]
    if resolution["disposition"] == "constructive":
        return target in selected and fallback not in selected
    return fallback in selected and target not in selected


def _check_execution_consequence(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> bool:
    resolution, action = _expected_obligation(scenario, obligation)
    if resolution["disposition"] == "fail_closed":
        return selected == {scenario["fail_closed_action_id"]}
    if action is None or resolution["action_id"] not in selected:
        return False
    if resolution["disposition"] == "constructive":
        return action["consequence"] == obligation["contract"]["execution_consequence"]
    return action["kind"] == "no_op" and action["consequence"] == {
        "add": [],
        "remove": [],
    }


def _check_priority(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> bool:
    resolution, action = _expected_obligation(scenario, obligation)
    if resolution["disposition"] == "fail_closed":
        return selected == {scenario["fail_closed_action_id"]}
    if action is None or resolution["action_id"] not in selected:
        return False
    target = _action_index(scenario)[obligation["action"]["action_id"]]
    selected_on_resource = {
        row["action_id"]
        for row in scenario["candidates"]
        if row["action_id"] in selected and row["resource"] == target["resource"]
    }
    resolved = resolve_scenario(scenario)["obligations"]
    expected_on_resource = {
        row["action_id"]
        for row in resolved.values()
        if row["disposition"] == "constructive"
        and _action_index(scenario)[row["action_id"]]["resource"] == target["resource"]
    }
    return selected_on_resource == expected_on_resource


FIELD_CHECKERS = {
    "prerequisite": _check_prerequisite,
    "authority": _check_authority,
    "fallback": _check_fallback,
    "execution_consequence": _check_execution_consequence,
    "priority": _check_priority,
}


def check_obligation(
    scenario: Mapping[str, Any],
    selected_action_ids: Sequence[str],
    obligation_id: str,
) -> JsonDict:
    """Run every typed field check for one obligation identifier."""

    obligations = {row["obligation_id"]: row for row in scenario["obligations"]}
    if obligation_id not in obligations:
        return {"fields": {field: False for field in OBLIGATION_FIELDS}, "passed": False}
    obligation = obligations[obligation_id]
    selected = set(selected_action_ids)
    fields = {
        field: FIELD_CHECKERS[field](scenario, selected, obligation) for field in OBLIGATION_FIELDS
    }
    return {"fields": fields, "passed": all(fields.values())}


def check_joint(scenario: Mapping[str, Any], raw: bytes) -> JsonDict:
    """Check all obligations and the one exact action set without repair."""

    try:
        selected = parse_candidate_response(raw)
    except FixtureError as exc:
        return {
            "obligation_checks": {},
            "parse_error": exc.code,
            "parsed": False,
            "passed": False,
            "selected_action_ids": [],
        }
    obligation_checks = {
        row["obligation_id"]: check_obligation(scenario, selected, row["obligation_id"])
        for row in scenario["obligations"]
    }
    expected = tuple(sorted(scenario["legal_action_ids"]))
    passed = selected == expected and all(row["passed"] for row in obligation_checks.values())
    return {
        "obligation_checks": obligation_checks,
        "parse_error": None,
        "parsed": True,
        "passed": passed,
        "selected_action_ids": list(selected),
    }


def _candidate_case_bytes(action_ids: Sequence[str]) -> bytes:
    return canonical_bytes({"selected_action_ids": list(action_ids)})


def run_checker_mutations(scenario: Mapping[str, Any]) -> JsonDict:
    """Exercise legal, violation, omission, conflict, and reorder cases."""

    legal = list(scenario["legal_action_ids"])
    nonlegal = [row["action_id"] for row in scenario["candidates"] if row["action_id"] not in legal]
    violation = sorted({*legal[1:], nonlegal[0]})
    omission = legal[1:]
    conflict = sorted({*legal, nonlegal[0]})
    candidates = {
        "legal": legal,
        "violation": violation,
        "omission": omission,
        "conflict": conflict,
        "reorder": list(reversed(legal)),
    }
    expected = {
        "legal": True,
        "violation": False,
        "omission": False,
        "conflict": False,
        "reorder": True,
    }
    cases: JsonDict = {}
    for case_id, action_ids in candidates.items():
        raw = _candidate_case_bytes(action_ids)
        result = check_joint(scenario, raw)
        cases[case_id] = {
            "candidate_sha256": sha256_bytes(raw),
            "joint_passed": result["passed"],
            "obligation_checks": result["obligation_checks"],
        }
    return {
        "all_expected": all(cases[key]["joint_passed"] is value for key, value in expected.items()),
        "cases": cases,
        "scenario_id": scenario["scenario_id"],
    }


def audit_prompt_leakage(scenarios: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reject fixture-only labels, answer bytes, and unequal prompt information."""

    violations: list[JsonDict] = []
    for scenario in scenarios:
        legal_response = _candidate_case_bytes(scenario["legal_action_ids"]).decode()
        for arm, prompt in scenario["prompts"].items():
            lowered = prompt.lower()
            for term in FORBIDDEN_PROMPT_TERMS:
                if term in lowered:
                    violations.append(
                        {
                            "arm": arm,
                            "reason": f"forbidden_vocabulary:{term}",
                            "scenario_id": scenario["scenario_id"],
                        }
                    )
            if legal_response in prompt:
                violations.append(
                    {
                        "arm": arm,
                        "reason": "exact_response_leakage",
                        "scenario_id": scenario["scenario_id"],
                    }
                )
        try:
            typed = extract_prompt_information(scenario["prompts"]["typed"])
            compressed = extract_prompt_information(scenario["prompts"]["compressed"])
        except (FixtureError, KeyError, TypeError, ValueError):
            typed, compressed = None, None
        if typed != compressed or typed != prompt_information(scenario):
            violations.append(
                {
                    "arm": "pair",
                    "reason": "information_mismatch",
                    "scenario_id": scenario["scenario_id"],
                }
            )
    return {
        "equal_information_pair_count": len(scenarios),
        "forbidden_terms": list(FORBIDDEN_PROMPT_TERMS),
        "passed": not violations,
        "prompt_count": len(scenarios) * 2,
        "violations": violations,
    }


def _scenario_manifest(scenarios: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    fields = (
        "scenario_id",
        "template_id",
        "permutation_id",
        "obligation_count",
        "dependency_mode",
        "semantic_class",
        "legal_action_set_id",
        "scenario_hash",
    )
    return [{field: row[field] for field in fields} for row in scenarios]


def _prompt_arm_manifest(scenarios: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    manifest: list[JsonDict] = []
    for scenario in scenarios:
        for arm, prompt in scenario["prompts"].items():
            prompt_bytes = prompt.encode()
            manifest.append(
                {
                    "arm_id": f"{scenario['scenario_id']}:{arm}",
                    "information_sha256": scenario["prompt_information_sha256"],
                    "output_schema_sha256": sha256_json(OUTPUT_SCHEMA),
                    "prompt_byte_length": len(prompt_bytes),
                    "prompt_sha256": sha256_bytes(prompt_bytes),
                    "representation": arm,
                    "scenario_id": scenario["scenario_id"],
                }
            )
    return manifest


def _checker_manifest(scenarios: Sequence[Mapping[str, Any]]) -> JsonDict:
    obligation_bindings = [
        {
            "checker_id": f"{scenario['scenario_id']}:{obligation['obligation_id']}",
            "obligation_id": obligation["obligation_id"],
            "scenario_id": scenario["scenario_id"],
        }
        for scenario in scenarios
        for obligation in scenario["obligations"]
    ]
    return {
        "field_checkers": [
            {"checker_id": f"field:{field}", "field": field} for field in OBLIGATION_FIELDS
        ],
        "joint_checker": {"checker_id": "joint:all_obligations"},
        "joint_checker_count": len(scenarios),
        "joint_scenario_ids": [row["scenario_id"] for row in scenarios],
        "obligation_bindings": obligation_bindings,
        "obligation_checker_count": len(obligation_bindings),
        "parser": "canonical JSON selected_action_ids set",
    }


def _legal_action_headroom(scenarios: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_count: JsonDict = {}
    all_required_nonzero = True
    for count in OBLIGATION_COUNTS:
        rows = [row for row in scenarios if row["obligation_count"] == count]
        required = [row for row in rows if row["semantic_class"] == "constructive"]
        nonzero = sum(
            any(
                action["kind"] == "constructive" and action["action_id"] in row["legal_action_ids"]
                for action in row["candidates"]
            )
            for row in required
        )
        passes = nonzero == len(required)
        all_required_nonzero = all_required_nonzero and passes
        by_count[str(count)] = {
            "constructive_headroom_nonzero_count": nonzero,
            "required_scenario_count": len(required),
            "passed": passes,
        }
    return {
        "all_legal_sets_nonempty": all(bool(row["legal_action_ids"]) for row in scenarios),
        "all_required_nonzero": all_required_nonzero,
        "by_count": by_count,
    }


def _implementation_hashes(root: Path) -> JsonDict:
    return {
        "module": {"path": str(MODULE_PATH), "sha256": _sha256_file(root / MODULE_PATH)},
        "wrapper": {"path": str(WRAPPER_PATH), "sha256": _sha256_file(root / WRAPPER_PATH)},
    }


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: list[JsonDict],
    sources: Mapping[str, Any],
    implementations: Mapping[str, Any],
) -> JsonDict:
    failed = next((row for row in checks if not row["passed"]), None)
    gate_summary = (
        {
            "passed": False,
            "failed_check": failed["check"],
            "expected": failed["expected"],
            "observed": failed["observed"],
        }
        if failed
        else {"passed": True, "failed_check": None, "expected": True, "observed": True}
    )
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "6832",
        "run_date": run_date,
        "status": "complete_blocked_operational_obligation_saturation_fixture",
        "field_principles": {},
        "preconditions_checked": checks,
        "inference_substrate": (
            "deterministic CPU exact-checker transactional fixture; "
            "deterministic CPU fixture generation, no LLM"
        ),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "implementation_hashes": deepcopy(dict(implementations)),
        "obligation_schema": deepcopy(EXPECTED_OBLIGATION_SCHEMA),
        "obligation_counts": list(OBLIGATION_COUNTS),
        "scenario_manifest": [],
        "prompt_arm_manifest": [],
        "scenarios": [],
        "checker_manifest": {},
        "checker_mutation_results": [],
        "leakage_audit": {"passed": False, "violations": ["precondition_blocked"]},
        "legal_action_headroom": {"all_required_nonzero": False},
        "operational_saturation_fixture_ready": False,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_operational_obligation_saturation_fixture",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic content while excluding measured task duration."""

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
    worktree_status: str | None = None,
) -> JsonDict:
    """Build a complete fixture or stop after the exact source checks."""

    exp6811_bytes = _read_source_bytes(repo_root, EXP6811_PATH)
    exp6831_bytes = _read_source_bytes(repo_root, EXP6831_PATH)
    status = _owned_worktree_status(repo_root) if worktree_status is None else worktree_status
    checks = evaluate_preconditions(exp6811_bytes, exp6831_bytes, worktree_status=status)
    sources = _source_identities(exp6811_bytes, exp6831_bytes)
    implementations = _implementation_hashes(repo_root)
    artifact = _base_artifact(
        run_date=run_date,
        duration_s=duration_s,
        checks=checks,
        sources=sources,
        implementations=implementations,
    )
    if not all(row["passed"] for row in checks):
        return _finish(artifact)

    scenarios = generate_scenarios()
    mutations = [run_checker_mutations(row) for row in scenarios]
    leakage = audit_prompt_leakage(scenarios)
    headroom = _legal_action_headroom(scenarios)
    count_balance = all(
        len([row for row in scenarios if row["obligation_count"] == count]) == 30
        and sum(
            row["dependency_mode"] == "independent"
            for row in scenarios
            if row["obligation_count"] == count
        )
        == 15
        for count in OBLIGATION_COUNTS
    )
    stable_scenario_hashes = all(
        row["scenario_hash"]
        == sha256_json({key: value for key, value in row.items() if key != "scenario_hash"})
        for row in scenarios
    )
    checker_complete = all(row["all_expected"] for row in mutations)
    ready = (
        count_balance
        and stable_scenario_hashes
        and checker_complete
        and leakage["passed"]
        and headroom["all_required_nonzero"]
        and headroom["all_legal_sets_nonempty"]
    )
    artifact.update(
        {
            "status": "complete_operational_obligation_saturation_fixture",
            "scenario_manifest": _scenario_manifest(scenarios),
            "prompt_arm_manifest": _prompt_arm_manifest(scenarios),
            "scenarios": scenarios,
            "checker_manifest": _checker_manifest(scenarios),
            "checker_mutation_results": mutations,
            "leakage_audit": leakage,
            "legal_action_headroom": headroom,
            "operational_saturation_fixture_ready": ready,
            "gate_check_summary": {
                "passed": ready,
                "failed_check": None if ready else "fixture_readiness",
                "expected": {
                    "checker_complete": True,
                    "count_balance": True,
                    "headroom": True,
                    "leakage": True,
                    "stable_scenario_hashes": True,
                },
                "observed": {
                    "checker_complete": checker_complete,
                    "count_balance": count_balance,
                    "headroom": headroom["all_required_nonzero"],
                    "leakage": leakage["passed"],
                    "stable_scenario_hashes": stable_scenario_hashes,
                },
            },
            "verdict_class": "null" if ready else "partial",
            "honest_verdict": (
                "complete_operational_obligation_saturation_fixture: deterministic source-free "
                "fixture ready; no model ran"
                if ready
                else "complete_operational_obligation_saturation_fixture: fixture checks incomplete; "
                "no model ran"
            ),
        }
    )
    return _finish(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate both complete and complete-blocked terminal artifacts."""

    errors: list[str] = []
    if set(artifact) != set(FIELD_PRINCIPLES):
        errors.append("top-level fields differ from the declared contract")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be a nonnegative number")
    ready = artifact.get("operational_saturation_fixture_ready") is True
    blocked = artifact.get("status") == "complete_blocked_operational_obligation_saturation_fixture"
    gate = artifact.get("gate_check_summary") or {}
    if blocked:
        if ready or gate.get("passed") is not False or not gate.get("failed_check"):
            errors.append("blocked artifact lacks an exact failed gate")
        if artifact.get("scenarios") != [] or artifact.get("checker_mutation_results") != []:
            errors.append("blocked artifact generated fixture rows")
        if artifact.get("verdict_class") != "blocked" or artifact.get("honest_verdict") != (
            "complete_blocked_operational_obligation_saturation_fixture"
        ):
            errors.append("blocked terminal verdict mismatch")
    elif not ready:
        errors.append("complete artifact is not ready")
    else:
        if gate.get("passed") is not True or gate.get("failed_check") is not None:
            errors.append("ready artifact has a failed gate")
        if len(artifact.get("scenarios") or []) != 150:
            errors.append("ready artifact scenario count mismatch")
        if len(artifact.get("prompt_arm_manifest") or []) != 300:
            errors.append("ready artifact prompt count mismatch")
        if len(artifact.get("checker_mutation_results") or []) != 150:
            errors.append("ready artifact checker count mismatch")
        if not (artifact.get("leakage_audit") or {}).get("passed"):
            errors.append("ready artifact failed leakage audit")
        if not (artifact.get("legal_action_headroom") or {}).get("all_required_nonzero"):
            errors.append("ready artifact lacks required action headroom")
        if artifact.get("verdict_class") != "null" or not str(
            artifact.get("honest_verdict") or ""
        ).startswith("complete_"):
            errors.append("ready terminal verdict mismatch")
    return errors


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Atomically replace only the caller-selected fixture artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(artifact) + b"\n")
    os.replace(temporary, path)


def execute(
    root: Path,
    run_date: str,
    output_path: Path,
    *,
    worktree_status: str | None = None,
) -> JsonDict:
    """Measure, build, validate, and atomically write one fixture."""

    if re.fullmatch(r"\d{8}", run_date) is None:
        raise FixtureError("invalid_run_date")
    try:
        datetime.strptime(run_date, "%Y%m%d")
    except ValueError as exc:
        raise FixtureError("invalid_run_date") from exc
    started = time.perf_counter()
    artifact = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=0.0,
        worktree_status=worktree_status,
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise FixtureError("invalid_artifact", "; ".join(errors))
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture command or validate its existing output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.root / OUTPUT_PATH
    try:
        if args.validate:
            artifact = json.loads(output.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(f"valid: {output}")
            return 0
        artifact = execute(args.root, args.date, output)
    except (FixtureError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"artifact": str(output), "verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - covered by the required command.
    raise SystemExit(main())
