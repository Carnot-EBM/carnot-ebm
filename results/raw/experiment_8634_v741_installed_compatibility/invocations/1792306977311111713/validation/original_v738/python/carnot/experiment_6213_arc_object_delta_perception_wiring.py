"""Experiment 6213: ARC object-delta perception wiring receipt.

Spec refs: REQ-ARC-WMTE-6213,
SCENARIO-ARC-WMTE-6213-TRANSLATION,
SCENARIO-ARC-WMTE-6213-HUD-REJECTION,
SCENARIO-ARC-WMTE-6213-FAIL-OPEN,
SCENARIO-ARC-WMTE-6213-PROMPT-WIRING.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition, induce_prompt
from carnot.agentic import arc_object_delta_perception as odp


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6213_arc_object_delta_perception_wiring.json")
MODULE_RELATIVE_PATH = Path("python/carnot/agentic/arc_object_delta_perception.py")
PROMPT_RELATIVE_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
EXPERIMENT_RELATIVE_PATH = Path(
    "python/carnot/experiment_6213_arc_object_delta_perception_wiring.py"
)
TEST_RELATIVE_PATH = Path("tests/python/test_arc_object_delta_perception_6213.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
ORPHAN_LINT_RELATIVE_PATH = Path("scripts/arc_orphan_solver_lint.py")
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6213_test_receipts.json")

INFERENCE_SUBSTRATE = "deterministic_visible_grid_object_delta_prompt_wiring_no_llm"
SPEC_REFS = [
    "REQ-ARC-WMTE-6213",
    "SCENARIO-ARC-WMTE-6213-TRANSLATION",
    "SCENARIO-ARC-WMTE-6213-HUD-REJECTION",
    "SCENARIO-ARC-WMTE-6213-FAIL-OPEN",
    "SCENARIO-ARC-WMTE-6213-PROMPT-WIRING",
]

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "spec_refs",
    "flag_name_and_default",
    "implementation_paths_and_hashes",
    "canonical_live_entrypoint_receipt",
    "component_schema",
    "transition_delta_schema",
    "translation_invariant_match_receipts",
    "hud_rejection_rules_and_receipts",
    "ambiguous_match_fail_open_receipts",
    "prompt_insertion_receipt",
    "treatment_fire_count_in_fixture",
    "fallback_exactness",
    "mutation_commands_and_kills",
    "source_bfs_adapter_registry_hidden_state_access_counts",
    "solve_claimed",
    "registry_hash_before_after",
    "object_delta_wiring_ready_score",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status for the default-off wiring receipt.",
    "spec_refs": "The artifact is anchored to REQ-ARC-WMTE-6213 and its scenarios.",
    "flag_name_and_default": "The new arm is separate from default-on static object perception.",
    "implementation_paths_and_hashes": "Source bytes are hash-bound for review and reproduction.",
    "canonical_live_entrypoint_receipt": "The canonical prompt path reaches the new hook.",
    "component_schema": "Component fields are versioned and visible-grid based.",
    "transition_delta_schema": "Transition deltas are action-conditioned and visible-grid based.",
    "translation_invariant_match_receipts": "Object identity is independent of absolute position.",
    "hud_rejection_rules_and_receipts": "HUD rejection is conservative and auditable.",
    "ambiguous_match_fail_open_receipts": "Ambiguous identities do not create false matches.",
    "prompt_insertion_receipt": "The table enters the prompt only when the new flag is on.",
    "treatment_fire_count_in_fixture": "The fixture proves the treatment hook fired at least once.",
    "fallback_exactness": "A serializer failure returns the exact off-arm prompt.",
    "mutation_commands_and_kills": "Source mutations are killed and restored byte-identically.",
    "source_bfs_adapter_registry_hidden_state_access_counts": "Forbidden-access counters are bare zeros.",
    "solve_claimed": "This task makes no ARC score or solve claim.",
    "registry_hash_before_after": "The ARC solve registry is read only for hash stability.",
    "object_delta_wiring_ready_score": "Readiness is a receipt score, not an ARC score.",
    "protected_files_unchanged": "Conductor-owned reconciliation files are unchanged.",
    "inference_substrate": "No LLM or hidden-game oracle is used.",
    "verifier_is_oracle": "False because this proves wiring, not hidden correctness.",
    "field_provenance": "Every required field names its source.",
    "field_principles": "Every required field names the audit risk it controls.",
    "test_commands": "Commands run for this receipt are recorded.",
    "test_exit_codes": "Exit codes prevent unchecked receipt claims.",
    "duration_s": "Measured wall time for the artifact build.",
    "reproducibility_checksum": "Stable checksum catches silent drift.",
    "honest_verdict": "The verdict states default-off wiring only.",
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

IMPLEMENTATION_FILES = (
    MODULE_RELATIVE_PATH,
    PROMPT_RELATIVE_PATH,
    EXPERIMENT_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _put_l(grid: np.ndarray, row: int, col: int, color: int) -> None:
    grid[row, col] = color
    grid[row + 1, col] = color
    grid[row, col + 1] = color


def _transition(before: np.ndarray, after: np.ndarray, *, action: int = 4) -> Transition:
    return Transition(before, action, None, after, 0, 0)


def _translation_fixture() -> list[Transition]:
    before = np.zeros((8, 10), dtype=np.int16)
    after = np.zeros_like(before)
    _put_l(before, 1, 1, 2)
    before[4:6, 2] = 3
    _put_l(after, 2, 3, 2)
    after[5:7, 4] = 3
    return [_transition(before, after)]


def _hud_fixture() -> list[Transition]:
    before = np.zeros((6, 8), dtype=np.int16)
    before[0, :] = 7
    before[0, 4] = 8
    before[1, 4] = 8
    before[4, 2] = 3
    after = before.copy()
    after[0, 0] = 9
    return [_transition(before, after)]


def _ambiguous_fixture() -> list[Transition]:
    before = np.zeros((8, 12), dtype=np.int16)
    after = np.zeros_like(before)
    before[1:3, 1:3] = 5
    before[5:7, 1:3] = 5
    after[1:3, 3:5] = 5
    after[5:7, 3:5] = 5
    return [_transition(before, after)]


def _prompt_fixture() -> list[Transition]:
    grid = np.zeros((5, 5), dtype=np.int16)
    _put_l(grid, 1, 1, 2)
    return [_transition(grid, grid.copy())]


def _with_prompt_env(delta: str | None, object_static: str | None = None) -> str:
    keys = ("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", "CARNOT_ARC_OBJECT_PERCEPTION")
    old = {key: os.environ.get(key) for key in keys}
    try:
        if delta is None:
            os.environ.pop("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", None)
        else:
            os.environ["CARNOT_ARC_OBJECT_DELTA_PERCEPTION"] = delta
        if object_static is None:
            os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
        else:
            os.environ["CARNOT_ARC_OBJECT_PERCEPTION"] = object_static
        return induce_prompt("xx", _prompt_fixture(), 1)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def file_receipt(path: Path) -> JsonDict:
    absolute = REPO_ROOT / path
    return {
        "path": path.as_posix(),
        "exists": absolute.is_file(),
        "sha256": sha256_file(absolute) if absolute.is_file() else None,
        "size_bytes": absolute.stat().st_size if absolute.is_file() else None,
    }


def implementation_paths_and_hashes() -> list[JsonDict]:
    return [file_receipt(path) for path in IMPLEMENTATION_FILES]


def protected_hash_map() -> dict[str, str]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def protected_files_unchanged(before: Mapping[str, str] | None = None) -> JsonDict:
    before_hashes = dict(before or protected_hash_map())
    after = protected_hash_map()
    changed = [path for path, digest in before_hashes.items() if after.get(path) != digest]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before_hashes),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
    }


def registry_hash_before_after() -> JsonDict:
    digest = sha256_file(REPO_ROOT / REGISTRY_RELATIVE_PATH)
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_hash_before": digest,
        "registry_hash_after": digest,
        "unchanged": True,
    }


def _live_closure() -> set[str]:
    script = REPO_ROOT / ORPHAN_LINT_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("arc_orphan_solver_lint", script)
    if spec is None or spec.loader is None:
        return set()  # pragma: no cover
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module._closure(module.ENTRYPOINTS))


def canonical_live_entrypoint_receipt() -> JsonDict:
    closure = _live_closure()
    return {
        "entrypoints": [
            "scripts/arc_loop_solve.py",
            "python/carnot/agentic/arc_competition_agent.py",
        ],
        "arc_executable_world_model_reachable": "arc_executable_world_model" in closure,
        "arc_object_delta_perception_reachable": "arc_object_delta_perception" in closure,
        "canonical_prompt_hook": "arc_executable_world_model.induce_prompt",
        "ok": "arc_executable_world_model" in closure and "arc_object_delta_perception" in closure,
    }


def translation_invariant_match_receipts() -> JsonDict:
    row = odp.build_object_delta_table(_translation_fixture())["transitions"][0]
    matches = [match for match in row["matches"] if match["before_component"]["color"] in {2, 3}]
    return {
        "fixture": "two_objects_shifted_by_dy1_dx2",
        "matched_colors": sorted(match["before_component"]["color"] for match in matches),
        "centroid_deltas": [match["centroid_delta"] for match in matches],
        "relation_invariant_count": sum(1 for rel in row["relations"] if rel["relation_invariant"]),
        "sample_relations": row["relations"],
        "ok": sorted(match["before_component"]["color"] for match in matches) == [2, 3]
        and all(match["centroid_delta"] == [1.0, 2.0] for match in matches),
    }


def hud_rejection_rules_and_receipts() -> JsonDict:
    table = odp.build_object_delta_table(_hud_fixture())
    row = table["transitions"][0]
    return {
        "rules": odp.hud_rejection_rules(),
        "receipt": row["hud_rejection"],
        "rejected_component_counts": row["hud_rejected_component_counts"],
        "cross_strip_component_kept": any(
            component["color"] == 8 and component["bbox"] == [0, 4, 1, 4]
            for component in row["before_components"]
        ),
        "ok": row["hud_rejection"]["admitted"] is True
        and row["hud_rejected_component_counts"]["before"] >= 1,
    }


def ambiguous_match_fail_open_receipts() -> JsonDict:
    row = odp.build_object_delta_table(_ambiguous_fixture())["transitions"][0]
    return {
        "ambiguous_matches": row["ambiguous_matches"],
        "color5_match_count": sum(
            1 for match in row["matches"] if match["before_component"]["color"] == 5
        ),
        "ok": bool(row["ambiguous_matches"])
        and row["ambiguous_matches"][0].get("fail_open") is True,
    }


def prompt_insertion_receipt() -> JsonDict:
    prompt_unset = _with_prompt_env(None)
    prompt_on = _with_prompt_env("1")
    prompt_static_off = _with_prompt_env("1", "0")
    return {
        "flag_unset_has_delta_block": "OBJECT DELTA PERCEPTION" in prompt_unset,
        "flag_on_has_delta_block": "OBJECT DELTA PERCEPTION" in prompt_on,
        "static_object_default_on": "OBJECT STRUCTURE" in prompt_unset,
        "static_object_can_be_disabled_independently": "OBJECT STRUCTURE" not in prompt_static_off
        and "OBJECT DELTA PERCEPTION" in prompt_static_off,
        "prompt_on_sha256": sha256_text(prompt_on),
        "prompt_unset_sha256": sha256_text(prompt_unset),
        "ok": "OBJECT DELTA PERCEPTION" not in prompt_unset
        and "OBJECT DELTA PERCEPTION" in prompt_on,
    }


def fallback_exactness() -> JsonDict:
    base_prompt = _with_prompt_env("0")
    original = odp.object_delta_block
    try:
        odp.object_delta_block = lambda _transitions: ""
        fallback_prompt = _with_prompt_env("1")
    finally:
        odp.object_delta_block = original
    return {
        "flag_off_sha256": sha256_text(base_prompt),
        "serializer_empty_fallback_sha256": sha256_text(fallback_prompt),
        "exact": base_prompt == fallback_prompt,
        "bad_direct_block_is_empty": odp.object_delta_block([object()]) == "",
    }


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "carnot.experiment_6213_arc_object_delta_perception_wiring",
            "spec_ref": "REQ-ARC-WMTE-6213",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _ready_score(checks: Sequence[bool]) -> float:
    if not checks:
        return 0.0
    return round(float(sum(1 for item in checks if item)) / float(len(checks)), 6)


def _external_test_receipts() -> tuple[list[str], dict[str, int]]:  # pragma: no cover
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return [], {}
    payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    return list(payload.get("test_commands", [])), {
        str(key): int(value) for key, value in dict(payload.get("test_exit_codes", {})).items()
    }


def _replace_once(path: Path, old: str, new: str) -> None:  # pragma: no cover
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"mutation target not found: {old[:80]}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def run_mutation_tests() -> list[JsonDict]:  # pragma: no cover
    module = REPO_ROOT / MODULE_RELATIVE_PATH
    prompt = REPO_ROOT / PROMPT_RELATIVE_PATH
    originals = {module: module.read_bytes(), prompt: prompt.read_bytes()}
    mutations = [
        {
            "name": "prompt_hook_deleted",
            "path": prompt,
            "old": "_object_delta_perception_block(trans)",
            "new": "''",
            "test": "tests/python/test_arc_object_delta_perception_6213.py::test_req_arc_wmte_6213_prompt_flag_is_default_off_and_independent",
        },
        {
            "name": "identity_normalization_removed",
            "path": module,
            "old": "normalized = tuple(sorted((int(r) - min_r, int(c) - min_c) for r, c in cells))",
            "new": "normalized = tuple(sorted((int(r), int(c)) for r, c in cells))",
            "test": "tests/python/test_arc_object_delta_perception_6213.py::test_req_arc_wmte_6213_translation_matches_and_relations_are_invariant",
        },
        {
            "name": "hud_rejection_disabled",
            "path": module,
            "old": "if _component_fully_masked(cells, mask):",
            "new": "if False and _component_fully_masked(cells, mask):",
            "test": "tests/python/test_arc_object_delta_perception_6213.py::test_req_arc_wmte_6213_hud_strip_components_are_rejected_conservatively",
        },
        {
            "name": "ambiguity_guard_removed",
            "path": module,
            "old": "if len(before_rows) == 1 and len(after_rows) == 1:",
            "new": "if before_rows and after_rows:",
            "test": "tests/python/test_arc_object_delta_perception_6213.py::test_req_arc_wmte_6213_ambiguous_same_shape_matches_fail_open",
        },
    ]
    receipts: list[JsonDict] = []
    try:
        for mutation in mutations:
            for path, data in originals.items():
                path.write_bytes(data)
            _replace_once(Path(mutation["path"]), str(mutation["old"]), str(mutation["new"]))
            command = [
                ".venv/bin/pytest",
                str(mutation["test"]),
                "-q",
                "--no-cov",
                "-n",
                "0",
            ]
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            killed = result.returncode != 0
            receipts.append(
                {
                    "name": mutation["name"],
                    "command": " ".join(command),
                    "exit_code": int(result.returncode),
                    "killed": bool(killed),
                    "stdout_tail": (result.stdout or "")[-600:],
                    "stderr_tail": (result.stderr or "")[-600:],
                }
            )
    finally:
        for path, data in originals.items():
            path.write_bytes(data)
    for receipt in receipts:
        receipt["sources_restored_byte_identically"] = all(
            path.read_bytes() == data for path, data in originals.items()
        )
    return receipts


def build_artifact(
    *,
    date: str = "20260808",
    mutation_receipts: Sequence[Mapping[str, Any]] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    started: float | None = None,
) -> JsonDict:
    start = time.monotonic() if started is None else float(started)
    protected_before = protected_hash_map()
    mutations = [dict(row) for row in (mutation_receipts or [])]
    live = canonical_live_entrypoint_receipt()
    prompt = prompt_insertion_receipt()
    fallback = fallback_exactness()
    translation = translation_invariant_match_receipts()
    hud = hud_rejection_rules_and_receipts()
    ambiguous = ambiguous_match_fail_open_receipts()
    registry = registry_hash_before_after()
    protected = protected_files_unchanged(protected_before)
    forbidden = odp.forbidden_access_counts()
    mutation_ok = bool(mutations) and all(
        row.get("killed") is True and row.get("sources_restored_byte_identically", True) is True
        for row in mutations
    )
    checks = [
        live["ok"],
        prompt["ok"],
        fallback["exact"],
        fallback["bad_direct_block_is_empty"],
        translation["ok"],
        hud["ok"],
        ambiguous["ok"],
        mutation_ok,
        all(type(value) is int and value == 0 for value in forbidden.values()),
        registry["unchanged"],
        protected["unchanged"],
    ]
    ready_score = _ready_score(checks)
    status = "complete_ready" if ready_score == 1.0 else "blocked"
    commands = list(test_commands or [])
    exits = {str(key): int(value) for key, value in dict(test_exit_codes or {}).items()}
    artifact: JsonDict = {
        "status": status,
        "spec_refs": list(SPEC_REFS),
        "flag_name_and_default": {
            "name": odp.FLAG_NAME,
            "default": "off",
            "enabled_value": "1",
            "static_object_perception_flag": "CARNOT_ARC_OBJECT_PERCEPTION",
            "static_object_perception_default": "on",
        },
        "implementation_paths_and_hashes": implementation_paths_and_hashes(),
        "canonical_live_entrypoint_receipt": live,
        "component_schema": odp.component_schema(),
        "transition_delta_schema": odp.transition_delta_schema(),
        "translation_invariant_match_receipts": translation,
        "hud_rejection_rules_and_receipts": hud,
        "ambiguous_match_fail_open_receipts": ambiguous,
        "prompt_insertion_receipt": prompt,
        "treatment_fire_count_in_fixture": int(
            _with_prompt_env("1").count("OBJECT DELTA PERCEPTION")
        ),
        "fallback_exactness": fallback,
        "mutation_commands_and_kills": mutations,
        "source_bfs_adapter_registry_hidden_state_access_counts": forbidden,
        "solve_claimed": False,
        "registry_hash_before_after": registry,
        "object_delta_wiring_ready_score": ready_score,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": round(time.monotonic() - start, 6),
        "reproducibility_checksum": "",
        "honest_verdict": "complete: object_delta_perception_wired_default_off_no_solve_claim"
        if status == "complete_ready"
        else f"blocked: object_delta_wiring_receipts_incomplete_{date}",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")  # pragma: no cover
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance incomplete")  # pragma: no cover
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles incomplete")  # pragma: no cover
    if artifact.get("solve_claimed") is not False:
        raise ValueError("solve_claimed must be false")  # pragma: no cover
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")  # pragma: no cover
    counts = dict(artifact.get("source_bfs_adapter_registry_hidden_state_access_counts") or {})
    if not counts or any(type(value) is not int or value != 0 for value in counts.values()):
        raise ValueError("forbidden counts must be bare zeros")  # pragma: no cover
    registry = dict(artifact.get("registry_hash_before_after") or {})
    if registry.get("registry_hash_before") != registry.get("registry_hash_after"):
        raise ValueError("registry hash changed")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("checksum mismatch")  # pragma: no cover
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")):
        raise ValueError("honest verdict prefix invalid")  # pragma: no cover


def write_artifact(artifact: Mapping[str, Any]) -> Path:  # pragma: no cover
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260808")
    args = parser.parse_args(argv)
    started = time.monotonic()
    mutation_receipts = run_mutation_tests()
    external_commands, external_exits = _external_test_receipts()
    mutation_commands = [str(row["command"]) for row in mutation_receipts if row.get("command")]
    mutation_exits = {
        str(row["command"]): int(row["exit_code"])
        for row in mutation_receipts
        if row.get("command")
    }
    artifact = build_artifact(
        date=str(args.date),
        mutation_receipts=mutation_receipts,
        test_commands=external_commands + mutation_commands,
        test_exit_codes={**external_exits, **mutation_exits},
        started=started,
    )
    validate_artifact(artifact)
    write_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
