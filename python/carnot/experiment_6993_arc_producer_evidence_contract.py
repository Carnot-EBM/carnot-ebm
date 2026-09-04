"""Build the deterministic ARC producer evidence contract artifact.

Spec ref: REQ-ARC-WMTE-6993. The experiment uses byte fixtures only. It does
not create an ARC environment, call a service, or start an LLM.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.agentic import arc_executable_world_model as world
from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.agentic.arc_producer_evidence import (
    EVIDENCE_ENVELOPE_SCHEMA,
    EVIDENCE_MANIFEST_SCHEMA,
    LIVE_TRANSITION_SOURCE_KIND,
    REQUIRED_ENVELOPE_FIELDS,
    EvidencePublishInterrupted,
    canonical_json_bytes,
    produce_engine_evidence,
    read_evidence_manifest,
    sha256_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "results" / "experiment_6993_arc_producer_evidence_contract.json"
INFERENCE_SUBSTRATE = "deterministic_arc_producer_contract_fixture_no_llm"
RANDOM_SEED = 6_993_202_609_04
STAMP = "20260904T120000_000000"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "envelope_schema",
    "required_field_rows",
    "rows",
    "producer_write_rows",
    "atomic_publish_rows",
    "interruption_rows",
    "tamper_rows",
    "legacy_row_rows",
    "transition_source_rows",
    "manifest_row_rows",
    "prompt_hash_rows",
    "transition_hash_rows",
    "engine_hash_rows",
    "environment_hash_rows",
    "scorer_hash_rows",
    "policy_hash_rows",
    "factory_hash_rows",
    "envelope_hash_rows",
    "agent_factory_trace_rows",
    "live_routing_fixture_rows",
    "action_influence_fixture_rows",
    "focused_test_receipts",
    "arc_producer_contract_complete_score",
    "arc_live_path_fixture_ready_score",
    "solve_provenance_applicable",
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
    "model_quality_claimed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A principle states why every evidence field exists.",
    "preconditions_checked": "Preflight checks stop missing producer parts from looking measured.",
    "inference_substrate": "The substrate separates deterministic fixtures from model inference.",
    "duration_s": "Measured time shows that the contract checks executed.",
    "source_artifact_hashes": "Source hashes bind the experiment to exact implementation bytes.",
    "envelope_schema": "A named schema makes future compatibility decisions explicit.",
    "required_field_rows": "Field rows prove that each required envelope value was emitted.",
    "rows": "All manifest decisions stay visible for independent review.",
    "producer_write_rows": "Write rows prove that evidence existed before synthesis.",
    "atomic_publish_rows": "Publish rows prove that the manifest marker appeared last.",
    "interruption_rows": "Interrupted rows prove that partial work cannot become eligible.",
    "tamper_rows": "Changed-byte rows prove that every immutable source is checked.",
    "legacy_row_rows": "Legacy rows preserve readability without granting false eligibility.",
    "transition_source_rows": "Source rows separate live observations from prohibited evidence.",
    "manifest_row_rows": "Manifest rows bind each run marker to its canonical content.",
    "prompt_hash_rows": "Prompt rows bind generation to the exact raw request bytes.",
    "transition_hash_rows": "Transition rows bind values and ordering before synthesis.",
    "engine_hash_rows": "Engine rows identify the exact executable candidate.",
    "environment_hash_rows": "Environment rows bind the runtime receipt used by the producer.",
    "scorer_hash_rows": "Scorer rows fix the measurement definition at creation time.",
    "policy_hash_rows": "Policy rows fix the shipped routing code at creation time.",
    "factory_hash_rows": "Factory rows fix the constructor that reaches the live policy.",
    "envelope_hash_rows": "Envelope rows detect schema-content changes without a hash cycle.",
    "agent_factory_trace_rows": "Trace rows prove that the shipped factory created E3 policy.",
    "live_routing_fixture_rows": "Routing rows prove that the envelope engine reached the real seam.",
    "action_influence_fixture_rows": "Influence rows require a measured candidate-score change.",
    "focused_test_receipts": "Receipts expose which deterministic checks support the two scores.",
    "arc_producer_contract_complete_score": "Contract credit requires every integrity class to pass.",
    "arc_live_path_fixture_ready_score": "Path credit requires factory reachability and score influence.",
    "solve_provenance_applicable": "A producer contract does not contain solve provenance.",
    "solve_claimed": "An engine write does not prove a game solve.",
    "level_claimed": "A candidate-score change does not prove level completion.",
    "registry_updated": "Fixture work must leave the solve registry unchanged.",
    "submitted_to_leaderboard": "Local fixture execution is not an external submission.",
    "model_quality_claimed": "Evidence completeness does not establish model quality.",
    "random_seed": "A fixed seed makes the fixture identity reproducible.",
    "reproducibility_checksum": "A timing-free checksum detects scientific-content drift.",
    "gate_check_summary": "Each failed gate states its expected and observed values.",
    "verifier_is_oracle": "False keeps evidence validation distinct from game control.",
    "verdict_class": "A closed verdict class prevents blocked work from reading as positive.",
    "honest_verdict": "A terminal prefix makes the result machine-classifiable.",
}

PROMPT = b"fixture raw prompt\x00\xff\r\n"
ENGINE = b"""import numpy as np

def engine(grid, action, data):
    result = np.asarray(grid).copy()
    if int(action) == 2:
        result[0, 0] = 7
    return result

def is_level_complete(grid):
    return False
"""
TRANSITIONS = (
    {
        "grid": [[0, 0], [0, 0]],
        "action": 1,
        "data": None,
        "next_grid": [[0, 0], [0, 0]],
        "level_before": 0,
        "level_after": 0,
    },
    {
        "grid": [[0, 0], [0, 0]],
        "action": 2,
        "data": {"x": 0, "y": 0},
        "next_grid": [[7, 0], [0, 0]],
        "level_before": 0,
        "level_after": 0,
    },
)


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> dict[str, Any]:
    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed if passed is None else bool(passed),
    }


def _failed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "failed_check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
        }
        for row in rows
        if row.get("passed") is not True
    ]


def _file_hash(path: Path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _produce(
    root: Path,
    game: str,
    run_id: str,
    source_kind: str | None = LIVE_TRANSITION_SOURCE_KIND,
    **kwargs,
):
    return produce_engine_evidence(
        store_root=root,
        game=game,
        raw_prompt=PROMPT,
        transitions=TRANSITIONS,
        transition_source_kind=source_kind,
        environment_receipt={"environment": "deterministic_stub", "game": game},
        synthesize_engine=lambda: ENGINE,
        run_id=run_id,
        timestamp=STAMP,
        **kwargs,
    )


def _trace(symbol: str, value: Any, called: bool) -> dict[str, Any]:
    path = inspect.getsourcefile(value)
    try:
        line = inspect.getsourcelines(value)[1]
    except (OSError, TypeError):
        line = None
    return {
        "symbol": symbol,
        "path": str(Path(path).resolve().relative_to(REPO_ROOT)) if path else None,
        "line": line,
        "called_by_fixture": called,
        "terminal": True,
    }


def _factory_fixture(
    root: Path, game: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    called = {
        "make_carnot_agent": False,
        "E3AgentPolicy": False,
        "load_engine": False,
        "_world_model_candidates": False,
    }

    class FixtureBase:
        def __init__(self) -> None:
            self.game_id = game

    agent_type = make_carnot_agent(FixtureBase, cascade=True, proposer=object())
    called["make_carnot_agent"] = True
    agent = agent_type()
    called["E3AgentPolicy"] = isinstance(agent._policy, E3AgentPolicy)
    old_root = world.E3_DIR
    world.E3_DIR = root
    try:
        engine, goal = world.load_engine(game)
        called["load_engine"] = True
    finally:
        world.E3_DIR = old_root
    candidates = agent._policy._world_model_candidates(engine, goal)
    called["_world_model_candidates"] = True
    routed = next(row for row in candidates if row.name == "loaded_world_model.py")
    grid = np.zeros((2, 2), dtype=np.int64)
    no_engine_score = int(np.count_nonzero(grid != grid))
    envelope_engine_score = int(np.count_nonzero(np.asarray(routed.engine(grid, 2, None)) != grid))
    trace = [
        _trace("make_carnot_agent", make_carnot_agent, called["make_carnot_agent"]),
        _trace("E3AgentPolicy", E3AgentPolicy, called["E3AgentPolicy"]),
        _trace("load_engine", world.load_engine, called["load_engine"]),
        _trace(
            "_world_model_candidates",
            E3AgentPolicy._world_model_candidates,
            called["_world_model_candidates"],
        ),
    ]
    routing = {
        "factory_constructed_e3_policy": called["E3AgentPolicy"],
        "selected_candidate_name": routed.name,
        "exact_engine_hash": sha256_bytes((root / game / "world_model.py").read_bytes()),
        "terminal": True,
    }
    influence = {
        "action": 2,
        "no_engine_candidate_score": no_engine_score,
        "envelope_engine_candidate_score": envelope_engine_score,
        "score_changed": envelope_engine_score != no_engine_score,
        "future_frame_used": False,
        "terminal": True,
    }
    return trace, routing, influence


def _empty_artifact() -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "envelope_schema": {
            "envelope": EVIDENCE_ENVELOPE_SCHEMA,
            "manifest": EVIDENCE_MANIFEST_SCHEMA,
            "envelope_hash_projection_excludes": ["envelope_sha256", "manifest_row_sha256"],
            "manifest_hash_projection_excludes": ["manifest_row_sha256"],
        },
        "required_field_rows": [],
        "rows": [],
        "producer_write_rows": [],
        "atomic_publish_rows": [],
        "interruption_rows": [],
        "tamper_rows": [],
        "legacy_row_rows": [],
        "transition_source_rows": [],
        "manifest_row_rows": [],
        "prompt_hash_rows": [],
        "transition_hash_rows": [],
        "engine_hash_rows": [],
        "environment_hash_rows": [],
        "scorer_hash_rows": [],
        "policy_hash_rows": [],
        "factory_hash_rows": [],
        "envelope_hash_rows": [],
        "agent_factory_trace_rows": [],
        "live_routing_fixture_rows": [],
        "action_influence_fixture_rows": [],
        "focused_test_receipts": [],
        "arc_producer_contract_complete_score": 0,
        "arc_live_path_fixture_ready_score": 0,
        "solve_provenance_applicable": False,
        "solve_claimed": False,
        "level_claimed": False,
        "registry_updated": False,
        "submitted_to_leaderboard": False,
        "model_quality_claimed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked: blocked_arc_producer_evidence_contract",
    }
    return artifact


def build_artifact(
    *, output_path: Path = OUTPUT_PATH, execution_date: str = "20260904"
) -> dict[str, Any]:
    """Run all local contract fixtures and write one stable result document."""

    started = time.perf_counter()
    artifact = _empty_artifact()
    registry = REPO_ROOT / "ops" / "arc_solve_registry.yaml"
    registry_before = _file_hash(registry)
    spec = REPO_ROOT / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    producer = REPO_ROOT / "python" / "carnot" / "agentic" / "arc_producer_evidence.py"
    policy = REPO_ROOT / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
    scorer = REPO_ROOT / "scripts" / "arc_e3_induced_model_quality.py"
    sources = {
        "spec": spec,
        "producer": producer,
        "world_model_producer": REPO_ROOT
        / "python"
        / "carnot"
        / "agentic"
        / "arc_executable_world_model.py",
        "live_policy_and_factory": policy,
        "scorer": scorer,
        "solve_registry": registry,
    }
    artifact["source_artifact_hashes"] = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _file_hash(path)}
        for name, path in sources.items()
    }
    preconditions = [
        _gate("execution_date", "20260904", execution_date),
        _gate("producer_importable", True, callable(produce_engine_evidence)),
        _gate("e3_policy_importable", True, E3AgentPolicy is not None),
        _gate("factory_importable", True, callable(make_carnot_agent)),
        _gate("attempt_manifest_schema", EVIDENCE_MANIFEST_SCHEMA, EVIDENCE_MANIFEST_SCHEMA),
        _gate("requirement_present", True, "REQ-ARC-WMTE-6993" in spec.read_text(encoding="utf-8")),
        _gate(
            "arc_live_path_lint_present",
            True,
            (REPO_ROOT / "scripts" / "arc_orphan_solver_lint.py").is_file(),
        ),
        _gate(
            "arc_liveness_lint_present",
            True,
            (REPO_ROOT / "scripts" / "arc_llm_on_liveness_lint.py").is_file(),
        ),
    ]
    artifact["preconditions_checked"] = preconditions
    if _failed(preconditions):
        artifact["gate_check_summary"] = _failed(preconditions)
        return _finish_artifact(artifact, Path(output_path), started)

    with tempfile.TemporaryDirectory(prefix="carnot-exp6993-") as temporary:
        root = Path(temporary)
        writable = root / "writable"
        writable.write_bytes(b"ok")
        writable_gate = _gate(
            "temporary_fixture_directory_writable",
            "ok",
            writable.read_bytes().decode("ascii"),
        )
        artifact["preconditions_checked"].append(writable_gate)
        if not writable_gate["passed"]:  # pragma: no cover - same-file write/read cannot differ.
            artifact["gate_check_summary"] = _failed(artifact["preconditions_checked"])
            return _finish_artifact(artifact, Path(output_path), started)

        staged: dict[str, Any] = {}

        def synthesize() -> bytes:
            stage = root / "success" / "attempts" / ".evidence-staging" / "success-run"
            staged.update(
                {
                    "prompt_before_synthesis": (stage / "prompt.raw").read_bytes() == PROMPT,
                    "transitions_before_synthesis": (stage / "transitions.jsonl").is_file(),
                    "engine_absent_before_synthesis": not (stage / "engine.py").exists(),
                }
            )
            return ENGINE

        success = produce_engine_evidence(
            store_root=root,
            game="success",
            raw_prompt=PROMPT,
            transitions=TRANSITIONS,
            transition_source_kind=LIVE_TRANSITION_SOURCE_KIND,
            environment_receipt={"environment": "deterministic_stub", "game": "success"},
            synthesize_engine=synthesize,
            run_id="success-run",
            timestamp=STAMP,
        )
        success_validation = read_evidence_manifest(root, "success")[0]
        envelope = json.loads(success.envelope_path.read_bytes())
        artifact["rows"].append(success_validation)
        artifact["producer_write_rows"].append({**staged, "terminal": True})
        artifact["atomic_publish_rows"].append(
            {
                "run_id": "success-run",
                "bundle_published": success.envelope_path.is_file(),
                "engine_published": success.canonical_engine_path.is_file(),
                "manifest_marker_count": len(success.manifest_path.read_bytes().splitlines()),
                "eligible": success_validation["eligible"],
                "terminal": True,
            }
        )
        artifact["required_field_rows"] = [
            {"field": field, "present": field in envelope, "terminal": True}
            for field in REQUIRED_ENVELOPE_FIELDS
        ]

        for failpoint in (
            "after_engine_temp",
            "after_bundle_publish",
            "after_engine_publish",
            "before_manifest",
        ):
            game = "interrupt_" + failpoint
            interrupted = False
            try:
                _produce(root, game, game + "-run", failpoint=failpoint)
            except EvidencePublishInterrupted:
                interrupted = True
            visible = read_evidence_manifest(root, game)
            artifact["interruption_rows"].append(
                {
                    "failpoint": failpoint,
                    "interrupted": interrupted,
                    "eligible_count": sum(row["eligible"] for row in visible),
                    "manifest_row_count": len(visible),
                    "terminal": True,
                }
            )

        tamper_fields = {
            "raw_prompt_path": b"changed",
            "transition_jsonl_path": b"{}\n",
            "engine_path": b"\n# changed\n",
            "envelope": b" ",
        }
        for index, (field, suffix) in enumerate(tamper_fields.items()):
            game = f"tamper_{index}"
            result = _produce(root, game, game + "-run")
            if field == "envelope":
                target = result.envelope_path
            else:
                document = json.loads(result.envelope_path.read_bytes())
                target = root / document[field]
            target.write_bytes(target.read_bytes() + suffix)
            validation = read_evidence_manifest(root, game)[0]
            artifact["tamper_rows"].append(
                {
                    "target": field,
                    "eligible": validation["eligible"],
                    "rejection_reasons": validation["rejection_reasons"],
                    "terminal": True,
                }
            )

        legacy_result = _produce(root, "legacy_duplicate", "duplicate-run")
        original = legacy_result.manifest_path.read_bytes()
        legacy = canonical_json_bytes(
            {"ts": "20260901T000000_000000", "file": "wm_old.py", "sha256_16": "0" * 16}
        )
        legacy_result.manifest_path.write_bytes(legacy + b"\n" + original + original)
        legacy_rows = read_evidence_manifest(root, "legacy_duplicate")
        artifact["legacy_row_rows"] = [
            {
                "classification": row["classification"],
                "run_id": row["run_id"],
                "eligible": row["eligible"],
                "rejection_reasons": row["rejection_reasons"],
                "terminal": True,
            }
            for row in legacy_rows
        ]

        for index, source_kind in enumerate(
            (
                LIVE_TRANSITION_SOURCE_KIND,
                "game_source",
                "hand_adapter",
                "offline_bfs",
                "synthetic_hidden_transition",
                None,
                "unknown_source",
            )
        ):
            game = f"source_{index}"
            _produce(root, game, game + "-run", source_kind)
            validation = read_evidence_manifest(root, game)[0]
            artifact["transition_source_rows"].append(
                {
                    "transition_source_kind": source_kind,
                    "eligible": validation["eligible"],
                    "terminal": True,
                }
            )

        hash_rows = {
            "prompt_hash_rows": ("raw_prompt_sha256", success.prompt_path),
            "transition_hash_rows": ("transition_sha256", success.transition_path),
            "engine_hash_rows": ("engine_sha256", success.engine_path),
            "environment_hash_rows": (
                "environment_receipt_sha256",
                root / envelope["environment_receipt_path"],
            ),
            "scorer_hash_rows": ("scorer_sha256", root / envelope["scorer_path"]),
            "policy_hash_rows": ("live_policy_sha256", root / envelope["live_policy_path"]),
            "factory_hash_rows": (
                "agent_factory_sha256",
                root / envelope["agent_factory_path"],
            ),
        }
        for artifact_field, (envelope_field, path) in hash_rows.items():
            observed = _file_hash(path)
            artifact[artifact_field] = [
                {
                    "run_id": "success-run",
                    "expected_hash": envelope[envelope_field],
                    "observed_hash": observed,
                    "passed": envelope[envelope_field] == observed,
                    "terminal": True,
                }
            ]
        artifact["manifest_row_rows"] = [
            {
                "run_id": "success-run",
                "expected_hash": envelope["manifest_row_sha256"],
                "observed_hash": _file_hash(success.manifest_row_path),
                "passed": envelope["manifest_row_sha256"] == _file_hash(success.manifest_row_path),
                "terminal": True,
            }
        ]
        envelope_projection = next(
            check["observed_value"]
            for check in success_validation["checks"]
            if check["check"] == "envelope_hash"
        )
        artifact["envelope_hash_rows"] = [
            {
                "run_id": "success-run",
                "expected_hash": envelope["envelope_sha256"],
                "observed_hash": envelope_projection,
                "passed": envelope["envelope_sha256"] == envelope_projection,
                "terminal": True,
            }
        ]

        trace, routing, influence = _factory_fixture(root, "success")
        artifact["agent_factory_trace_rows"] = trace
        artifact["live_routing_fixture_rows"] = [routing]
        artifact["action_influence_fixture_rows"] = [influence]

    atomic_ok = (
        success_validation["eligible"]
        and all(staged.values())
        and all(
            row["eligible_count"] == 0 and row["manifest_row_count"] == 0
            for row in artifact["interruption_rows"]
        )
    )
    tamper_ok = all(row["eligible"] is False for row in artifact["tamper_rows"])
    legacy_ok = artifact["legacy_row_rows"][0]["classification"] == "legacy" and all(
        row["eligible"] is False for row in artifact["legacy_row_rows"]
    )
    source_ok = all(
        row["eligible"] is (row["transition_source_kind"] == LIVE_TRANSITION_SOURCE_KIND)
        for row in artifact["transition_source_rows"]
    )
    hashes_ok = all(
        row["passed"]
        for field in (
            "prompt_hash_rows",
            "transition_hash_rows",
            "engine_hash_rows",
            "environment_hash_rows",
            "scorer_hash_rows",
            "policy_hash_rows",
            "factory_hash_rows",
            "manifest_row_rows",
            "envelope_hash_rows",
        )
        for row in artifact[field]
    )
    schema_ok = all(row["present"] for row in artifact["required_field_rows"])
    contract_ok = atomic_ok and tamper_ok and legacy_ok and source_ok and hashes_ok and schema_ok
    path_ok = (
        all(row["called_by_fixture"] for row in artifact["agent_factory_trace_rows"])
        and artifact["live_routing_fixture_rows"][0]["selected_candidate_name"]
        == "loaded_world_model.py"
        and artifact["action_influence_fixture_rows"][0]["score_changed"]
    )
    artifact["arc_producer_contract_complete_score"] = int(contract_ok)
    artifact["arc_live_path_fixture_ready_score"] = int(path_ok)
    artifact["focused_test_receipts"] = [
        {"check": "atomicity_eligibility_tamper_legacy_schema", "passed": contract_ok},
        {"check": "shipped_factory_e3_candidate_score_influence", "passed": path_ok},
    ]
    registry_after = _file_hash(registry)
    artifact["registry_updated"] = registry_before != registry_after
    gates = [
        _gate(
            "arc_producer_contract_complete_score",
            1,
            artifact["arc_producer_contract_complete_score"],
        ),
        _gate(
            "arc_live_path_fixture_ready_score", 1, artifact["arc_live_path_fixture_ready_score"]
        ),
        _gate("solve_registry_unchanged", registry_before, registry_after),
    ]
    artifact["gate_check_summary"] = _failed(gates)
    if not artifact["gate_check_summary"]:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "positive: arc_producer_evidence_contract_complete"
    else:
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial: arc_producer_evidence_contract_incomplete"
    return _finish_artifact(artifact, Path(output_path), started)


def _finish_artifact(artifact: dict[str, Any], output_path: Path, started: float) -> dict[str, Any]:
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    stable = dict(artifact)
    stable.pop("duration_s", None)
    stable.pop("reproducibility_checksum", None)
    artifact["reproducibility_checksum"] = sha256_bytes(canonical_json_bytes(stable))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, output_path)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(output_path=args.output, execution_date=args.date)
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in (
                    "verdict_class",
                    "honest_verdict",
                    "arc_producer_contract_complete_score",
                    "arc_live_path_fixture_ready_score",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["verdict_class"] == "positive" else 1


if __name__ == "__main__":  # pragma: no cover - the wrapper is the tested command surface.
    raise SystemExit(main())
