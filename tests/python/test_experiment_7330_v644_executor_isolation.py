"""Tests for REQ-CL-7330 and SCENARIO-CL-7330-*.

The tests use temporary result roots. They never rewrite the repository's
terminal research record.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import threading
from typing import Any

import pytest

from carnot import experiment_7330_v644_executor_isolation as experiment
from carnot import experiment_7330_v644_public_learner as public


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_EXECUTOR = ROOT / "scripts/experiments/experiment_7330_v644_private_executor.py"


@pytest.fixture(scope="module")
def built_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Build the real two-process fixture once for the behavioral tests."""

    output_root = tmp_path_factory.mktemp("exp7330-output")
    artifact = experiment.build_artifact(ROOT, output_root, progress=False)
    return output_root, artifact


def _request(request_id: str = "public-request") -> dict[str, Any]:
    return {
        "request_id": request_id,
        "version_token": "opaque-token-one",
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 1], "b": [0, 1]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 1, "b": 1},
        "horizon": 3,
    }


def test_public_learner_serializes_and_versions_without_private_imports() -> None:
    """REQ-CL-7330: public state is canonical and version-scoped."""

    learner = public.PublicConstraintLearner("opaque-token-one")
    request = _request()
    atom = public.make_pair_atom(
        "opaque-token-one",
        "a",
        "b",
        0,
        {"query_id": "q1", "accepted": False, "sequence": 1},
    )
    learner.admit_atom(atom, current_query_index=1)
    assert learner.propose(request)["assignments"] == {"a": 0, "b": 1}
    restored = public.PublicConstraintLearner.from_state_bytes(learner.state_bytes())
    assert restored.state_bytes() == learner.state_bytes()
    restored.activate_version("opaque-token-two")
    assert restored.active_atoms() == []

    source = (ROOT / "python/carnot/experiment_7330_v644_public_learner.py").read_text()
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    imported_modules = {node.module for node in imports}
    assert "carnot.experiment_7323_v643_addition_prototype" not in imported_modules
    assert all("private_executor" not in str(module) for module in imported_modules)


def test_public_validation_and_compound_rejection_stay_conservative() -> None:
    """SCENARIO-CL-7330-COMPOUND: one compound Boolean creates no pair atom."""

    learner = public.PublicConstraintLearner("opaque-token-one")
    request = _request()
    with pytest.raises(public.PublicLearningError, match="activity_set"):
        learner.propose({**request, "allowed_starts": {"a": [0, 1]}})

    plan = public.make_plan(request["request_id"], {"a": 0, "b": 0})
    answers = iter([True])
    result = learner.localize_rejection(request, plan, lambda _plan, _reason: next(answers))
    assert result == []
    assert learner.active_atoms() == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("pop", "request_id", None), "request_fields"),
        (("set", "activities", []), "activity_count"),
        (("set", "activities", ["a", "a"]), "activity_identity"),
        (("set", "horizon", 0), "horizon"),
        (("set_nested", "allowed_starts", {"a": [], "b": [0]}), "allowed_starts"),
        (("set_nested", "durations", {"a": 0, "b": 1}), "duration"),
        (("set_nested", "weights", {"a": 0, "b": 1}), "weight"),
        (("set", "horizon", 1), "window"),
    ],
)
def test_public_request_validation_rejects_each_ambiguous_shape(
    change: tuple[str, str, Any], message: str
) -> None:
    """REQ-CL-7330: every malformed public field fails explicitly."""

    request = _request()
    operation, field, value = change
    if operation == "pop":
        request.pop(field)
    else:
        request[field] = value
    with pytest.raises(public.PublicLearningError, match=message):
        public.validate_public_request(request)


def test_public_state_and_atom_failure_modes_are_atomic() -> None:
    """REQ-CL-7330: corrupt, early, stale, and oversized evidence is rejected."""

    with pytest.raises(public.PublicLearningError, match="state_json"):
        public.PublicConstraintLearner.from_state_bytes(b"{")
    with pytest.raises(public.PublicLearningError, match="state_schema"):
        public.PublicConstraintLearner.from_state_bytes(b"{}")
    canonical = public.PublicConstraintLearner("token").state_bytes()
    with pytest.raises(public.PublicLearningError, match="state_not_canonical"):
        public.PublicConstraintLearner.from_state_bytes(canonical.replace(b",", b", ", 1))
    oversized_state = {
        "schema": public.STATE_SCHEMA,
        "active_version_token": "token",
        "atoms": [],
        "uncertain_compounds": ["x" * public.STATE_CAP_BYTES],
    }
    with pytest.raises(public.PublicLearningError, match="persistent_state_cap"):
        public.PublicConstraintLearner.from_state_bytes(public.canonical_bytes(oversized_state))

    base = public.make_pair_atom(
        "token", "a", "b", 0, {"query_id": "q", "accepted": False, "sequence": 2}
    )
    learner = public.PublicConstraintLearner("other")
    with pytest.raises(public.PublicLearningError, match="version_mismatch"):
        learner.admit_atom(base, current_query_index=2)
    learner = public.PublicConstraintLearner("token")
    missing = {**base, "query_receipts": []}
    with pytest.raises(public.PublicLearningError, match="query_receipt"):
        learner.admit_atom(missing, current_query_index=2)
    accepted = public.make_pair_atom(
        "token", "a", "b", 0, {"query_id": "q", "accepted": True, "sequence": 1}
    )
    with pytest.raises(public.PublicLearningError, match="non_rejection_witness"):
        learner.admit_atom(accepted, current_query_index=1)
    with pytest.raises(public.PublicLearningError, match="early_evidence"):
        learner.admit_atom(base, current_query_index=1)
    changed = {**base, "minimum_gap": 2}
    with pytest.raises(public.PublicLearningError, match="atom_hash"):
        learner.admit_atom(changed, current_query_index=2)
    admitted = learner.admit_atom(base, current_query_index=2)
    assert learner.admit_atom(base, current_query_index=2) == admitted
    tiny = public.PublicConstraintLearner("token", memory_cap_bytes=10)
    with pytest.raises(public.PublicLearningError, match="persistent_state_cap"):
        tiny.admit_atom(base, current_query_index=2)


def test_public_solver_and_localizer_cover_reverse_gap_and_caps() -> None:
    """SCENARIO-CL-7330-COMPOUND: localization stays bounded and conservative."""

    request = _request()
    learner = public.PublicConstraintLearner("opaque-token-one")
    reverse = public.make_plan(request["request_id"], {"a": 1, "b": 0})
    assert public._interval_gap(request, reverse["assignments"], "a", "b") == 0
    plan = public.make_plan(request["request_id"], {"a": 0, "b": 0})
    atoms = learner.localize_rejection(request, plan, lambda _plan, _reason: False)
    assert len(atoms) == 1
    assert learner.propose(request)["assignments"] == {"a": 0, "b": 1}
    assert public._atom_allows_plan(atoms[0], request, public.make_plan("x", {"a": 0}))

    impossible = public.PublicConstraintLearner("opaque-token-one")
    impossible.admit_atom(
        public.make_pair_atom(
            "opaque-token-one",
            "a",
            "b",
            99,
            {"query_id": "q", "accepted": False, "sequence": 1},
        ),
        current_query_index=1,
    )
    with pytest.raises(public.PublicLearningError, match="no_public_candidate"):
        impossible.propose(request)
    tiny = public.PublicConstraintLearner("opaque-token-one", memory_cap_bytes=100)
    with pytest.raises(public.PublicLearningError, match="persistent_state_cap"):
        tiny.localize_rejection(request, plan, lambda _plan, _reason: True)


def _socket_reply_server(endpoint: Path, replies: list[dict[str, Any]]) -> threading.Thread:
    ready = threading.Event()

    def serve() -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(endpoint))
        server.listen(1)
        ready.set()
        connection, _ = server.accept()
        reader = connection.makefile("r", encoding="utf-8")
        writer = connection.makefile("w", encoding="utf-8")
        for reply in replies:
            request = json.loads(reader.readline())
            row = dict(reply)
            if row.pop("copy_query_id", False):
                row["query_id"] = request["query_id"]
            writer.write(json.dumps(row) + "\n")
            writer.flush()
        writer.close()
        reader.close()
        connection.close()
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(2)
    return thread


def test_socket_oracle_checks_budget_shape_and_boolean(tmp_path: Path) -> None:
    """SCENARIO-CL-7330-BOUNDARY: IPC accepts only the declared Boolean response."""

    plan = public.make_plan("r", {"a": 0})
    endpoint = tmp_path / "valid.sock"
    thread = _socket_reply_server(endpoint, [{"copy_query_id": True, "accepted": True}])
    oracle = public._SocketOracle(endpoint, "r")
    oracle.set_request("r")
    assert oracle.query(plan, "main") is True
    oracle.close()
    thread.join()

    for index, reply in enumerate(
        (
            {"query_id": "wrong", "accepted": True},
            {"copy_query_id": True, "accepted": 1},
        )
    ):
        endpoint = tmp_path / f"invalid-{index}.sock"
        thread = _socket_reply_server(endpoint, [reply])
        oracle = public._SocketOracle(endpoint, "r")
        with pytest.raises(
            public.PublicLearningError,
            match="response_shape" if index == 0 else "response_boolean",
        ):
            oracle.query(plan, "main")
        oracle.close()
        thread.join()

    endpoint = tmp_path / "budget.sock"
    thread = _socket_reply_server(endpoint, [])
    oracle = public._SocketOracle(endpoint, "r")
    oracle.call_count = public.QUERY_BUDGET
    with pytest.raises(public.PublicLearningError, match="query_budget"):
        oracle.query(plan, "main")
    oracle.close()
    thread.join()


def test_direct_worker_keeps_child_logic_under_changed_module_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7330-BOUNDARY: direct coverage mirrors the real worker protocol."""

    output_root, _artifact = built_fixture
    paths = experiment.ExperimentPaths.for_output_root(output_root)

    class FakeOracle:
        def __init__(self, _endpoint: Path, request_id: str) -> None:
            self.request_id = request_id
            self.call_count = 0
            self.total_call_count = 0
            self.response_keys = {"accepted", "query_id"}
            self.first_main = True
            self.first_pair = True

        def set_request(self, request_id: str) -> None:
            self.request_id = request_id
            self.call_count = 0

        def query(self, _plan: dict[str, Any], reason: str) -> bool:
            self.call_count += 1
            self.total_call_count += 1
            if reason == "compound_full":
                return False
            if reason == "localization_pair" and self.request_id == "compound-challenge":
                return True
            if reason == "main" and self.first_main:
                self.first_main = False
                return False
            if reason == "localization_pair" and self.first_pair:
                self.first_pair = False
                return False
            return True

        def close(self) -> None:
            return None

    monkeypatch.setattr(public, "_SocketOracle", FakeOracle)
    output = tmp_path / "direct-worker.json"
    evidence = public.run_worker(paths.public_manifest, tmp_path / "fake.sock", output)
    assert len(evidence["rows"]) == 48
    assert output.is_file()


def test_private_executor_controls_are_exhaustive_and_suffix_independent() -> None:
    """SCENARIO-CL-7330-EXECUTOR: independent evaluator controls fail closed."""

    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-u", str(PRIVATE_EXECUTOR), "--self-test"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    controls = json.loads(completed.stdout.strip().splitlines()[-1])
    assert controls["passed"] is True
    assert controls["exhaustive_tiny_domain"]["enumerated"] == 8
    assert controls["exhaustive_tiny_domain"]["accepted"] > 0
    assert controls["exhaustive_tiny_domain"]["rejected"] > 0
    assert controls["malformed_rejection_count"] >= 5
    assert controls["different_rules_change_label"] is True
    assert controls["same_rules_different_tokens_same_label"] is True
    assert controls["old_suffix_lookup_mismatch_exposed"] is True

    tree = ast.parse(PRIVATE_EXECUTOR.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules = {node.module or "" for node in imports}
    assert all("public_learner" not in module for module in modules)
    assert all("experiment_7323" not in module for module in modules)


def test_manifests_have_fresh_cohorts_balanced_twins_and_separate_seals(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7330-PANEL: cohort counts and live twins are fixed."""

    output_root, artifact = built_fixture
    paths = experiment.ExperimentPaths.for_output_root(output_root)
    public_manifest = json.loads(paths.public_manifest.read_text(encoding="utf-8"))
    private_manifest = json.loads(paths.private_manifest.read_text(encoding="utf-8"))
    assert len(public_manifest["development_streams"]) == 4
    assert len(public_manifest["held_out_streams"]) == 16
    assert {row["cohort"] for row in public_manifest["held_out_streams"]} == {
        "stable_rules",
        "announced_changes",
        "return_to_prior_version",
        "unannounced_changes",
    }
    assert all(len(row["requests"]) == 12 for row in public_manifest["held_out_streams"])
    assert len(public_manifest["live_proposal_panel"]) == 24
    for cohort in {row["cohort"] for row in public_manifest["live_proposal_panel"]}:
        cohort_rows = [
            row for row in public_manifest["live_proposal_panel"] if row["cohort"] == cohort
        ]
        assert len(cohort_rows) == 6
        assert sum(row["presentation_order"] == "original_first" for row in cohort_rows) == 3
    assert all(
        row["original"]["request_id"] != row["twin"]["request_id"]
        for row in public_manifest["live_proposal_panel"]
    )
    public_text = paths.public_manifest.read_text(encoding="utf-8")
    assert "private_rules" not in public_text
    assert "acceptance_witness" not in public_text
    assert private_manifest["public_manifest_sha256"] == public.sha256_file(paths.public_manifest)
    assert (
        artifact["public_manifest"]["public_sha256"]
        != artifact["public_manifest"]["evaluator_only_sha256"]
    )


def test_real_process_boundary_exposes_only_boolean_responses(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7330-BOUNDARY: learner files and imports exclude private rules."""

    _output_root, artifact = built_fixture
    receipt = artifact["executor_boundary_receipt"]
    assert receipt["process_separation_passed"] is True
    assert receipt["learner_private_access_count"] == 0
    assert receipt["held_out_or_live_query_count"] == 0
    assert receipt["evaluator_response_keys"] == ["accepted", "query_id"]
    assert receipt["hostile_process_security_claim"] is False
    assert artifact["compound_conflict_challenge"]["full_rejected"] is True
    assert artifact["compound_conflict_challenge"]["all_pair_projections_accepted"] is True
    assert artifact["compound_conflict_challenge"]["learned_atom_count"] == 0


def test_independent_reduction_and_terminal_contract(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7330-TERMINAL: raw rows independently own readiness."""

    output_root, artifact = built_fixture
    paths = experiment.ExperimentPaths.for_output_root(output_root)
    reduction = experiment.independent_reduce(paths)
    assert reduction["row_count"] == 48
    assert reduction["completed_units"] == 48
    assert reduction["censored_units"] == 0
    assert reduction["held_out_or_live_query_count"] == 0
    assert artifact["executor_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert experiment.validate_artifact(artifact, check_files=True, require_validation=False) == []
    assert artifact["reproducibility_checksum"] == experiment.reproducibility_checksum(artifact)


def test_tampering_and_failed_preconditions_cannot_keep_readiness(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7330: failed gates and changed rows fail closed."""

    output_root, artifact = built_fixture
    tampered = deepcopy(artifact)
    tampered["rows"][0]["executor_calls"] += 1
    assert "reproducibility_checksum" in experiment.validate_artifact(
        tampered, check_files=False, require_validation=False
    )

    checks, hashes = experiment.collect_preconditions(
        ROOT, overrides={"current_task_not_quarantined": False}
    )
    blocked = experiment.build_blocked_artifact(checks, hashes)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["executor_fixture_ready_score"] == 0
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["first_failure"]["field"] == "experiment_7330"
    assert experiment.validate_artifact(blocked, check_files=False, require_validation=False) == []
    blocked_from_builder = experiment.build_artifact(
        ROOT,
        output_root / "blocked",
        progress=False,
        precondition_overrides={"current_task_not_quarantined": False},
    )
    assert blocked_from_builder["status"] == "blocked"


def test_manifest_mutations_and_validation_receipts_fail_closed(
    built_fixture: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """SCENARIO-CL-7330-PANEL: every seal and count mutation is diagnosed."""

    output_root, artifact = built_fixture
    paths = experiment.ExperimentPaths.for_output_root(output_root)
    manifest = json.loads(paths.public_manifest.read_text(encoding="utf-8"))
    mutations = {
        "development_stream_count": lambda row: row["development_streams"].pop(),
        "held_out_stream_count": lambda row: row["held_out_streams"].pop(),
        "request_count": lambda row: row["development_streams"][0]["requests"].pop(),
        "cohorts": lambda row: row["held_out_streams"][0].update(cohort="wrong"),
        "cohort_size": lambda row: row["held_out_streams"][0].update(cohort="announced_changes"),
        "live_pair_count": lambda row: row["live_proposal_panel"].pop(),
        "live_balance": lambda row: row["live_proposal_panel"][0].update(
            presentation_order="twin_first"
        ),
        "public_manifest_hash": lambda row: row.update(manifest_hash="sha256:bad"),
        "activity_count": lambda row: row["development_streams"][0]["requests"][0].update(
            activities=["a"]
        ),
        "unique_requests": lambda row: row["development_streams"][0]["requests"].__setitem__(
            1, deepcopy(row["development_streams"][0]["requests"][0])
        ),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(manifest)
        mutate(changed)
        assert expected in experiment._manifest_errors(changed)

    invalid = deepcopy(artifact)
    invalid["required_checks_passed"] = False
    invalid["validation_receipts"] = [{"command": ""}]
    invalid["reproducibility_checksum"] = experiment.reproducibility_checksum(invalid)
    errors = experiment.validate_artifact(invalid, check_files=False, require_validation=True)
    assert {"validation_receipts", "required_checks_passed", "required_receipts"} <= set(errors)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", invalid)


def test_loader_child_failure_and_raw_hash_mutation_are_detected(
    built_fixture: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """REQ-CL-7330: malformed objects, child failures, and changed sidecars reject."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected_object"):
        experiment._load_object(non_object)
    with pytest.raises(RuntimeError, match="child_failed"):
        experiment._run_streamed(
            [str(ROOT / ".venv/bin/python"), "-c", "raise SystemExit(3)"],
            ROOT,
            tmp_path / "failed-child.log",
        )

    output_root, artifact = built_fixture
    paths = experiment.ExperimentPaths.for_output_root(output_root)
    original = paths.public_manifest.read_bytes()
    try:
        paths.public_manifest.write_bytes(original + b" ")
        assert "raw_hash_public_manifest" in experiment.validate_artifact(
            artifact, check_files=True, require_validation=False
        )
    finally:
        paths.public_manifest.write_bytes(original)


def test_failed_fixture_gate_sets_disqualified_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7330-TERMINAL: a current fixture failure zeros readiness."""

    monkeypatch.setattr(experiment, "_manifest_errors", lambda _manifest: ["injected"])
    artifact = experiment.build_artifact(ROOT, tmp_path, progress=False)
    assert artifact["executor_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified")


def test_atomic_writer_and_cold_file_validation(
    tmp_path: Path, built_fixture: tuple[Path, dict[str, Any]]
) -> None:
    """REQ-CL-7330: only a validated complete JSON object is published."""

    _output_root, artifact = built_fixture
    destination = tmp_path / "terminal.json"
    experiment.write_artifact(destination, artifact, require_validation=False)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert experiment.sha256_file(destination).startswith("sha256:")
