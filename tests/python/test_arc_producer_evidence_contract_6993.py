"""Prospective ARC engine evidence tests (REQ-ARC-WMTE-6993).

The tests use only local byte fixtures. They never create an ARC environment or
start a model server.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_6993_arc_producer_evidence_contract as experiment
from carnot.agentic import arc_executable_world_model as world
from carnot.agentic import arc_producer_evidence as evidence
from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.agentic.arc_producer_evidence import (
    EVIDENCE_ENVELOPE_SCHEMA,
    REQUIRED_ENVELOPE_FIELDS,
    EngineEvidenceTransaction,
    EvidencePublishInterrupted,
    produce_engine_evidence,
    read_evidence_manifest,
    sha256_bytes,
)
from carnot.experiment_6993_arc_producer_evidence_contract import build_artifact


GAME = "fixture6993"
STAMP = "20260904T120000_000000"
PROMPT = b"raw\x00prompt\xffbytes\r\n"
ENGINE = b"""import numpy as np

def engine(grid, action, data):
    result = np.asarray(grid).copy()
    if int(action) == 2:
        result[0, 0] = 7
    return result

def is_level_complete(grid):
    return False
"""
TRANSITIONS = [
    {
        "index": 0,
        "grid": [[0, 0], [0, 0]],
        "action": 1,
        "data": None,
        "next_grid": [[0, 0], [0, 0]],
        "level_before": 0,
        "level_after": 0,
    },
    {
        "index": 1,
        "grid": [[0, 0], [0, 0]],
        "action": 2,
        "data": {"y": 0, "x": 0},
        "next_grid": [[7, 0], [0, 0]],
        "level_before": 0,
        "level_after": 0,
    },
]


def _produce(
    root: Path,
    *,
    game: str = GAME,
    run_id: str = "run-success",
    source_kind: str | None = "live_agent_attempts",
    failpoint: str | None = None,
    engine_factory=None,
):
    return produce_engine_evidence(
        store_root=root,
        game=game,
        raw_prompt=PROMPT,
        transitions=TRANSITIONS,
        transition_source_kind=source_kind,
        environment_receipt={"environment": "deterministic_stub", "game": game},
        synthesize_engine=engine_factory or (lambda: ENGINE),
        run_id=run_id,
        timestamp=STAMP,
        failpoint=failpoint,
    )


def _only_validation(root: Path, game: str = GAME) -> dict:
    rows = read_evidence_manifest(root, game)
    assert len(rows) == 1
    return rows[0]


def test_atomic_write_preserves_raw_prompt_transition_order_and_hashes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6993-ATOMIC-PUBLISH: stage sources before synthesis."""

    def synthesize() -> bytes:
        staging = tmp_path / GAME / "attempts" / ".evidence-staging" / "run-success"
        assert (staging / "prompt.raw").read_bytes() == PROMPT
        rows = [
            json.loads(line) for line in (staging / "transitions.jsonl").read_bytes().splitlines()
        ]
        assert [row["index"] for row in rows] == [0, 1]
        assert [row["action"] for row in rows] == [1, 2]
        assert rows[1]["data"] == {"x": 0, "y": 0}
        assert not (staging / "engine.py").exists()
        return ENGINE

    result = _produce(tmp_path, engine_factory=synthesize)
    validation = _only_validation(tmp_path)
    envelope = json.loads(result.envelope_path.read_text(encoding="utf-8"))

    assert validation["eligible"] is True
    assert set(REQUIRED_ENVELOPE_FIELDS) <= set(envelope)
    assert envelope["schema"] == EVIDENCE_ENVELOPE_SCHEMA
    assert result.prompt_path.read_bytes() == PROMPT
    assert envelope["raw_prompt_sha256"] == sha256_bytes(PROMPT)
    assert envelope["transition_count"] == 2
    assert envelope["engine_sha256"] == sha256_bytes(ENGINE)
    for field in (
        "environment_receipt_sha256",
        "scorer_sha256",
        "live_policy_sha256",
        "agent_factory_sha256",
        "manifest_row_sha256",
        "envelope_sha256",
    ):
        assert str(envelope[field]).startswith("sha256:")
        assert len(envelope[field]) == 71
    manifest_lines = result.manifest_path.read_bytes().splitlines()
    assert len(manifest_lines) == 1
    assert json.loads(manifest_lines[0])["run_id"] == "run-success"
    assert not list(result.manifest_path.parent.rglob("*.tmp"))


@pytest.mark.parametrize(
    "failpoint",
    ["after_engine_temp", "after_bundle_publish", "after_engine_publish", "before_manifest"],
)
def test_interrupted_publish_never_becomes_eligible(tmp_path: Path, failpoint: str) -> None:
    """SCENARIO-ARC-WMTE-6993-INTERRUPTION: no final row means no complete evidence."""

    with pytest.raises(EvidencePublishInterrupted, match=failpoint):
        _produce(tmp_path, run_id=f"run-{failpoint}", failpoint=failpoint)

    assert read_evidence_manifest(tmp_path, GAME) == []
    manifest = tmp_path / GAME / "attempts" / "manifest.jsonl"
    assert not manifest.exists() or manifest.read_bytes() == b""


@pytest.mark.parametrize(
    ("field", "suffix"),
    [
        ("raw_prompt_path", b"changed"),
        ("transition_jsonl_path", b"{}\n"),
        ("engine_path", b"\n# changed\n"),
        ("envelope_path", b" "),
    ],
)
def test_tampered_bytes_are_ineligible(tmp_path: Path, field: str, suffix: bytes) -> None:
    """SCENARIO-ARC-WMTE-6993-HASH-AND-PATH-INTEGRITY: changed bytes fail closed."""

    result = _produce(tmp_path)
    if field == "envelope_path":
        path = result.envelope_path
    else:
        envelope = json.loads(result.envelope_path.read_text(encoding="utf-8"))
        path = tmp_path / envelope[field]
    path.write_bytes(path.read_bytes() + suffix)

    validation = _only_validation(tmp_path)
    assert validation["eligible"] is False
    assert validation["rejection_reasons"]
    assert all(
        set(reason) == {"failed_check", "expected_value", "observed_value"}
        for reason in validation["rejection_reasons"]
    )


def test_escaped_source_path_is_ineligible_even_when_bytes_match(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6993-HASH-AND-PATH-INTEGRITY: stale paths cannot escape."""

    result = _produce(tmp_path)
    envelope = json.loads(result.envelope_path.read_text(encoding="utf-8"))
    prompt = tmp_path / envelope["raw_prompt_path"]
    outside = tmp_path.parent / "outside-prompt.raw"
    outside.write_bytes(prompt.read_bytes())
    prompt.unlink()
    prompt.symlink_to(outside)

    validation = _only_validation(tmp_path)
    assert validation["eligible"] is False
    assert any(
        reason["failed_check"] == "raw_prompt_path_safe"
        for reason in validation["rejection_reasons"]
    )


@pytest.mark.parametrize(
    "source_kind",
    [
        "game_source",
        "hand_adapter",
        "offline_bfs",
        "synthetic_hidden_transition",
        None,
        "unknown_source",
    ],
)
def test_only_live_agent_attempts_are_live_evidence(
    tmp_path: Path, source_kind: str | None
) -> None:
    """SCENARIO-ARC-WMTE-6993-SOURCE-PROVENANCE: reject non-live sources."""

    _produce(tmp_path, source_kind=source_kind)
    validation = _only_validation(tmp_path)

    assert validation["eligible"] is False
    assert any(
        reason["failed_check"] == "transition_source_kind"
        and reason["expected_value"] == "live_agent_attempts"
        and reason["observed_value"] == source_kind
        for reason in validation["rejection_reasons"]
    )


def test_legacy_and_duplicate_runs_remain_visible_but_ineligible(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6993-LEGACY-AND-DUPLICATE: old and repeated rows fail closed."""

    result = _produce(tmp_path)
    original = result.manifest_path.read_bytes()
    legacy = json.dumps(
        {"ts": "20260901T000000_000000", "file": "wm_old.py", "sha256_16": "0" * 16}
    ).encode("utf-8")
    result.manifest_path.write_bytes(legacy + b"\n" + original + original)

    validations = read_evidence_manifest(tmp_path, GAME)
    assert len(validations) == 3
    assert validations[0]["classification"] == "legacy"
    assert validations[0]["eligible"] is False
    duplicates = [row for row in validations if row["run_id"] == "run-success"]
    assert len(duplicates) == 2
    assert all(row["eligible"] is False for row in duplicates)
    assert all(
        any(reason["failed_check"] == "unique_run_id" for reason in row["rejection_reasons"])
        for row in duplicates
    )


def test_complete_envelope_reaches_shipped_factory_and_changes_action_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-6993-FACTORY-REACHABILITY: the real seam uses fixture bytes."""

    result = _produce(tmp_path)
    assert _only_validation(tmp_path)["eligible"] is True

    class FixtureBase:
        def __init__(self) -> None:
            self.game_id = GAME

    agent_type = make_carnot_agent(FixtureBase, cascade=True, proposer=object())
    agent = agent_type()
    assert isinstance(agent._policy, E3AgentPolicy)
    monkeypatch.setattr(world, "E3_DIR", tmp_path)
    engine, goal = world.load_engine(GAME)
    candidates = agent._policy._world_model_candidates(engine, goal)
    routed = next(row for row in candidates if row.name == "loaded_world_model.py")
    grid = np.zeros((2, 2), dtype=np.int64)
    control_score = int(np.count_nonzero(grid != grid))
    fixture_score = int(np.count_nonzero(np.asarray(routed.engine(grid, 2, None)) != grid))

    assert result.engine_path.read_bytes() == ENGINE
    assert fixture_score == 1
    assert fixture_score != control_score


def test_local_proposer_write_seam_publishes_the_pending_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-6993-ATOMIC-PUBLISH: the shipped writer owns publication."""

    proposer = object.__new__(world.LocalGGUFProposer)
    proposer.producer_transition_source_kind = "live_agent_attempts"
    proposer.producer_environment_receipt = {
        "environment": "deterministic_stub",
        "game": GAME,
    }
    proposer.producer_transitions = list(TRANSITIONS)
    monkeypatch.setattr(world, "E3_DIR", tmp_path)
    monkeypatch.setattr(
        world.LocalGGUFProposer,
        "_effective_model_label",
        lambda self: "deterministic_stub",
    )

    proposer._begin_engine_evidence(GAME, PROMPT, TRANSITIONS, run_id="writer-run", timestamp=STAMP)
    ok, _message = proposer._write_world_model(GAME, ENGINE.decode("utf-8"))

    validations = read_evidence_manifest(tmp_path, GAME)
    assert ok is True
    assert len(validations) == 1
    assert validations[0]["eligible"] is True
    assert proposer.last_attempt_archive["evidence_envelope_path"]
    assert (tmp_path / GAME / "world_model.py").read_bytes() == ENGINE


def test_e3_policy_marks_only_its_own_attempt_transitions_as_live() -> None:
    """SCENARIO-ARC-WMTE-6993-SOURCE-PROVENANCE: E3 supplies the live receipt."""

    policy = object.__new__(E3AgentPolicy)
    policy.short = GAME
    proposer = SimpleNamespace()

    policy._configure_producer_evidence(proposer, TRANSITIONS)

    assert proposer.producer_transition_source_kind == "live_agent_attempts"
    assert proposer.producer_transitions == list(TRANSITIONS)
    assert proposer.producer_environment_receipt["policy"] == "E3AgentPolicy"
    assert proposer.producer_environment_receipt["transition_count"] == len(TRANSITIONS)


def test_artifact_has_required_rows_scores_and_false_solve_fields(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6993-NO-SOLVE: fixture success makes no solve claim."""

    artifact = build_artifact(output_path=tmp_path / "artifact.json", execution_date="20260904")

    assert artifact["arc_producer_contract_complete_score"] == 1
    assert artifact["arc_live_path_fixture_ready_score"] == 1
    assert artifact["inference_substrate"] == "deterministic_arc_producer_contract_fixture_no_llm"
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive:")
    assert artifact["solve_provenance_applicable"] is False
    for field in (
        "solve_claimed",
        "level_claimed",
        "registry_updated",
        "submitted_to_leaderboard",
        "model_quality_claimed",
        "verifier_is_oracle",
    ):
        assert artifact[field] is False
    assert artifact["gate_check_summary"] == []
    assert set(artifact["field_principles"]) == set(artifact["required_artifact_fields"])


def test_object_transitions_default_ids_and_source_overrides(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6993: defaults and array-like values retain deterministic bytes."""

    class ItemOnly:
        def item(self) -> int:
            return 4

    transition = SimpleNamespace(
        grid=np.asarray([[0]], dtype=np.int64),
        action=np.int64(2),
        data={"value": ItemOnly()},
        next_grid=np.asarray([[7]], dtype=np.int64),
        level_before=np.int64(0),
        level_after=np.int64(0),
    )
    source = tmp_path / "source.py"
    source.write_bytes(b"SOURCE\n")
    result = produce_engine_evidence(
        store_root=tmp_path,
        game="defaults",
        raw_prompt=b"p",
        transitions=[transition],
        transition_source_kind="live_agent_attempts",
        environment_receipt={"scalar": ItemOnly()},
        synthesize_engine=lambda: ENGINE.decode("utf-8"),
        source_paths={name: source for name in ("scorer", "live_policy", "agent_factory")},
    )
    row = json.loads(result.transition_path.read_bytes().splitlines()[0])

    assert result.row["run_id"].startswith("arc-")
    assert row["grid"] == [[0]]
    assert row["data"] == {"value": 4}
    assert json.loads((result.engine_path.parent / "environment_receipt.json").read_bytes()) == {
        "scalar": 4
    }
    assert (result.engine_path.parent / "scorer.py").read_bytes() == b"SOURCE\n"


def test_unsafe_and_repeated_transaction_identifiers_are_rejected(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6993-HASH-AND-PATH-INTEGRITY: identifiers stay local and unique."""

    begin = dict(
        store_root=tmp_path,
        game="safe",
        raw_prompt=b"p",
        transitions=[],
        transition_source_kind="live_agent_attempts",
        environment_receipt={},
        run_id="same-run",
        timestamp=STAMP,
    )
    EngineEvidenceTransaction.begin(**begin)
    with pytest.raises(FileExistsError, match="already exists"):
        EngineEvidenceTransaction.begin(**begin)
    with pytest.raises(ValueError, match="unsafe game"):
        EngineEvidenceTransaction.begin(**{**begin, "game": "../escape", "run_id": "other"})
    with pytest.raises(ValueError, match="unsafe run_id"):
        EngineEvidenceTransaction.begin(**{**begin, "game": "other", "run_id": "../escape"})


def test_short_manifest_append_and_missing_sources_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-6993-INTERRUPTION: short appends and missing bytes are ineligible."""

    transaction = EngineEvidenceTransaction.begin(
        store_root=tmp_path,
        game="short_write",
        raw_prompt=b"p",
        transitions=TRANSITIONS,
        transition_source_kind="live_agent_attempts",
        environment_receipt={},
        run_id="short-run",
        timestamp=STAMP,
    )
    with monkeypatch.context() as local_patch:
        local_patch.setattr(evidence.os, "write", lambda descriptor, value: len(value) - 1)
        with pytest.raises(OSError, match="short manifest append"):
            transaction.publish(ENGINE)
    assert read_evidence_manifest(tmp_path, "short_write") == []

    result = _produce(tmp_path, game="missing_source", run_id="missing-run")
    result.prompt_path.unlink()
    validation = _only_validation(tmp_path, "missing_source")
    assert validation["eligible"] is False
    assert any(
        row["failed_check"] == "raw_prompt_sha256" for row in validation["rejection_reasons"]
    )


def test_reader_handles_invalid_paths_envelopes_transitions_and_rows(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6993: unreadable input is visible and never eligible."""

    assert evidence._safe_path(tmp_path, None) is None
    assert evidence._safe_path(tmp_path, str(tmp_path / "absolute")) is None
    assert evidence._read_envelope(tmp_path, {}) == (None, None)
    assert evidence._read_envelope(tmp_path, {"envelope": {"path": "missing"}}) == (None, None)
    malformed = tmp_path / "bad-envelope.json"
    malformed.write_bytes(b"not-json")
    assert evidence._read_envelope(tmp_path, {"envelope": {"path": malformed.name}}) == (None, None)
    malformed.write_bytes(b"[]")
    assert evidence._read_envelope(tmp_path, {"envelope": {"path": malformed.name}})[0] is None

    result = _produce(tmp_path, game="bad_transition", run_id="bad-transition-run")
    result.transition_path.write_bytes(b"not-json\n")
    validation = _only_validation(tmp_path, "bad_transition")
    assert validation["eligible"] is False
    assert any(row["failed_check"] == "transition_count" for row in validation["rejection_reasons"])

    manifest = tmp_path / "malformed" / "attempts" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"{bad\n[]\n")
    malformed_rows = read_evidence_manifest(tmp_path, "malformed")
    assert [row["classification"] for row in malformed_rows] == ["malformed", "malformed"]
    assert all(row["eligible"] is False for row in malformed_rows)


def test_artifact_blocked_partial_helpers_and_command_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-6993: blocked and partial artifacts keep terminal verdict prefixes."""

    assert experiment._file_hash(tmp_path / "missing") is None
    monkeypatch.setattr(
        experiment.inspect, "getsourcelines", lambda value: (_ for _ in ()).throw(OSError())
    )
    trace = experiment._trace("make_carnot_agent", make_carnot_agent, True)
    assert trace["line"] is None

    blocked = build_artifact(output_path=tmp_path / "blocked.json", execution_date="wrong")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["failed_check"] == "execution_date"

    monkeypatch.setattr(
        experiment,
        "_factory_fixture",
        lambda root, game: (
            [{"called_by_fixture": False}],
            {"selected_candidate_name": "loaded_world_model.py"},
            {"score_changed": False},
        ),
    )
    partial = build_artifact(output_path=tmp_path / "partial.json", execution_date="20260904")
    assert partial["verdict_class"] == "partial"
    assert partial["honest_verdict"].startswith("partial:")

    monkeypatch.setattr(
        experiment,
        "build_artifact",
        lambda **kwargs: {
            "verdict_class": "positive",
            "honest_verdict": "positive: ok",
            "arc_producer_contract_complete_score": 1,
            "arc_live_path_fixture_ready_score": 1,
        },
    )
    assert experiment.main(["--date", "20260904", "--output", str(tmp_path / "main.json")]) == 0
    assert '"verdict_class": "positive"' in capsys.readouterr().out
    monkeypatch.setattr(
        experiment,
        "build_artifact",
        lambda **kwargs: {
            "verdict_class": "blocked",
            "honest_verdict": "blocked: no",
            "arc_producer_contract_complete_score": 0,
            "arc_live_path_fixture_ready_score": 0,
        },
    )
    assert experiment.main(["--date", "bad", "--output", str(tmp_path / "main-bad.json")]) == 1
