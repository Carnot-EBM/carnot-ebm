"""Focused tests for REQ-ARC-WMTE-6921 banked supervisor credit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6921_arc_dynamic_supervisor_banked_credit as exp
from carnot.agentic import arc_solve_artifact_discipline as discipline
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER


REPO = Path(__file__).resolve().parents[2]


def test_req_arc_wmte_6921_registers_exact_no_llm_audit_substrate() -> None:
    """REQ-ARC-WMTE-6921 gives deterministic receipt audit its reviewed fast floor."""

    substrate = "deterministic_cpu_audit_of_live_arc_receipts_no_new_llm"
    artifact = {
        "honest_verdict": "complete_no_new_eligible_receipts",
        "duration_s": 0.001,
        "inference_substrate": substrate,
    }

    assert discipline.ARC_DYNAMIC_SUPERVISOR_BANKED_CREDIT_SUBSTRATE == substrate
    assert discipline.duration_floor_s(substrate) == 0.0001
    assert discipline.validate_arc_solve_artifact(artifact) == []

    import importlib.util
    import sys

    path = REPO / "scripts" / "adversarial_verify.py"
    module_spec = importlib.util.spec_from_file_location("adversarial_verify_exp6921", path)
    assert module_spec is not None and module_spec.loader is not None
    verifier = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = verifier
    module_spec.loader.exec_module(verifier)
    classification = verifier._classify_inference_substrate(artifact)
    assert classification == {
        "kind": "no_llm",
        "declared_value": substrate,
        "matched_value": substrate,
        "source": "top_level_inference_substrate",
    }


def _receipt_row(
    *,
    game: str = "r11l",
    seed: int = 7,
    arm: str = "E3_default_llmon",
    levels: int = 1,
    level_actions: list[int] | None = None,
    redirects: list[dict] | None = None,
    mode: str = "applied",
    error: str | None = None,
    solve_provenance: str = "live_agent_self_discovery",
) -> dict:
    """Build one small live row with durable level timing."""

    if level_actions is None:
        level_actions = [200] if levels else []
    if redirects is None:
        redirects = [
            {
                "arm": "drop_goal_bias",
                "action_index": 100,
                "level": 0,
                "resolved_by_levelup": True,
                "actions_to_levelup": 100,
            }
        ]
    if error is not None:
        receipt = {"error": error}
    elif mode == "shadow":
        receipt = {
            "enabled": False,
            "mode": "shadow",
            "would_have_redirects": redirects,
            "would_have_arm_outcomes": {},
        }
    else:
        receipt = {
            "enabled": True,
            "mode": "applied",
            "redirects": redirects,
            "arm_outcomes": {
                item["arm"]: {
                    "fired": 1,
                    "helped": int(item.get("resolved_by_levelup") is True),
                }
                for item in redirects
            },
            "actions_observed": 240,
            "stagnations_unredirected": 0,
            "window": 120,
        }
    return {
        "game": game,
        "seed": seed,
        "arm": arm,
        "levels": levels,
        "reached": levels,
        "actions": 240,
        "charged_actions": 240,
        "level_up_charged": level_actions,
        "level_progress": bool(level_actions),
        "solve_provenance": solve_provenance,
        "trajectory_supervisor": receipt,
    }


def _document(rows: list[dict], *, complete: bool = True, run_id: str = "run-7") -> dict:
    """Wrap rows in the canonical scored-path result shape."""

    return {
        "experiment": "arc_leaderboard_eval",
        "policy": "e3",
        "budget": 400,
        "random_seed": 7,
        "run_id": run_id,
        "source_commit": "a" * 40,
        "complete": complete,
        "honest_verdict": (
            "complete_leaderboard_eval" if complete else "partial_1_of_2_games_run_in_progress"
        ),
        "per_game": rows,
    }


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _prepare_root(tmp_path: Path) -> tuple[list[dict], dict[str, str]]:
    """Create all preconditions without touching repository evidence."""

    for relative in exp.CANONICAL_ENTRYPOINTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture entrypoint\n", encoding="utf-8")
    (tmp_path / exp.SCORED_PATH_RUN_DIRECTORY).mkdir(parents=True)
    (tmp_path / "lever-runs").mkdir()
    _write_json(
        tmp_path / exp.LEDGER_PATH,
        {
            "schema": exp.LEDGER_SCHEMA,
            "created_at": None,
            "updated_at": None,
            "entries": {},
            "recommendation": None,
        },
    )
    registry = tmp_path / exp.REGISTRY_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(yaml.safe_dump({"schema_version": 1, "games": []}), encoding="utf-8")
    expected: dict[str, str] = {}
    for name, relative in exp.PRIOR_ARTIFACT_PATHS.items():
        path = _write_json(tmp_path / relative, {"artifact": name, "terminal": True})
        expected[name] = exp.sha256_file(path)
    roots = [
        {"kind": "scored_path", "path": exp.SCORED_PATH_RUN_DIRECTORY.as_posix()},
        {"kind": "lever_harness", "path": "lever-runs"},
    ]
    return roots, expected


def _build(tmp_path: Path, roots: list[dict], expected: dict[str, str]) -> dict:
    return exp.build_artifact(
        tmp_path,
        run_date="20260903",
        duration_s=0.25,
        discovery_roots=roots,
        expected_prior_hashes=expected,
        source_commit_fallback="b" * 40,
    )


def test_req_arc_wmte_6921_spec_owns_required_contract() -> None:
    """REQ-ARC-WMTE-6921 declares all focused scenarios and fields."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-ARC-WMTE-6921", 1)[1]
    for anchor in (
        "SCENARIO-ARC-WMTE-6921-HARD-CODED-SOURCE",
        "SCENARIO-ARC-WMTE-6921-NESTED-CLONE-AND-COPY-DEDUPE",
        "SCENARIO-ARC-WMTE-6921-SHADOW-AND-ERROR-DENOMINATORS",
        "SCENARIO-ARC-WMTE-6921-TRANSIENT-PROGRESS-IS-NOT-CREDIT",
        "SCENARIO-ARC-WMTE-6921-BANKED-LEVEL-CREDIT",
        "SCENARIO-ARC-WMTE-6921-ACTIONS-MISMATCH-AND-ORDER",
        "SCENARIO-ARC-WMTE-6921-COMPETING-REDIRECTS",
        "SCENARIO-ARC-WMTE-6921-NO-NEW-ROW",
        "SCENARIO-ARC-WMTE-6921-AUTOMATIC-MUTATION-FORBIDDEN",
        "SCENARIO-ARC-WMTE-6921-ARTIFACT",
    ):
        assert anchor in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section


def test_scenario_6921_hard_coded_source_regression(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-HARD-CODED-SOURCE finds arbitrary names by content."""

    roots, expected = _prepare_root(tmp_path)
    path = _write_json(
        tmp_path / exp.SCORED_PATH_RUN_DIRECTORY / "future-name-not-in-source.data.json",
        _document([_receipt_row()]),
    )

    artifact = _build(tmp_path, roots, expected)

    assert artifact["arc_supervisor_audit_complete_score"] == 1
    assert artifact["new_eligible_receipt_count"] == 1
    assert artifact["discovered_file_rows"][0]["path"] == path.relative_to(tmp_path).as_posix()
    assert artifact["applied_receipt_rows"][0]["source_commit"] == "a" * 40


def test_scenario_6921_nested_clone_and_copied_row_dedupe(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-NESTED-CLONE-AND-COPY-DEDUPE blocks inflation."""

    roots, expected = _prepare_root(tmp_path)
    row = _receipt_row()
    _write_json(tmp_path / "lever-runs" / "original.json", {"rows": [row]})
    _write_json(tmp_path / "lever-runs" / "copy.json", {"rows": [deepcopy(row)]})
    clone = tmp_path / "lever-runs" / "nested-clone"
    clone.mkdir()
    (clone / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    _write_json(clone / "hidden.json", {"rows": [_receipt_row(seed=99)]})

    artifact = _build(tmp_path, roots, expected)

    assert len(artifact["applied_receipt_rows"]) == 1
    assert any(row["disposition"] == "duplicate_copy" for row in artifact["dedupe_rows"])
    assert all("nested-clone" not in row["path"] for row in artifact["discovered_file_rows"])


def test_scenario_6921_shadow_and_error_admission(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-SHADOW-AND-ERROR-DENOMINATORS keeps non-effects."""

    roots, expected = _prepare_root(tmp_path)
    rows = [_receipt_row(), _receipt_row(seed=8, mode="shadow"), _receipt_row(seed=9, error="boom")]
    _write_json(tmp_path / "lever-runs" / "mixed.json", {"rows": rows})

    artifact = _build(tmp_path, roots, expected)

    assert len(artifact["applied_receipt_rows"]) == 1
    assert len(artifact["shadow_receipt_rows"]) == 1
    assert len(artifact["error_receipt_rows"]) == 1
    assert artifact["refinement_input_receipt_count"] == 1


def test_scenario_6921_transient_progress_has_no_banked_credit(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-TRANSIENT-PROGRESS-IS-NOT-CREDIT rejects helped."""

    roots, expected = _prepare_root(tmp_path)
    row = _receipt_row(levels=0, level_actions=[150])
    _write_json(tmp_path / "lever-runs" / "transient.json", {"rows": [row]})

    artifact = _build(tmp_path, roots, expected)
    redirect = artifact["redirect_rows"][0]

    assert redirect["old_credit"] is True
    assert redirect["banked_credit"] is False
    assert redirect["actions_to_banked_progress"] is None
    assert redirect["censored"] is True
    assert redirect["no_progress_outcome"] is True
    assert artifact["banked_progress_event_count"] == 0


def test_scenario_6921_banked_level_credit(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-BANKED-LEVEL-CREDIT computes strict later credit."""

    roots, expected = _prepare_root(tmp_path)
    _write_json(tmp_path / "lever-runs" / "banked.json", {"rows": [_receipt_row()]})

    artifact = _build(tmp_path, roots, expected)
    redirect = artifact["redirect_rows"][0]

    assert redirect["banked_credit"] is True
    assert redirect["actions_to_banked_progress"] == 100
    assert redirect["banked_transition_action_index"] == 200
    assert artifact["banked_level_transition_rows"][0]["level_after"] == 1


def test_scenario_6921_actions_mismatch_and_post_redirect_order(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-ACTIONS-MISMATCH-AND-ORDER trusts replay."""

    roots, expected = _prepare_root(tmp_path)
    redirect = {
        "arm": "drop_goal_bias",
        "action_index": 100,
        "level": 1,
        "resolved_by_levelup": True,
        "actions_to_levelup": 20,
    }
    row = _receipt_row(levels=2, level_actions=[90, 180], redirects=[redirect])
    _write_json(tmp_path / "lever-runs" / "ordered.json", {"rows": [row]})

    artifact = _build(tmp_path, roots, expected)
    replay = artifact["redirect_rows"][0]

    assert replay["banked_transition_action_index"] == 180
    assert replay["actions_to_banked_progress"] == 80
    assert replay["actions_to_progress_mismatch"] is True
    assert artifact["actions_to_progress_rows"][0]["authoritative_source"] == "banked_replay"


def test_scenario_6921_competing_redirects_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-COMPETING-REDIRECTS exposes shared credit."""

    roots, expected = _prepare_root(tmp_path)
    redirects = [
        {
            "arm": "drop_goal_bias",
            "action_index": 100,
            "level": 0,
            "resolved_by_levelup": True,
            "actions_to_levelup": 100,
        },
        {
            "arm": "allow_reinduction",
            "action_index": 120,
            "level": 0,
            "resolved_by_levelup": True,
            "actions_to_levelup": 80,
        },
    ]
    _write_json(
        tmp_path / "lever-runs" / "competing.json",
        {"rows": [_receipt_row(redirects=redirects)]},
    )

    artifact = _build(tmp_path, roots, expected)

    assert len(artifact["competing_redirect_rows"]) == 2
    assert all(row["competing_redirect_count"] == 1 for row in artifact["redirect_rows"])
    assert artifact["refinement_policy"]["recommendation_only"] is True


def test_scenario_6921_no_new_row_state(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-NO-NEW-ROW reports an honest terminal null."""

    roots, expected = _prepare_root(tmp_path)
    row = _receipt_row()
    row_id = exp.canonical_row_id(row)
    ledger_path = tmp_path / exp.LEDGER_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["entries"][row_id] = {"receipt_id": row_id}
    _write_json(ledger_path, ledger)
    _write_json(tmp_path / "lever-runs" / "already-known.json", {"rows": [row]})

    artifact = _build(tmp_path, roots, expected)

    assert artifact["new_eligible_receipt_count"] == 0
    assert artifact["banked_credit_eligible_score"] == 0
    assert artifact["honest_verdict"] == "complete_no_new_eligible_receipts"


def test_scenario_6921_frozen_floor_controls_eligibility(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6921 keeps the existing evidence floor unchanged."""

    roots, expected = _prepare_root(tmp_path)
    redirects = [
        {
            "arm": "drop_goal_bias",
            "action_index": index,
            "level": 0,
            "resolved_by_levelup": False,
            "actions_to_levelup": None,
        }
        for index in range(10, 110, 10)
    ]
    _write_json(
        tmp_path / "lever-runs" / "floor.json",
        {"rows": [_receipt_row(level_actions=[200], redirects=redirects)]},
    )

    artifact = _build(tmp_path, roots, expected)

    arm = next(row for row in artifact["per_arm_rows"] if row["arm"] == "drop_goal_bias")
    assert arm["fired"] == exp.MIN_FIRED_PER_ARM
    assert arm["meets_floor"] is True
    assert artifact["banked_credit_eligible_score"] == 1


def test_scenario_6921_automatic_arm_mutation_is_forbidden(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-AUTOMATIC-MUTATION-FORBIDDEN leaves sources unchanged."""

    roots, expected = _prepare_root(tmp_path)
    _write_json(tmp_path / "lever-runs" / "one.json", {"rows": [_receipt_row()]})
    ledger_path = tmp_path / exp.LEDGER_PATH
    registry_path = tmp_path / exp.REGISTRY_PATH
    before = (tuple(ARM_ORDER), ledger_path.read_bytes(), registry_path.read_bytes())

    artifact = _build(tmp_path, roots, expected)

    after = (tuple(ARM_ORDER), ledger_path.read_bytes(), registry_path.read_bytes())
    assert artifact["automatic_arm_mutation_count"] == 0
    assert artifact["arc_run_launch_count"] == 0
    assert artifact["game_adapter_mutation_count"] == 0
    assert before == after


def test_req_arc_wmte_6921_precondition_failure_is_complete_blocked(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6921 writes the exact failed hash gate before discovery."""

    roots, expected = _prepare_root(tmp_path)
    expected["experiment_6844"] = "sha256:" + "0" * 64

    artifact = _build(tmp_path, roots, expected)

    assert artifact["arc_supervisor_audit_complete_score"] == 0
    assert artifact["honest_verdict"] == "complete_blocked_arc_dynamic_supervisor_banked_credit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "prior_artifact_hash:experiment_6844"
    assert artifact["gate_check_summary"]["expected"] == expected["experiment_6844"]


def test_scenario_6921_artifact_schema_checksum_and_atomic_writer(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-ARTIFACT validates and writes one stable document."""

    roots, expected = _prepare_root(tmp_path)
    _write_json(tmp_path / "lever-runs" / "one.json", {"rows": [_receipt_row()]})
    artifact = _build(tmp_path, roots, expected)
    output = tmp_path / exp.OUTPUT_PATH

    assert exp.validate_artifact(artifact) == []
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    exp.write_artifact_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["solve_claim"] is False
    assert artifact["new_solve_count"] == 0
    assert all(
        row["solve_provenance"] == "live_agent_self_discovery" for row in artifact["per_game_rows"]
    )


def test_req_arc_wmte_6921_helper_defenses_and_frame_fallback(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6921 fails closed on malformed and non-live source shapes."""

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("[", encoding="utf-8")
    assert exp._read_json(invalid_json)[1].startswith("JSONDecodeError:")
    assert exp._read_yaml(invalid_yaml)[1].startswith("ParserError:")
    assert exp._relative_or_absolute(Path("/outside"), tmp_path) == "/outside"
    assert exp._git_head(REPO) != "unrecorded_source_commit"
    assert exp._git_head(tmp_path) == "unrecorded_source_commit"
    assert exp._default_discovery_roots(tmp_path, {"entries": {}})[1]["path"] == (
        "missing_ledger_source_directory"
    )
    ledger_roots = exp._default_discovery_roots(
        tmp_path,
        {"entries": {"one": {"source": str(tmp_path / "lever" / "rows.json")}}},
    )
    assert Path(ledger_roots[1]["path"]) == tmp_path / "lever"
    assert exp._source_hash_row("missing", tmp_path / "missing", tmp_path)["exists"] is False
    assert exp._source_rows([{"game": "x"}, "skip"])[1] == "bare_rows"
    assert exp._source_rows("bad") == ([], None)
    assert exp._document_terminal([], None) is False
    assert exp._document_terminal({"complete": False}, "lever_rows") is False
    assert exp._run_id({}, tmp_path / "run.partial.json") == "run"

    roots, expected = _prepare_root(tmp_path)
    no_receipt = _write_json(tmp_path / "lever-runs" / "ordinary.json", {"rows": [{}]})
    discovered = exp.discover_receipts(
        tmp_path,
        [roots[1], roots[1]],
        source_commit_fallback="f" * 40,
    )
    assert discovered["files"] == []
    assert no_receipt.exists()

    base_candidate = {
        "row": _receipt_row(),
        "terminal": True,
        "load_error": None,
    }
    mutations = (
        ({"load_error": "bad"}, "unreadable_json"),
        ({"terminal": False}, "nonterminal_receipt_file"),
        ({"row": _receipt_row(solve_provenance="development_proxy")}, "development_proxy"),
        ({"row": {**_receipt_row(), "read_game_source": True}}, "source_reading"),
        ({"row": {**_receipt_row(), "outer_loop_re": True}}, "outer_loop_re"),
        ({"row": {**_receipt_row(), "llm_on_row_valid": False}}, "invalid_live_harness_row"),
        (
            {"row": {**_receipt_row(), "trajectory_supervisor": {"enabled": False}}},
            "receipt_other",
        ),
    )
    for mutation, reason in mutations:
        candidate = {**base_candidate, **mutation}
        assert exp._provenance_disposition(candidate)[1] == reason

    frame_row = _receipt_row(levels=2, level_actions=[])
    frame_row["frame_sequence"] = [
        "skip",
        {"levels_completed": "bad"},
        {"levels_completed": 1, "frame_index": 9},
        {"levels_completed": 2, "charged_action_index": 20},
        {"levels_completed": 3, "charged_action_index": 30},
    ]
    candidate = {
        "row": frame_row,
        "canonical_row_id": exp.canonical_row_id(frame_row),
        "run_id": "frames",
    }
    assert [row["action_index"] for row in exp._banked_transitions(candidate)] == [10, 20]

    bad_redirect_row = _receipt_row()
    bad_redirect_row["trajectory_supervisor"]["redirects"] = [{"arm": None}, "skip"]
    bad_redirect_row["trajectory_supervisor"]["arm_outcomes"] = {}
    replay = exp.replay_banked_credit(
        [
            {
                "row": bad_redirect_row,
                "canonical_row_id": exp.canonical_row_id(bad_redirect_row),
                "run_id": "bad-redirect",
                "path": "fixture.json",
            }
        ]
    )
    assert replay["redirects"] == []

    with pytest.raises(ValueError, match="missing field principles"):
        exp._finish_artifact({"unknown_field": True})
    assert expected


def test_req_arc_wmte_6921_new_arm_receipt_stays_recommendation_only(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6921 can report but cannot implement a new curated arm."""

    roots, expected = _prepare_root(tmp_path)
    redirects = [
        {
            "arm": arm,
            "action_index": 10 + index * 10,
            "level": 0,
            "resolved_by_levelup": False,
            "actions_to_levelup": None,
        }
        for index, arm in enumerate(ARM_ORDER)
    ]
    row = _receipt_row(levels=0, level_actions=[], redirects=redirects)
    row["trajectory_supervisor"]["stagnations_unredirected"] = 1
    _write_json(tmp_path / "lever-runs" / "all-arms.json", {"rows": [row]})

    artifact = _build(tmp_path, roots, expected)

    assert artifact["refinement_recommendation_rows"][0]["kind"] == "new_arm_specification"
    assert artifact["automatic_arm_mutation_count"] == 0


def test_req_arc_wmte_6921_validator_rejects_each_contract_mutation(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6921 validates every safety boundary independently."""

    roots, expected = _prepare_root(tmp_path)
    _write_json(tmp_path / "lever-runs" / "one.json", {"rows": [_receipt_row()]})
    clean = _build(tmp_path, roots, expected)
    mutations = (
        (lambda row: row.pop("rows"), "missing required field: rows"),
        (lambda row: row.update(field_principles={}), "field_principles must cover"),
        (lambda row: row.update(inference_substrate="wrong"), "wrong inference_substrate"),
        (lambda row: row.update(automatic_arm_mutation_count=1), "automatic arm mutation"),
        (lambda row: row.update(verifier_is_oracle=True), "verifier_is_oracle"),
        (lambda row: row.update(verdict_class="wrong"), "invalid verdict_class"),
        (lambda row: row.update(honest_verdict="partial"), "complete_ terminal prefix"),
        (lambda row: row.update(reproducibility_checksum="bad"), "checksum mismatch"),
        (lambda row: row.update(solve_claim=True), "must not claim a new solve"),
        (lambda row: row.update(arc_run_launch_count=1), "must not run ARC"),
        (lambda row: row.update(registry_mutation_count=1), "must not mutate the registry"),
        (
            lambda row: row["per_game_rows"][0].update(solve_provenance="development_proxy"),
            "every game outcome row",
        ),
        (
            lambda row: row["refinement_policy"].update(recommendation_only=False),
            "recommendation-only",
        ),
        (
            lambda row: row["shadow_receipt_rows"].append(row["applied_receipt_rows"][0]),
            "shadow or error row",
        ),
        (lambda row: row.update(gate_check_summary=[]), "gate_check_summary must be an object"),
        (
            lambda row: row.update(gate_check_summary={"passed": False}),
            "failed gate summary lacks exact values",
        ),
    )
    for mutation, message in mutations:
        changed = deepcopy(clean)
        mutation(changed)
        assert any(message in error for error in exp.validate_artifact(changed))


def test_req_arc_wmte_6921_main_writes_and_reports_validation_errors(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6921-ARTIFACT covers command success and failure paths."""

    roots, expected = _prepare_root(tmp_path)
    receipt_path = _write_json(tmp_path / "lever-runs" / "one.json", {"rows": [_receipt_row()]})
    ledger = json.loads((tmp_path / exp.LEDGER_PATH).read_text(encoding="utf-8"))
    ledger["entries"]["source"] = {"source": str(receipt_path)}
    _write_json(tmp_path / exp.LEDGER_PATH, ledger)
    prior_hashes = exp.EXPECTED_PRIOR_ARTIFACT_HASHES
    validator = exp.validate_artifact
    try:
        exp.EXPECTED_PRIOR_ARTIFACT_HASHES = expected
        assert exp.main(["--date", "20260903", "--root", str(tmp_path)]) == 0
        assert (tmp_path / exp.OUTPUT_PATH).is_file()
        exp.validate_artifact = lambda artifact: ["forced validation failure"]
        assert (
            exp.main(
                [
                    "--date",
                    "20260903",
                    "--root",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--scored-root",
                    roots[0]["path"],
                    "--lever-root",
                    roots[1]["path"],
                ]
            )
            == 1
        )
    finally:
        exp.EXPECTED_PRIOR_ARTIFACT_HASHES = prior_hashes
        exp.validate_artifact = validator
