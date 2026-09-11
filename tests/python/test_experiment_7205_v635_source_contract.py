"""Focused tests for REQ-REPORT-7205 and SCENARIO-REPORT-7205-*.

All writer tests use private paths. They never change the repository's research
record while they exercise the receipt builder.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
import yaml

from carnot import experiment_7205_v635_source_contract as mod


ROOT = Path(__file__).resolve().parents[2]
IDS = (
    "exp7205-source-contract",
    "exp7206-arc-volume-a",
    "exp7207-arc-volume-b",
    "exp7208-span-fixture",
    "exp7209-span-canary",
    "exp7210-span-capture",
    "exp7211-span-value-audit",
    "exp7212-refinement-fixture",
    "exp7213-refinement-learning",
    "exp7214-refinement-cold-audit",
    "exp7215-down-up-prototype",
    "exp7216-down-up-quality",
    "exp7217-abi-board-readiness",
    "exp7218-capstone",
)


def _prompt(index: int) -> str:
    """Declare one gate producer field in its actual required field block."""

    producer = '- ready_score: principle: "The producer completed."\n' if index == 4 else ""
    return (
        "CONCRETE STEPS:\n"
        "8. REQUIRED ARTIFACT FIELDS:\n"
        '- rows: principle: "Keep every unit."\n' + producer + "Run command: fixture\n"
    )


def _roadmap(*, milestone: str = mod.MILESTONE) -> dict[str, Any]:
    """Build an isolated V635 YAML contract with one real structured gate."""

    tasks: list[dict[str, Any]] = []
    for index, task_id in enumerate(IDS, 1):
        task: dict[str, Any] = {
            "id": task_id,
            "title": f"Task {7204 + index}",
            "track": "research",
            "priority": "high",
            "requires_gpu": False,
            "milestone": milestone,
            "deliverable": f"results/experiment_{7204 + index}.json",
            "prompt": _prompt(index),
            "prior_failures": [
                {
                    "experiment_id": f"exp{7000 + index}-prior",
                    "verdict": "complete_null_fixture",
                    "addressed_by": "Use a changed fixture mechanism.",
                    "retire_if_same_verdict": True,
                }
            ],
        }
        if index == 5:
            task["gated_on"] = [
                {
                    "upstream": IDS[3],
                    "artifact_field": "ready_score",
                    "op": "==",
                    "value": 1,
                }
            ]
        tasks.append(task)
    return {
        "milestone": milestone,
        "milestone_title": "fixture",
        "milestone_doc": str(mod.DESIGN_PATH),
        "tasks": tasks,
    }


def _markdown(*, milestone: str = mod.MILESTONE) -> str:
    """Build the Markdown source without reading any value from the YAML fixture."""

    rows = []
    for index, task_id in enumerate(IDS, 1):
        gate = "none"
        if index == 5:
            gate = f"`{IDS[3]}.ready_score == 1`"
        rows.append(
            f"| {index} | `{task_id}` | Task {7204 + index} | "
            f"`results/experiment_{7204 + index}.json` | {gate} |"
        )
    return "\n".join(
        (
            "# V635 fixture",
            "",
            f"**Milestone:** `{milestone}`",
            "",
            "## Exact Task Contract",
            "",
            "| Order | Task ID | Exact title | Deliverable | Structured gate |",
            "|---:|---|---|---|---|",
            *rows,
        )
    )


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return primary-page metadata without making a network request in tests."""

    arxiv_id = next((paper for paper in mod.ARXIV_SOURCES if paper in url), None)
    if arxiv_id:
        source = mod.ARXIV_SOURCES[arxiv_id]
        body = (
            f"<h1>Title:{source['title']}</h1>"
            f"<div>[v1] {source['submitted_date']} 00:00:00 UTC</div>"
        )
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {},
            "body": body,
            "error": None,
        }
    return {
        "ok": True,
        "status_code": 200,
        "url": url,
        "headers": {},
        "body": "No post-planning dated release.",
        "error": None,
    }


def _write_inputs(
    root: Path,
    *,
    active: dict[str, Any] | None = None,
    staged: dict[str, Any] | None = None,
    markdown: str | None = None,
) -> tuple[Path, Path, Path]:
    """Create private SCENARIO-REPORT-7205-PREFLIGHT source files."""

    files: dict[Path, str] = {
        mod.DESIGN_PATH: markdown or _markdown(),
        mod.SPEC_PATH: "REQ-REPORT-V635-PLAN\nREQ-REPORT-7205\n",
        mod.REFERENCE_PATH: "<!-- V635-PLANNER-REFRESH-20260911-START -->\n"
        + "\n".join(mod.ARXIV_SOURCES)
        + "\nhttps://extropic.ai/writing/z1t"
        + "\n<!-- V635-PLANNER-REFRESH-20260911-END -->\n",
        mod.EXCLUSION_PATH: "retired: []\n",
        Path("AGENTS.md"): "instructions\n",
        Path("CLAUDE.md"): "instructions\n",
        Path("CODEX.md"): "instructions\n",
        Path("research-program.md"): "program\n",
        Path("ops/e2e-test-plan.md"): "E2E plan\n",
    }
    for relative in mod.TOOL_PATHS + (mod.MODULE_PATH, mod.WRAPPER_PATH, mod.TEST_PATH):
        files[relative] = "source\n"
    for relative in mod.SOURCE_PATHS:
        files.setdefault(relative, "source\n")
    files[mod.V634_RECEIPT_PATH] = json.dumps(
        {
            "status": "complete",
            "honest_verdict": "complete_positive_v634_source_contract_exact_agreement",
            "source_contract_complete_score": 1,
            "flagged_adversarial": False,
        }
    )
    files[mod.V634_CAPSTONE_PATH] = json.dumps(
        {
            "status": "complete",
            "honest_verdict": "complete_null_v634_evidence_matrix",
            "capstone_complete_score": 1,
            "flagged_adversarial": False,
        }
    )
    if active is not None:
        files[mod.ACTIVE_ROADMAP_PATH] = yaml.safe_dump(active, sort_keys=False)
    if staged is not None:
        files[mod.NEXT_ROADMAP_PATH] = yaml.safe_dump(staged, sort_keys=False)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return (
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )


def test_req_report_7205_spec_precedes_implementation() -> None:
    """REQ-REPORT-7205 names every required field and focused scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7205") :]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for scenario in (
        "PARITY",
        "PREFLIGHT",
        "GATES",
        "QUARANTINE",
        "WRAPPERS",
        "SOURCES",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7205-{scenario}" in section


def test_scenario_report_7205_parity_is_independent_and_exact() -> None:
    """SCENARIO-REPORT-7205-PARITY accepts fourteen exact independent rows."""

    result = mod.evaluate_contract(_markdown(), _roadmap())
    assert result["passed"] is True
    assert result["expected_id_order"] == list(IDS)
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert len(result["contract_rows"]) == 14
    assert all(row["passed"] for row in result["contract_rows"])
    assert all(row["passed"] for row in result["gate_producer_rows"])
    assert all(row["passed"] for row in result["prior_failure_rows"])


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        ("missing", lambda value: value["tasks"].pop()),
        (
            "order",
            lambda value: value["tasks"].__setitem__(
                slice(1, 3), [value["tasks"][2], value["tasks"][1]]
            ),
        ),
        ("title", lambda value: value["tasks"][0].__setitem__("title", "changed")),
        (
            "deliverable",
            lambda value: value["tasks"][0].__setitem__("deliverable", "results/changed.json"),
        ),
        ("gate", lambda value: value["tasks"][4]["gated_on"][0].__setitem__("value", 0)),
        (
            "prior_failure",
            lambda value: value["tasks"][0]["prior_failures"][0].pop("retire_if_same_verdict"),
        ),
    ),
)
def test_req_report_7205_contract_mutations_fail(name: str, mutation: Any) -> None:
    """REQ-REPORT-7205 rejects every required contract mismatch."""

    roadmap = _roadmap()
    mutation(roadmap)
    assert mod.evaluate_contract(_markdown(), roadmap)["passed"] is False, name

    malformed = mod._prior_failure_rows([{"id": "fixture", "prior_failures": {}}])
    assert malformed[0]["checks"] == {"list_shape": False}


def test_req_report_7205_requires_exact_producer_field() -> None:
    """REQ-REPORT-7205 rejects a gate field absent from its producer block."""

    roadmap = _roadmap()
    roadmap["tasks"][3]["prompt"] = roadmap["tasks"][3]["prompt"].replace(
        "ready_score", "other_score"
    )
    result = mod.evaluate_contract(_markdown(), roadmap)
    assert result["passed"] is False
    assert result["gate_producer_rows"][0]["checks"]["bare_field_declared"] is False


def test_scenario_report_7205_preflight_selects_only_v635(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7205-PREFLIGHT prefers matching active, then staged YAML."""

    stale = _roadmap(milestone="2026.09.633")
    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=stale, staged=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert artifact["yaml_authority_path"] == str(mod.NEXT_ROADMAP_PATH)
    assert artifact["source_contract_complete_score"] == 1
    assert output.is_file() and checkpoint.is_file()
    assert mod.validate_artifact(artifact) == []

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(_roadmap(), sort_keys=False), encoding="utf-8"
    )
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert artifact["yaml_authority_path"] == str(mod.ACTIVE_ROADMAP_PATH)


def test_scenario_report_7205_preflight_blocks_without_v635(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7205-PREFLIGHT emits a diagnosed external block."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap(milestone="2026.09.633"))
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["field"] == "milestone"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7205_gate_edges_and_quarantine(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7205-GATES and QUARANTINE record real evaluator behavior."""

    rows = mod.run_gate_validation_fixtures(tmp_path)
    by_case = {row["case"]: row for row in rows}
    assert by_case["field_one"]["conductor_passed"] is True
    assert by_case["field_zero"]["conductor_passed"] is False
    assert by_case["missing_field"]["conductor_passed"] is False
    assert by_case["missing_file"]["conductor_passed"] is False
    assert by_case["missing_file"]["fixture_path"] is None
    quarantined = by_case["quarantined_field_one"]
    assert quarantined["conductor_passed"] is True
    assert quarantined["experiment_precondition_passed"] is False
    assert quarantined["validation_input"] is True


def test_scenario_report_7205_only_actual_principle_wrappers_unwrap() -> None:
    """SCENARIO-REPORT-7205-WRAPPERS preserves arbitrary domain mappings."""

    wrapper = {"principle": "Keep the reason beside the value.", "value": 1}
    assert mod._unwrap_principle_value(wrapper) == 1
    assert mod.upstream_precondition({"ready": wrapper}, "ready", 1)["passed"] is True

    domain_mapping = {"value": 1, "domain": "not a principle wrapper"}
    assert mod._unwrap_principle_value(domain_mapping) is domain_mapping
    row = mod.upstream_precondition({"ready": domain_mapping}, "ready", 1)
    assert row["actual"] == domain_mapping
    assert row["passed"] is False


def test_scenario_report_7205_sources_preserve_access_limits() -> None:
    """SCENARIO-REPORT-7205-SOURCES keeps real URLs and cached boundaries."""

    rows = mod.collect_source_method_rows(_fake_fetch)
    assert len([row for row in rows if row["source_type"] == "primary_paper"]) == 4
    assert {row["source_id"] for row in rows if row["source_type"] == "primary_paper"} == {
        f"arxiv:{paper}" for paper in mod.ARXIV_SOURCES
    }
    assert all(row["url"].startswith("https://") for row in rows)
    assert all(row["access_outcome"] == "http_200" for row in rows)
    assert all(row["post_planning_delta"] is False for row in rows)
    assert all(row["planning_date"] for row in rows)

    def unavailable(url: str) -> dict[str, Any]:
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "headers": {},
            "body": "",
            "error": "offline",
        }

    limited = mod.collect_source_method_rows(unavailable)
    assert all(row["access_outcome"] == "unavailable_cached_primary_evidence" for row in limited)
    assert all(row["method_boundary"] for row in limited)


def test_req_report_7205_rejects_quarantined_and_failed_history(tmp_path: Path) -> None:
    """REQ-REPORT-7205 never promotes quarantined or failed V634 history."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    (tmp_path / mod.V634_RECEIPT_PATH).write_text(
        json.dumps(
            {
                "status": "complete",
                "honest_verdict": "complete_positive_v634_source_contract_exact_agreement",
                "source_contract_complete_score": 1,
                "flagged_adversarial": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / mod.V634_CAPSTONE_PATH).write_text(
        json.dumps(
            {
                "status": "complete",
                "honest_verdict": "complete_null_v634_evidence_matrix",
                "capstone_complete_score": 0,
                "flagged_adversarial": False,
            }
        ),
        encoding="utf-8",
    )
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    history = {row["source_path"]: row for row in artifact["upstream_intake_rows"]}
    assert history[str(mod.V634_RECEIPT_PATH)]["quarantined"] is True
    assert history[str(mod.V634_RECEIPT_PATH)]["accepted_for_evidence"] is False
    assert history[str(mod.V634_CAPSTONE_PATH)]["known_failed_value"] is True
    assert history[str(mod.V634_CAPSTONE_PATH)]["accepted_for_evidence"] is False
    assert artifact["contract_authorities"] == [
        str(mod.DESIGN_PATH),
        str(mod.ACTIVE_ROADMAP_PATH),
    ]


def test_scenario_report_7205_artifact_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7205-ARTIFACT recomputes score, rows, and checksum."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    for mutation in (
        lambda value: value.__setitem__("status", "running"),
        lambda value: value.__setitem__("source_contract_complete_score", 0),
        lambda value: value["contract_rows"][0].__setitem__("passed", False),
        lambda value: value.__setitem__("rows", []),
        lambda value: value.__setitem__("model_invoked", True),
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
    ):
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod.validate_artifact(changed)


def test_req_report_7205_date_and_commands_are_scoped() -> None:
    """REQ-REPORT-7205 fixes the date and lists each required structural check."""

    assert mod._date_argument("20260911") == "20260911"
    with pytest.raises(ValueError):
        mod._date_argument("20260909x")
    names = [
        name
        for name, _command in mod._validation_commands(
            ROOT, ROOT / mod.DEFAULT_OUTPUT_PATH, mod.ACTIVE_ROADMAP_PATH
        )
    ]
    assert names[:6] == [
        "roadmap_schema",
        "prior_failure",
        "exclusion_manifest",
        "invented_paths",
        "arc_floor",
        "gate_declarations",
    ]


def test_req_report_7205_local_error_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7205 records write, hash, spec, and history input failures."""

    assert mod._source_hashes(tmp_path, None)
    assert all(value is None for value in mod._source_hashes(tmp_path, None).values())

    def fail_write(*_args: object, **_kwargs: object) -> object:
        raise OSError("read-only")

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", fail_write)
    assert mod._check_writable_directory(tmp_path) == (False, "OSError: read-only")
    monkeypatch.undo()

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    (tmp_path / mod.SPEC_PATH).unlink()
    rows, authority, roadmap, yaml_bytes = mod._preconditions(tmp_path, output, raw_dir, checkpoint)
    assert authority == mod.ACTIVE_ROADMAP_PATH and roadmap and yaml_bytes
    assert next(row for row in rows if row["check"] == "driving_requirement")["available"] is False

    (tmp_path / mod.REFERENCE_PATH).unlink()
    rows, _authority, _roadmap_doc, _yaml_bytes = mod._preconditions(
        tmp_path, output, raw_dir, checkpoint
    )
    source_table = next(row for row in rows if row["check"] == "planning_source_table")
    assert source_table["available"] is False

    (tmp_path / mod.V634_RECEIPT_PATH).write_text("[]", encoding="utf-8")
    (tmp_path / mod.V634_CAPSTONE_PATH).write_text("not-json", encoding="utf-8")
    intake = mod._upstream_intake_rows(tmp_path)
    assert all(row["accepted_for_evidence"] is False for row in intake)
    assert all(row["source_sha256"] is None for row in intake)


def test_req_report_7205_fetch_receipts_cover_http_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7205 keeps successful, HTTP-error, and transport outcomes."""

    class Response:
        status = 200
        headers = {"Last-Modified": "Thu, 10 Sep 2026 00:00:00 GMT"}

        @staticmethod
        def read() -> bytes:
            return b"[v2] Mon, 7 Sep 2026 11:12:03 UTC"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: Response())
    ok = mod._fetch_url("https://example.test/ok")
    assert ok["ok"] is True and ok["status_code"] == 200
    assert mod._observed_version_and_date(ok["body"], None) == ("v2", "2026-09-07")

    error = HTTPError(
        "https://example.test/error",
        429,
        "limited",
        {"Retry-After": "1"},
        io.BytesIO(b"limited"),
    )
    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    failed = mod._fetch_url("https://example.test/error")
    assert failed["status_code"] == 429 and failed["body"] == "limited"

    monkeypatch.setattr(
        mod, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline"))
    )
    offline = mod._fetch_url("https://example.test/offline")
    assert offline["status_code"] is None and "offline" in offline["error"]

    def raises(_url: str) -> dict[str, Any]:
        raise RuntimeError("fetcher failed")

    rows = mod.collect_source_method_rows(raises)
    assert all(row["access_outcome"] == "unavailable_cached_primary_evidence" for row in rows)


def test_req_report_7205_activation_and_score_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7205 rejects malformed activation and stored score evidence."""

    assert mod._activation_complete(None) is False
    assert mod._failed_precondition({}) is None
    invalid_rows = mod._activation_rows({"tasks": []})
    assert invalid_rows[0]["passed"] is False

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    cases = []
    for mutation in (
        lambda value: value.__setitem__("contract_rows", None),
        lambda value: value.__setitem__("markdown_milestone", "stale"),
        lambda value: value["source_method_rows"][0].__setitem__("url", ""),
        lambda value: value.__setitem__("activation_validation_rows", []),
        lambda value: value.__setitem__("upstream_intake_rows", []),
        lambda value: value.__setitem__(
            "validation_command_rows", [{"name": "roadmap_schema", "passed": False}]
        ),
    ):
        changed = deepcopy(artifact)
        mutation(changed)
        cases.append(changed)
    assert all(mod._score_from_artifact(changed) == 0 for changed in cases)


def test_req_report_7205_failure_summaries_cover_each_class(tmp_path: Path) -> None:
    """REQ-REPORT-7205 names contract, producer, activation, command, and fallback failures."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    mutations = (
        ("yaml_milestone", lambda value: value.__setitem__("yaml_milestone", "stale")),
        ("contract_order_1", lambda value: value["contract_rows"][0].__setitem__("passed", False)),
        (
            "gate_producer_contract",
            lambda value: value["gate_producer_rows"][0].__setitem__("passed", False),
        ),
        (
            "prior_failure_contract",
            lambda value: value["prior_failure_rows"][0].__setitem__("passed", False),
        ),
        (
            "activation_validation",
            lambda value: value.__setitem__("activation_validation_rows", []),
        ),
        (
            "validation:roadmap_schema",
            lambda value: value.__setitem__(
                "validation_command_rows",
                [{"name": "roadmap_schema", "passed": False, "exit_code": 1}],
            ),
        ),
        (
            "stored_source_contract_evidence",
            lambda value: value["source_method_rows"].clear(),
        ),
    )
    for expected, mutation in mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod._failure_summary(changed)["failed_check"] == expected


def test_scenario_report_7205_artifact_validator_covers_invalid_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7205-ARTIFACT rejects every independent field class."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert mod.validate_artifact(None) == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    missing.pop("status")
    assert mod.validate_artifact(missing) == ["missing_required_field:status"]
    mutations = (
        lambda value: value.__setitem__("field_principles", {}),
        lambda value: value.__setitem__("run_date", "20260909"),
        lambda value: value.__setitem__("execution_venue", "unknown"),
        lambda value: value.__setitem__("execution_host", ""),
        lambda value: value.__setitem__("duration_s", 0),
        lambda value: value.__setitem__("random_seed", 0),
        lambda value: value.__setitem__("verifier_is_oracle", True),
        lambda value: value.__setitem__("source_artifact_hashes", {}),
        lambda value: value.__setitem__("sample_size_budget", {}),
        lambda value: value["raw_source_rows"][0].__setitem__("hash_matches", False),
        lambda value: value.__setitem__("validation_command_rows", [{"name": "wrong"}]),
    )
    for mutation in mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod.validate_artifact(changed)


def test_req_report_7205_validation_stream_and_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7205 stores subprocess receipts and runs both CLI outcomes."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    monkeypatch.setattr(
        mod,
        "_validation_commands",
        lambda *_args: (("roadmap_schema", ["ok"]), ("prior_failure", ["bad"])),
    )
    receipts = iter(
        (
            {"exit_code": 0, "duration_s": 0.1, "output": "ok", "timed_out": False},
            {"exit_code": 1, "duration_s": 0.1, "output": "bad", "timed_out": False},
        )
    )
    monkeypatch.setattr(mod, "_run_streaming_command", lambda *_args, **_kwargs: next(receipts))
    rows = mod.run_validation_commands(tmp_path, checkpoint, artifact, 0.0)
    assert [row["passed"] for row in rows] == [True, False]

    valid = deepcopy(artifact)
    valid["validation_command_rows"] = []
    mod._apply_terminal_state(valid)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: valid)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    assert mod.main(["--date", mod.RUN_DATE]) == 0
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["bad"])
    assert (
        mod.main(
            [
                "--date",
                mod.RUN_DATE,
                "--output",
                str(output),
                "--raw-dir",
                str(raw_dir),
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 1
    )


def test_req_report_7205_build_records_contract_parse_failure(tmp_path: Path) -> None:
    """REQ-REPORT-7205 produces a disqualified receipt for readable malformed design text."""

    output, raw_dir, checkpoint = _write_inputs(
        tmp_path, active=_roadmap(), markdown="readable but missing the contract table"
    )
    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["source_contract_complete_score"] == 0


def test_req_report_7205_build_runs_validation_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7205 invokes the validation phase when the caller requests it."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    called: list[bool] = []

    def validation(
        _root: Path, _checkpoint: Path, _artifact: dict[str, Any], _started: float
    ) -> list[dict[str, Any]]:
        called.append(True)
        return []

    monkeypatch.setattr(mod, "run_validation_commands", validation)
    mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=True,
    )
    assert called == [True]


def test_req_report_7205_date_rejects_other_valid_day() -> None:
    """REQ-REPORT-7205 refuses a valid date that is not the execution date."""

    with pytest.raises(argparse.ArgumentTypeError):
        mod._date_argument("20260909")
