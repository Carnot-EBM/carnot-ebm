"""Tests for the poison-test-cascade pretest triage module (Exp 5194).

Spec refs: REQ-AUTO-5194, SCENARIO-AUTO-5194-PRIMARY,
SCENARIO-AUTO-5194-PRECISION, SCENARIO-AUTO-5194-HISTORICAL.

The primary fixture is the ACTUAL milestone-2026.07.475 failure that lost 10 of
12 tasks: ``test_experiment_5182_...::test_ondisk_deliverable_is_valid`` reading a
``results/experiment_5182_...json`` deliverable its sibling ``main()`` never wrote
(reconstructed from ops/conductor-log.md + the live pytest rendering).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import pretest_triage as pt


# --- fixtures reproducing the real .475 signature -------------------------------

_D5182 = "results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json"
_T5182 = "tests/python/test_experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.py"


def _v475_output() -> str:
    """The exact shape of the .475 pretest failure blob (FileNotFoundError)."""
    return (
        "=================================== FAILURES ===================================\n"
        "_______________________ test_ondisk_deliverable_is_valid _______________________\n"
        "    def test_ondisk_deliverable_is_valid() -> None:\n"
        f"        path = REPO / mod.RESULT_RELATIVE_PATH\n"
        '>       art = json.loads(path.read_text(encoding="utf-8"))\n'
        f"{_T5182}:495: \n"
        "self = PosixPath('/home/ianblenke/github.com/ianblenke/carnot/"
        f"{_D5182}'), mode = 'r'\n"
        f"E       FileNotFoundError: [Errno 2] No such file or directory: "
        f"'/home/ianblenke/github.com/ianblenke/carnot/{_D5182}'\n"
        "=========================== short test summary info ============================\n"
        f"FAILED {_T5182}::test_ondisk_deliverable_is_valid\n"
        "1 failed, 116 passed, 15 warnings in 12.58s\n"
    )


def _roadmap() -> dict[str, str]:
    return {_D5182: "exp5182-diffusiongemma-meta-tensor-rootcause-fix-v475"}


# --- SCENARIO-AUTO-5194-PRIMARY -------------------------------------------------


def test_detects_the_475_signature(tmp_path: Path) -> None:
    """REQ-AUTO-5194: the .475 FileNotFoundError-on-deliverable is matched to one node."""
    res = pt.detect_poison_cascade(
        _v475_output(), repo_root=tmp_path, roadmap_deliverables=_roadmap()
    )
    assert res.matched is True
    assert res.all_failures_explained is True
    assert len(res.matches) == 1
    m = res.matches[0]
    assert m.nodeid == f"{_T5182}::test_ondisk_deliverable_is_valid"
    assert m.deliverable_path == _D5182
    assert m.producing_task_id == "exp5182-diffusiongemma-meta-tensor-rootcause-fix-v475"
    assert "self-expires" in m.reason


def test_rendered_xfail_is_self_expiring_and_scoped(tmp_path: Path) -> None:
    """SCENARIO-AUTO-5194-PRIMARY: the xfail condition is the deliverable's absence."""
    res = pt.detect_poison_cascade(
        _v475_output(), repo_root=tmp_path, roadmap_deliverables=_roadmap()
    )
    deco = pt.render_xfail_decorator(res.matches[0])
    assert "mark.xfail" in deco
    assert "condition=not _pretest_triage_os.path.exists" in deco
    assert _D5182 in deco
    assert "strict=False" in deco
    assert "pretest-triage:xfail" in deco  # idempotency sentinel present


# --- SCENARIO-AUTO-5194-PRECISION: genuine failures are NOT masked --------------


def test_unrelated_assertion_failure_does_not_match(tmp_path: Path) -> None:
    """A plain assertion failure (no results path) keeps blocking."""
    out = (
        "=================================== FAILURES ===================================\n"
        "____________________________ test_math ____________________________\n"
        "    def test_math() -> None:\n"
        ">       assert add(2, 2) == 5\n"
        "E       assert 4 == 5\n"
        "tests/python/test_experiment_4242_math.py:7: AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/python/test_experiment_4242_math.py::test_math\n"
    )
    res = pt.detect_poison_cascade(out, repo_root=tmp_path, roadmap_deliverables=_roadmap())
    assert res.matched is False
    assert res.unmatched_failures == ("tests/python/test_experiment_4242_math.py::test_math",)
    assert res.all_failures_explained is False


def test_results_path_not_in_roadmap_does_not_match(tmp_path: Path) -> None:
    """A test reading a results path that is NOT a declared deliverable stays broken."""
    out = (
        "=================================== FAILURES ===================================\n"
        "____________________ test_ondisk_deliverable_is_valid ____________________\n"
        ">       art = json.loads(path.read_text())\n"
        "tests/python/test_experiment_9999_typo.py:9: \n"
        "E       FileNotFoundError: [Errno 2] No such file or directory: "
        "'/repo/results/experiment_9999_TYPO.json'\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/python/test_experiment_9999_typo.py::test_ondisk_deliverable_is_valid\n"
    )
    res = pt.detect_poison_cascade(out, repo_root=tmp_path, roadmap_deliverables=_roadmap())
    assert res.matched is False


def test_deliverable_present_on_disk_does_not_match(tmp_path: Path) -> None:
    """If the deliverable EXISTS, a failure is a real assertion bug, not a wait."""
    (tmp_path / "results").mkdir()
    (tmp_path / _D5182).write_text("{}", encoding="utf-8")
    res = pt.detect_poison_cascade(
        _v475_output(), repo_root=tmp_path, roadmap_deliverables=_roadmap()
    )
    assert res.matched is False
    assert res.unmatched_failures  # still blocks


def test_missing_absence_marker_does_not_match(tmp_path: Path) -> None:
    """A test that names a pending deliverable but fails on a VALUE (no FileNotFound)."""
    out = (
        "=================================== FAILURES ===================================\n"
        "____________________ test_ondisk_deliverable_is_valid ____________________\n"
        ">       assert art['diffusiongemma_loadable'] is True\n"
        f"        # loaded from {_D5182}\n"
        "E       assert False is True\n"
        f"{_T5182}:497: AssertionError\n"
        "=========================== short test summary info ============================\n"
        f"FAILED {_T5182}::test_ondisk_deliverable_is_valid\n"
    )
    res = pt.detect_poison_cascade(out, repo_root=tmp_path, roadmap_deliverables=_roadmap())
    assert res.matched is False


def test_core_test_is_never_triaged(tmp_path: Path) -> None:
    """SCENARIO-AUTO-5194-PRECISION: a core/shared test failing the gate keeps blocking."""
    out = (
        "=================================== FAILURES ===================================\n"
        "____________________ test_reads_pending ____________________\n"
        "tests/python/test_pipeline_extract.py:9: \n"
        f"E       FileNotFoundError: [Errno 2] No such file or directory: '/repo/{_D5182}'\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/python/test_pipeline_extract.py::test_reads_pending\n"
    )
    res = pt.detect_poison_cascade(out, repo_root=tmp_path, roadmap_deliverables=_roadmap())
    assert res.matched is False


def test_mixed_real_and_poison_failure_still_blocks(tmp_path: Path) -> None:
    """A poison wait mixed with a genuine failure -> all_failures_explained is False."""
    out = _v475_output().replace(
        "1 failed, 116 passed, 15 warnings in 12.58s\n",
        "FAILED tests/python/test_experiment_4242_math.py::test_math\n"
        "2 failed, 116 passed in 12.58s\n",
    )
    # add a real FAILURES block for the math test
    out = out.replace(
        "=========================== short test summary info",
        "____________________________ test_math ____________________________\n"
        ">       assert add(2, 2) == 5\n"
        "E       assert 4 == 5\n"
        "tests/python/test_experiment_4242_math.py:7: AssertionError\n"
        "=========================== short test summary info",
    )
    res = pt.detect_poison_cascade(out, repo_root=tmp_path, roadmap_deliverables=_roadmap())
    assert res.matched is True  # the poison wait IS detected
    assert res.all_failures_explained is False  # but a real failure remains
    assert "tests/python/test_experiment_4242_math.py::test_math" in res.unmatched_failures


# --- roadmap loading from disk --------------------------------------------------


def _write_roadmap(tmp_path: Path, body: str) -> None:
    (tmp_path / "research-roadmap.yaml").write_text(body, encoding="utf-8")


def test_load_roadmap_deliverables_from_disk(tmp_path: Path) -> None:
    """REQ-AUTO-5194: deliverables + milestone are read from research-roadmap.yaml."""
    _write_roadmap(
        tmp_path,
        "milestone: 2026.07.476\n"
        "tasks:\n"
        f"  - id: exp5182-x\n    deliverable: {_D5182}\n"
        "  - id: nodeliv\n    title: no deliverable\n"
        "  - id: notjson\n    deliverable: docs/report.md\n"
        "  - not-a-mapping\n",
    )
    mapping, milestone = pt.load_roadmap_deliverables(tmp_path)
    assert milestone == "2026.07.476"
    assert mapping == {_D5182: "exp5182-x"}


def test_detect_loads_roadmap_when_not_passed(tmp_path: Path) -> None:
    """detect_poison_cascade reads the roadmap itself when no mapping is supplied."""
    _write_roadmap(
        tmp_path,
        f"milestone: 2026.07.476\ntasks:\n  - id: exp5182-x\n    deliverable: {_D5182}\n",
    )
    res = pt.detect_poison_cascade(_v475_output(), repo_root=tmp_path)
    assert res.matched is True
    assert res.roadmap_milestone == "2026.07.476"
    assert res.matches[0].producing_task_id == "exp5182-x"


def test_load_roadmap_missing_file_is_empty(tmp_path: Path) -> None:
    mapping, milestone = pt.load_roadmap_deliverables(tmp_path)
    assert mapping == {}
    assert milestone is None


def test_load_roadmap_no_milestone_key(tmp_path: Path) -> None:
    _write_roadmap(tmp_path, "tasks: []\n")
    _, milestone = pt.load_roadmap_deliverables(tmp_path)
    assert milestone is None


def test_load_yaml_parse_error_and_nondict(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("::: not: [valid", encoding="utf-8")
    assert pt._load_yaml(bad) == {}
    lst = tmp_path / "list.yaml"
    lst.write_text("- a\n- b\n", encoding="utf-8")
    assert pt._load_yaml(lst) == {}


# --- unit coverage: helpers -----------------------------------------------------


def test_normalize_rel_path_variants(tmp_path: Path) -> None:
    assert pt._normalize_rel_path(None, tmp_path) is None
    assert pt._normalize_rel_path("   ", tmp_path) is None
    assert pt._normalize_rel_path("./", tmp_path) is None
    assert pt._normalize_rel_path("results/x.json", tmp_path) == "results/x.json"
    assert pt._normalize_rel_path("./results/x.json", tmp_path) == "results/x.json"
    inside = tmp_path / "results" / "x.json"
    assert pt._normalize_rel_path(str(inside), tmp_path) == "results/x.json"
    # absolute path outside the repo root -> returned as-is (ValueError branch)
    assert pt._normalize_rel_path("/other/results/x.json", tmp_path) == "/other/results/x.json"


def test_is_experiment_specific_test() -> None:
    assert pt.is_experiment_specific_test("tests/python/test_experiment_1_a.py") is True
    assert pt.is_experiment_specific_test("tests/python/test_exp_a.py") is True
    assert pt.is_experiment_specific_test("tests/python/test_pipeline_extract.py") is False
    assert pt.is_experiment_specific_test("tests/python/test_experiment_1.txt") is False
    assert pt.is_experiment_specific_test("scripts/test_experiment_1.py") is False
    assert pt.is_experiment_specific_test("tests/python/quarantine/test_experiment_1.py") is False
    # a test_-prefixed path that nonetheless sits under a quarantine dir (defensive guard)
    assert pt.is_experiment_specific_test("tests/python/test_experiment_9/quarantine/x.py") is False


def test_display_name_matches_nodeid() -> None:
    node = "tests/python/test_experiment_1_a.py::test_x"
    assert pt._display_name_matches_nodeid("test_x", node) is True
    assert pt._display_name_matches_nodeid("test_y", node) is False
    cls = "tests/python/test_experiment_1_a.py::TestC::test_m"
    assert pt._display_name_matches_nodeid("TestC.test_m", cls) is True
    param = "tests/python/test_experiment_1_a.py::test_x[case1]"
    assert pt._display_name_matches_nodeid("test_x[case1]", param) is True
    # a display name with no "::" nodeid (defensive edge)
    assert pt._display_name_matches_nodeid("bare", "bare") is True


def test_parse_summary_failures_nodeid_optional() -> None:
    out = (
        "FAILED tests/python/test_experiment_1_a.py::test_x\n"
        "ERROR tests/python/test_experiment_2_b.py::test_y - some reason\n"
        "FAILED tests/python/test_experiment_3_c.py\n"  # no ::nodeid
        "not a summary line\n"
    )
    got = pt._parse_summary_failures(out)
    assert (
        "tests/python/test_experiment_1_a.py",
        "tests/python/test_experiment_1_a.py::test_x",
    ) in got
    assert (
        "tests/python/test_experiment_2_b.py",
        "tests/python/test_experiment_2_b.py::test_y",
    ) in got
    assert ("tests/python/test_experiment_3_c.py", None) in got
    assert len(got) == 3


def test_collect_evidence_fallback_and_recovery(tmp_path: Path) -> None:
    """Cover the header-mismatch fallback + the traceback-only recovery branches."""
    # Fallback: summary nodeid func 'test_actual' but header says 'test_renamed'.
    fallback = (
        "=================================== FAILURES ===================================\n"
        "____________________ test_renamed ____________________\n"
        "tests/python/test_experiment_7777_x.py:9: in test_actual\n"
        "E       FileNotFoundError: results/experiment_7777_x.json\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/python/test_experiment_7777_x.py::test_actual\n"
    )
    ev = pt._collect_failure_evidence(fallback)
    assert ev[0].nodeid == "tests/python/test_experiment_7777_x.py::test_actual"
    assert ev[0].result_paths == ["results/experiment_7777_x.json"]
    assert ev[0].has_absence_marker is True

    # Recovery: a section with a traceback file but NO FAILED/ERROR summary line.
    recovery = (
        "=================================== ERRORS =====================================\n"
        "____________________ test_orphan ____________________\n"
        "tests/python/test_experiment_8888_y.py:3: in <module>\n"
        "E       FileNotFoundError: results/experiment_8888_y.json\n"
        "=========================== short test summary info ============================\n"
        "1 error in 0.10s\n"
    )
    ev2 = pt._collect_failure_evidence(recovery)
    assert ev2[0].nodeid == "tests/python/test_experiment_8888_y.py::test_orphan"
    assert ev2[0].has_absence_marker is True


def test_collect_evidence_recovery_skips_sectionless_and_dup() -> None:
    """Cover the recovery-loop guards: a section with no traceback file, and a dup node."""
    # A leftover FAILURES section with NO 'tests/...py:NN:' traceback line and no
    # summary entry -> the recovery loop must skip it (no file to recover).
    sectionless = (
        "=================================== FAILURES ===================================\n"
        "____________________ test_no_file ____________________\n"
        "some prose with no test-file path at all\n"
        "=========================== short test summary info ============================\n"
        "1 failed in 0.1s\n"
    )
    assert pt._collect_failure_evidence(sectionless) == []

    # Two sections for the SAME node; the summary consumes one, the recovery loop
    # meets the second and must skip it as an already-seen node.
    dup = (
        "=================================== FAILURES ===================================\n"
        "____________________ test_dup ____________________\n"
        "tests/python/test_experiment_1_a.py:9: \n"
        "E       FileNotFoundError: results/experiment_1_a.json\n"
        "____________________ test_dup ____________________\n"
        "tests/python/test_experiment_1_a.py:5: \n"
        "E       FileNotFoundError: results/experiment_1_a.json\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/python/test_experiment_1_a.py::test_dup\n"
    )
    ev = pt._collect_failure_evidence(dup)
    assert [e.nodeid for e in ev] == ["tests/python/test_experiment_1_a.py::test_dup"]


def test_collect_evidence_dedups_repeated_summary() -> None:
    out = (
        "FAILED tests/python/test_experiment_1_a.py::test_x\n"
        "FAILED tests/python/test_experiment_1_a.py::test_x\n"
    )
    ev = pt._collect_failure_evidence(out)
    assert len(ev) == 1


def test_collect_evidence_nodeid_none_becomes_unknown() -> None:
    out = "FAILED tests/python/test_experiment_3_c.py\n"
    ev = pt._collect_failure_evidence(out)
    assert ev[0].nodeid == "tests/python/test_experiment_3_c.py::<unknown>"


def test_iter_failure_sections_ignores_preamble_and_no_name() -> None:
    out = (
        "=================== test session starts ===================\n"
        "collected 3 items\n"
        "=================================== FAILURES ===================================\n"
        "leading line before any header\n"
        "___ test_a ___\n"
        "body a\n"
        "=================================== warnings summary ===========================\n"
    )
    sections = list(pt._iter_failure_sections(out))
    assert sections == [("test_a", "body a")]


# --- xfail application -----------------------------------------------------------


def _make_test_file(tmp_path: Path, rel: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '"""demo."""\n'
        "from __future__ import annotations\n"
        "import json\n"
        "\n"
        "def test_ondisk_deliverable_is_valid() -> None:\n"
        "    assert True\n",
        encoding="utf-8",
    )
    return p


def _match_for(tmp_path: Path) -> pt.TriageMatch:
    return pt.TriageMatch(
        test_file=_T5182,
        nodeid=f"{_T5182}::test_ondisk_deliverable_is_valid",
        deliverable_path=_D5182,
        producing_task_id="exp5182-x",
        reason="r",
    )


def test_apply_xfail_inserts_and_is_idempotent(tmp_path: Path) -> None:
    """apply_xfail adds the runtime block + a scoped decorator, and does not double-add."""
    _make_test_file(tmp_path, _T5182)
    result = pt.TriageResult(matched=True, matches=(_match_for(tmp_path),))

    modified = pt.apply_xfail(result, repo_root=tmp_path)
    assert modified == [_T5182]
    text = (tmp_path / _T5182).read_text(encoding="utf-8")
    assert pt._RUNTIME_SENTINEL in text
    assert "_pretest_triage_pytest.mark.xfail" in text
    assert text.count("pretest-triage:xfail") == 1
    # the decorator sits directly above the def
    lines = text.splitlines()
    def_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("def test_ondisk_deliverable_is_valid(")
    )
    assert any("mark.xfail" in ln for ln in lines[max(0, def_idx - 8) : def_idx])

    # second application is a no-op (idempotent)
    modified2 = pt.apply_xfail(result, repo_root=tmp_path)
    assert modified2 == []
    assert (tmp_path / _T5182).read_text(encoding="utf-8").count("pretest-triage:xfail") == 1


def test_apply_xfail_skips_missing_file_and_missing_def(tmp_path: Path) -> None:
    # file missing entirely
    result_missing = pt.TriageResult(matched=True, matches=(_match_for(tmp_path),))
    assert pt.apply_xfail(result_missing, repo_root=tmp_path) == []

    # file present but the def cannot be located -> left untouched
    other = "tests/python/test_experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.py"
    (tmp_path / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (tmp_path / other).write_text("def test_something_else():\n    assert True\n", encoding="utf-8")
    assert pt.apply_xfail(result_missing, repo_root=tmp_path) == []
    assert "mark.xfail" not in (tmp_path / other).read_text(encoding="utf-8")


def test_ensure_runtime_block_only_once() -> None:
    lines = ["import os", "x = 1"]
    once = pt._ensure_runtime_block(lines)
    assert once.count(pt._RUNTIME_SENTINEL) == 1
    twice = pt._ensure_runtime_block(once)
    assert twice.count(pt._RUNTIME_SENTINEL) == 1


def test_nodeid_function_variants() -> None:
    assert pt._nodeid_function("f.py::test_x") == "test_x"
    assert pt._nodeid_function("f.py::TestC::test_m") == "test_m"
    assert pt._nodeid_function("f.py::test_x[case]") == "test_x"
    assert pt._nodeid_function("bare") == "bare"


def test_find_and_sentinel_helpers() -> None:
    lines = ["import os", "", "def test_a():", "    pass"]
    assert pt._find_module_level_def(lines, "def test_a(") == 2
    assert pt._find_module_level_def(lines, "def test_missing(") is None
    marked = ["# pretest-triage:xfail t -> d", "def test_a():"]
    assert pt._sentinel_already_above(marked, 1, "pretest-triage:xfail t -> d") is True
    assert pt._sentinel_already_above(["def test_a():"], 0, "sent") is False


# --- SCENARIO-AUTO-5194-HISTORICAL: honest retrospective ------------------------


def test_historical_incidents_honest_classification() -> None:
    """SCENARIO-AUTO-5194-HISTORICAL: 1/4 exact match; 3/4 honestly the sibling class."""
    rows = pt.validate_historical_incidents()
    by_id = {r["experiment_id"]: r for r in rows}
    assert set(by_id) == {"exp5182", "exp3521", "exp3544", "exp3612"}
    assert by_id["exp5182"]["detector_matches"] is True
    assert by_id["exp5182"]["poison_class"] == "deliverable_read"
    for eid in ("exp3521", "exp3544", "exp3612"):
        assert by_id[eid]["detector_matches"] is False
        assert by_id[eid]["poison_class"] == "verdict_assertion"


def test_synthetic_deliverable_read_variant_would_match(tmp_path: Path) -> None:
    """The deliverable-read VARIANT of an earlier incident WOULD be caught (demonstration)."""
    deliverable = "results/experiment_3521_demo.json"
    out = pt._synthetic_output_for("exp3521", deliverable)
    res = pt.detect_poison_cascade(
        out, repo_root=tmp_path, roadmap_deliverables={deliverable: "exp3521-demo"}
    )
    assert res.matched is True
    assert res.matches[0].deliverable_path == deliverable


# --- CLI ------------------------------------------------------------------------


def test_cli_validate_historical(capsys: pytest.CaptureFixture[str]) -> None:
    rc = pt._cli(["--validate-historical"])
    assert rc == 0
    assert "exp5182" in capsys.readouterr().out


def test_cli_triage_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_roadmap(
        tmp_path,
        f"milestone: 2026.07.476\ntasks:\n  - id: exp5182-x\n    deliverable: {_D5182}\n",
    )
    out_file = tmp_path / "pytest.out"
    out_file.write_text(_v475_output(), encoding="utf-8")
    rc = pt._cli([str(out_file), "--repo-root", str(tmp_path)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert '"matched": true' in printed
    assert "exp5182-x" in printed


def test_cli_requires_an_argument() -> None:
    with pytest.raises(SystemExit):
        pt._cli([])
