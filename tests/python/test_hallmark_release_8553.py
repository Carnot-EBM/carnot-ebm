"""REQ-REPORT-8553 / REQ-VERIFY-8553: released labels do not imply human truth."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hallmark_release_8553 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_inventory_and_rejection() -> None:
    """SCENARIO-REPORT-8553-RELEASE: preserve counts, origins and sibling overlap."""
    data = e.fixture()
    result = e.reduce(data)
    assert result["release_audit_ready_score"] == 1
    assert result["independent_label_ready_score"] == 0
    assert result["admission_decision"]["decision"] == "reject_for_carnot_independent_labels"
    assert result["intended_count"] == 5
    assert sum(r["count"] for r in result["label_provenance_rows"]) == 5
    assert result["base_work_overlap_rows"][0]["shared_base_identity_count"] == 1
    assert len(result["rows"]) == 5
    assert result["prediction_reductions"][0]["committed"]["TP"] == 1
    assert result["prediction_reductions"][0]["committed"]["FN"] == 1
    assert result["prediction_reductions"][0]["abstention_aware"]["denominator"] == 1
    assert result["prediction_reductions"][0]["abstained_count"] == 1


@pytest.mark.parametrize(
    "change", ["missing", "count", "id", "label", "predictions", "unknown", "empty"]
)
def test_incomplete_and_unknown(change: str) -> None:
    """SCENARIO-REPORT-8553-RELEASE: unavailable bytes never become a clean corpus."""
    data = e.fixture()
    if change == "missing":
        data["missing_resources"] = ["release absent"]
    elif change == "count":
        data["splits"]["a"].pop()
    elif change == "id":
        data["splits"]["a"][1]["bibtex_key"] = "a0"
    elif change == "label":
        data["splits"]["a"][0]["label"] = None
    elif change == "predictions":
        data["predictions"][0]["rows"].pop()
    elif change == "unknown":
        data["splits"]["a"][0]["generation_method"] = "unlisted"
    else:
        data["splits"] = {}
    result = e.reduce(data)
    assert result["independent_label_ready_score"] == 0
    if change == "unknown":
        assert any(r["origin"] == "unknown" for r in result["rows"])
    else:
        assert result["gate_check_summary"] and result["release_audit_ready_score"] == 0


def test_prediction_rejection_and_absence() -> None:
    """SCENARIO-REPORT-8553-RELEASE: complete denominators and explicit versions are required."""
    data = e.fixture()
    data["predictions"] = []
    assert e.reduce(data)["prediction_reductions"] == []
    for key in ["version", "split"]:
        changed = e.fixture()
        changed["predictions"][0][key] = ""
        assert e.reduce(changed)["gate_check_summary"]
    changed = e.fixture()
    changed["predictions"][0]["rows"][0]["label"] = "invalid"
    assert e.reduce(changed)["gate_check_summary"]
    changed = e.fixture()
    changed["predictions"][0]["rows"][1]["bibtex_key"] = "a0"
    assert e.reduce(changed)["gate_check_summary"]


def test_bytes_and_replay(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: a repaired envelope cannot repair a false reduction."""
    source = tmp_path / "source.json"
    atomic_json(source, e.fixture())
    ref = e.reference(source)
    assert e.read(ref) == e.fixture()
    value = e.build(e.fixture(), ref, [], tmp_path / "out.json")
    out = tmp_path / "value.json"
    atomic_json(out, value)
    assert e.replay(out)["passed"]
    changed = deepcopy(value)
    changed["independent_label_ready_score"] = 1
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(out, changed)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(out)
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "bad"
    atomic_json(out, changed)
    with pytest.raises(ValueError, match="checksum"):
        e.replay(out)
    source.write_text("{}")
    with pytest.raises(ValueError, match="source_hash"):
        e.read(ref)
    with pytest.raises(FileNotFoundError):
        e.read({"path": str(tmp_path / "absent"), "sha256": "x"})
    value = e.build(e.fixture(), ref, [{"passed": False}], out)
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]
    blocked = e.fixture()
    blocked["missing_resources"] = ["absent"]
    assert e.build(blocked, ref, [], out)["verdict_class"] == "blocked"


def test_origin_and_confusion() -> None:
    """SCENARIO-REPORT-8553-RELEASE: review provenance differs from generation method."""
    for method, origin in [
        ("scraped", "database_cross_check"),
        ("perturbation", "deterministic_perturbation"),
        ("adversarial", "deterministic_perturbation"),
        ("llm_generated", "unknown"),
        ("real_world", "unknown"),
        ("canary", "deterministic_perturbation"),
    ]:
        assert e.origin({"generation_method": method, "label": "HALLUCINATED"}) == origin
    assert (
        e.origin(
            {
                "source": "walters_wilder_2023",
                "generation_method": "llm_generated",
                "label": "VALID",
            }
        )
        == "explicit_human_adjudication"
    )
    assert (
        e.origin({"generation_method": "perturbation", "label": "VALID"}) == "database_cross_check"
    )
    data = e.fixture()
    data["predictions"][0]["rows"] = [
        {"bibtex_key": "a0", "label": "VALID"},
        {"bibtex_key": "a1", "label": "HALLUCINATED"},
    ]
    assert e.reduce(data)["prediction_reductions"][0]["committed"]["FP"] == 0


def test_input_custody() -> None:
    """SCENARIO-REPORT-8553-RELEASE: author git blobs anchor local release files."""
    from carnot.reporting import hallmark_release_inputs_8553 as inputs

    data = inputs.load(inputs.MANIFEST)
    result = e.reduce(data)
    assert result["count_reconciliation"]["observed_public_core"] == 2072
    assert {r["split"]: r["count"] for r in result["split_inventory"]}["stress_test"] == 122
    assert len(result["prediction_reductions"]) == 2
    assert result["release_audit_ready_score"] == 1
    assert len(result["rows"]) > 2072


def test_private_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: actual private children publish and reject tampering."""
    from carnot.reporting import hallmark_release_runner_8553 as runner

    data = tmp_path / "fixture.json"
    atomic_json(data, e.fixture())
    out = tmp_path / "results" / f"{e.NAME}.json"
    command = [sys.executable, str(e.ROOT / e.CLI)]
    published = subprocess.run(
        [*command, "--fixture", str(data), "--output", str(out)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert published.returncode == 0, published.stdout + published.stderr
    for args, code in [
        (["--cold-replay", str(out)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 1),
        (["--date", "bad"], 2),
    ]:
        result = subprocess.run([*command, *args], capture_output=True, text=True, timeout=60)
        assert result.returncode == code, result.stdout + result.stderr
    value = json.loads(out.read_bytes())
    value["rows"][0]["label"] = "VALID"
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(out, value)
    assert runner.main(["--cold-replay", str(out)]) == 1
    assert runner.main(["--fixture", str(tmp_path / "absent"), "--output", str(out)]) == 1


def test_primitive_and_owned_tampering(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: independently pinned bytes defeat primitive rehashing."""
    from carnot.reporting import hallmark_release_inputs_8553 as inputs

    data = inputs.load(inputs.MANIFEST)
    source = tmp_path / "primitive.json"
    atomic_json(source, data)
    value = e.build(data, e.reference(source), [], tmp_path / "out.json")
    path = tmp_path / "out.json"
    atomic_json(path, value)
    assert e.replay(path)["passed"]
    data["splits"]["dev_public"][0]["label"] = "VALID"
    atomic_json(source, data)
    value = e.build(data, e.reference(source), [], path)
    atomic_json(path, value)
    with pytest.raises(ValueError, match="primitive_drift"):
        e.replay(path)
    data = e.fixture()
    atomic_json(source, data)
    value = e.build(data, e.reference(source), [], path)
    value["validation_receipts"] = [{"passed": False}]
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(path, value)
    with pytest.raises(ValueError, match="owned_check_drift"):
        e.replay(path)
    value = e.build(data, e.reference(source), [{"passed": False}], path)
    atomic_json(path, value)
    assert e.replay(path)["passed"]


def test_input_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: absent release files block; modified author blobs reject."""
    from carnot.reporting import hallmark_release_inputs_8553 as inputs

    original = e.read
    records = original(e.reference(inputs.MANIFEST))

    def manifest_read(ref):
        return records if ref["sha256"] == inputs.MANIFEST_PIN else original(ref)

    monkeypatch.setattr(e, "read", manifest_read)
    records[0]["missing"] = True
    records[1]["path"] = "/var/tmp/carnot8553-absent-release"
    assert inputs.load(inputs.MANIFEST)["missing_resources"]
    records[2]["commit"] = "altered"
    with pytest.raises(ValueError, match="author_blob"):
        inputs.load(inputs.MANIFEST)


def test_terminal_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: owned required failure disqualifies the private result."""
    from carnot.reporting import hallmark_release_runner_8553 as runner

    data = tmp_path / "fixture.json"
    atomic_json(data, e.fixture())
    out = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(
        runner,
        "validate",
        lambda *a: dict(passed=True, checks=[{"passed": False}], adversarial_findings=[]),
    )
    assert runner.main(["--fixture", str(data), "--output", str(out)]) == 1
    assert json.loads(out.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **k: {"passed": False})
    assert runner.main(["--fixture", str(data), "--output", str(out)]) == 1


def test_all_scoring_stances_and_inventory() -> None:
    """SCENARIO-REPORT-8553-RELEASE: preserve conservative, aggressive and selective counts."""
    data = e.fixture()
    r = e.reduce(data)["prediction_reductions"][0]
    assert r["conservative"]["FN"] == 1
    assert r["aggressive"]["TP"] == 2
    assert r["selective"]["denominator"] == 1
    assert r["coverage_numerator"] == 1 and r["coverage_denominator"] == 2
    data["splits"]["a"][1]["label"] = "VALID"
    data["predictions"][0]["rows"][1]["label"] = "VALID"
    assert e.reduce(data)["prediction_reductions"][0]["conservative"]["TN"] == 1
    data["predictions"][0]["rows"][1]["label"] = "HALLUCINATED"
    assert e.reduce(data)["prediction_reductions"][0]["conservative"]["FP"] == 1


def test_alias_overlap_is_not_independent_work() -> None:
    """SCENARIO-REPORT-8553-RELEASE: two aliases of one base paper count as one work."""
    data = e.fixture()
    for split in ["a", "b"]:
        data["splits"][split][0]["fields"]["doi"] = "10.1/shared"
    result = e.reduce(data)["base_work_overlap_rows"][0]
    assert result["shared_base_identity_count"] == 1
    assert result["shared_identity_alias_count"] == 2
    assert result["exact_citation_overlap_count"] == 1


def test_original_input_seals(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: copied instructions and authority keep original hashes."""
    from carnot.reporting import hallmark_release_inputs_8553 as inputs

    original = e.read
    refs = original(e.reference(inputs.RAW / "input_manifest.json"))

    def input_read(ref):
        return refs if ref["sha256"] == inputs.INPUT_PIN else original(ref)

    monkeypatch.setattr(e, "read", input_read)
    refs.append(dict(missing=True, original_path="missing_original"))
    assert "missing_original" in inputs.load(inputs.MANIFEST)["missing_resources"]
    refs[0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="original_input_hash"):
        inputs.load(inputs.MANIFEST)


@pytest.mark.parametrize(
    "key,value",
    [
        ("independent_generalization_score", 1),
        ("generalized_learning_benefit_score", 1),
        ("MODEL_SPECS", [{"model": "unrun"}]),
        ("preserved_quarantines", {}),
        ("verdict_class", "positive"),
    ],
)
def test_fixed_claim_tampering(tmp_path: Path, key: str, value) -> None:
    """SCENARIO-VERIFY-8553-CONTROLS: repaired hashes cannot invent scientific readiness."""
    data = e.fixture()
    data["preserved_quarantines"] = {"historic": "must stay closed"}
    source = tmp_path / "source.json"
    atomic_json(source, data)
    result = e.build(data, e.reference(source), [], tmp_path / "out.json")
    result[key] = value
    result["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in result.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "tampered.json"
    atomic_json(path, result)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(path)
