"""REQ-VERIFY-8567: keep all metrics visible while proving discrete relations."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import count_semantics_8566 as sealed
from scripts import adversarial_verify as av


def candidate(corpus, case):
    """Claims bind unchanged numeric metrics, so bad proof cannot hide values."""
    n = case["value"]
    values = dict(observed_alpha=n, observed_beta=n)
    if case["id"] == "unequal":
        values = dict(observed_alpha=2, observed_beta=1)
    if "left" in case:
        left = deepcopy(case["left"])
        if "raw_numeric_json" in case:
            left["value"] = json.loads(case["raw_numeric_json"])
        values["discrete_count_proof"] = dict(
            version=corpus["contract"]["version"],
            tables=corpus["tables"],
            claims=dict(observed_alpha=left, observed_beta=case["right"]),
        )
    return values


def flags(value):
    """Inspect every diagnostic rather than trusting an artifact's own verdict."""
    result = []
    av.check_tautology(value, result)
    return [row.to_dict() for row in result]


@pytest.fixture(scope="module")
def corpus():
    return sealed.corpus()


@pytest.mark.parametrize("index", range(17))
def test_all_sealed_controls(corpus, index):
    """SCENARIO-VERIFY-8567-CONTROLS: sealed positives and negatives remain fixed."""
    case = corpus["cases"][index]
    result = flags(candidate(corpus, case))
    critical = [r for r in result if r["severity"] == "critical"]
    if case["expected"] == "valid_discrete_identity":
        assert not critical
        assert any("structural count relation" in r["detail"] for r in result)
    elif case["expected"] == "valid_unequal":
        assert not result
    else:
        assert critical


def test_missing_proof_and_copied_continuous(corpus):
    """REQ-VERIFY-8567: missing proof and integer-looking floats remain critical."""
    good = candidate(corpus, corpus["cases"][0])
    assert flags(dict(observed_alpha=2400, observed_beta=2400))[0]["severity"] == "critical"
    for n in [0.913123456, 2400.0, "2400", True]:
        changed = deepcopy(good)
        changed.update(observed_alpha=n, observed_beta=n)
        assert not any("structural count relation" in r["detail"] for r in flags(changed))
    good["unrelated_loss"] = good["copied_loss"] = 0.913123456
    assert any(r["severity"] == "critical" for r in flags(good))


def test_rehashed_and_malformed_rows(corpus):
    """SCENARIO-VERIFY-8567-CONTROLS: hashes cannot certify inconsistent primitives."""
    good = candidate(corpus, corpus["cases"][0])
    for mutation in ["duplicate", "inactive", "missing", "identity", "boolean", "shape"]:
        changed = deepcopy(good)
        proof = changed["discrete_count_proof"]
        rows = proof["tables"]["2400"]["rows"]
        if mutation == "duplicate":
            rows[1] = rows[0]
        elif mutation == "inactive":
            rows[0]["applied"] = False
        elif mutation == "missing":
            rows.clear()
        elif mutation == "identity":
            rows[0]["namespace"] = 1
        elif mutation == "boolean":
            rows[0]["admitted"] = 1
        else:
            rows[0]["after"]["parameters"] = []
        for claim in proof["claims"].values():
            claim["table_sha256"] = sealed.digest(proof["tables"]["2400"])
        assert any(r["severity"] == "critical" for r in flags(changed))


def test_actual_cli_and_import(corpus, tmp_path):
    """SCENARIO-VERIFY-8567-CONTROLS: genuine CLI and imported consumer agree."""
    path = tmp_path / "candidate.json"
    value = candidate(corpus, corpus["cases"][0])
    path.write_text(json.dumps(value))
    result = subprocess.run(
        [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert av.verify_artifact(path)["flags"] == json.loads(result.stdout)["reports"][0]["flags"]
    assert not any(r["severity"] == "critical" for r in av.verify_artifact(path)["flags"])


@pytest.mark.parametrize("proof_state", ["valid", "corrupt", "missing"])
def test_standalone_cli_without_pythonpath(corpus, tmp_path, proof_state):
    """SCENARIO-VERIFY-8567-STANDALONE-CLI: cold imports preserve proof diagnostics."""
    value = candidate(corpus, corpus["cases"][0])
    if proof_state == "corrupt":
        value["discrete_count_proof"]["claims"]["observed_alpha"]["table_sha256"] = "wrong"
    elif proof_state == "missing":
        del value["discrete_count_proof"]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    result = subprocess.run(
        [sys.executable, str(Path(av.__file__).resolve()), "--json", str(path)],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout, result.stderr
    actual = json.loads(result.stdout)["reports"][0]["flags"]
    assert actual == av.verify_artifact(path)["flags"]
    assert any(row["severity"] == "critical" for row in actual) == (proof_state != "valid")
    assert any("structural count relation" in row["detail"] for row in actual) == (
        proof_state == "valid"
    )


def test_bounded_proof_rejections(corpus, monkeypatch):
    """REQ-VERIFY-8567: malformed and over-budget evidence never earns proof."""
    from carnot.reporting.discrete_count_proof import independent_totals, proven_counts

    for value in [None, {}, dict(version="wrong")]:
        assert proven_counts(dict(discrete_count_proof=value)) == {}
    good = candidate(corpus, corpus["cases"][0])
    for claims, tables in [([], {}), ({}, []), ({str(i): {} for i in range(65)}, {})]:
        changed = deepcopy(good)
        changed["discrete_count_proof"].update(claims=claims, tables=tables)
        assert proven_counts(changed) == {}
    row = corpus["tables"]["2400"]["rows"][0]
    for rows in [
        [],
        [row] * 50001,
        [row, row],
        [dict(row, namespace="")],
        [dict(row, before=dict(parameters="bad", decay_scale=1))],
        [dict(row, assignment_indices=[0] * 50001)],
    ]:
        with pytest.raises((ValueError, TypeError)):
            independent_totals(rows)
    with pytest.raises(ValueError, match="event_type"):
        independent_totals([dict(row, admitted=1)])
    with monkeypatch.context() as patch:
        patch.setattr(sealed, "reduce", lambda *args: dict(completed_count=999))
        assert proven_counts(good) == {}


def test_original_seven_relations_use_private_proof():
    """SCENARIO-VERIFY-8567-CONTROLS: original names and values survive all seven diagnoses."""
    from carnot.reporting import count_validator_8567 as study

    upstream, work, _ = study.authenticate()
    row = study.original_proof(work)
    assert len(row["old_findings"]) == 7
    assert len(row["new_findings"]) == 7
    assert all(f["severity"] == "warn" for f in row["new_findings"])
    assert row["unchanged_metrics"] and row["verdict_class"] == "disqualified"
