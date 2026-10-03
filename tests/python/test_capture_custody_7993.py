"""REQ-REPORT-7993 and REQ-VERIFY-7993: reconstruct immutable stream custody."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import capture_custody_7993 as custody
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.adversarial_verify import _classify_current_task_inference_claim as classify

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results/raw/experiment_7993_v693_capture_custody/historical_receipt_manifest.json"


@pytest.fixture(scope="module")
def bundle():
    return custody.load_bundle(json.loads(HISTORY.read_text()))


def test_authenticated_reconstruction_and_scope(bundle):
    """SCENARIO-REPORT-7993-CUSTODY: all independent sources rejoin."""
    value = custody.reduce_bundle(bundle)
    assert len(value["rows"]) == 224
    assert len(value["recovered_role_rows"]) == 212
    assert len(value["excluded_rows"]) == 12
    assert value["sample_size_budget"]["independent"] == 212
    assert value["historical_token_totals"]["completion_tokens"] > 0
    assert {k: len(v) for k, v in value["by_role"].items()} == dict(
        calibration_replay=29, online_update=91, online_admission=61, retention=31
    )
    assert all(r["scope"] == "historical" for r in value["invocation_scope_rows"])
    assert len(value["invocation_scope_rows"]) == 212
    assert value["scope_contradiction_diagnosis"]["negative_evidence"][0]["path"] == (
        "resumed_invocation_counts.generation_calls_attempted"
    )
    assert value == custody.reconstruct(bundle["history"])


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "response",
        "request",
        "role",
        "parse",
        "timestamp",
        "token",
        "server",
        "label",
        "source",
        "count",
        "identity",
        "receipt_duplicate",
        "receipt_missing",
        "load",
        "offload",
        "finished",
        "public_roster",
    ],
)
def test_reconstruction_rejects_conflicting_evidence(bundle, mutation):
    """SCENARIO-REPORT-7993-CUSTODY: drift cannot become a fabricated zero."""
    data = deepcopy(bundle)
    row = next(r for r in data["rows"] if r["started"])
    if mutation == "duplicate":
        data["rows"].append(deepcopy(row))
    elif mutation == "missing":
        data["rows"].remove(row)
    elif mutation == "response":
        row["raw_response"]["choices"][0]["message"]["content"] = "{}"
    elif mutation == "request":
        row["request"]["messages"][1]["content"] = "{}"
    elif mutation == "role":
        row["role"] = "retention"
    elif mutation == "parse":
        row["parsed"]["probability"] = 0.5
    elif mutation == "timestamp":
        row["invocation_finished_at"] = "2020-01-01T00:00:00+00:00"
    elif mutation == "token":
        data["runtime"]["runtime_receipts"][0]["output_tokens"] += 1
    elif mutation == "server":
        data["runtime"]["runtime_receipts"][0]["server_identity"]["pid"] += 1
    elif mutation == "label":
        data["labels"][row["role"]]["rows"][0]["response_sha256"] = "sha256:bad"
    elif mutation == "source":
        data["public"][row["role"]]["request_rows"][0]["source_bytes"] = "00"
    elif mutation == "count":
        data["candidate"]["model_invocation_counts"]["generation_calls_completed"] = 211
    elif mutation == "identity":
        data["runtime"]["gguf_sha256"] = "sha256:bad"
    elif mutation == "receipt_duplicate":
        data["runtime"]["runtime_receipts"].append(data["runtime"]["runtime_receipts"][0])
    elif mutation == "receipt_missing":
        data["runtime"]["runtime_receipts"].pop()
    elif mutation == "load":
        data["runtime"]["model_loads_completed"] = 0
    elif mutation == "offload":
        data["runtime"]["model_identity_receipt"]["offload_layers"] = [0, 66]
    elif mutation == "finished":
        data["candidate"]["finished_at"] = "2020-01-01T00:00:00+00:00"
    else:
        data["public"]["calibration_replay"]["request_rows"].pop()
    with pytest.raises(custody.CustodyError) as caught:
        custody.reduce_bundle(data)
    assert {
        "upstream_id",
        "path",
        "hash",
        "field",
        "op",
        "expected",
        "observed",
    } <= caught.value.check.keys()


def test_missing_and_tampered_external_bytes(tmp_path):
    """REQ-REPORT-7993: missing fields are contract errors, never zero counts."""
    item = dict(path=str(tmp_path / "missing.json"), sha256="sha256:expected")
    with pytest.raises(custody.CustodyError, match="sha256"):
        custody.checked(item)
    path = tmp_path / "record.json"
    atomic_json(path, dict(value=1))
    item = dict(path=str(path), sha256=sha256_file(path))
    assert custody.checked(item) == path
    path.write_text("{}")
    with pytest.raises(custody.CustodyError, match="sha256"):
        custody.checked(item)
    with pytest.raises(custody.CustodyError, match="contract"):
        custody.reconstruct({})


def test_actual_provenance_classifier(bundle):
    """SCENARIO-VERIFY-7993-SCOPE: original live contradiction stays rejected."""
    live = deepcopy(bundle["candidate"])
    assert classify(live)["state"] == "contradictory"
    live["resumed_invocation_counts"]["scope"] = "historical"
    assert classify(live)["state"] == "live_inference"
    cached = dict(
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(generation_calls_attempted=0, generation_calls_completed=0),
    )
    assert classify(cached)["state"] == "no_live_inference"
    resumed = deepcopy(cached)
    resumed["historical_invocations"] = dict(scope="historical", generation_calls_completed=212)
    assert classify(resumed)["state"] == "no_live_inference"
    mixed = deepcopy(live)
    mixed["historical_invocations"] = resumed["historical_invocations"]
    assert classify(mixed)["state"] == "live_inference"
    blocked = deepcopy(cached)
    blocked.update(verdict_class="blocked", honest_verdict="complete_blocked_missing_custody")
    assert classify(blocked)["state"] in {"no_live_inference", "blocked"}
    live["model_invocation_counts"]["generation_calls_completed"] = 0
    assert classify(live)["state"] == "contradictory"


def test_reconstruct_preserves_byte_failure_and_archived_code(tmp_path):
    """REQ-VERIFY-7993: preserved producer code is checked against its archive."""
    history = json.loads(HISTORY.read_text())
    missing = deepcopy(history)
    missing["references"]["candidate"]["path"] = str(tmp_path / "absent")
    with pytest.raises(custody.CustodyError, match="sha256"):
        custody.reconstruct(missing)
    archived = json.loads(Path(history["references"]["code_archive"]["path"]).read_text())
    archived["references"][0]["sha256"] = "sha256:tampered"
    path = tmp_path / "archive.json"
    atomic_json(path, archived)
    history["references"]["code_archive"] = dict(path=str(path), sha256=sha256_file(path))
    with pytest.raises(custody.CustodyError, match="sha256"):
        custody.load_bundle(history)
