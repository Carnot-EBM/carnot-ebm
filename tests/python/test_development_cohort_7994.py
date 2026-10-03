"""REQ-VERIFY-7994, REQ-SELF-7994: label isolation and frozen chronology."""

import copy
import json
from pathlib import Path

import pytest

from carnot.verify import development_cohort_7994 as d
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def training(root: Path, size: int = 384):
    data = root / "data/ragtruth"
    data.mkdir(parents=True)
    sources, responses = [], []
    for i in range(size):
        sources.append(dict(source_id=str(i), source_info=f"Record {i} has {i} units."))
        responses.append(
            dict(
                id=str(i),
                source_id=str(i),
                split="train",
                response=f"It has {i} units.",
                quality="good",
                labels=[],
            )
        )
    for name, rows in [("source_info", sources), ("response", responses)]:
        (data / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    with (data / "response.jsonl").open("a") as stream:
        stream.write('{"split":"test", BROKEN TEST MUST NEVER DECODE}\n')
    return sources, responses


def test_selection_and_metadata_invariance(tmp_path):
    sources, responses = training(tmp_path, 386)
    a, b = d.public_training(tmp_path)
    assert len(a) == 386 and len(b) == 386
    roster, public = d.select(a, b, {"0"}, {d.f.normalized(sources[1]["source_info"].encode())})
    assert len(roster) == 384
    assert not {"0", "1"} & {r["source_id"] for r in roster}
    assert [r["role"] for r in roster].count("stream") == 256
    changed = [dict(r, labels=[999], model="changed", role="changed") for r in responses]
    assert d.select(
        sources, changed, {"0"}, {d.f.normalized(sources[1]["source_info"].encode())}
    ) == (roster, public)
    assert canonical_hash([d.f.extract(p) for p in public]) == canonical_hash(
        [d.f.extract(p) for p in public]
    )
    assert len(d.schedule(roster, set())) == 256
    assert d.schedule([], set())[0]["status"] == "unavailable"
    duplicate = copy.deepcopy(roster)
    duplicate[-1]["normalized_source_hash"] = duplicate[0]["normalized_source_hash"]
    with pytest.raises(ValueError, match="duplicate_role"):
        d.check_roles(duplicate)
    responses.insert(0, dict(responses[1], id="000"))
    assert d.select(sources, responses, set(), set())[0]


def test_access_delay_retention_and_invalid_public(tmp_path):
    row = dict(role="stream", slot=3, y=1)
    with pytest.raises(ValueError, match="future_label"):
        d.authorize(row, "feedback", 22, None)
    d.authorize(row, "feedback", 23, None)
    with pytest.raises(ValueError, match="role_access"):
        d.authorize(row, "calibrate", 99, None)
    d.authorize(dict(role="calibration"), "calibrate", 0, None)
    with pytest.raises(ValueError, match="audit_seal"):
        d.authorize(dict(role="retention"), "audit", 0, None)
    seal = tmp_path / "audit.json"
    atomic_json(seal, dict(independent_learning_audit=True))
    d.authorize(dict(role="retention"), "audit", 0, d.c.reference(seal))
    with pytest.raises(ValueError, match="role_access"):
        d.authorize(dict(role="retention"), "feedback", 99, None)
    sources, responses = training(tmp_path, 2)
    responses[0]["response"] = ""
    roster, public = d.select(sources, responses, set(), set())
    assert len(roster) == 2
    assert any(d.f.extract(r)["abstention"] for r in public)


def test_historical_exclusion_and_evaluator_api(tmp_path):
    """REQ-VERIFY-7994: original and explicit historical evaluation IDs fail admission."""
    sources, responses = training(tmp_path, 3)
    responses.extend([dict(responses[0], split="test"), dict(responses[0], source_id="missing")])
    sources[1]["source_info"] = dict(description="structured source")
    roster, public = d.select(sources, responses, set(), set())
    original = tmp_path / "original.json"
    atomic_json(original, dict(request_rows=public[:1]))
    plan = dict(
        upstream=dict(
            original_public=d.c.reference(original),
            rows=[dict(source_id="original")],
            exposure_audit=dict(
                discovered_exclusions=[
                    dict(
                        path="/history/evaluation.json",
                        source_ids=["2"],
                        source_hashes=["hash"],
                        sha256="old",
                    )
                ]
            ),
        )
    )
    ids, hashes, history = d.exclusions(plan, fixture=True)
    assert {"original", "2"} <= ids and hashes and history
    with pytest.raises(ValueError, match="original_roster_contract"):
        d.exclusions(plan)
    path = tmp_path / "labels.json"
    atomic_json(
        path,
        dict(
            rows=[
                dict(family_id="stream", role="stream", slot=0, y=1),
                dict(family_id="calibration", role="calibration", y=0),
            ]
        ),
    )
    assert d.read_label(path, "calibration", "calibrate", 0) == 0
    with pytest.raises(ValueError, match="future_label"):
        d.read_label(path, "stream", "feedback", 19)
    assert d.read_label(path, "stream", "feedback", 20) == 1
    before = d.seal(tmp_path / "before", roster, public)
    metadata = json.loads(path.read_text())
    metadata["rows"].reverse()
    for row in metadata["rows"]:
        row.update(y=1 - row["y"], role="changed", model="changed")
    atomic_json(path, metadata)
    after = d.seal(tmp_path / "after", roster, public)
    a = json.loads(before.read_text())
    b = json.loads(after.read_text())
    assert d.c.checked(a["features"]).read_bytes() == d.c.checked(b["features"]).read_bytes()
