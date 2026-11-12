"""REQ-REPORT-9029-V771: private responses test custody, never source success."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from carnot.reporting import source_identity_9029 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

sys.path.insert(0, str(e.ROOT))


def response(title: str = "Canonical", qid: str = "Q1") -> bytes:
    """Keep a minimal recorded API shape, including both alias mechanisms."""
    return (
        json.dumps(
            dict(
                query=dict(
                    normalized=[dict(from_="Alias_name", to="Alias name")],
                    redirects=[{"from": "Alias name", "to": title}],
                    pages={"1": dict(title=title, pageprops=dict(wikibase_item=qid))},
                )
            )
        )
        .replace('"from_"', '"from"')
        .encode()
    )


def receipt(
    tmp_path: Path, host: str = "en.wikipedia.org", title: str = "Alias name"
) -> dict[str, Any]:
    """Authenticate exact private body bytes and the original request URL."""
    path = tmp_path / (host + ".json")
    path.write_bytes(response())
    return dict(e.ref(path), url=e.request_url(host, [title]), status=200, error=None)


class Clock:
    """Advance virtual wait time without delaying private tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, delay: float) -> None:
        self.value += delay


def test_identity_and_authenticated_redirects(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-RESUME: redirects merge without folding title case."""
    assert (
        e.identity("https://EN.wikipedia.org/wiki/Alias_name#part") == "en.wikipedia.org/Alias name"
    )
    assert e.identity("https://en.wikipedia.org/wiki/aBC") != e.identity(
        "https://en.wikipedia.org/wiki/Abc"
    )
    for bad in [
        "https://evil.example/wiki/X",
        "https://en.wikipedia.org.evil/wiki/X",
        "https://en.wikipedia.org/wiki/",
        "http://en.wikipedia.org/wiki/X",
    ]:
        with pytest.raises(ValueError):
            e.identity(bad)
    r = receipt(tmp_path)
    values = e.parse_receipt(r)
    assert values["en.wikipedia.org/Alias name"]["wikidata_id"] == "Q1"
    assert values["en.wikipedia.org/Alias name"]["resolved_title"] == "Canonical"
    r["url"] = r["url"].replace("pageprops", "extracts")
    with pytest.raises(ValueError, match="request_identity"):
        e.parse_receipt(r)
    r = receipt(tmp_path)
    Path(r["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed_bytes"):
        e.parse_receipt(r)


def test_resume_and_cache_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-RESUME: a later block cannot discard prior success."""
    clock = Clock()
    calls: list[str] = []

    def fetch(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        calls.append(url)
        assert timeout <= 30 and limit <= 268435456 and "maxlag=5" in url
        return 200, {}, response(), None

    urls = [
        "https://en.wikipedia.org/wiki/Alias_name",
        "https://en.wikipedia.org/wiki/Alias%20name",
    ]
    state = e.acquire(urls, [], tmp_path / "cache", fetch=fetch, now=clock.now, sleep=clock.sleep)
    assert len(calls) == 1 and not state["unresolved"]
    resumed = e.acquire(urls, [], tmp_path / "cache", fetch=fetch, now=clock.now, sleep=clock.sleep)
    assert len(calls) == 1 and resumed["cache_reuse_count"] == 1
    Path(state["receipts"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed_bytes"):
        e.acquire(urls, [], tmp_path / "cache", fetch=fetch)


@pytest.mark.parametrize("status,error", [(429, "HTTP429"), (503, "maxlag"), (0, "timeout")])
def test_retry_after_timeout_and_stop(tmp_path: Path, status: int, error: str) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: three tries then stop, without host rotation."""
    clock = Clock()
    calls: list[str] = []

    def fetch(url: str, timeout: float, limit: int) -> tuple[int, dict[str, str], bytes, str]:
        calls.append(url)
        return status, {"Retry-After": "3"}, b"rejected original", error

    state = e.acquire(
        ["https://de.wikipedia.org/wiki/X", "https://en.wikipedia.org/wiki/Y"],
        [],
        tmp_path,
        fetch=fetch,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert len(calls) == 3 and all("de.wikipedia" in u for u in calls)
    assert clock.value >= 6 and len(state["unresolved"]) == 2
    assert state["blocked"]["status"] == status and state["retry_counts"] == 2
    assert all(Path(r["path"]).read_bytes() == b"rejected original" for r in state["receipts"])


def test_budget_and_successful_prefix(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: byte and clock ceilings never grow after a block."""
    clock = Clock()
    urls = [f"https://en.wikipedia.org/wiki/T{i}" for i in range(51)]

    def fetch(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        return 200, {}, response("T0"), None

    state = e.acquire(
        urls, [], tmp_path / "bytes", fetch=fetch, now=clock.now, sleep=clock.sleep, byte_limit=1
    )
    assert state["blocked"]["error"] == "byte_budget" and state["bytes_downloaded"] <= 1
    state = e.acquire(
        urls, [], tmp_path / "time", fetch=fetch, now=clock.now, sleep=clock.sleep, budget=0
    )
    assert state["blocked"]["error"] == "time_budget" and not state["receipts"]

    def limited(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        return 429, {"Retry-After": "1000"}, b"blocked", "HTTP429"

    state = e.acquire(
        urls, [], tmp_path / "wait", fetch=limited, now=clock.now, sleep=clock.sleep, budget=2
    )
    assert len(state["receipts"]) == 1 and clock.value <= 2


def test_retry_dates_and_receipt_errors(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: HTTP dates and malformed metadata fail closed."""
    assert e.retry_delay("Thu, 01 Jan 1970 00:00:10 GMT", 1, wall=0) == 10
    assert e.retry_delay("bad", 2, wall=0) == 2
    assert e.retry_delay(None, 3, wall=0) == 4
    r = receipt(tmp_path)
    r["status"] = 429
    assert e.parse_receipt(r) == {}
    r["status"] = 200
    Path(r["path"]).write_bytes(b'{"error":{"code":"maxlag"}}')
    r.update(e.ref(Path(r["path"])))
    assert e.parse_receipt(r) == {}
    Path(r["path"]).write_bytes(b'{"query":{"pages":{"1":{"title":"Alias name"}}}}')
    r.update(e.ref(Path(r["path"])))
    assert e.parse_receipt(r)["en.wikipedia.org/Alias name"]["wikidata_id"] is None


def public_rows() -> list[dict[str, Any]]:
    """Public metadata contains no output, judgment or annotation fields."""
    return [
        dict(
            id=str(i),
            lang=lang,
            split="test",
            model_input="question " + str(i),
            wikipedia_url=f"https://{lang}.wikipedia.org/wiki/T{i}",
        )
        for i, lang in enumerate(e.LANGUAGES)
    ]


def test_components_cross_language_and_unknown_exposure() -> None:
    """SCENARIO-REPORT-9029-CLOSURE: shared QIDs merge, unknown never means unseen."""
    rows = public_rows()
    resolutions = {
        e.identity(r["wikipedia_url"]): dict(wikidata_id=f"Q{i + 1}", resolved_title=f"T{i}")
        for i, r in enumerate(rows)
    }
    resolutions[e.identity(rows[1]["wikipedia_url"])]["wikidata_id"] = "Q1"
    inventory = dict(exposed_keys=[], historical_reference_urls=[rows[0]["wikipedia_url"]], rows=[])
    result = e.reduce_metadata(rows, inventory, resolutions)
    assert len(result["source_group_rows"]) == 13
    assert result["excluded_count"] == 1 and result["completed_count"] == 12
    assert result["source_identity_ready_score"] == 1
    inventory["historical_reference_urls"].append("https://en.wikipedia.org/wiki/Unknown")
    result = e.reduce_metadata(rows, inventory, resolutions)
    assert (
        result["source_identity_ready_score"] == 0
        and result["exposure_inventory"]["unresolved_count"] == 1
    )
    assert all(r["exclusion_reason"] != "unseen" for r in result["rows"])
    resolutions.pop(e.identity(rows[-1]["wikipedia_url"]))
    assert e.reduce_metadata(rows, inventory, resolutions)["unresolved_source_groups"]
    inventory["historical_reference_urls"] = []
    inventory["unknown_exposure_rows"] = ["missing_item_identity"]
    assert e.reduce_metadata(rows, inventory, resolutions)["source_identity_ready_score"] == 0
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        e.reduce_metadata(rows, inventory, resolutions)


def test_pinned_inputs_and_no_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9029-RESUME: authenticate real pinned public columns only."""
    data = e.load_inputs(e.DEFAULT_INPUT)
    assert len(data["records"]) == 2401 and all("annotations" not in r for r in data["records"])
    assert len(data["inherited_receipts"]) == 11
    changed = tmp_path / "manifest.json"
    value = json.loads(e.DEFAULT_INPUT.read_bytes())
    value["revision"] = "changed"
    atomic_json(changed, value)
    with pytest.raises(ValueError, match="release_manifest"):
        e.load_inputs(changed)
    monkeypatch.setattr(e, "MANIFEST_PIN", e.ref(changed)["sha256"])
    with pytest.raises(ValueError, match="release_revision"):
        e.load_inputs(changed)


def test_fetch_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: HTTP and transport failures retain their bytes."""
    import io

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise HTTPError(
            "https://en.wikipedia.org",
            429,
            "limited",
            {"Retry-After": "5"},
            io.BytesIO(b"original"),
        )

    monkeypatch.setattr(e, "urlopen", failed)
    status, headers, body, error = e.fetch("https://en.wikipedia.org", 30, 100)
    assert status == 429 and body == b"original" and error

    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise URLError("timed out")

    monkeypatch.setattr(e, "urlopen", timeout)
    assert e.fetch("https://en.wikipedia.org", 30, 100)[0] == 0


def test_runner_block_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: private CLI custody rejects rehashed changes."""
    from carnot.reporting import source_identity_runner_9029 as r

    scope = r.scope_manifest([])
    ref = r.measure(tmp_path / "missing.json", tmp_path / (e.NAME + ".json"), "20261112", scope)
    result = r.build(ref, {})
    assert result["verdict_class"] == "disqualified" and result["source_identity_ready_score"] == 0
    assert result["gate_check_summary"][0]["passed"] is False
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, result)
    assert r.replay(candidate)["replay_passed"]
    result["completed_count"] += 1
    result["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in result.items() if k != "reproducibility_checksum"}
    )
    atomic_json(candidate, result)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)
    monkeypatch.setattr(r, "receipts_pass", lambda *args: True)
    result = r.build(ref, {})
    assert result["verdict_class"] == "blocked" and result["honest_verdict"].startswith(
        "complete_blocked_"
    )


def test_metadata_measure_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: replay rebuilds public components without network."""
    from carnot.reporting import source_identity_runner_9029 as r

    rows = public_rows()
    data = dict(
        records=rows,
        inventory=dict(exposed_keys=[], historical_reference_urls=[], rows=[]),
        inherited_receipts=[],
        references=[],
    )
    monkeypatch.setattr(e, "load_inputs", lambda p: deepcopy(data))
    responses = []
    for i, row in enumerate(rows):
        p = tmp_path / (str(i) + ".body")
        p.write_bytes(response("T" + str(i), "Q" + str(i + 1)))
        responses.append(
            dict(
                e.ref(p),
                status=200,
                error=None,
                url=e.request_url(row["lang"] + ".wikipedia.org", ["T" + str(i)]),
            )
        )
    state = dict(
        resolutions={k: v for receipt in responses for k, v in e.parse_receipt(receipt).items()},
        receipts=responses,
        unresolved=[],
        blocked=None,
        cache_reuse_count=14,
        retry_counts=0,
        bytes_downloaded=0,
        network_duration_s=0,
    )
    monkeypatch.setattr(e, "acquire", lambda *args: state)
    scope = r.scope_manifest([])
    ref = r.measure(e.DEFAULT_INPUT, tmp_path / (e.NAME + ".json"), "20261112", scope)
    monkeypatch.setattr(r, "receipts_pass", lambda *args: True)
    result = r.build(ref, {})
    assert result["source_identity_ready_score"] == 1 and result["verdict_class"] == "null"
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, result)
    assert r.replay(candidate)["passed"]
    Path(responses[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        r.replay(candidate)


def test_transport_success_and_authority_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9029-RESUME: successful transport still needs pinned authority."""
    import io

    class Stream(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/json"}

    monkeypatch.setattr(e, "urlopen", lambda *a, **kw: Stream(b"hello"))
    assert e.fetch("https://en.wikipedia.org", 30, 100) == (200, Stream.headers, b"hello", None)
    assert e.fetch("https://en.wikipedia.org", 30, 2)[-1] == "byte_budget"
    manifest = json.loads(e.DEFAULT_INPUT.read_bytes())
    changed = tmp_path / "manifest.json"
    monkeypatch.setattr(e, "UPSTREAM_PIN", "changed")
    with pytest.raises(ValueError, match="upstream_primary"):
        e.load_inputs(e.DEFAULT_INPUT)
    monkeypatch.setattr(e, "UPSTREAM_PIN", e.ref(e.UPSTREAM)["sha256"])
    manifest["files"]["policy"]["sha256"] = "changed"
    atomic_json(changed, manifest)
    monkeypatch.setattr(e, "MANIFEST_PIN", e.ref(changed)["sha256"])
    with pytest.raises(ValueError, match="pinned_release"):
        e.load_inputs(changed)
    manifest = json.loads(e.DEFAULT_INPUT.read_bytes())
    metadata = tmp_path / "metadata.json"
    atomic_json(metadata, dict(sha=e.REVISION, cardData=dict(license="wrong")))
    manifest["metadata"] = e.ref(metadata)
    atomic_json(changed, manifest)
    monkeypatch.setattr(e, "MANIFEST_PIN", e.ref(changed)["sha256"])
    with pytest.raises(ValueError, match="author_metadata"):
        e.load_inputs(changed)
    r = receipt(tmp_path)
    r["url"] = e.request_url("en.wikipedia.org", [str(n) for n in range(51)])
    with pytest.raises(ValueError, match="batch_size"):
        e.parse_receipt(r)


def test_recovery_and_prefix_resume(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-RESUME: retry recovery and later failure preserve successes."""
    clock = Clock()
    calls = 0

    def fetch(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 429, {}, b"limited", "HTTP429"
        if "de.wikipedia" in url:
            return 200, {}, response("D"), None
        return 403, {}, b"denied", "HTTP403"

    urls = ["https://de.wikipedia.org/wiki/D", "https://en.wikipedia.org/wiki/E"]
    state = e.acquire(urls, [], tmp_path, fetch=fetch, now=clock.now, sleep=clock.sleep)
    assert state["resolutions"]["de.wikipedia.org/D"]["wikidata_id"] == "Q1"
    assert state["blocked"]["status"] == 403 and len(state["receipts"]) == 3
    resumed = e.acquire(urls, [], tmp_path, fetch=fetch, now=clock.now, sleep=clock.sleep)
    assert resumed["cache_reuse_count"] == 1 and calls == 4


def test_real_private_cli_missing_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: actual foreign-directory CLI and scanners execute."""
    import os
    import subprocess

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (e.NAME + ".json")
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]

    def run(args: list[str], expected: int) -> None:
        e.progress("before_private_cli", 0, 1)
        result = subprocess.run(
            argv + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
        )
        e.progress("after_private_cli", 1, 0)
        assert result.returncode == expected, result.stdout + result.stderr

    run(["--date", "wrong"], 1)
    run(["--inputs", str(tmp_path / "missing.json"), "--output", str(output)], 0)
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and not value["labels_accessed"]
    run(["--cold-replay", str(output)], 0)
    value["completed_count"] = 123
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    run(["--cold-replay", str(output)], 1)
    changed = tmp_path / "changed.json"
    atomic_json(changed, dict(revision="changed"))
    run(["--inputs", str(changed), "--output", str(output)], 0)


def test_receipt_acceptance_and_forgery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: failed, stale or edited check receipts cannot qualify."""
    from carnot.reporting import source_identity_runner_9029 as r

    log = tmp_path / "log"
    log.write_text("normal exit\n")
    commands = [dict(name=name, argv=[name]) for name in r.CHECKS]
    scope = r.scope_manifest(commands)
    checks = [
        dict(
            name=c["name"],
            argv=c["argv"],
            passed=True,
            normal_exit=True,
            timed_out=False,
            exit_code=0,
            started_monotonic_ns=scope["frozen_ns"],
            ended_monotonic_ns=scope["frozen_ns"] + 1,
            scope_sha256=canonical_hash(scope),
            stdout_path=str(log),
            stderr_path=str(log),
            stdout_sha256=e.ref(log)["sha256"],
            stderr_sha256=e.ref(log)["sha256"],
        )
        for c in commands
    ]
    monkeypatch.setattr(r, "verify_measured", lambda *a: True)
    validation = dict(checks=checks, owned_statement_coverage={})
    assert r.receipts_pass(scope, validation)
    for field, value in [
        ("passed", False),
        ("normal_exit", False),
        ("timed_out", True),
        ("exit_code", 1),
        ("argv", ["changed"]),
        ("scope_sha256", "changed"),
        ("started_monotonic_ns", 0),
        ("stdout_sha256", "changed"),
    ]:
        changed = deepcopy(validation)
        changed["checks"][0][field] = value
        assert not r.receipts_pass(scope, changed)
    assert not r.receipts_pass(scope, {})


def test_environment_block_and_component_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: resource failures and recomputed sidecars cannot qualify."""
    from carnot.reporting import source_identity_runner_9029 as r

    monkeypatch.setattr(r.sys, "prefix", str(tmp_path))
    ref = r.measure(
        e.DEFAULT_INPUT, tmp_path / (e.NAME + ".json"), "20261112", r.scope_manifest([])
    )
    value = r.build(ref, {})
    assert value["gate_check_summary"][0]["observed"] == "virtualenv_or_resources"
    primitive = json.loads(Path(ref["path"]).read_bytes())
    primitive["reduction"]["completed_count"] += 1
    atomic_json(Path(ref["path"]), primitive)
    value["primitive_reference"] = e.ref(Path(ref["path"]))
    value["raw_shard_hashes"][0] = value["primitive_reference"]
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="component_drift"):
        r.replay(candidate)


def test_terminal_network_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: a measured HTTP operand remains the first terminal gate."""
    from carnot.reporting import source_identity_runner_9029 as r

    rows = public_rows()
    data = dict(
        records=rows,
        inventory=dict(exposed_keys=[], historical_reference_urls=[], rows=[]),
        inherited_receipts=[],
        references=[],
    )
    monkeypatch.setattr(e, "load_inputs", lambda p: data)
    state = dict(
        resolutions={},
        receipts=[],
        unresolved=[e.identity(x["wikipedia_url"]) for x in rows],
        blocked=dict(status=429, error="HTTP429", url="https://de.wikipedia.org/w/api.php"),
        cache_reuse_count=0,
        retry_counts=2,
        bytes_downloaded=10,
        network_duration_s=3,
    )
    monkeypatch.setattr(e, "acquire", lambda *a: state)
    ref = r.measure(
        e.DEFAULT_INPUT, tmp_path / (e.NAME + ".json"), "20261112", r.scope_manifest([])
    )
    monkeypatch.setattr(r, "receipts_pass", lambda *a: True)
    value = r.build(ref, {})
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"][0]["observed"] == 429
    assert value["source_identity_ready_score"] == 0


def test_replay_uses_pinned_public_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: rehashed public metadata cannot replace the pinned author rows."""
    from carnot.reporting import source_identity_runner_9029 as r

    data = dict(
        records=public_rows(),
        inventory=dict(exposed_keys=[], historical_reference_urls=[], rows=[]),
        inherited_receipts=[],
        references=[],
    )
    monkeypatch.setattr(e, "load_inputs", lambda p: deepcopy(data))
    state = dict(
        resolutions={},
        receipts=[],
        unresolved=[],
        blocked=None,
        cache_reuse_count=0,
        retry_counts=0,
        bytes_downloaded=0,
        network_duration_s=0,
    )
    monkeypatch.setattr(e, "acquire", lambda *a: state)
    ref = r.measure(
        e.DEFAULT_INPUT, tmp_path / (e.NAME + ".json"), "20261112", r.scope_manifest([])
    )
    value = r.build(ref, {})
    metadata_path = Path(value["metadata_manifest"]["path"])
    metadata = json.loads(metadata_path.read_bytes())
    metadata["records"][0]["model_input"] = "forged"
    atomic_json(metadata_path, metadata)
    primitive = json.loads(Path(ref["path"]).read_bytes())
    primitive["metadata_manifest"] = e.ref(metadata_path)
    primitive["reduction"] = e.reduce_metadata(metadata["records"], metadata["inventory"], {})
    atomic_json(Path(ref["path"]), primitive)
    value = r.build(e.ref(Path(ref["path"])), {})
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="public_metadata_drift"):
        r.replay(candidate)


def test_maxlag_and_request_hash(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: HTTP200 maxlag retries and altered request hashes reject."""
    r = receipt(tmp_path)
    r["request_sha256"] = "changed"
    with pytest.raises(ValueError, match="request_hash"):
        e.parse_receipt(r)
    calls = 0
    clock = Clock()

    def fetch(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 200, {"Retry-After": "2"}, b'{"error":{"code":"maxlag"}}', None
        return 200, {}, response(), None

    state = e.acquire(
        ["https://en.wikipedia.org/wiki/Alias_name"],
        [],
        tmp_path / "cache",
        fetch=fetch,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert (
        calls == 2
        and state["blocked"] is None
        and not state["unresolved"]
        and state["retry_counts"] == 1
    )


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"error":{"code":"badtitle"}}'])
def test_invalid_metadata_keeps_rejected_bytes(tmp_path: Path, body: bytes) -> None:
    """SCENARIO-REPORT-9029-RESUME: malformed external bodies are retained terminal evidence."""
    clock = Clock()

    def fetch(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        return 200, {}, body, None

    state = e.acquire(
        ["https://en.wikipedia.org/wiki/T"],
        [],
        tmp_path,
        fetch=fetch,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert state["blocked"] and state["blocked"]["error"]
    assert Path(state["receipts"][0]["path"]).read_bytes() == body


def test_recorded_exp9015_responses_in_private_cache(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-RESUME: actual retained HTTP200/429 bytes replay in private scratch."""
    from urllib.parse import quote

    inherited = e.load_inputs(e.DEFAULT_INPUT)["inherited_receipts"]
    copied = []
    values = {}
    for i, recorded in enumerate(inherited):
        body = Path(recorded["path"]).read_bytes()
        private = tmp_path / (str(i) + ".body")
        private.write_bytes(body)
        row = dict(recorded, **e.ref(private))
        assert row["sha256"] == recorded["sha256"]
        values.update(e.parse_receipt(row))
        copied.append(row)
    assert (
        copied[-1]["status"] == 429
        and Path(copied[-1]["path"]).read_bytes() == Path(inherited[-1]["path"]).read_bytes()
    )
    urls = [
        "https://" + key.split("/", 1)[0] + "/wiki/" + quote(key.split("/", 1)[1])
        for key, v in values.items()
        if v["wikidata_id"]
    ]

    def forbidden(*args: Any) -> Any:
        raise AssertionError("authenticated cache must avoid network")

    state = e.acquire(urls, copied, tmp_path / "cache", fetch=forbidden)
    assert (
        state["cache_reuse_count"] == len(urls)
        and not state["unresolved"]
        and state["bytes_downloaded"] == 0
    )


def test_rehashed_readiness_cannot_hide_unresolved_metadata(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-CLOSURE: rehashed primitive readiness cannot override missing identities."""
    from carnot.reporting import source_identity_runner_9029 as r

    reference = r.measure(
        tmp_path / "missing.json", tmp_path / (e.NAME + ".json"), "20261112", r.scope_manifest([])
    )
    primitive = json.loads(Path(reference["path"]).read_bytes())
    primitive["reduction"]["source_identity_ready_score"] = 1
    atomic_json(Path(reference["path"]), primitive)
    value = r.build(e.ref(Path(reference["path"])), {})
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="readiness_drift"):
        r.replay(candidate)
    primitive["reduction"]["source_identity_ready_score"] = 0
    primitive["reduction"]["gate_check_summary"][-1]["passed"] = True
    atomic_json(Path(reference["path"]), primitive)
    value = r.build(e.ref(Path(reference["path"])), {})
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="gate_drift"):
        r.replay(candidate)
    primitive["reduction"]["gate_check_summary"][-1]["passed"] = False
    primitive["acquisition"]["resolutions"] = {"forged": {"wikidata_id": "Q1"}}
    atomic_json(Path(reference["path"]), primitive)
    value = r.build(e.ref(Path(reference["path"])), {})
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="resolution_drift"):
        r.replay(candidate)


def test_all_pinned_url_shapes_and_literal_title_characters(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-RESUME: Chinese views and literal title punctuation preserve identity."""
    from urllib.parse import quote

    assert e.identity("https://zh.wikipedia.org/zh-cn/病例對照研究") == e.identity(
        "https://zh.wikipedia.org/wiki/病例對照研究"
    )
    data = e.load_inputs(e.DEFAULT_INPUT)
    assert all(e.identity(row["wikipedia_url"]) for row in data["records"])
    with pytest.raises(ValueError):
        e.identity("https://en.wikipedia.org/zh-cn/Topic")
    for title in ["A?B", "A%20B"]:
        body = tmp_path / (quote(title, safe="") + ".body")
        body.write_bytes(response(title))
        receipt = dict(
            e.ref(body), url=e.request_url("en.wikipedia.org", [title]), status=200, error=None
        )
        values = e.parse_receipt(receipt)
        assert (
            values[e.identity("https://en.wikipedia.org/wiki/" + quote(title, safe=""))][
                "wikidata_id"
            ]
            == "Q1"
        )


def test_backoff_budget_counts_only_executed_retries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9029-BOUNDS: a retry beyond the deadline is not an executed request."""
    clock = Clock()

    def fetch(
        url: str, timeout: float, limit: int
    ) -> tuple[int, dict[str, str], bytes, str | None]:
        return 429, {"Retry-After": "1000"}, b"blocked", "HTTP429"

    state = e.acquire(
        ["https://en.wikipedia.org/wiki/T"],
        [],
        tmp_path,
        fetch=fetch,
        now=clock.now,
        sleep=clock.sleep,
        budget=2,
    )
    assert len(state["receipts"]) == 1 and state["retry_counts"] == 0
