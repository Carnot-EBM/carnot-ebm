"""REQ-REPORT-9029-V771: resolve public identities without opening human judgments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import time
from typing import Any
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit
from urllib.request import Request, urlopen

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.human_disagreement_source_9015 import LANGUAGES, PINNED
from carnot.reporting.human_disagreement_source_9015 import REVISION as REVISION, gate as gate
from carnot.reporting.human_source_authority_9001 import grouped, normalized
from carnot.reporting.natural_label_eligibility_8975 import ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9029_v771_source_identity_resolution"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/source_identity_9029.py",
    "python/carnot/reporting/source_identity_runner_9029.py",
    CLI,
]
TEST = "tests/python/test_source_identity_9029.py"
DEFAULT_INPUT = (
    ROOT / "results/raw/experiment_9015_v770_human_disagreement_source/release/manifest.json"
)
MANIFEST_PIN = "sha256:f4c8569364fe220a2ee8c17ad22a8d8664a9aa051b5251c4f1bf4f717af3daf9"
UPSTREAM = ROOT / "results/experiment_9015_v770_human_disagreement_source.json"
UPSTREAM_PIN = "sha256:94fefec6e66dbbca4974fbfdab9f5c470bfc515a62d1d0314f1793169872d96a"
MODEL_SPECS: list[Json] = []
USER_AGENT = "CarnotSourceIdentity/9029 (https://github.com/ianblenke/carnot; metadata research)"
Fetch = Callable[[str, float, int], tuple[int, dict[str, str], bytes, str | None]]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a blocked request cannot look like progress."""
    print(f"[exp9029] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(reference: Json) -> None:
    """Reject changed bytes before accepting either a successful or rejected response."""
    if ref(Path(reference["path"]))["sha256"] != reference["sha256"] or reference["sha256"] is None:
        raise ValueError("changed_bytes:" + reference["path"])


def load_inputs(manifest: Path) -> Json:
    """Read only public parquet columns from the exact retained author release."""
    if ref(manifest)["sha256"] != MANIFEST_PIN:
        raise ValueError("release_manifest:" + str(ref(manifest)["sha256"]))
    if ref(UPSTREAM)["sha256"] != UPSTREAM_PIN:
        raise ValueError("upstream_primary:" + str(ref(UPSTREAM)["sha256"]))
    data = json.loads(manifest.read_bytes())
    if data["revision"] != REVISION or data["scope"] != "author_release":
        raise ValueError("release_revision:" + str(data["revision"]))
    for key, pin in PINNED.items():
        if data["files"][key]["sha256"] != "sha256:" + pin:
            raise ValueError("pinned_release:" + key)
    inventory = json.loads(Path(data["exposure_inventory"]["path"]).read_bytes())
    refs = [
        ref(manifest),
        ref(UPSTREAM),
        *data["files"].values(),
        data["metadata"],
        data["resolutions"],
        data["exposure_inventory"],
        *data["custody_references"],
        *data["resolution_receipts"],
        *inventory["rows"],
    ]
    refs += [dict(path=r["path"], sha256=r["sha256"]) for r in data["source_access_receipts"]]
    for reference in refs:
        authenticate(reference)
    metadata = json.loads(Path(data["metadata"]["path"]).read_bytes())
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "cc-by-4.0":
        raise ValueError("author_metadata")
    import pyarrow.parquet as pq

    reader: Any = pq.read_table
    records = [
        dict(row, split=split)
        for split in ["validation", "test"]
        for row in reader(
            data["files"][split]["path"], columns=["id", "lang", "model_input", "wikipedia_url"]
        ).to_pylist()
    ]
    receipts = [
        r for r in data["source_access_receipts"] if urlsplit(r["url"]).path == "/w/api.php"
    ]
    for receipt in receipts:
        parse_receipt(receipt)
    return dict(records=records, inventory=inventory, inherited_receipts=receipts, references=refs)


def identity(url: str) -> str:
    """Normalize URL spelling while preserving case-sensitive Wikipedia title text."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path = (
        re.sub(r"^/zh-(?:cn|tw|hk|mo|sg|my|hans|hant)/", "/wiki/", parsed.path)
        if host == "zh.wikipedia.org"
        else parsed.path
    )
    title = unicodedata.normalize(
        "NFC", unquote(path.removeprefix("/wiki/")).replace("_", " ")
    ).strip()
    if (
        parsed.scheme != "https"
        or not re.fullmatch(r"[a-z]+\.wikipedia\.org", host)
        or not path.startswith("/wiki/")
        or not title
        or parsed.port
    ):
        raise ValueError("wikipedia_identity:" + url)
    return host + "/" + title


def request_url(host: str, titles: list[str]) -> str:
    """Batch metadata only; the fixed maxlag protects a busy Wikimedia server."""
    return (
        "https://"
        + host
        + "/w/api.php?"
        + urlencode(
            dict(
                action="query",
                prop="pageprops",
                ppprop="wikibase_item",
                titles="|".join(titles),
                format="json",
                redirects=1,
                maxlag=5,
            )
        )
    )


def parse_receipt(receipt: Json) -> Json:
    """Bind normalized and redirected titles to authenticated original request bytes."""
    authenticate(receipt)
    if receipt.get("request_sha256", canonical_hash(receipt["url"])) != canonical_hash(
        receipt["url"]
    ):
        raise ValueError("request_hash")
    parsed = urlsplit(receipt["url"])
    query = parse_qs(parsed.query)
    required = dict(
        action="query", prop="pageprops", ppprop="wikibase_item", format="json", redirects="1"
    )
    if (
        parsed.scheme != "https"
        or parsed.path != "/w/api.php"
        or any(query.get(k) != [v] for k, v in required.items())
    ):
        raise ValueError("request_identity")
    host = parsed.hostname
    titles = query["titles"][0].split("|")
    if len(titles) > 50:
        raise ValueError("request_identity_batch_size")
    for title in titles:
        identity("https://" + str(host) + "/wiki/" + quote(title, safe=""))
    if receipt["status"] != 200 or receipt.get("error"):
        return {}
    data = json.loads(Path(receipt["path"]).read_bytes())
    if "error" in data:
        return {}
    public = data["query"]
    aliases = {
        r["from"]: r["to"] for key in ["normalized", "redirects"] for r in public.get(key, [])
    }
    pages = {p["title"]: p for p in public["pages"].values()}
    values = {}
    for original in titles:
        title, seen = original, set()
        while title in aliases and title not in seen:
            seen.add(title)
            title = aliases[title]
        page = pages.get(title, {})
        qid = page.get("pageprops", {}).get("wikibase_item")
        values[identity("https://" + str(host) + "/wiki/" + quote(original, safe=""))] = dict(
            wikidata_id=qid
            if isinstance(qid, str) and re.fullmatch(r"Q[1-9][0-9]*", qid)
            else None,
            resolved_title=title,
            normalized_titles=public.get("normalized", []),
            redirects=public.get("redirects", []),
            request_sha256=canonical_hash(receipt["url"]),
            response_reference=dict(path=receipt["path"], sha256=receipt["sha256"]),
        )
    return values


def fetch(url: str, timeout: float, limit: int) -> tuple[int, dict[str, str], bytes, str | None]:
    """Keep HTTP error bodies; a bounded read prevents an oversized response from exhausting disk."""
    try:
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as stream:
            body = stream.read(limit)
            return (
                stream.status,
                dict(stream.headers),
                body,
                "byte_budget" if len(body) >= limit else None,
            )
    except HTTPError as error:
        body = error.read(limit)
        return error.code, dict(error.headers), body, "HTTP" + str(error.code)
    except (URLError, TimeoutError, OSError) as error:
        return 0, {}, b"", str(error)


def retry_delay(value: str | None, attempt: int, *, wall: float | None = None) -> float:
    """Honor both Retry-After forms without shortening exponential backoff."""
    delay = float(2 ** (attempt - 1))
    if value:
        try:
            delay = max(delay, float(value))
        except ValueError:
            try:
                delay = max(
                    delay,
                    parsedate_to_datetime(value).timestamp()
                    - (time.time() if wall is None else wall),
                )
            except (ValueError, TypeError, OverflowError):
                pass
    return delay


def acquire(
    urls: list[str],
    inherited: list[Json],
    raw: Path,
    *,
    fetch: Fetch = fetch,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    budget: float = 600,
    byte_limit: int = 268435456,
) -> Json:
    """Resume authenticated batches, checkpoint successes and stop globally at a terminal block."""
    raw.mkdir(parents=True, exist_ok=True)
    checkpoint = raw / "checkpoint.json"
    prior = json.loads(checkpoint.read_bytes()) if checkpoint.exists() else dict(receipts=[])
    receipts = [*inherited, *prior["receipts"]]
    initial_receipt_count = len(receipts)
    resolutions: Json = {}
    for receipt in receipts:
        resolutions.update(parse_receipt(receipt))
    required = sorted({identity(url) for url in urls})
    reuse = sum(bool(resolutions.get(key, {}).get("wikidata_id")) for key in required)
    pending: dict[str, list[str]] = defaultdict(list)
    for key in required:
        if not resolutions.get(key, {}).get("wikidata_id"):
            host, title = key.split("/", 1)
            pending[host].append(title)
    owned = list(prior["receipts"])
    downloaded, retries, blocked = 0, 0, None
    start, last_end = now(), now() - 1
    progress("authenticated_cache", reuse, len(required) - reuse)
    for host, titles in sorted(pending.items()):
        for offset in range(0, len(titles), 50):
            url = request_url(host, titles[offset : offset + 50])
            for attempt in range(1, 4):
                delay = max(0.0, last_end + 1 - now())
                if attempt > 1:
                    delay = max(
                        delay,
                        retry_delay(
                            next(
                                (
                                    v
                                    for k, v in receipt["headers"].items()
                                    if k.lower() == "retry-after"
                                ),
                                None,
                            ),
                            attempt - 1,
                        ),
                    )
                remaining = budget - (now() - start)
                if remaining <= delay or downloaded >= byte_limit:
                    blocked = dict(
                        status=receipt["status"] if attempt > 1 else None,
                        error="time_budget" if remaining <= delay else "byte_budget",
                        url=url,
                    )
                    break
                while delay > 0:
                    done = sum(bool(resolutions.get(k, {}).get("wikidata_id")) for k in required)
                    progress("request_wait", done, len(required) - done)
                    interval = min(delay, 30)
                    sleep(interval)
                    delay -= interval
                progress(
                    "before_network",
                    sum(bool(resolutions.get(k, {}).get("wikidata_id")) for k in required),
                    sum(not resolutions.get(k, {}).get("wikidata_id") for k in required),
                )
                retries += int(attempt > 1)
                status, headers, body, error = fetch(
                    url, min(30, budget - (now() - start)), byte_limit - downloaded
                )
                body = body[: byte_limit - downloaded]
                downloaded += len(body)
                last_end = now()
                if status == 200 and not error and downloaded < byte_limit:
                    try:
                        payload = json.loads(body)
                        if "error" in payload:
                            error = str(payload["error"].get("code", "api_error"))
                    except (ValueError, TypeError):
                        error = "invalid_json"
                body_path = raw / (f"response-{len(owned):04d}.body")
                temporary = body_path.with_suffix(".tmp")
                temporary.write_bytes(body)
                temporary.replace(body_path)
                receipt = dict(
                    ref(body_path),
                    url=url,
                    request_sha256=canonical_hash(url),
                    status=status,
                    headers=headers,
                    error=error,
                    bytes=len(body),
                    attempt=attempt,
                    timeout_s=min(30, remaining),
                    user_agent=USER_AGENT,
                    finished_elapsed_s=last_end - start,
                )
                if downloaded >= byte_limit:
                    receipt["error"] = "byte_budget"
                owned.append(receipt)
                receipts.append(receipt)
                atomic_json(checkpoint, dict(receipts=owned))
                progress("after_network", len(owned), 0)
                try:
                    values = parse_receipt(receipt)
                except (ValueError, KeyError, TypeError) as invalid:
                    receipt["error"] = "invalid_metadata:" + str(invalid)
                    atomic_json(checkpoint, dict(receipts=owned))
                    values = {}
                if status == 200 and not receipt["error"] and values:
                    resolutions.update(values)
                    blocked = None
                    break
                blocked = receipt
                if (
                    (status not in {0, 429, 503} and error != "maxlag")
                    or attempt == 3
                    or downloaded >= byte_limit
                ):
                    break
            if blocked:
                break
        if blocked:
            break
    unresolved = [key for key in required if not resolutions.get(key, {}).get("wikidata_id")]
    progress("metadata_acquisition_complete", len(required) - len(unresolved), len(unresolved))
    return dict(
        resolutions=resolutions,
        receipts=receipts,
        unresolved=unresolved,
        blocked=blocked,
        cache_reuse_count=reuse,
        retry_counts=retries,
        bytes_downloaded=downloaded,
        network_duration_s=now() - start,
        initial_receipt_count=initial_receipt_count,
    )


def reduce_metadata(records: list[Json], inventory: Json, resolutions: Json) -> Json:
    """Use the frozen document parser on identity hashes, excluding prior and unknown exposure."""
    exposed = set(inventory["exposed_keys"])
    history_unknown = []
    for url in inventory["historical_reference_urls"]:
        key = identity(url)
        qid = resolutions.get(key, {}).get("wikidata_id")
        exposed.add("url:" + key)
        if qid:
            exposed.add("qid:" + qid)
        else:
            history_unknown.append(key)
    exposed |= {"url:" + identity(k[4:]) for k in list(exposed) if k.startswith("url:https://")}
    public, sources = {}, []
    for row in records:
        key = identity(row["wikipedia_url"])
        qid = resolutions.get(key, {}).get("wikidata_id")
        keys = ["url:" + key, "prompt:" + canonical_hash(normalized(row["model_input"]))]
        if qid:
            keys.append("qid:" + qid)
        public[row["id"]] = dict(
            unit_id=row["id"],
            language=row["lang"].lower(),
            split=row["split"],
            source_keys=keys,
            resolved=bool(qid),
            prompt_present=bool(row["model_input"]),
        )
        sources.append(
            dict(
                source_id=row["id"],
                task_type="Summary",
                prompt=row["id"],
                source_info="\n\n".join(
                    f"passage {n}: " + canonical_hash(k) for n, k in enumerate(keys)
                ),
            )
        )
    components, _ = grouped(sources, [])
    unknown = bool(history_unknown or inventory.get("unknown_exposure_rows"))
    groups, rows = [], []
    for component in components:
        members = [public[ident] for ident in component["source_ids"]]
        keys = sorted({key for member in members for key in member["source_keys"]})
        resolved = all(member["resolved"] and member["prompt_present"] for member in members)
        reason = (
            "prior_exposure"
            if set(keys) & exposed
            else "unresolved_source_identity"
            if not resolved
            else "unknown_exposure"
            if unknown
            else None
        )
        groups.append(
            dict(
                group_id=component["group_id"],
                source_memberships=members,
                source_keys=keys,
                identity_resolved=resolved,
                exclusion_reason=reason,
            )
        )
        rows.append(
            dict(
                unit_id=component["group_id"],
                group_id=component["group_id"],
                arm="public_metadata",
                condition="source_identity",
                metric="source_identity_authenticated",
                value=int(resolved),
                numerator=int(resolved),
                denominator=1,
                status="completed"
                if reason is None
                else "excluded"
                if reason == "prior_exposure"
                else "censored",
                exclusion_reason=reason,
            )
        )
    languages = sorted({r["lang"].lower() for r in records})
    unresolved = [g["group_id"] for g in groups if not g["identity_resolved"]]
    gates = [
        gate("release_identity_closure", 0, len(unresolved)),
        gate("historical_identity_closure", 0, len(history_unknown)),
        gate("unknown_exposure", [], inventory.get("unknown_exposure_rows", [])),
        gate("all_fourteen_languages", LANGUAGES, languages),
    ]
    return dict(
        rows=rows,
        source_group_rows=groups,
        intended_count=len(rows),
        independent_count=len(groups),
        completed_count=sum(r["status"] == "completed" for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=0,
        source_identity_ready_score=int(all(g["passed"] for g in gates)),
        unresolved_source_groups=unresolved,
        exposure_inventory=dict(
            original_inventory=inventory,
            exposed_keys=sorted(exposed),
            unresolved_identities=history_unknown,
            unresolved_count=len(history_unknown),
            cross_language_exposure_resolution_complete=not unknown,
            planning_prose_is_exposure=False,
        ),
        gate_check_summary=gates,
    )
