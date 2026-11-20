from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20261120"
MILESTONE = "2026.11.788"
TASK = "exp9256-corpus-prefilter"
EXP_ID = 9256
OUT = "results/experiment_9256_v788_corpus_prefilter.json"
RAW = "results/raw/experiment_9256_v788_corpus_prefilter"
BASELINE_TARGET_REL = (
    "results/raw/experiment_9256_v788_corpus_prefilter/baseline/"
    "test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
)
BASELINE_TARGET_SHA256 = "sha256:30dde5c4fd63097d24010d781f45441f9fca9db4cb6d3c126882a18d94d5d845"
SEED = 92562026
BUDGET_S = 4800.0
MEASUREMENT_S = 3300.0
VERIFICATION_S = 1200.0
PUBLICATION_S = 300.0
DEFAULT_CHUNK_SIZE = 262144
SENTINEL_CONSTANTS = (
    "ARC_LIVE_AGENT_NO_LLM_SUBSTRATE",
    "ARC_LIVE_AGENT_ENV_INTERACTION_SUBSTRATE",
    "ARC_SUBMITTED_KERNEL_OFFLINE_FROZEN_POLICY_REPLAY_SUBSTRATE",
    "ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE",
    "ARC_SUPERVISOR_AB_NO_LLM_SUBSTRATE",
)
STRATUM_BOUNDS = (
    ("le_4kb", 0, 4096),
    ("4kb_to_64kb", 4097, 65536),
    ("64kb_to_1mb", 65537, 1048576),
    ("1mb_to_16mb", 1048577, 16777216),
    ("gt_16mb", 16777217, 1 << 62),
)
OWNED = (
    "python/carnot/reporting/v788_corpus_prefilter.py",
    "python/carnot/reporting/v788_corpus_prefilter_runner.py",
    "scripts/experiments/experiment_9256_v788_corpus_prefilter.py",
)
OWNED_TESTS = (
    "tests/python/test_v788_corpus_prefilter_9256.py",
    "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py",
)
MODIFIED = (
    "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py",
    "python/carnot/reporting/causal_validation_9167.py",
    "python/carnot/reporting/validation_schedule_9139.py",
)
UPSTREAM_PINS = {
    "exp9255-contract": (
        "results/experiment_9255_v788_contract.json",
        "sha256:7c669d5df1331eb5d22211d22bca58db4959b76af6abcaa814ca5de5a5ea24ef",
    ),
    "exp9181-causal-validation": (
        "results/experiment_9181_v782_causal_validation.json",
        "sha256:dbdb988f3ce372334dc1634309556987d26f05c73739eeae574624ea601d2934",
    ),
    "exp9241-v787-contract": (
        "results/experiment_9241_v787_contract.json",
        "sha256:32cdfef7df5fb0b829ef9eba8d68a44a0e6b7947afbd402d24129dae27a92f5b",
    ),
    "exp9242-v787-corpus-prefilter": (
        "results/experiment_9242_corpus_prefilter.json",
        "sha256:9e5cf741802b6c1202c6bbbb5ff3b008929d2fbd39b079ff64a00a2d50e63063",
    ),
}
_AV_CACHE: dict[str, Any] = {}
_UPSTREAM_CACHE: dict[str, dict[str, Any]] = {}
_SLIM_KEYS = (
    "honest_verdict",
    "verdict_class",
    "contract_ready_score",
    "replay_authority_ready_score",
    "required_checks_passed",
    "causal_fix_ready_score",
    "protocol_references",
    "validation_receipts",
)


def _load_slim_json(raw_b: bytes, sha_obs: str) -> dict[str, Any]:
    if sha_obs not in _UPSTREAM_CACHE:
        parsed = json.loads(raw_b.decode("utf-8"))
        _UPSTREAM_CACHE[sha_obs] = {k: parsed[k] for k in _SLIM_KEYS if k in parsed}
    return _UPSTREAM_CACHE[sha_obs]


def log(msg: str) -> None:
    print(f"[exp9256] {msg}", flush=True)


def digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def file_ref(path: Path, root: Path | None = ROOT) -> dict[str, Any]:
    rel = (
        path.relative_to(root).as_posix()
        if (root is not None and path.is_relative_to(root))
        else path.resolve().as_posix()
    )
    raw = path.read_bytes()
    return {"path": rel, "sha256": digest_bytes(raw), "size_bytes": len(raw)}


def load_av(root: Path = ROOT) -> Any:
    target = (root / "scripts/adversarial_verify.py").resolve()
    key = str(target)
    if key not in _AV_CACHE:
        spec = importlib.util.spec_from_file_location("adversarial_verify_9256", target)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _AV_CACHE[key] = mod
    return _AV_CACHE[key]


def derive_sentinels(av: Any) -> dict[str, Any]:
    sentinels: list[tuple[str, str, bytes]] = []
    for name in SENTINEL_CONSTANTS:
        val = str(getattr(av, name))
        raw = val.encode("ascii")
        sentinels.append((name, val, raw))
    max_len = max(len(item[2]) for item in sentinels)
    return {
        "sentinels": tuple(sentinels),
        "max_needle_len": max_len,
        "min_overlap_bytes": max_len - 1,
        "fast_common_subneedle": b"live_",
    }


def stratum_for_size(size_bytes: int) -> str:
    for label, lo, hi in STRATUM_BOUNDS[:-1]:
        if lo <= size_bytes <= hi:
            return label
    return STRATUM_BOUNDS[-1][0]


def stream_prefilter_path(
    path: Path,
    sentinels: tuple[tuple[str, str, bytes], ...],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    include_unicode_escape: bool = True,
) -> dict[str, Any]:
    """Read to EOF so a negative decision and byte identity cover the entire file."""
    max_len = max((len(raw) for _, _, raw in sentinels), default=73)
    overlap = max(72, max_len - 1)
    step = max(chunk_size, overlap + 1)
    use_fast_prefix = all(b"live_" in raw for _, _, raw in sentinels)
    tail = b""
    bytes_read = 0
    chunks_read = 0
    matched_sentinel: str | None = None
    unicode_escape_seen = False
    digest = hashlib.sha256()
    heartbeat = time.monotonic()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(step)
            if not chunk:
                break
            bytes_read += len(chunk)
            digest.update(chunk)
            chunks_read += 1
            buf = tail + chunk
            if not use_fast_prefix or b"live_" in buf:
                for const_name, _, needle in sentinels:
                    if needle in buf:
                        matched_sentinel = const_name
                        break
            if (
                matched_sentinel is None
                and include_unicode_escape
                and b"\\" in buf
                and b"\\u" in buf
            ):
                unicode_escape_seen = True
            tail = buf[-overlap:]
            now = time.monotonic()
            if now - heartbeat >= 60:
                log(f"stream_heartbeat path={path} bytes_read={bytes_read}")
                heartbeat = now
            if now >= float(os.environ.get("CARNOT_9256_DEADLINE", "inf")):
                raise TimeoutError(f"stream_deadline:{path}")
    should_decode = matched_sentinel is not None or unicode_escape_seen
    decode_reason = (
        f"sentinel:{matched_sentinel}"
        if matched_sentinel is not None
        else (
            "unicode_escape" if unicode_escape_seen else "eof_zero_sentinels_zero_unicode_escapes"
        )
    )
    return {
        "bytes_read": bytes_read,
        "sha256": f"sha256:{digest.hexdigest()}",
        "chunks_read": chunks_read,
        "overlap_bytes": overlap,
        "matched_sentinel": matched_sentinel,
        "unicode_escape_seen": unicode_escape_seen,
        "should_decode": should_decode,
        "decode_reason": decode_reason,
    }


def _classify_decoded(data: Any, av: Any) -> tuple[str, bool, bool, float | None]:
    if not isinstance(data, dict):
        return ("non_dict_json", False, False, None)
    is_pos = bool(av._is_arc_live_agent_no_llm(data))
    dur = data.get("duration_s")
    dur_val = float(dur) if isinstance(dur, (int, float)) and not isinstance(dur, bool) else None
    floor_s = float(getattr(av, "ARC_LIVE_AGENT_NO_LLM_DURATION_FLOOR_S", 0.01))
    is_off = bool(is_pos and dur_val is not None and dur_val < floor_s)
    return ("decoded_positive" if is_pos else "decoded_negative", is_pos, is_off, dur_val)


def original_evaluate_path(path: Path, av: Any) -> dict[str, Any]:
    """Keep the original decoder and classifier so the filter cannot change their answers."""
    t0 = time.perf_counter()
    read_s = decode_s = classifier_s = 0.0
    try:
        raw_text = path.read_text(encoding="utf-8")
        read_s = time.perf_counter() - t0
        size_b = len(raw_text.encode("utf-8"))
        decode_start = time.perf_counter()
        data = json.loads(raw_text)
        decode_s = time.perf_counter() - decode_start
    except Exception:
        return {
            "bytes_read": path.stat().st_size,
            "decode_reason": "unconditional_original_json_loads",
            "decode_status": "malformed_or_unreadable",
            "is_positive": False,
            "is_offender": False,
            "duration_val": None,
            "elapsed_s": time.perf_counter() - t0,
            "read_s": read_s,
            "decode_s": decode_s,
            "classifier_s": classifier_s,
        }
    classify_start = time.perf_counter()
    status, is_pos, is_off, dur_val = _classify_decoded(data, av)
    classifier_s = time.perf_counter() - classify_start
    return {
        "bytes_read": size_b,
        "decode_reason": "unconditional_original_json_loads",
        "decode_status": status,
        "is_positive": is_pos,
        "is_offender": is_off,
        "duration_val": dur_val,
        "elapsed_s": time.perf_counter() - t0,
        "read_s": read_s,
        "decode_s": decode_s,
        "classifier_s": classifier_s,
    }


def prefilter_evaluate_path(
    path: Path,
    av: Any,
    *,
    sentinels: tuple[tuple[str, str, bytes], ...] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    include_unicode_escape: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    active = derive_sentinels(av)["sentinels"] if sentinels is None else sentinels
    pre = stream_prefilter_path(
        path,
        active,
        chunk_size=chunk_size,
        include_unicode_escape=include_unicode_escape,
    )
    if not pre["should_decode"]:
        return {
            **pre,
            "decode_status": "skipped_eof_negative",
            "is_positive": False,
            "is_offender": False,
            "duration_val": None,
            "elapsed_s": time.perf_counter() - t0,
        }
    decoded = original_evaluate_path(path, av)
    return {
        **pre,
        **decoded,
        "sha256": pre["sha256"],
        "decode_reason": pre["decode_reason"],
        "bytes_read": pre["bytes_read"] + decoded["bytes_read"],
        "stream_bytes_read": pre["bytes_read"],
        "elapsed_s": time.perf_counter() - t0,
    }


def scan_corpus(
    results_dir: Path,
    av: Any | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_every: int = 25000,
    scan_output: Path | None = None,
    stop_monotonic: float | None = None,
) -> dict[str, Any]:
    av_mod = load_av() if av is None else av
    s_info = derive_sentinels(av_mod)
    sentinels = s_info["sentinels"]
    t0 = time.perf_counter()
    log(f"enumerate_manifest_start dir={results_dir}")
    all_paths = sorted(p for p in results_dir.rglob("*.json") if p.is_file())
    total_files = len(all_paths)
    log(f"scan_corpus_start dir={results_dir} total_files={total_files}")

    stop_monotonic = (
        stop_monotonic
        if stop_monotonic is not None
        else float(os.environ.get("CARNOT_9256_DEADLINE", "inf"))
    )
    pop_hasher = hashlib.sha256()
    stratum_Quick: dict[str, dict[str, Any]] = {
        label: {
            "stratum": label,
            "file_count": 0,
            "total_bytes": 0,
            "decoded_count": 0,
            "safely_negative_eof_count": 0,
            "malformed_or_nondict_count": 0,
            "positive_count": 0,
            "offender_count": 0,
            "bytes_read": 0,
            "elapsed_s": 0.0,
            "first_path": None,
            "last_path": None,
        }
        for label, _, _ in STRATUM_BOUNDS
    }
    manifest_rows: list[dict[str, Any]] = []
    for path in all_paths:
        st = path.stat()
        rel = path.relative_to(results_dir).as_posix()
        manifest_rows.append(
            {
                "path": rel,
                "abs_path": str(path),
                "size_bytes": st.st_size,
                "stratum": stratum_for_size(st.st_size),
                "sha256": None,
                "stat_identity": (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
            }
        )
        pop_hasher.update(f"{rel}\t{st.st_size}\n".encode())
    stratum_samples: dict[str, int] = {label: 0 for label, _, _ in STRATUM_BOUNDS}
    scan_rows: list[dict[str, Any]] = []
    offenders: list[tuple[str, Any]] = []

    visited = 0
    decoded_count = 0
    safely_neg_count = 0
    malformed_count = 0
    nondict_count = 0
    checked = 0
    total_bytes = 0
    total_bytes_read = 0
    changed_files = 0
    parity_discrepancies = 0
    heartbeat = time.monotonic()
    env_out = os.environ.get("CARNOT_9256_SCAN_OUTPUT")
    out_target = scan_output or (Path(env_out) if env_out else None)

    for idx, path in enumerate(all_paths):
        if stop_monotonic is not None and time.monotonic() >= stop_monotonic:
            break
        rel = path.relative_to(results_dir).as_posix()
        st_before = path.stat()
        size_b = int(st_before.st_size)
        stratum = stratum_for_size(size_b)

        try:
            res = prefilter_evaluate_path(path, av_mod, sentinels=sentinels, chunk_size=chunk_size)
        except TimeoutError:
            log(f"scan_censored_at_deadline path={path}")
            break
        st_after = path.stat()
        identity = (
            st_before.st_ino,
            st_before.st_size,
            st_before.st_mtime_ns,
            st_before.st_ctime_ns,
        )
        after_identity = (
            st_after.st_ino,
            st_after.st_size,
            st_after.st_mtime_ns,
            st_after.st_ctime_ns,
        )
        changed_files += int(identity != after_identity)
        changed_files += int(identity != tuple(manifest_rows[idx]["stat_identity"]))
        manifest_rows[idx]["sha256"] = res["sha256"]
        visited += 1
        total_bytes += size_b
        total_bytes_read += int(res["bytes_read"])

        s_bucket = stratum_Quick[stratum]
        s_bucket["file_count"] += 1
        s_bucket["total_bytes"] += size_b
        s_bucket["bytes_read"] += int(res["bytes_read"])
        s_bucket["elapsed_s"] += float(res["elapsed_s"])
        if s_bucket["first_path"] is None:
            s_bucket["first_path"] = rel
        s_bucket["last_path"] = rel

        status = res["decode_status"]
        if status == "skipped_eof_negative":
            safely_neg_count += 1
            s_bucket["safely_negative_eof_count"] += 1
        elif status == "malformed_or_unreadable":
            decoded_count += 1
            malformed_count += 1
            s_bucket["decoded_count"] += 1
            s_bucket["malformed_or_nondict_count"] += 1
        elif status == "non_dict_json":
            decoded_count += 1
            nondict_count += 1
            s_bucket["decoded_count"] += 1
            s_bucket["malformed_or_nondict_count"] += 1
        else:
            decoded_count += 1
            s_bucket["decoded_count"] += 1
            if res["is_positive"]:
                checked += 1
                s_bucket["positive_count"] += 1
                if res["is_offender"]:
                    offenders.append((rel, res["duration_val"]))
                    s_bucket["offender_count"] += 1

        keep_sample = stratum_samples[stratum] < (2 if stratum == "gt_16mb" else 16)
        if keep_sample:
            stratum_samples[stratum] += 1
        orig = original_evaluate_path(path, av_mod) if keep_sample and size_b <= 1_048_576 else res
        p_match = (
            orig["is_positive"] == res["is_positive"] and orig["is_offender"] == res["is_offender"]
        )
        parity_discrepancies += int(not p_match)
        scan_rows.append(
            {
                "path": rel,
                "stratum": stratum,
                "size_bytes": size_b,
                "bytes_read": int(res["bytes_read"]),
                "decode_reason": res["decode_reason"],
                "decode_status": status,
                "original_positive": orig["is_positive"],
                "candidate_positive": res["is_positive"],
                "original_offender": orig["is_offender"],
                "candidate_offender": res["is_offender"],
                "parity_match": p_match,
                "comparison_basis": "original_decoder"
                if orig is not res
                else ("same_unchanged_decoder" if res["should_decode"] else "eof_absence_proof"),
                "duration_s": round(float(res["elapsed_s"]), 6),
                "censor_reason": "none",
            }
        )

        if (
            time.monotonic() - heartbeat >= 60
            or (idx + 1) % progress_every == 0
            or (idx + 1) == total_files
        ):
            log(
                f"scan_progress visited={visited}/{total_files} decoded={decoded_count} "
                f"safely_negative={safely_neg_count} checked={checked} offenders={len(offenders)}"
            )
            heartbeat = time.monotonic()

    for row in manifest_rows:
        path = Path(row["abs_path"])
        if not path.exists():
            changed_files += 1
            continue
        st = path.stat()
        changed_files += int(
            (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != tuple(row["stat_identity"])
        )
    final_paths = sorted(
        p.relative_to(results_dir).as_posix() for p in results_dir.rglob("*.json") if p.is_file()
    )
    initial_paths = [p.relative_to(results_dir).as_posix() for p in all_paths]
    added_paths = sorted(set(final_paths) - set(initial_paths))
    removed_paths = sorted(set(initial_paths) - set(final_paths))

    unfinished = total_files - visited
    for row in manifest_rows[visited:]:
        scan_rows.append(
            {
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "bytes_read": 0,
                "decode_reason": "deadline",
                "decode_status": "unfinished",
                "original_positive": None,
                "candidate_positive": None,
                "original_offender": None,
                "candidate_offender": None,
                "parity_match": None,
                "duration_s": 0.0,
                "censor_reason": "measurement_deadline",
            }
        )
    summary = {
        "results_dir": str(results_dir),
        "total_files": total_files,
        "visited_files": visited,
        "unfinished_files": unfinished,
        "changed_files": changed_files,
        "candidate_bytes_stable": changed_files == 0
        and unfinished == 0
        and not added_paths
        and not removed_paths,
        "added_paths": added_paths,
        "removed_paths": removed_paths,
        "total_size_bytes": total_bytes,
        "total_bytes_read": total_bytes_read,
        "population_sha256": f"sha256:{pop_hasher.hexdigest()}",
        "decoded_count": decoded_count,
        "safely_negative_eof_count": safely_neg_count,
        "malformed_count": malformed_count,
        "nondict_count": nondict_count,
        "checked": checked,
        "offenders": offenders,
        "parity_discrepancies": parity_discrepancies,
        "semantic_parity_exact": parity_discrepancies == 0 and unfinished == 0,
        "duration_s": round(time.perf_counter() - t0, 6),
        "stratum_summary": list(stratum_Quick.values()),
        "manifest_rows": manifest_rows,
        "scan_rows": scan_rows,
    }
    if out_target is not None:
        out_target.parent.mkdir(parents=True, exist_ok=True)
        out_target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_parity_and_mutation_controls(
    av: Any | None = None,
    tmp_root: Path | None = None,
) -> dict[str, Any]:
    av_mod = load_av() if av is None else av
    s_info = derive_sentinels(av_mod)
    sentinels = s_info["sentinels"]
    base = (
        Path(tempfile.mkdtemp(prefix="carnot-9256-parity-", dir="/var/tmp"))
        if tmp_root is None
        else tmp_root
    )
    base.mkdir(parents=True, exist_ok=True)
    chunk_size = 128
    pad = "p" * (chunk_size - 42)

    fixtures: list[tuple[str, str, bytes]] = []
    for idx, (const_name, text, _) in enumerate(sentinels):
        fixtures.append(
            (
                f"sentinel_{idx}_above",
                const_name,
                json.dumps({"inference_substrate": text, "duration_s": 0.05}).encode("utf-8"),
            )
        )
        fixtures.append(
            (
                f"sentinel_{idx}_below_cross_chunk",
                const_name,
                json.dumps({"pad": pad, "inference_substrate": text, "duration_s": 0.001}).encode(
                    "utf-8"
                ),
            )
        )
    s0 = sentinels[0][1]
    s1 = sentinels[1][1]
    for idx, sep in enumerate((" -- note", "; note", ": note", ", note", ". note", " note")):
        fixtures.append(
            (
                f"suffix_accepted_{idx}",
                "suffix_boundary",
                json.dumps({"inference_substrate": f"{s0}{sep}", "duration_s": 0.02}).encode(
                    "utf-8"
                ),
            )
        )
    for idx, bad_sep in enumerate(("_v2", "2", "-extra")):
        fixtures.append(
            (
                f"suffix_rejected_{idx}",
                "suffix_boundary",
                json.dumps({"inference_substrate": f"{s0}{bad_sep}", "duration_s": 0.001}).encode(
                    "utf-8"
                ),
            )
        )
    fixtures.extend(
        [
            (
                "principle_wrapped_valid",
                "principle_wrapped",
                json.dumps(
                    {
                        "inference_substrate": {"value": s1, "principle": "honest"},
                        "duration_s": 0.002,
                    }
                ).encode("utf-8"),
            ),
            (
                "principle_wrapped_missing_value",
                "principle_wrapped",
                json.dumps({"inference_substrate": {"principle": s1}, "duration_s": 0.001}).encode(
                    "utf-8"
                ),
            ),
            (
                "unicode_escaped_key_and_value",
                "unicode_escape",
                (
                    '{"pad":"'
                    + ("u" * (chunk_size - 11))
                    + '","inference\\u005fsubstrate":"live\\u005fagent_environment_interaction","duration_s":0.002}'
                ).encode("utf-8"),
            ),
            (
                "duplicate_keys_last_wins_positive",
                "duplicate_keys",
                (
                    '{"inference_substrate":"no_model_load","inference_substrate":"'
                    + s1
                    + '","duration_s":0.002}'
                ).encode("utf-8"),
            ),
            (
                "duplicate_keys_last_wins_negative",
                "duplicate_keys",
                (
                    '{"inference_substrate":"'
                    + s1
                    + '","inference_substrate":"no_model_load","duration_s":0.002}'
                ).encode("utf-8"),
            ),
            (
                "nested_decoy_negative",
                "nested_decoy",
                json.dumps(
                    {
                        "nested": {"inference_substrate": s0},
                        "inference_substrate": "no_model_load",
                        "duration_s": 0.001,
                    }
                ).encode("utf-8"),
            ),
            (
                "malformed_utf8_with_sentinel",
                "malformed",
                b'{"inference_substrate":"' + s1.encode("ascii") + b'",\xff\xfe}',
            ),
            (
                "malformed_json_without_sentinel",
                "malformed",
                b'{"inference_substrate":"no_model_load",',
            ),
            (
                "nondict_list_with_sentinel",
                "nondict",
                json.dumps([{"inference_substrate": s0, "duration_s": 0.001}]).encode("utf-8"),
            ),
            (
                "nondict_scalar_string",
                "nondict",
                json.dumps(s0).encode("utf-8"),
            ),
            (
                "safely_negative_large_eof",
                "eof_negative",
                json.dumps(
                    {"pad": "n" * 600, "inference_substrate": "no_model_load", "duration_s": 0.1}
                ).encode("utf-8"),
            ),
        ]
    )

    fixture_paths: list[tuple[str, str, Path]] = []
    parity_rows: list[dict[str, Any]] = []
    discrepancies = 0
    for name, category, payload in fixtures:
        fpath = base / f"{name}.json"
        fpath.write_bytes(payload)
        fixture_paths.append((name, category, fpath))
        orig = original_evaluate_path(fpath, av_mod)
        cand = prefilter_evaluate_path(
            fpath,
            av_mod,
            sentinels=sentinels,
            chunk_size=chunk_size,
        )
        matched = (
            orig["is_positive"] == cand["is_positive"]
            and orig["is_offender"] == cand["is_offender"]
        )
        discrepancies += int(not matched)
        parity_rows.append(
            {
                "fixture": name,
                "category": category,
                "decode_reason": cand["decode_reason"],
                "decode_status": cand["decode_status"],
                "original_positive": orig["is_positive"],
                "candidate_positive": cand["is_positive"],
                "original_offender": orig["is_offender"],
                "candidate_offender": cand["is_offender"],
                "parity_match": matched,
            }
        )

    mutation_controls: list[dict[str, Any]] = []
    for const_name, _, _ in sentinels:
        reduced = tuple(item for item in sentinels if item[0] != const_name)
        missed = 0
        for _, _, fpath in fixture_paths:
            orig = original_evaluate_path(fpath, av_mod)
            mut = prefilter_evaluate_path(
                fpath,
                av_mod,
                sentinels=reduced,
                chunk_size=chunk_size,
            )
            missed += int(orig["is_positive"] and not mut["is_positive"])
        mutation_controls.append(
            {
                "control_id": f"delete_{const_name}",
                "mutated_operand": const_name,
                "missed_positive_count": missed,
                "mutation_detected": missed > 0,
            }
        )
    missed_u = 0
    for _, _, fpath in fixture_paths:
        orig = original_evaluate_path(fpath, av_mod)
        mut_u = prefilter_evaluate_path(
            fpath,
            av_mod,
            sentinels=sentinels,
            chunk_size=chunk_size,
            include_unicode_escape=False,
        )
        missed_u += int(orig["is_positive"] and not mut_u["is_positive"])
    mutation_controls.append(
        {
            "control_id": "delete_unicode_escape_fallback",
            "mutated_operand": "\\u",
            "missed_positive_count": missed_u,
            "mutation_detected": missed_u > 0,
        }
    )
    return {
        "fixture_count": len(parity_rows),
        "parity_discrepancies": discrepancies,
        "semantic_parity_exact": discrepancies == 0,
        "parity_rows": parity_rows,
        "mutation_controls": mutation_controls,
        "all_mutations_detected": all(item["mutation_detected"] for item in mutation_controls),
    }


def measure_stratum_costs(
    root: Path,
    manifest_rows: list[dict[str, Any]],
    av: Any | None = None,
    *,
    per_stratum_limit: int = 16,
) -> dict[str, Any]:
    av_mod = load_av(root) if av is None else av
    sentinels = derive_sentinels(av_mod)["sentinels"]
    by_stratum: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in STRATUM_BOUNDS}
    existing_rows = [row for row in manifest_rows if Path(row["abs_path"]).exists()]
    for row in sorted(existing_rows, key=lambda item: (int(item["size_bytes"]), str(item["path"]))):
        bucket = by_stratum[str(row["stratum"])]
        limit = min(per_stratum_limit, 1) if row["stratum"] == "gt_16mb" else per_stratum_limit
        if len(bucket) < limit:
            bucket.append(row)

    strata_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for label, lo, hi in STRATUM_BOUNDS:
        selected = by_stratum[label]
        orig_s = 0.0
        pref_s = 0.0
        orig_dec = 0
        pref_dec = 0
        neg_eof = 0
        sample_b = 0
        discrepancies = 0
        for row in selected:
            fpath = Path(row["abs_path"])
            log(f"stratum_benchmark_before stratum={label} path={fpath}")
            before = fpath.stat()
            orig = original_evaluate_path(fpath, av_mod)
            pref = prefilter_evaluate_path(fpath, av_mod, sentinels=sentinels)
            after = fpath.stat()
            unit_rows.append(
                {
                    "path": row["path"],
                    "stratum": label,
                    "size_bytes": row["size_bytes"],
                    "original_clock_s": orig["elapsed_s"],
                    "prefilter_clock_s": pref["elapsed_s"],
                    "original_read_s": orig["read_s"],
                    "original_decode_s": orig["decode_s"],
                    "original_classifier_s": orig["classifier_s"],
                    "candidate_read_s": pref.get("read_s", 0.0),
                    "candidate_decode_s": pref.get("decode_s", 0.0),
                    "candidate_classifier_s": pref.get("classifier_s", 0.0),
                    "same_input_stable": (before.st_mtime_ns, before.st_ctime_ns, before.st_size)
                    == (after.st_mtime_ns, after.st_ctime_ns, after.st_size),
                    "censor_reason": None,
                }
            )
            log(f"stratum_benchmark_after stratum={label} path={fpath}")
            orig_s += float(orig["elapsed_s"])
            pref_s += float(pref["elapsed_s"])
            orig_dec += 1
            pref_dec += int(pref["decode_status"] != "skipped_eof_negative")
            neg_eof += int(pref["decode_status"] == "skipped_eof_negative")
            sample_b += int(row["size_bytes"])
            discrepancies += int(
                orig["is_positive"] != pref["is_positive"]
                or orig["is_offender"] != pref["is_offender"]
            )
        strata_rows.append(
            {
                "stratum": label,
                "min_bytes": lo,
                "max_bytes": hi,
                "sample_files": len(selected),
                "sample_bytes": sample_b,
                "original_clock_s": round(orig_s, 6),
                "prefilter_clock_s": round(pref_s, 6),
                "delta_clock_s": round(pref_s - orig_s, 6),
                "original_decoded_count": orig_dec,
                "prefilter_decoded_count": pref_dec,
                "safely_negative_eof_count": neg_eof,
                "parity_discrepancies": discrepancies,
            }
        )
    return {
        "full_corpus_speedup_claimed": False,
        "unit_rows": unit_rows,
        "censored_baseline_note": (
            "Exp9181 direct_consumers was SIGTERM-censored at 1224.316s inside "
            "test_no_artifact_on_disk_is_newly_flagged_by_closing_this_hole; "
            "no full-corpus speedup ratio is claimed from a censored baseline."
        ),
        "strata": strata_rows,
    }


def _gate_field(
    data: dict[str, Any] | None,
    *,
    check: str,
    upstream: str,
    rel_path: str,
    sha256: str | None,
    field: str,
    expected: Any,
) -> dict[str, Any]:
    if data is None:
        obs: Any = "missing_file"
        ok = False
    elif field not in data:
        obs = "missing_field"
        ok = False
    else:
        obs = data[field]
        ok = type(obs) is type(expected) and obs == expected
    return {
        "check": check,
        "upstream": upstream,
        "path": rel_path,
        "sha256": sha256,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": obs,
        "passed": ok,
    }


def inspect(
    root: Path,
    raw: Path,
    *,
    authenticate_only: bool = False,
    scan_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw.mkdir(parents=True, exist_ok=True)
    av_mod = load_av()
    mem_text = Path("/proc/meminfo").read_text("utf-8")
    mem_kb = next(
        int(line.split()[1]) for line in mem_text.splitlines() if line.startswith("MemAvailable:")
    )
    disk_free = shutil.disk_usage(raw).free
    preconditions = [
        {
            "resource": "cpu_cores",
            "available": (os.cpu_count() or 1) >= 1,
            "observed": os.cpu_count() or 1,
        },
        {
            "resource": "mem_available_mb",
            "available": mem_kb // 1024 >= 512,
            "observed": mem_kb // 1024,
        },
        {
            "resource": "disk_free_mb",
            "available": disk_free // (1024 * 1024) >= 512,
            "observed": disk_free // (1024 * 1024),
        },
        {
            "resource": "required_tools",
            "available": (root / "scripts/adversarial_verify.py").exists()
            and (root / "scripts/verdict_row_consistency_lint.py").exists()
            and all(
                os.access(ROOT / ".venv/bin" / name, os.X_OK)
                for name in ("python", "pytest", "ruff", "mypy")
            ),
            "observed": {
                name: str(ROOT / ".venv/bin" / name)
                for name in ("python", "pytest", "ruff", "mypy")
            },
        },
    ]

    loaded: dict[str, dict[str, Any] | None] = {}
    source_refs: list[dict[str, Any]] = []
    gate_checks: list[dict[str, Any]] = []
    for upstream, (rel, exp_sha) in UPSTREAM_PINS.items():
        p_file = root / rel
        if p_file.exists():
            raw_b = p_file.read_bytes()
            sha_obs = digest_bytes(raw_b)
            loaded[upstream] = _load_slim_json(raw_b, sha_obs)
            source_refs.append(
                {
                    "upstream": upstream,
                    "path": rel,
                    "sha256": sha_obs,
                    "expected_sha256": exp_sha,
                    "size_bytes": len(raw_b),
                    "imported_fields": [
                        "honest_verdict",
                        "verdict_class",
                        "contract_ready_score",
                        "replay_authority_ready_score",
                        "causal_fix_ready_score",
                        "validation_receipts",
                        "protocol_references",
                    ],
                }
            )
        else:
            sha_obs = "missing_file"
            loaded[upstream] = None
        gate_checks.append(
            {
                "check": f"{upstream.split('-')[0]}_sha256",
                "upstream": upstream,
                "path": rel,
                "sha256": None if sha_obs == "missing_file" else sha_obs,
                "artifact_field": None,
                "op": "sha256==",
                "expected": exp_sha,
                "observed": sha_obs,
                "passed": sha_obs == exp_sha,
            }
        )

    c9255 = loaded["exp9255-contract"]
    sha9255 = gate_checks[0]["sha256"]
    for field_name, exp_val in (
        ("contract_ready_score", 1),
        ("replay_authority_ready_score", 1),
        ("required_checks_passed", True),
    ):
        gate_checks.append(
            _gate_field(
                c9255,
                check=f"exp9255_{field_name}",
                upstream="exp9255-contract",
                rel_path=UPSTREAM_PINS["exp9255-contract"][0],
                sha256=sha9255,
                field=field_name,
                expected=exp_val,
            )
        )
    c9181 = loaded["exp9181-causal-validation"]
    gate_checks.append(
        _gate_field(
            c9181,
            check="exp9181_causal_fix_ready_score",
            upstream="exp9181-causal-validation",
            rel_path=UPSTREAM_PINS["exp9181-causal-validation"][0],
            sha256=gate_checks[1]["sha256"],
            field="causal_fix_ready_score",
            expected=0,
        )
    )
    gate_checks.append(
        _gate_field(
            c9241 := loaded["exp9241-v787-contract"],
            check="exp9241_honest_verdict",
            upstream="exp9241-v787-contract",
            rel_path=UPSTREAM_PINS["exp9241-v787-contract"][0],
            sha256=gate_checks[2]["sha256"],
            field="honest_verdict",
            expected="complete_blocked_v787_contract",
        )
    )
    gate_checks.append(
        _gate_field(
            loaded["exp9242-v787-corpus-prefilter"],
            check="exp9242_honest_verdict",
            upstream="exp9242-v787-corpus-prefilter",
            rel_path=UPSTREAM_PINS["exp9242-v787-corpus-prefilter"][0],
            sha256=gate_checks[3]["sha256"],
            field="honest_verdict",
            expected="blocked_gate_check_failed",
        )
    )

    historical = next(
        (
            row
            for row in (c9181 or {}).get("validation_receipts", [])
            if row.get("name") == "direct_consumers"
        ),
        {},
    )
    for field, expected in (("exit_code", -15), ("duration_s", 1224.3159505779913)):
        gate_checks.append(
            _gate_field(
                historical,
                check=f"exp9181_direct_consumers_{field}",
                upstream="exp9181-causal-validation",
                rel_path=UPSTREAM_PINS["exp9181-causal-validation"][0],
                sha256=gate_checks[1]["sha256"],
                field=field,
                expected=expected,
            )
        )
    for key in ("receipt_reference", "events_reference"):
        reference = historical.get(key, {})
        path = Path(reference.get("path", "/missing-9256"))
        observed = file_ref(path, None)["sha256"] if path.is_file() else "missing_file"
        gate_checks.append(
            {
                "check": f"exp9181_{key}",
                "upstream": "exp9181-causal-validation",
                "path": str(path),
                "sha256": observed,
                "artifact_field": "sha256",
                "op": "==",
                "expected": reference.get("sha256", "missing_field"),
                "observed": observed,
                "passed": observed == reference.get("sha256"),
            }
        )

    scan_cache = raw / "corpus_scan_summary.json"
    results_dir = root / "results"
    gates_passed = all(row["passed"] for row in gate_checks) and all(
        row["available"] for row in preconditions
    )
    if gates_passed and not authenticate_only and results_dir.exists():
        corpus_scan = (
            scan_summary
            if scan_summary is not None
            else scan_corpus(results_dir, av_mod, scan_output=scan_cache)
        )
    else:
        corpus_scan = {
            "results_dir": str(results_dir),
            "total_files": 0,
            "visited_files": 0,
            "unfinished_files": 0,
            "changed_files": 0,
            "candidate_bytes_stable": False,
            "total_size_bytes": 0,
            "total_bytes_read": 0,
            "population_sha256": digest_bytes(b""),
            "decoded_count": 0,
            "safely_negative_eof_count": 0,
            "malformed_count": 0,
            "nondict_count": 0,
            "checked": 0,
            "offenders": [],
            "parity_discrepancies": 0,
            "semantic_parity_exact": False,
            "duration_s": 0.0,
            "stratum_summary": [],
            "manifest_rows": [],
            "scan_rows": [],
        }
    controls = (
        run_parity_and_mutation_controls(av_mod)
        if gates_passed and not authenticate_only
        else {
            "fixture_count": 0,
            "parity_discrepancies": 0,
            "semantic_parity_exact": False,
            "parity_rows": [],
            "mutation_controls": [],
            "all_mutations_detected": False,
        }
    )
    stratum_costs = (
        measure_stratum_costs(root, corpus_scan["manifest_rows"], av_mod)
        if gates_passed and not authenticate_only
        else {"full_corpus_speedup_claimed": False, "strata": []}
    )
    if c9255 and "protocol_references" in c9255:
        proto_refs = c9255["protocol_references"]
    else:
        proto_refs = {}
    code_hashes = [
        file_ref(ROOT / rel, ROOT)
        for rel in (
            *OWNED,
            *OWNED_TESTS,
            *MODIFIED,
            "scripts/adversarial_verify.py",
            "pyproject.toml",
        )
        if (ROOT / rel).exists()
    ]
    return {
        "preconditions": preconditions,
        "gate_checks": gate_checks,
        "source_references": source_refs,
        "protocol_references": proto_refs,
        "code_config_hashes": code_hashes,
        "sentinel_info": {
            "constants": [
                {"name": name, "value": text, "byte_length": len(raw_b)}
                for name, text, raw_b in derive_sentinels(av_mod)["sentinels"]
            ],
            "max_needle_len": 73,
            "min_overlap_bytes": 72,
        },
        "exp9181_timeout_diagnostics": {
            "upstream_path": UPSTREAM_PINS["exp9181-causal-validation"][0],
            "direct_consumers_exit_code": historical.get("exit_code"),
            "direct_consumers_duration_s": historical.get("duration_s"),
            "receipt_reference": historical.get("receipt_reference"),
            "events_reference": historical.get("events_reference"),
            "unfinished_nodeid": (
                "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py::"
                "test_no_artifact_on_disk_is_newly_flagged_by_closing_this_hole"
            ),
            "baseline_target_test_sha256": BASELINE_TARGET_SHA256,
        },
        "corpus_scan": corpus_scan,
        "controls": controls,
        "stratum_costs": stratum_costs,
    }
