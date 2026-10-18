"""Exp 3811 abstention cross-surface parity smoke.

This is a wiring smoke, not an accuracy claim.  It compares the certified
Exp 3771 abstention operating point across verify API, CLI/batch, and HTTP/REST
surfaces using cached FoVer-style verifier-scoring candidates.

Spec: REQ-SPOE-3811, SCENARIO-SPOE-3811.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any

from carnot.pipeline import abstention_http_rest as rest
from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3811_abstention_cross_surface_parity_smoke.json")
CERTIFIED_THRESHOLD_REL_PATH = Path(
    "results/experiment_3771_certified_abstention_operating_point.json"
)
EXP3779_REL_PATH = Path("results/experiment_3779_abstention_operating_point_product_wiring.json")
EXP3789_REL_PATH = Path("results/experiment_3789_abstention_cli_batch_surface.json")
EXP3810_REL_PATH = Path("results/experiment_3810_abstention_http_rest_surface_v2.json")
FOVER_CORPUS_REL_PATH = Path("data/fover_corpus_v4.json")
RANDOM_SEED = 3811
SURFACES = ("verify_api", "cli", "http_rest")
FLOAT_TOLERANCE = 1e-6
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached triples across three surfaces, no live LLM)."
)
SCORING_VERIFIERS = (
    "controlled_invariance_executor_v2",
    "executable_monitor_runtime_adapter",
    "ast_structure_verifier",
    "code_structural_dependency_verifier",
)
COMPLETE_VERDICT_TEMPLATE = (
    "complete: abstention_cross_surface_parity_smoke_all_surfaces_agree_"
    "{agree}_n{n}_verify_api_cli_http_rest_no_surface_drift"
)
FIXED_CONFIDENT_EXAMPLE_IDS = (
    "math-156-0",
    "math-159-1",
    "math-160-2",
    "math-161-3",
    "math-164-4",
)
FIXED_ABSTAIN_EXAMPLE_IDS = (
    "math-math_v3_1422-200",
    "math-math_85-29",
    "math-math_192-51",
    "math-math_432-83",
    "math-math_v3_792-133",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "surfaces_compared",
    "all_surfaces_agree",
    "n_candidates_compared",
    "mismatches",
    "certified_threshold_used",
    "tests_assert_real_behavior",
    "cited_upstream_artifacts",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the parity outcome; blocked_<resource> if a precondition failed."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates "
        "(principle: scores cached triples across three surfaces, no live LLM)."
    ),
    "surfaces_compared": (
        "BARE list -- the three surfaces compared (verify_api, cli, http_rest); "
        "records the Rule-4 coverage."
    ),
    "all_surfaces_agree": (
        "BARE bool -- every candidate's verdict + metadata matched across all "
        "three surfaces."
    ),
    "n_candidates_compared": (
        "BARE int, >=10 -- sample-size hygiene spanning confident + abstain."
    ),
    "mismatches": (
        "BARE list -- any per-candidate cross-surface mismatch; a drift finding "
        "surfaced honestly."
    ),
    "certified_threshold_used": (
        "The shared threshold from Exp 3771 that all three surfaces loaded."
    ),
    "tests_assert_real_behavior": (
        "BARE bool, true -- shipped tests assert the real parity behavior."
    ),
    "cited_upstream_artifacts": (
        "Provenance for the three surfaces + the threshold."
    ),
    "model_specs": (
        "Names the 4 verifiers + the certified-threshold source -- honest substrate."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def select_cached_fover_candidates(root: Path | str, n_candidates: int = 10) -> list[JsonDict]:
    """Return a fixed cached FoVer candidate sample spanning both verdicts."""

    root_path = Path(root)
    examples, status = spd.load_cached_labeled_examples(
        root_path,
        use_balanced_code_corpus=True,
    )
    math_status = status.get("math", {})
    corpus_path = root_path / FOVER_CORPUS_REL_PATH
    if math_status.get("status") != "loaded" or not corpus_path.exists():
        raise FileNotFoundError(str(corpus_path.resolve()))

    by_id = {example.example_id: example for example in examples if example.domain == "math"}
    wanted = list(FIXED_CONFIDENT_EXAMPLE_IDS + FIXED_ABSTAIN_EXAMPLE_IDS)
    if n_candidates > len(wanted):
        extras = [
            example_id
            for example_id in sorted(by_id)
            if example_id not in set(wanted)
        ]
        wanted.extend(extras[: n_candidates - len(wanted)])
    wanted = wanted[:n_candidates]

    missing = [example_id for example_id in wanted if example_id not in by_id]
    if missing:
        raise FileNotFoundError(f"cached FoVer examples missing: {missing}")

    candidates: list[JsonDict] = []
    abstain_ids = set(FIXED_ABSTAIN_EXAMPLE_IDS)
    for example_id in wanted:
        example = by_id[example_id]
        candidates.append(
            {
                "candidate_id": example.example_id,
                "domain": "math",
                "text": f"cached FoVer verifier-scoring candidate {example.example_id}",
                "confidence_error": round(float(example.confidence_error), 6),
                "ensemble_energy": round(float(example.ensemble_energy), 6),
                "source_corpus_path": str(corpus_path.resolve()),
                "expected_verdict": "abstain" if example_id in abstain_ids else "confident",
            }
        )
    return candidates


def compare_surfaces(
    root: Path | str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    executable: str,
    certified_threshold_path: Path | str,
) -> JsonDict:
    """Call verify API, CLI, and HTTP surfaces and compare normalized rows."""

    root_path = Path(root)
    threshold_path = Path(certified_threshold_path).resolve()
    surface_rows = {
        "verify_api": normalize_verify_api_response(
            run_verify_api_surface(root_path, candidates)
        ),
        "cli": normalize_cli_response(run_cli_surface(root_path, candidates, executable)),
        "http_rest": normalize_http_response(
            run_http_surface(root_path, candidates, threshold_path)
        ),
    }
    return compare_normalized_surface_rows(surface_rows)


def run_verify_api_surface(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Score candidates through the in-process verify API surface."""

    return spd.score_candidates(
        [dict(candidate) for candidate in candidates],
        root=root,
        default_domain="math",
        abstention_mode=True,
    )


def run_cli_surface(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
    executable: str,
) -> JsonDict:
    """Score candidates through the packaged CLI batch surface."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp3811-") as tmp:
        candidates_file = Path(tmp) / "candidates.json"
        candidates_file.write_text(
            json.dumps([dict(candidate) for candidate in candidates], sort_keys=True),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                executable,
                "-m",
                "carnot.cli",
                "verify-batch",
                "--candidates-file",
                str(candidates_file),
                "--domain",
                "math",
                "--abstention-mode",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"CLI surface failed rc={completed.returncode}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def run_http_surface(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
    threshold_path: Path,
) -> JsonDict:
    """Score candidates through the local HTTP/REST endpoint."""

    server = rest.make_server(
        ("127.0.0.1", 0),
        root=root,
        certified_threshold_path=threshold_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        status, payload = _post_json(
            str(host),
            int(port),
            {
                "domain": "math",
                "abstention_mode": True,
                "candidates": [dict(candidate) for candidate in candidates],
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
    if status != 200:
        raise RuntimeError(f"HTTP surface failed status={status}: {payload}")
    return payload


def normalize_verify_api_response(response: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Normalize verify API rows to the parity fields."""

    return {
        str(row["candidate_id"]): _normalize_product_row(row)
        for row in response.get("scores", [])
        if isinstance(row, Mapping)
    }


def normalize_cli_response(response: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Normalize CLI batch rows to the parity fields."""

    return normalize_verify_api_response(response)


def normalize_http_response(response: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Normalize HTTP/REST rows to the parity fields."""

    normalized: dict[str, JsonDict] = {}
    for row in response.get("scores", []):
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row["candidate_id"])
        normalized[candidate_id] = {
            "candidate_id": candidate_id,
            "verdict": str(row["verdict"]),
            "coverage": _float(row["coverage"]),
            "risk": _float(row["risk"]),
            "delta": _float(row["delta"]),
            "threshold": _float(row["threshold"]),
        }
    return normalized


def compare_normalized_surface_rows(
    surface_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> JsonDict:
    """Compare normalized rows and return explicit mismatch records."""

    candidate_ids = sorted(
        {
            candidate_id
            for rows in surface_rows.values()
            for candidate_id in rows
        }
    )
    mismatches: list[JsonDict] = []
    for candidate_id in candidate_ids:
        for surface in SURFACES:
            if candidate_id not in surface_rows.get(surface, {}):
                mismatches.append(
                    {
                        "candidate_id": candidate_id,
                        "field": "presence",
                        "values_by_surface": {
                            item: candidate_id in rows
                            for item, rows in surface_rows.items()
                        },
                    }
                )
                break
        else:
            for field in ("verdict", "coverage", "risk", "delta", "threshold"):
                values = {
                    surface: surface_rows[surface][candidate_id][field]
                    for surface in SURFACES
                }
                if not _values_match(field, values):
                    mismatches.append(
                        {
                            "candidate_id": candidate_id,
                            "field": field,
                            "values_by_surface": values,
                        }
                    )

    canonical_rows = [
        dict(surface_rows["verify_api"][candidate_id])
        for candidate_id in candidate_ids
        if candidate_id in surface_rows.get("verify_api", {})
    ]
    return {
        "surfaces_compared": list(SURFACES),
        "all_surfaces_agree": not mismatches,
        "n_candidates_compared": len(candidate_ids),
        "mismatches": mismatches,
        "canonical_rows": canonical_rows,
        "surface_rows": {
            surface: dict(rows)
            for surface, rows in surface_rows.items()
        },
    }


def check_preconditions(
    root: Path | str,
    *,
    executable: str,
    certified_threshold_path: Path | str,
    exp3779_path: Path | str,
    exp3789_path: Path | str,
    exp3810_path: Path | str,
    fover_corpus_path: Path | str,
) -> tuple[JsonDict, abstention.CertifiedAbstentionConfig | None]:
    """Check resources required before claiming cross-surface parity."""

    root_path = Path(root)
    executable_path = Path(executable)
    threshold_path = Path(certified_threshold_path).resolve()
    fover_path = Path(fover_corpus_path).resolve()
    preconditions: JsonDict = {
        "interpreter": {
            "available": (
                executable_path.exists()
                and executable_path.name == "python"
                and ".venv" in executable_path.parts
            ),
            "value": str(executable_path),
            "expected_suffix": ".venv/bin/python",
        }
    }
    if preconditions["interpreter"]["available"]:
        probe = subprocess.run(
            [
                str(executable_path),
                "-c",
                (
                    "import carnot; "
                    "from carnot.pipeline.second_pair_detector import score_candidates; "
                    "from carnot.cli import cmd_verify_batch; "
                    "from carnot.pipeline.abstention_http_rest "
                    "import score_candidates_http_payload; "
                    "print('ok')"
                ),
            ],
            cwd=root_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        preconditions["package_import_and_surfaces"] = {
            "available": probe.returncode == 0,
            "returncode": probe.returncode,
            "stderr": probe.stderr.strip(),
        }
    else:
        preconditions["package_import_and_surfaces"] = {
            "available": False,
            "detail": "interpreter unavailable",
        }

    config: abstention.CertifiedAbstentionConfig | None
    try:
        config = abstention.load_certified_abstention_config(threshold_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        preconditions["certified_threshold"] = {
            "available": False,
            "path": str(threshold_path),
            "detail": f"{type(exc).__name__}: {exc}",
        }
        config = None
    else:
        preconditions["certified_threshold"] = {
            "available": True,
            "path": config.threshold_source,
            "selected_threshold": config.threshold,
        }

    for name, path_value in (
        ("upstream_exp3779_verify_api", exp3779_path),
        ("upstream_exp3789_cli", exp3789_path),
    ):
        path = Path(path_value).resolve()
        preconditions[name] = {"available": path.exists(), "path": str(path)}

    exp3810 = Path(exp3810_path).resolve()
    try:
        exp3810_payload = json.loads(exp3810.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        preconditions["http_rest_surface"] = {
            "available": False,
            "path": str(exp3810),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    else:
        preconditions["http_rest_surface"] = {
            "available": exp3810_payload.get("http_rest_surface_added") is True,
            "path": str(exp3810),
            "http_rest_surface_added": exp3810_payload.get("http_rest_surface_added"),
        }

    if fover_path.exists():
        try:
            rows = json.loads(fover_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            preconditions["fover_corpus"] = {
                "available": False,
                "path": str(fover_path),
                "detail": f"JSONDecodeError: {exc}",
            }
        else:
            preconditions["fover_corpus"] = {
                "available": isinstance(rows, list) and bool(rows),
                "path": str(fover_path),
                "n_examples": len(rows) if isinstance(rows, list) else 0,
            }
    else:
        preconditions["fover_corpus"] = {
            "available": False,
            "path": str(fover_path),
        }
    return preconditions, config


def first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    """Return the first blocked_<resource> verdict implied by preconditions."""

    for name, result in preconditions.items():
        if not isinstance(result, Mapping) or result.get("available"):
            continue
        if name == "certified_threshold":
            return "blocked_no_certified_threshold"
        if name == "fover_corpus":
            return "blocked_fover_corpus_missing"
        if name == "http_rest_surface":
            return "blocked_http_surface_not_landed"
        if name == "package_import_and_surfaces":
            return "blocked_surface_import"
        if name.startswith("upstream_"):
            return "blocked_upstream_artifact_missing"
        return f"blocked_{name}"
    return None


def build_artifact(
    *,
    verdict: str,
    duration_s: float,
    threshold_config: abstention.CertifiedAbstentionConfig | None,
    preconditions: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    root: Path,
    output_path: Path,
    threshold_path: Path,
    exp3779_path: Path,
    exp3789_path: Path,
    exp3810_path: Path,
    fover_corpus_path: Path,
) -> JsonDict:
    """Assemble the Exp 3811 terminal artifact."""

    comparison_payload = dict(comparison or {})
    all_agree = bool(comparison_payload.get("all_surfaces_agree", False))
    n_compared = int(comparison_payload.get("n_candidates_compared", 0))
    mismatches = list(comparison_payload.get("mismatches", []))
    artifact: JsonDict = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "surfaces_compared": list(SURFACES),
        "all_surfaces_agree": all_agree,
        "n_candidates_compared": n_compared,
        "mismatches": mismatches,
        "certified_threshold_used": (
            None if threshold_config is None else round(float(threshold_config.threshold), 6)
        ),
        "tests_assert_real_behavior": True,
        "cited_upstream_artifacts": {
            "verify_api": str(exp3779_path.resolve()),
            "cli": str(exp3789_path.resolve()),
            "http_rest": str(exp3810_path.resolve()),
            "certified_threshold": str(threshold_path.resolve()),
        },
        "model_specs": model_specs(threshold_path),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": round(max(0.0, duration_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions),
        "methodology": {
            "label": "WIRING smoke, not an accuracy claim",
            "candidate_source": str(fover_corpus_path.resolve()),
            "random_seed": RANDOM_SEED,
            "live_llm_inference": False,
            "selection": "fixed cached FoVer verifier-scoring candidates spanning verdicts",
        },
        "candidate_sample": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "source_corpus_path": candidate.get("source_corpus_path"),
                "expected_verdict": candidate.get("expected_verdict"),
            }
            for candidate in candidates
        ],
        "parity_rows": list(comparison_payload.get("canonical_rows", [])),
        "surface_rows": dict(comparison_payload.get("surface_rows", {})),
        "output_path": str(output_path.resolve()),
        "scripts_research_conductor_modified": False,
        "root": str(root.resolve()),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | None = None,
    executable: str | None = None,
    certified_threshold_path: Path | None = None,
    exp3779_path: Path | None = None,
    exp3789_path: Path | None = None,
    exp3810_path: Path | None = None,
    fover_corpus_path: Path | None = None,
    n_candidates: int = 10,
    candidate_selector: Callable[[Path, int], Sequence[Mapping[str, Any]]] = (
        select_cached_fover_candidates
    ),
    surface_runner: Callable[..., Mapping[str, Any]] = compare_surfaces,
) -> JsonDict:
    """Run Exp 3811 and write its terminal artifact."""

    start = time.perf_counter()
    root_path = Path(root)
    output = root_path / OUTPUT_REL_PATH if output_path is None else output_path
    threshold_path = (
        root_path / CERTIFIED_THRESHOLD_REL_PATH
        if certified_threshold_path is None
        else certified_threshold_path
    ).resolve()
    upstream_3779 = (
        root_path / EXP3779_REL_PATH if exp3779_path is None else exp3779_path
    ).resolve()
    upstream_3789 = (
        root_path / EXP3789_REL_PATH if exp3789_path is None else exp3789_path
    ).resolve()
    upstream_3810 = (
        root_path / EXP3810_REL_PATH if exp3810_path is None else exp3810_path
    ).resolve()
    fover_path = (
        root_path / FOVER_CORPUS_REL_PATH
        if fover_corpus_path is None
        else fover_corpus_path
    ).resolve()
    exe = executable or str(root_path / ".venv/bin/python")

    preconditions, threshold_config = check_preconditions(
        root_path,
        executable=exe,
        certified_threshold_path=threshold_path,
        exp3779_path=upstream_3779,
        exp3789_path=upstream_3789,
        exp3810_path=upstream_3810,
        fover_corpus_path=fover_path,
    )
    blocker = first_blocker(preconditions)
    candidates: Sequence[Mapping[str, Any]] = []
    comparison: Mapping[str, Any] | None = None
    if blocker is None:
        candidates = list(candidate_selector(root_path, n_candidates))
        comparison = dict(
            surface_runner(
                root_path,
                candidates,
                executable=exe,
                certified_threshold_path=threshold_path,
            )
        )
        verdict = COMPLETE_VERDICT_TEMPLATE.format(
            agree=str(bool(comparison.get("all_surfaces_agree"))).lower(),
            n=int(comparison.get("n_candidates_compared", len(candidates))),
        )
    else:
        verdict = blocker

    artifact = build_artifact(
        verdict=verdict,
        duration_s=time.perf_counter() - start,
        threshold_config=threshold_config,
        preconditions=preconditions,
        comparison=comparison,
        candidates=candidates,
        root=root_path,
        output_path=output,
        threshold_path=threshold_path,
        exp3779_path=upstream_3779,
        exp3789_path=upstream_3789,
        exp3810_path=upstream_3810,
        fover_corpus_path=fover_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def model_specs(threshold_path: Path) -> JsonDict:
    """Return verifier and threshold provenance for the parity smoke."""

    return {
        "verifiers": list(SCORING_VERIFIERS),
        "certified_threshold_source": str(threshold_path.resolve()),
        "surfaces": list(SURFACES),
        "verify_api_entrypoint": "carnot.pipeline.second_pair_detector.score_candidates",
        "cli_entrypoint": "python -m carnot.cli verify-batch",
        "http_entrypoint": f"POST {rest.POST_PATH}",
        "live_llm_inference": False,
        "wiring_smoke_not_accuracy_claim": True,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic artifact fields for drift detection."""

    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "surfaces_compared": artifact.get("surfaces_compared"),
        "all_surfaces_agree": artifact.get("all_surfaces_agree"),
        "n_candidates_compared": artifact.get("n_candidates_compared"),
        "mismatches": artifact.get("mismatches"),
        "certified_threshold_used": artifact.get("certified_threshold_used"),
        "candidate_sample": artifact.get("candidate_sample"),
        "parity_rows": artifact.get("parity_rows"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _normalize_product_row(row: Mapping[str, Any]) -> JsonDict:
    cert = row.get("certified_abstention")
    if not isinstance(cert, Mapping):
        raise ValueError(f"missing certified_abstention metadata for {row.get('candidate_id')}")
    route_to_review = bool(row.get("route_to_review", row.get("abstained", False)))
    candidate_id = str(row["candidate_id"])
    return {
        "candidate_id": candidate_id,
        "verdict": "abstain" if route_to_review else "confident",
        "coverage": _float(cert["coverage"]),
        "risk": _float(cert["certified_risk_bound"]),
        "delta": _float(cert["delta"]),
        "threshold": _float(cert["threshold"]),
    }


def _values_match(field: str, values: Mapping[str, Any]) -> bool:
    if field == "verdict":
        return len(set(values.values())) == 1
    numeric = [float(value) for value in values.values()]
    return max(numeric) - min(numeric) <= FLOAT_TOLERANCE


def _post_json(host: str, port: int, payload: object) -> tuple[int, JsonDict]:
    conn = http.client.HTTPConnection(host, port, timeout=10.0)
    try:
        conn.request(
            "POST",
            rest.POST_PATH,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return response.status, json.loads(data) if data else {}
    finally:
        conn.close()


def _float(value: Any) -> float:
    return round(float(value), 6)
