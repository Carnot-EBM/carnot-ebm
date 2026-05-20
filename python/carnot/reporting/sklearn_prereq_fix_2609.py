"""Build the Exp 2609 sklearn prerequisite artifact.

Spec: REQ-REPORT-2609, SCENARIO-REPORT-2609.

The conductor gates several verifier-recovery experiments on a small set of
environment facts: sklearn must import, the Carnot package must import from the
repo checkout, and the real FoVer corpus path should be recorded if it exists.
Keeping this logic in a helper module makes the one-time environment repair
auditable without baking the check into `scripts/research_conductor.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_2609_sklearn_prereq_fix.json"

SCHEMA = "carnot.sklearn_prereq_fix.v1"
EXPERIMENT_ID = "exp2609"
TITLE = "Python Package Prerequisites: Install sklearn + Verify PYTHONPATH"

PROMPT_CORPUS_PATHS = (
    "python/carnot/verify/foVer_corpus.jsonl",
    "data/foVer_corpus.jsonl",
)
CANONICAL_LOWERCASE_CORPUS_PATHS = ("data/fover_corpus.jsonl",)

REQUIRED_FIELDS = {
    "honest_verdict",
    "sklearn_available",
    "sklearn_version",
    "sklearn_already_installed",
    "carnot_import_ok",
    "foVer_corpus_found",
    "foVer_corpus_path",
    "preconditions_checked",
}


def utc_now() -> str:
    """Return a stable UTC timestamp so artifacts can be compared across runs."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def line_count(path: Path) -> int | None:
    """Count rows only when a corpus file exists.

    Missing corpus paths are common during recovery tasks. Returning `None`
    lets the artifact distinguish "the file was absent" from "the file existed
    but was too small to satisfy the downstream training gate."
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _corpus_check(project_root: Path, relative_path: str) -> dict[str, Any]:
    path = project_root / relative_path
    n_lines = line_count(path)
    available = n_lines is not None and n_lines > 100
    check: dict[str, Any] = {
        "resource": relative_path,
        "available": available,
        "check": f"wc -l {relative_path} 2>/dev/null",
        "n_lines": 0 if n_lines is None else n_lines,
    }
    if n_lines is None:
        check["error"] = "missing"
    return check


def discover_fover_corpus(project_root: Path = REPO_ROOT) -> tuple[bool, str, list[dict[str, Any]]]:
    """Resolve the FoVer corpus path needed by follow-on verifier recovery.

    The milestone prompt names two camel-case paths, but the checked-in corpus
    currently follows the repository's lowercase convention. The exact prompt
    paths are checked first for audit fidelity; the lowercase canonical path is
    checked afterward so the artifact does not hide a real corpus behind a
    case-sensitive filename mismatch.
    """
    checks: list[dict[str, Any]] = []
    found_path = "not_found"

    for relative_path in (*PROMPT_CORPUS_PATHS, *CANONICAL_LOWERCASE_CORPUS_PATHS):
        check = _corpus_check(project_root, relative_path)
        checks.append(check)
        if found_path == "not_found" and check["available"]:
            found_path = str(project_root / relative_path)

    return found_path != "not_found", found_path, checks


def _python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _verify_sklearn_import() -> tuple[bool, str, str | None]:
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only in broken envs
        return False, "not_installed", f"{type(exc).__name__}: {exc}"
    return True, str(sklearn.__version__), None


def _verify_carnot_import(project_root: Path) -> tuple[bool, str | None]:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(project_root / 'python')!r}); "
        "import carnot; "
        "print('carnot OK')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode == 0:
        return True, None
    error = (completed.stderr or completed.stdout or "unknown carnot import failure").strip()
    return False, error


def build_artifact(
    *,
    sklearn_available: bool,
    sklearn_version: str,
    sklearn_already_installed: bool,
    carnot_import_ok: bool,
    fover_corpus_found: bool,
    fover_corpus_path: str,
    preconditions_checked: Sequence[Mapping[str, Any]],
    install_method: str,
    install_success: bool,
    install_attempts: Sequence[Mapping[str, Any]],
    python_executable: str,
    python_version: str,
    carnot_import_error: str | None = None,
    sklearn_import_error: str | None = None,
) -> dict[str, Any]:
    """Assemble the terminal artifact with explicit downstream gate fields."""
    timestamp = utc_now()
    verdict_suffix = (
        "sklearn prerequisite resolved; import chain verified"
        if sklearn_available and carnot_import_ok
        else "sklearn prerequisite audit completed with blockers recorded"
    )
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 2609,
        "title": TITLE,
        "status": "complete",
        "honest_verdict": f"complete: {verdict_suffix}",
        "generated_at": timestamp,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "python_executable": python_executable,
        "python_version": python_version,
        "platform": platform.platform(),
        "sklearn_available": sklearn_available,
        "sklearn_version": sklearn_version if sklearn_available else "not_installed",
        "sklearn_already_installed": sklearn_already_installed,
        "sklearn_import_error": sklearn_import_error,
        "install_method": install_method,
        "install_success": install_success,
        "install_attempts": [dict(row) for row in install_attempts],
        "carnot_import_ok": carnot_import_ok,
        "carnot_import_error": carnot_import_error,
        "foVer_corpus_found": fover_corpus_found,
        "foVer_corpus_path": fover_corpus_path,
        "preconditions_checked": [dict(row) for row in preconditions_checked],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields the conductor and follow-on tasks consume."""
    missing = sorted(REQUIRED_FIELDS.difference(artifact))
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_")
    ):
        raise ValueError("honest_verdict must start with 'complete:' or 'complete_'")

    for field in (
        "sklearn_available",
        "sklearn_already_installed",
        "carnot_import_ok",
        "foVer_corpus_found",
    ):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be bool")

    for field in ("sklearn_version", "foVer_corpus_path"):
        if not isinstance(artifact[field], str) or not artifact[field]:
            raise ValueError(f"{field} must be a non-empty string")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, list) or not preconditions:
        raise ValueError("preconditions_checked must be a non-empty list")
    for row in preconditions:
        if not isinstance(row, Mapping):
            raise ValueError("preconditions_checked entries must be objects")
        for key in ("resource", "available", "check"):
            if key not in row:
                raise ValueError(f"preconditions_checked entry missing {key}")


def collect_live_artifact(
    *,
    sklearn_already_installed: bool,
    install_method: str,
    install_success: bool,
    install_attempts: Sequence[Mapping[str, Any]],
    project_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Collect live verification results after the prerequisite repair runs."""
    sklearn_available, sklearn_version, sklearn_error = _verify_sklearn_import()
    carnot_ok, carnot_error = _verify_carnot_import(project_root)
    corpus_found, corpus_path, corpus_checks = discover_fover_corpus(project_root)

    preconditions: list[dict[str, Any]] = [dict(row) for row in install_attempts]
    preconditions.append(
        {
            "resource": "sklearn.linear_model.LogisticRegression",
            "available": sklearn_available,
            "check": (
                "python -c \"import sklearn; "
                "from sklearn.linear_model import LogisticRegression; "
                "print(sklearn.__version__)\""
            ),
            "version": sklearn_version,
            "error": sklearn_error,
        }
    )
    preconditions.append(
        {
            "resource": "carnot",
            "available": carnot_ok,
            "check": (
                "python -c \"import sys; "
                f"sys.path.insert(0, '{PROJECT_ROOT_FOR_METADATA}/python'); "
                "import carnot; print('carnot OK')\""
            ),
            "error": carnot_error,
        }
    )
    preconditions.extend(corpus_checks)

    return build_artifact(
        sklearn_available=sklearn_available,
        sklearn_version=sklearn_version,
        sklearn_already_installed=sklearn_already_installed,
        carnot_import_ok=carnot_ok,
        fover_corpus_found=corpus_found,
        fover_corpus_path=corpus_path,
        preconditions_checked=preconditions,
        install_method=install_method,
        install_success=install_success,
        install_attempts=install_attempts,
        python_executable=sys.executable,
        python_version=_python_version(),
        carnot_import_error=carnot_error,
        sklearn_import_error=sklearn_error,
    )


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a sorted JSON artifact so diffs and conductor reads stay stable."""
    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_install_attempts(raw: str) -> list[dict[str, Any]]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("install attempts must be a JSON list")
    return [dict(row) for row in parsed]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--sklearn-already-installed", action="store_true")
    state.add_argument("--sklearn-was-missing", action="store_true")
    parser.add_argument("--install-method", required=True)
    parser.add_argument("--install-success", action="store_true")
    parser.add_argument("--install-attempts-json", type=_parse_install_attempts, default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    artifact = collect_live_artifact(
        sklearn_already_installed=args.sklearn_already_installed,
        install_method=args.install_method,
        install_success=args.install_success,
        install_attempts=args.install_attempts_json,
    )
    write_artifact(args.out, artifact)
    print(args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI smoke-tested by invocation.
    raise SystemExit(main())
