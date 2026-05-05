"""Run Exp 1390: submit the audited arXiv bundle or write manual steps.

The key safety rule in this module is that a missing credential is not treated
as a submission attempt. The runner either makes a real HTTP request to arXiv's
SWORD endpoint with a verified source archive, or it leaves the operator a
complete browser checklist that uses the same ready archive and metadata.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1390_arxiv_submission_sword_api.json"
DEFAULT_BUNDLE_PATH = DEFAULT_RESULTS_DIR / "arxiv_bundle_v11.tar.gz"
DEFAULT_CHECKLIST_PATH = Path("docs") / "arxiv-manual-submission-checklist.md"
DEFAULT_PAPER_PATH = Path("docs") / "arxiv-paper" / "main.tex"
DEFAULT_OPS_SERVER_PATH = Path("ops") / "server.md"

EXPERIMENT = "1390_arxiv_submission_sword_api"
SCHEMA = "arxiv_submission_sword_api_v1"
RUN_DATE = "20260505"
SWORD_DEPOSIT_URL = "https://arxiv.org/sword/deposit"
PRIMARY_CATEGORY = "cs.LG"
LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
AUTHOR_NAME = "Ian Blenke"
AUTHOR_EMAIL = "ian@blenke.com"

HttpPost = Callable[..., Any]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def _base_artifact(status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": status,
        "bundle_path": "results/arxiv_bundle_v11.tar.gz",
        "submission_attempted": False,
        "submission_method": "pending",
        "arxiv_id_if_submitted": None,
        "submission_result": "pending",
        "manual_checklist_generated": False,
        "manual_checklist_path": None,
        "honest_verdict": status,
    }


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Write the required interruption-safe artifact before doing real work."""

    return _write_json(
        Path(out_path),
        {
            **_base_artifact("in_progress"),
            "honest_verdict": (
                "Submission workflow started; bundle verification and credential discovery pending."
            ),
        },
    )


def _extract_braced_command(tex_text: str, command: str) -> str | None:
    marker = f"\\{command}"
    start = tex_text.find(marker)
    if start < 0:
        return None
    brace_start = tex_text.find("{", start + len(marker))
    if brace_start < 0:
        return None
    depth = 1
    chars: list[str] = []
    idx = brace_start + 1
    while idx < len(tex_text) and depth:
        char = tex_text[idx]
        previous = tex_text[idx - 1] if idx else ""
        if char == "{" and previous != "\\":
            depth += 1
        elif char == "}" and previous != "\\":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
        idx += 1
    return "".join(chars) if depth == 0 else None


def _extract_abstract(tex_text: str) -> str | None:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        tex_text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def _plain_latex(value: str) -> str:
    """Convert simple paper metadata from LaTeX markup to form-friendly text."""

    text = value.replace("\\\\", " ")
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\*|\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    return " ".join(text.split())


def _metadata_tex(value: str) -> str:
    """Collapse whitespace while preserving TeX math that arXiv can render."""

    text = value.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    return " ".join(text.split())


def load_submission_metadata(paper_path: Path | str = DEFAULT_PAPER_PATH) -> dict[str, Any]:
    """Load the title and abstract from the audited TeX paper.

    The author, category, and license are fixed by the experiment prompt. The
    title and abstract come from the paper source so the API and manual checklist
    cannot drift from the archive that the operator uploads.
    """

    path = Path(paper_path)
    tex_text = path.read_text(encoding="utf-8")
    raw_title = _extract_braced_command(tex_text, "title") or "Carnot"
    raw_abstract = _extract_abstract(tex_text) or ""
    return {
        "title": _plain_latex(raw_title),
        "abstract": _metadata_tex(raw_abstract),
        "authors": [{"name": AUTHOR_NAME, "email": AUTHOR_EMAIL}],
        "primary_category": PRIMARY_CATEGORY,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "metadata_source": str(path),
    }


def _parse_config_value(config_text: str, key: str) -> str | None:
    pattern = re.compile(rf"(?im)^\s*(?:export\s+)?{re.escape(key)}\s*(?:=|:)\s*['\"]?([^'\"\s#]+)")
    match = pattern.search(config_text)
    return match.group(1) if match else None


def discover_credentials(
    *,
    environ: Mapping[str, str] | None = None,
    ops_server_path: Path | str = DEFAULT_OPS_SERVER_PATH,
) -> dict[str, Any]:
    """Find non-interactive arXiv credentials without exposing secret values."""

    env = os.environ if environ is None else environ
    pairs = (
        ("ARXIV_SWORD_USERNAME", "ARXIV_SWORD_PASSWORD"),
        ("ARXIV_USERNAME", "ARXIV_PASSWORD"),
    )
    for username_key, password_key in pairs:
        username = env.get(username_key)
        password = env.get(password_key)
        if username and password:
            return {
                "available": True,
                "username": username,
                "password": password,
                "source": f"environment:{username_key}/{password_key}",
                "ops_server_exists": Path(ops_server_path).exists(),
            }

    path = Path(ops_server_path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        for username_key, password_key in pairs:
            username = _parse_config_value(text, username_key)
            password = _parse_config_value(text, password_key)
            if username and password:
                return {
                    "available": True,
                    "username": username,
                    "password": password,
                    "source": f"{path}:{username_key}/{password_key}",
                    "ops_server_exists": True,
                }
    return {
        "available": False,
        "username": None,
        "password": None,
        "source": None,
        "ops_server_exists": path.exists(),
    }


def _atom_metadata_xml(metadata: dict[str, Any]) -> str:
    author = metadata["authors"][0]
    return "\n".join(
        [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<entry xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:arxiv="http://arxiv.org/schemas/atom">',
            f"  <title>{escape(metadata['title'])}</title>",
            f"  <summary>{escape(metadata['abstract'])}</summary>",
            "  <author>",
            f"    <name>{escape(author['name'])}</name>",
            f"    <email>{escape(author['email'])}</email>",
            "  </author>",
            f'  <category term="{escape(metadata["primary_category"])}" '
            'scheme="http://arxiv.org/schemas/atom" />',
            f'  <arxiv:primary_category term="{escape(metadata["primary_category"])}" '
            'scheme="http://arxiv.org/schemas/atom" />',
            f"  <arxiv:license>{escape(metadata['license_url'])}</arxiv:license>",
            "</entry>",
        ]
    )


def _parse_arxiv_id(text: str) -> str | None:
    match = re.search(r"(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", text)
    return match.group(1) if match else None


def _tail(text: object, limit: int = 2000) -> str:
    value = str(text or "")
    return value[-limit:]


def default_http_post(**kwargs: Any) -> Any:
    """Call requests.post lazily so tests can inject a fake HTTP client."""

    import requests

    return requests.post(**kwargs)


def attempt_sword_submission(
    *,
    bundle_path: Path,
    metadata: dict[str, Any],
    credentials: dict[str, Any],
    http_post: HttpPost = default_http_post,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST the source archive and metadata to the arXiv SWORD endpoint."""

    atom_xml = _atom_metadata_xml(metadata)
    data = {
        "metadata": atom_xml,
        "title": metadata["title"],
        "abstract": metadata["abstract"],
        "authors": f"{AUTHOR_NAME} <{AUTHOR_EMAIL}>",
        "category": metadata["primary_category"],
        "license": metadata["license"],
    }
    files = {
        "file": (
            bundle_path.name,
            bundle_path.read_bytes(),
            "application/gzip",
        )
    }
    headers = {
        "Slug": f"carnot-arxiv-v11-{RUN_DATE}",
        "In-Progress": "false",
    }
    try:
        response = http_post(
            url=SWORD_DEPOSIT_URL,
            auth=(credentials["username"], credentials["password"]),
            files=files,
            data=data,
            headers=headers,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - exercised by real failures.
        return {
            "submission_attempted": True,
            "submission_method": "sword_api",
            "submission_result": f"sword_api_exception_{type(exc).__name__}",
            "arxiv_id_if_submitted": None,
            "sword_endpoint": SWORD_DEPOSIT_URL,
            "sword_exception": str(exc),
        }

    status_code = int(getattr(response, "status_code", 0) or 0)
    response_text = str(getattr(response, "text", "") or "")
    response_headers = dict(getattr(response, "headers", {}) or {})
    arxiv_id = _parse_arxiv_id(response_text + "\n" + json.dumps(response_headers))
    ok = 200 <= status_code < 300
    return {
        "submission_attempted": True,
        "submission_method": "sword_api",
        "submission_result": "submitted"
        if ok and arxiv_id
        else (
            f"sword_api_success_http_{status_code}_no_arxiv_id"
            if ok
            else f"sword_api_failed_http_{status_code}"
        ),
        "arxiv_id_if_submitted": arxiv_id,
        "sword_endpoint": SWORD_DEPOSIT_URL,
        "sword_response_status_code": status_code,
        "sword_response_text_tail": _tail(response_text),
        "sword_response_headers": response_headers,
    }


def _manual_checklist_text(
    *,
    root: Path,
    bundle_path: Path,
    metadata: dict[str, Any],
) -> str:
    rel_bundle = _relative_path(bundle_path, root)
    abs_bundle = str(bundle_path.resolve())
    abstract = metadata["abstract"]
    return f"""# Carnot arXiv Manual Submission Checklist

Run date: 2026-05-05

Upload URL: https://arxiv.org/submit

Ready bundle:
- Relative path: `{rel_bundle}`
- Absolute path: `{abs_bundle}`
- Verified non-empty source archive: yes

## Pre-Filled Metadata

Title:

```text
{metadata["title"]}
```

Authors:

```text
{AUTHOR_NAME} <{AUTHOR_EMAIL}>
```

Primary category:

```text
{metadata["primary_category"]}
```

License:

```text
{metadata["license"]} ({metadata["license_url"]})
```

Abstract:

```text
{abstract}
```

Comments:

```text
Position paper draft v3; arXiv source bundle v11 prepared 2026-05-05.
```

Secondary categories, if the arXiv form offers them and the operator wants the
same routing as the existing metadata file:

```text
cs.AI, cs.NE, quant-ph
```

## Browser Upload Steps

1. Screen: Start. Open `https://arxiv.org/submit` and sign in to the operator arXiv account.
2. Screen: New submission. Choose to start a new submission and select the compressed TeX/source upload path.
3. Screen: Upload source. Upload `{abs_bundle}`.
4. Screen: Process source. Wait for AutoTeX to process the archive. If arXiv reports a fatal TeX error, stop and fix the local source before submitting.
5. Screen: Preview. Open the generated PDF preview and compare it with `docs/arxiv-paper/main.pdf`.
6. Screen: Classification. Set the primary category to `{metadata["primary_category"]}`.
7. Screen: Metadata. Paste the title, author, abstract, comments, and license exactly from the pre-filled metadata above.
8. Screen: License. Choose Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
9. Screen: Final review. Confirm figures, references, title, abstract, author, category, and license render correctly.
10. Screen: Submit. Submit the paper and record the returned arXiv identifier in `results/experiment_1390_arxiv_submission_sword_api.json`.
"""


def write_manual_checklist(
    *,
    root: Path,
    bundle_path: Path,
    metadata: dict[str, Any],
    checklist_path: Path,
) -> str:
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.write_text(
        _manual_checklist_text(root=root, bundle_path=bundle_path, metadata=metadata),
        encoding="utf-8",
    )
    return _relative_path(checklist_path, root)


def _blocked_artifact(
    blocker: str,
    *,
    root: Path,
    bundle_path: Path,
    bundle_size_bytes: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = _base_artifact("blocked")
    artifact.update(
        {
            "bundle_path": _relative_path(bundle_path, root),
            "bundle_size_bytes": bundle_size_bytes,
            "submission_attempted": False,
            "submission_method": "none",
            "submission_result": "not_attempted_bundle_missing_or_empty",
            "manual_checklist_generated": False,
            "manual_checklist_path": None,
            "remaining_blocker": blocker,
            "honest_verdict": "blocked_bundle_missing_or_empty",
        }
    )
    if extra:
        artifact.update(extra)
    return artifact


def run(
    *,
    project_root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    checklist_path: Path | str = DEFAULT_CHECKLIST_PATH,
    paper_path: Path | str = DEFAULT_PAPER_PATH,
    ops_server_path: Path | str = DEFAULT_OPS_SERVER_PATH,
    environ: Mapping[str, str] | None = None,
    http_post: HttpPost = default_http_post,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute Exp 1390 and write the final deliverable artifact."""

    root = Path(project_root)
    output = Path(out_path)
    bundle = Path(bundle_path)
    checklist = Path(checklist_path)
    paper = Path(paper_path)
    ops_server = Path(ops_server_path)
    if not bundle.is_absolute():
        bundle = root / bundle
    if not checklist.is_absolute():
        checklist = root / checklist
    if not paper.is_absolute():
        paper = root / paper
    if not ops_server.is_absolute():
        ops_server = root / ops_server
    if not output.is_absolute():
        output = root / output

    write_in_progress_artifact(output)

    bundle_size = bundle.stat().st_size if bundle.exists() else 0
    if bundle_size <= 0:
        return _write_json(
            output,
            _blocked_artifact(
                "results/arxiv_bundle_v11.tar.gz missing or empty",
                root=root,
                bundle_path=bundle,
                bundle_size_bytes=bundle_size,
            ),
        )

    metadata = load_submission_metadata(paper)
    credentials = discover_credentials(environ=environ, ops_server_path=ops_server)
    common = {
        "bundle_path": _relative_path(bundle, root),
        "bundle_size_bytes": bundle_size,
        "metadata": metadata,
        "ops_server_path": _relative_path(ops_server, root),
        "ops_server_exists": credentials["ops_server_exists"],
        "credential_source": credentials["source"],
    }

    if credentials["available"]:
        submission = attempt_sword_submission(
            bundle_path=bundle,
            metadata=metadata,
            credentials=credentials,
            http_post=http_post,
            timeout=timeout,
        )
        artifact = _base_artifact("complete")
        artifact.update(
            {
                **common,
                **submission,
                "manual_checklist_generated": False,
                "manual_checklist_path": None,
                "honest_verdict": "arxiv_submitted"
                if submission["arxiv_id_if_submitted"]
                else "sword_api_attempt_complete_no_arxiv_id_confirmed",
            }
        )
        return _write_json(output, artifact)

    checklist_rel = write_manual_checklist(
        root=root,
        bundle_path=bundle,
        metadata=metadata,
        checklist_path=checklist,
    )
    artifact = _base_artifact("complete")
    artifact.update(
        {
            **common,
            "submission_attempted": False,
            "submission_method": "manual_checklist_no_credentials",
            "arxiv_id_if_submitted": None,
            "submission_result": "manual_checklist_generated",
            "manual_checklist_generated": True,
            "manual_checklist_path": checklist_rel,
            "honest_verdict": (
                "credentials_missing_manual_submission_checklist_generated_ready_bundle_verified"
            ),
        }
    )
    return _write_json(output, artifact)


def main() -> int:
    artifact = run()
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "submission_attempted": artifact["submission_attempted"],
                "submission_result": artifact["submission_result"],
                "arxiv_id_if_submitted": artifact["arxiv_id_if_submitted"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
