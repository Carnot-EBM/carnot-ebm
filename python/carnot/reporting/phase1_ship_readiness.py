"""Build the Exp 1582 Phase 1 software ship-readiness ledger.

Spec: REQ-PUBLISH-024, SCENARIO-PUBLISH-026.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_REL_PATH = Path("results/experiment_1582_phase1_ship_readiness_ledger.json")
LEDGER_REL_PATH = Path("ops/phase1_ship_readiness.md")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "phase1_ship_readiness_ledger_ready",
    "pypi_package_ready",
    "hf_mirror_ready",
    "second_mirror_ready",
    "mcp_cli_docs_ready",
    "independent_reproducer_path_ready",
    "safe_local_smoke_ran",
    "blocking_items_count",
    "ledger_path",
    "honest_verdict",
)

REQUIRED_PER_TOKEN_EXPORTS = (
    "per-token-ebm-qwen3-06b",
    "per-token-ebm-qwen35-08b-nothink",
    "per-token-ebm-qwen35-08b-think",
    "per-token-ebm-lfm25-350m-nothink",
    "per-token-ebm-lfm25-12b-nothink",
    "per-token-ebm-bonsai-17b-nothink",
    "per-token-ebm-qwen35-2b-nothink",
    "per-token-ebm-qwen35-4b-nothink",
    "per-token-ebm-qwen35-9b-nothink",
    "per-token-ebm-qwen35-27b-nothink",
    "per-token-ebm-qwen35-35b-nothink",
    "per-token-ebm-gemma4-e2b-nothink",
    "per-token-ebm-gemma4-e2b-it-nothink",
    "per-token-ebm-gemma4-e4b-nothink",
    "per-token-ebm-gemma4-e4b-it-nothink",
    "per-token-ebm-gptoss-20b-nothink",
)

REQUIRED_SECOND_MIRROR_KEYS = (
    "vjepa_v2",
    "estimation_verifier_v1",
    "per_token_ebm_exports",
    "pypi_sdist",
)

DOCS_TO_SCAN_FOR_PUBLIC_INSTALLS = (
    "README.md",
    "docs/getting-started.md",
    "docs/usage-guide.md",
    "examples/README.md",
)


@dataclass(frozen=True)
class GateCheck:
    """One Phase 1 ship gate and the evidence behind its pass/fail result."""

    key: str
    ready: bool
    evidence: list[str]
    blockers: list[str]
    commands: list[str]


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_in_progress_artifact(out_path: Path | str = REPO_ROOT / RESULT_REL_PATH) -> dict[str, Any]:
    """REQ-PUBLISH-024: persist the started marker before inspecting release files."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "status": "in_progress",
            "phase1_ship_readiness_ledger_ready": False,
            "pypi_package_ready": False,
            "hf_mirror_ready": False,
            "second_mirror_ready": False,
            "mcp_cli_docs_ready": False,
            "independent_reproducer_path_ready": False,
            "safe_local_smoke_ran": False,
            "blocking_items_count": None,
            "ledger_path": str(LEDGER_REL_PATH),
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _flatten_dependencies(project: dict[str, Any]) -> set[str]:
    raw_deps: list[str] = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for values in optional.values():
        raw_deps.extend(values)
    names = set()
    for dep in raw_deps:
        name = re.split(r"[<>=!~;\[]", dep, maxsplit=1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _package_data_missing(root: Path, package_data: dict[str, list[str]]) -> list[str]:
    missing: list[str] = []
    for package, patterns in package_data.items():
        package_dir = root / "python" / package.replace(".", "/")
        for pattern in patterns:
            matches = list(package_dir.glob(pattern)) if "*" in pattern else [package_dir / pattern]
            if not any(path.exists() for path in matches):
                missing.append(f"{package}:{pattern}")
    return missing


def _wrong_public_install_refs(root: Path) -> list[str]:
    pattern = re.compile(r"pip install\s+carnot(?!(?:-ebm|-e\b))")
    refs: list[str] = []
    for rel_path in DOCS_TO_SCAN_FOR_PUBLIC_INSTALLS:
        text = _read_text(root / rel_path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                refs.append(f"{rel_path}:{line_no}: {line.strip()}")
    return refs


def inspect_pypi_package(root: Path) -> GateCheck:
    """Inspect local package metadata without publishing anything."""

    pyproject = _load_pyproject(root)
    project = pyproject.get("project", {})
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    package_data = setuptools.get("package-data", {})
    dependencies = _flatten_dependencies(project)
    blockers: list[str] = []
    evidence: list[str] = []

    if project.get("name") == "carnot-ebm":
        evidence.append("pyproject.toml uses PyPI distribution name carnot-ebm.")
    else:
        blockers.append("pyproject.toml must use project.name = 'carnot-ebm'.")

    if project.get("scripts", {}).get("carnot") == "carnot.cli:main":
        evidence.append("console script carnot resolves to carnot.cli:main.")
    else:
        blockers.append("project.scripts must expose carnot = 'carnot.cli:main'.")

    version_text = _read_text(root / "python/carnot/_version.py")
    if "__version__" in version_text and project.get("dynamic") == ["version"]:
        evidence.append("dynamic version resolves through python/carnot/_version.py.")
    else:
        blockers.append("dynamic version metadata must resolve to carnot._version.__version__.")

    license_text = _read_text(root / "LICENSE")
    if project.get("license", {}).get("text") == "Apache-2.0" and "Apache License" in license_text:
        evidence.append("Apache-2.0 metadata and LICENSE file are present.")
    else:
        blockers.append("Apache-2.0 project metadata and LICENSE file must both be present.")

    missing_package_data = _package_data_missing(root, package_data)
    if not missing_package_data:
        evidence.append("declared package-data globs resolve to local files.")
    else:
        blockers.append(f"declared package-data entries are missing: {missing_package_data}.")

    wrong_installs = _wrong_public_install_refs(root)
    if wrong_installs:
        blockers.append(
            "public docs still use the unavailable package name `pip install carnot`: "
            + "; ".join(wrong_installs)
        )
    else:
        evidence.append("public install docs use carnot-ebm rather than the unavailable carnot name.")

    if "huggingface-hub" in dependencies:
        evidence.append("huggingface-hub is declared for Hub model loading.")
    else:
        blockers.append(
            "HuggingFace model loading paths import huggingface_hub, but pyproject.toml does "
            "not declare huggingface-hub in dependencies or optional extras."
        )

    return GateCheck(
        key="pypi_package_ready",
        ready=not blockers,
        evidence=evidence,
        blockers=blockers,
        commands=[
            "python3 -m venv /tmp/carnot-phase1-package-smoke",
            "/tmp/carnot-phase1-package-smoke/bin/python -m pip install --upgrade pip",
            "/tmp/carnot-phase1-package-smoke/bin/python -m pip install --no-deps .",
            "/tmp/carnot-phase1-package-smoke/bin/python -c \"import carnot; print(carnot.__version__)\"",
            "/tmp/carnot-phase1-package-smoke/bin/carnot --help",
            "python -m build && twine check dist/*",
        ],
    )


def inspect_hf_mirror(root: Path) -> GateCheck:
    """Inspect local HuggingFace staging evidence and model-card references."""

    blockers: list[str] = []
    evidence: list[str] = []
    missing_export_files: list[str] = []
    for model_id in REQUIRED_PER_TOKEN_EXPORTS:
        for filename in ("README.md", "config.json", "model.safetensors", "training_metadata.json"):
            if not (root / "exports" / model_id / filename).exists():
                missing_export_files.append(f"exports/{model_id}/{filename}")

    if missing_export_files:
        blockers.append("per-token HuggingFace export staging is incomplete: " + ", ".join(missing_export_files))
    else:
        evidence.append(f"{len(REQUIRED_PER_TOKEN_EXPORTS)} per-token EBM export directories are locally staged.")

    if (root / "data/token_activations_tqa_qwen35.safetensors").exists():
        evidence.append("documented TruthfulQA activation dataset artifact is present.")
    else:
        blockers.append(
            "docs/huggingface-plan.md lists data/token_activations_tqa_qwen35.safetensors, "
            "but that dataset artifact is not present locally."
        )

    hf_plan = _read_text(root / "docs/huggingface-plan.md")
    if "Action needed" in hf_plan:
        blockers.append("docs/huggingface-plan.md still contains unresolved 'Action needed' export work.")
    else:
        evidence.append("docs/huggingface-plan.md has no unresolved 'Action needed' marker.")

    readme = _read_text(root / "README.md")
    if "Two Phase 1 research artifacts" in readme:
        blockers.append(
            "README.md still says only two Phase 1 research artifacts are published, "
            "which conflicts with later per-token/model-card references."
        )
    else:
        evidence.append("README.md no longer advertises the obsolete two-artifact HF inventory.")

    return GateCheck(
        key="hf_mirror_ready",
        ready=not blockers,
        evidence=evidence,
        blockers=blockers,
        commands=[
            "find exports -maxdepth 2 -type f | sort",
            "python - <<'PY'\nfrom pathlib import Path\nfor p in sorted(Path('exports').glob('per-token-ebm-*')):\n    print(p.name, all((p / f).exists() for f in ['README.md','config.json','model.safetensors','training_metadata.json']))\nPY",
            "huggingface-cli repo-files Carnot-EBM/<repo-id>",
        ],
    )


def inspect_second_mirror(root: Path) -> GateCheck:
    """Inspect second-channel mirror records without adding or pinning content."""

    registry = _read_json(root / "results/ipfs_mirrors.json")
    blockers: list[str] = []
    evidence: list[str] = []
    missing = [
        key
        for key in REQUIRED_SECOND_MIRROR_KEYS
        if not isinstance(registry.get(key), dict) or not registry[key].get("cid")
    ]
    if missing:
        blockers.append(
            "results/ipfs_mirrors.json lacks content-addressed CIDs for required keys: "
            + ", ".join(missing)
        )
    else:
        evidence.append("IPFS mirror registry contains CIDs for package and model artifact groups.")

    return GateCheck(
        key="second_mirror_ready",
        ready=not blockers,
        evidence=evidence,
        blockers=blockers,
        commands=[
            "ipfs add -r exports/per-token-ebm-*",
            "ipfs add dist/carnot_ebm-*.tar.gz",
            "python -m json.tool results/ipfs_mirrors.json",
        ],
    )


def inspect_mcp_cli_docs(root: Path) -> GateCheck:
    """Inspect CLI and MCP quick-start documentation for external integrators."""

    blockers: list[str] = []
    evidence: list[str] = []
    readme = _read_text(root / "README.md")
    usage = _read_text(root / "docs/usage-guide.md")
    mcp_example = _read_text(root / ".mcp.json.example")

    if "carnot verify " in usage and "carnot verify-code " in usage:
        evidence.append("docs/usage-guide.md includes CLI verify and verify-code examples.")
    else:
        blockers.append("docs/usage-guide.md must show both `carnot verify` and `carnot verify-code`.")

    if "python -m carnot.mcp" in usage or '"-m", "carnot.mcp"' in mcp_example:
        evidence.append("MCP docs include the packaged python -m carnot.mcp entry point.")
    else:
        blockers.append(
            ".mcp.json.example/docs still point at tools/verify-mcp/server.py instead of "
            "the packaged `python -m carnot.mcp` entry point."
        )

    if "exposes **7** tools" in readme:
        blockers.append("README.md reports 7 MCP tools, but python/carnot/mcp/server.py exposes 9.")
    else:
        evidence.append("README.md does not contain the stale 7-tool MCP count.")

    if (root / "docs/integrator-guide.md").exists():
        evidence.append("docs/integrator-guide.md exists for one-page external onboarding.")
    else:
        blockers.append("docs/integrator-guide.md is missing.")

    return GateCheck(
        key="mcp_cli_docs_ready",
        ready=not blockers,
        evidence=evidence,
        blockers=blockers,
        commands=[
            "carnot verify examples/math_funcs.py --func gcd --test '(12,8):4' --test '(7,13):1'",
            "carnot verify-code examples/math_funcs.py --func gcd --pbt",
            "python -m carnot.mcp",
        ],
    )


def inspect_independent_reproducer(safe_local_smoke_ran: bool) -> GateCheck:
    """Record whether a non-publishing reproducer path is ready to hand off."""

    blockers: list[str] = []
    evidence: list[str] = [
        "fresh-venv and CI reproducer commands are recorded in ops/phase1_ship_readiness.md."
    ]
    if safe_local_smoke_ran:
        evidence.append("safe local smoke command was run without publishing or credentials.")
    else:
        blockers.append("safe local fresh-venv package smoke has not been run in this ledger task.")

    return GateCheck(
        key="independent_reproducer_path_ready",
        ready=not blockers,
        evidence=evidence,
        blockers=blockers,
        commands=[
            "python3 -m venv /tmp/carnot-phase1-repro",
            "/tmp/carnot-phase1-repro/bin/python -m pip install --no-deps .",
            "/tmp/carnot-phase1-repro/bin/python -c \"import carnot; print(carnot.__version__)\"",
            "/tmp/carnot-phase1-repro/bin/carnot --help",
            "gh workflow run phase1-reproducer.yml",
        ],
    )


def _collect_blocking_items(checks: list[GateCheck]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for check in checks:
        for blocker in check.blockers:
            items.append({"gate": check.key, "details": blocker, "commands": check.commands})
    return items


def build_readiness_report(root: Path | str = REPO_ROOT, *, safe_local_smoke_ran: bool) -> dict[str, Any]:
    """SCENARIO-PUBLISH-026: compute the terminal readiness artifact."""

    root_path = Path(root)
    checks = [
        inspect_pypi_package(root_path),
        inspect_hf_mirror(root_path),
        inspect_second_mirror(root_path),
        inspect_mcp_cli_docs(root_path),
        inspect_independent_reproducer(safe_local_smoke_ran),
    ]
    blocking_items = _collect_blocking_items(checks)
    artifact: dict[str, Any] = {
        "status": "complete",
        "phase1_ship_readiness_ledger_ready": True,
        "safe_local_smoke_ran": safe_local_smoke_ran,
        "blocking_items_count": len(blocking_items),
        "ledger_path": str(LEDGER_REL_PATH),
        "blocking_items": blocking_items,
        "gate_evidence": {check.key: check.evidence for check in checks},
    }
    for check in checks:
        artifact[check.key] = check.ready
    artifact["honest_verdict"] = (
        "phase1_software_ship_ready"
        if not blocking_items
        else f"blocked_{len(blocking_items)}_items_remaining"
    )
    return artifact


def render_markdown(report: dict[str, Any]) -> str:
    """Render the human-readable ship ledger with exact next commands."""

    gate_rows = [
        ("PyPI package", report["pypi_package_ready"]),
        ("HuggingFace mirror", report["hf_mirror_ready"]),
        ("Second mirror", report["second_mirror_ready"]),
        ("MCP/CLI docs", report["mcp_cli_docs_ready"]),
        ("Independent reproducer path", report["independent_reproducer_path_ready"]),
        ("Safe local smoke", report["safe_local_smoke_ran"]),
    ]
    lines = [
        "# Phase 1 Ship Readiness",
        "",
        "Phase 1 is treated here as a software-operational ship gate only. Paper, arXiv, "
        "GateMate, PolarFire SoC, and other hardware validation are out of scope for this ledger.",
        "",
        f"Honest verdict: `{report['honest_verdict']}`",
        f"Blocking items: `{report['blocking_items_count']}`",
        "",
        "## Checklist",
        "",
        "| Gate | Status |",
        "|------|--------|",
    ]
    for name, ready in gate_rows:
        lines.append(f"| {name} | {'PASS' if ready else 'FAIL'} |")

    lines.extend(["", "## Exact Blockers", ""])
    blocking_items = report.get("blocking_items", [])
    if blocking_items:
        for idx, item in enumerate(blocking_items, start=1):
            lines.append(f"{idx}. `{item['gate']}`: {item['details']}")
            lines.append("")
            lines.append("   Commands:")
            for command in item["commands"]:
                lines.append(f"   - `{command}`")
            lines.append("")
    else:
        lines.append("No unresolved software ship blockers were detected by the local ledger.")
        lines.append("")

    lines.extend(
        [
            "## Independent Reproducer Plan",
            "",
            "Safe local smoke, no publishing:",
            "",
            "```bash",
            "python3 -m venv /tmp/carnot-phase1-repro",
            "/tmp/carnot-phase1-repro/bin/python -m pip install --upgrade pip",
            "/tmp/carnot-phase1-repro/bin/python -m pip install --no-deps .",
            "/tmp/carnot-phase1-repro/bin/python -c \"import carnot; print(carnot.__version__)\"",
            "/tmp/carnot-phase1-repro/bin/carnot --help",
            "```",
            "",
            "Independent path before declaring Phase 1 shipped:",
            "",
            "```bash",
            "python3 -m venv /tmp/carnot-phase1-independent",
            "/tmp/carnot-phase1-independent/bin/python -m pip install carnot-ebm",
            "/tmp/carnot-phase1-independent/bin/python -c \"import carnot; print(carnot.__version__)\"",
            "/tmp/carnot-phase1-independent/bin/carnot verify examples/math_funcs.py --func gcd --test '(12,8):4' --test '(7,13):1'",
            "```",
            "",
            "CI path before declaring Phase 1 shipped:",
            "",
            "```bash",
            "python3 -m build",
            "twine check dist/*",
            "python3 -m venv /tmp/carnot-wheel-smoke",
            "/tmp/carnot-wheel-smoke/bin/python -m pip install dist/*.whl",
            "/tmp/carnot-wheel-smoke/bin/carnot --help",
            "```",
            "",
            "## What Remains Before Phase 1 Ship",
            "",
        ]
    )
    if blocking_items:
        lines.append("Resolve every blocker above, then collect one independent reproducer log.")
    else:
        lines.append("Collect and archive one independent reproducer log before public announcement.")
    lines.append("")
    return "\n".join(lines)


def run(root: Path | str = REPO_ROOT, *, safe_local_smoke_ran: bool = False) -> dict[str, Any]:
    """Write the in-progress marker, markdown ledger, and terminal JSON artifact."""

    root_path = Path(root)
    result_path = root_path / RESULT_REL_PATH
    ledger_path = root_path / LEDGER_REL_PATH
    write_in_progress_artifact(result_path)
    artifact = build_readiness_report(root_path, safe_local_smoke_ran=safe_local_smoke_ran)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(render_markdown(artifact), encoding="utf-8")
    return _write_json(result_path, artifact)


def main(argv: list[str] | None = None) -> int:
    """Non-publishing CLI for regenerating the Phase 1 readiness ledger."""

    parser = argparse.ArgumentParser(description="Build Phase 1 ship readiness ledger.")
    parser.add_argument("root", nargs="?", default=str(REPO_ROOT), help="Repository root to inspect.")
    parser.add_argument(
        "--safe-local-smoke-ran",
        action="store_true",
        help="Record that the safe local no-deps venv smoke was run.",
    )
    args = parser.parse_args(argv)
    run(Path(args.root), safe_local_smoke_ran=args.safe_local_smoke_ran)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the console entry point.
    raise SystemExit(main())
