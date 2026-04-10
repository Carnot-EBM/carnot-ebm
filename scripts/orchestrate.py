#!/usr/bin/env python3
"""BMAD Sprint Orchestration Script for Carnot EBM Framework.

**What this does (in plain English):**
    This script automates the BMAD multi-agent development workflow. It reads
    a backlog of stories from an epic file, then for each story it:
    1. Builds a "sprint contract" (a YAML file listing what must be done)
    2. Launches a Generator agent (CLI subprocess) to implement the story
    3. Launches an Evaluator agent (CLI subprocess) to independently verify
    4. If the evaluator says "pass", the story is done and traceability is updated
    5. If the evaluator says "fail", the generator retries with the critique
    6. After all retries exhaust, the story is escalated to ops/known-issues.md

**Why this exists:**
    The orchestrator itself is NOT an LLM. It is a deterministic Python script
    that reads files, writes YAML, and invokes a configured agent CLI as
    subprocesses.
    Each agent invocation gets a fresh context window (no shared memory between
    generator and evaluator). This ensures the evaluator is truly independent --
    it never sees the generator's reasoning, only the artifacts it produced.

**How it differs from python/carnot/autoresearch/orchestrator.py:**
    That orchestrator runs the EBM hypothesis testing loop (propose -> sandbox ->
    evaluate -> log). THIS script orchestrates BMAD *software development* sprints
    where provider-backed agents play generator and evaluator roles on
    story-level work items.

**Usage:**
    python3 scripts/orchestrate.py --epic epics/epic-1.md
    python3 scripts/orchestrate.py --epic epics/epic-1.md --story UI-001
    python3 scripts/orchestrate.py --epic epics/epic-1.md --dry-run
    python3 scripts/orchestrate.py --epic epics/epic-1.md --max-retries 5

**Crash recovery:**
    State is persisted to .harness/state.yaml after every story transition.
    If the script is interrupted, re-running with the same --epic flag will
    resume from the last incomplete story.
"""

import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# PyYAML is required. We import it here and give a clear error if missing.
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install it with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging setup -- all orchestrator actions go to stdout with timestamps.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("orchestrate")


# ---------------------------------------------------------------------------
# Constants -- paths relative to the project root.
# ---------------------------------------------------------------------------
# The project root is determined by walking up from this script's location
# until we find a directory containing .harness/config.yaml.
def _find_project_root() -> Path:
    """Walk up from this script's directory to find the project root.

    The project root is identified by the presence of .harness/config.yaml.
    This lets the script work regardless of where it is invoked from.
    """
    candidate = Path(__file__).resolve().parent.parent
    # Check up to 5 levels (should be enough for any reasonable layout)
    for _ in range(5):
        if (candidate / ".harness" / "config.yaml").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: assume the script is in <root>/scripts/
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()
SUPPORTED_AGENT_TYPES = ("claude", "gemini", "codex", "opencode")
AGENT_BIN_ENV_VARS = {
    "claude": "CLAUDE_BIN",
    "gemini": "GEMINI_BIN",
    "codex": "CODEX_BIN",
    "opencode": "OPENCODE_BIN",
}
AGENT_DISPLAY_NAMES = {
    "claude": "Claude Code",
    "gemini": "Gemini CLI",
    "codex": "Codex CLI",
    "opencode": "OpenCode CLI",
}
AGENT_INSTALL_HINTS = {
    "claude": "Install it with: npm install -g @anthropic-ai/claude-code",
    "gemini": "Install or expose the Gemini CLI on PATH.",
    "codex": "Install or expose the Codex CLI on PATH.",
    "opencode": "Install or expose the OpenCode CLI on PATH.",
}
DEFAULT_PROJECT_INSTRUCTION_FILES = {
    "claude": "CLAUDE.md",
    "gemini": "GEMINI.md",
    "codex": "AGENTS.md",
    "opencode": "OPENCODE.md",
}
INLINE_PROJECT_INSTRUCTIONS = {"opencode"}


# ---------------------------------------------------------------------------
# Data structures -- plain dicts backed by YAML, no dataclasses needed.
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    """Load .harness/config.yaml and return the parsed dict.

    This file defines agent models, budgets, prompt paths, evaluation criteria,
    and directory locations. The orchestrator reads it once at startup and
    passes relevant sections to each phase.
    """
    config_path = PROJECT_ROOT / ".harness" / "config.yaml"
    if not config_path.exists():
        log.error("Config not found at %s", config_path)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_state() -> dict[str, Any]:
    """Load persisted orchestrator state from .harness/state.yaml.

    If no state file exists (first run), return an empty initial state.
    The state tracks which stories have been completed, which are in progress,
    and how many retries have been attempted on the current story.

    State schema:
        epic_file: str           -- path to the epic being executed
        run_id: str              -- UUID for this orchestration run
        started_at: str          -- ISO timestamp of run start
        stories_completed: list  -- slugs of stories that passed evaluation
        stories_failed: list     -- slugs that exhausted retries
        current_story: str|null  -- slug of story currently being worked on
        current_retry: int       -- retry count for current story (0-based)
        cost_total_usd: float    -- cumulative estimated cost
        history: list            -- log of all agent invocations
    """
    state_path = PROJECT_ROOT / ".harness" / "state.yaml"
    if state_path.exists():
        with open(state_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_state(state: dict[str, Any]) -> None:
    """Persist orchestrator state to .harness/state.yaml.

    Called after every story transition (start, pass, fail, retry) so that
    the orchestrator can resume after a crash. The YAML file is human-readable
    and can be manually edited if needed.
    """
    state_path = PROJECT_ROOT / ".harness" / "state.yaml"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)
    log.info("State saved to %s", state_path)


# ---------------------------------------------------------------------------
# Epic and story parsing -- extract story slugs from an epic markdown file.
# ---------------------------------------------------------------------------

def parse_epic(epic_path: Path) -> dict[str, Any]:
    """Parse an epic markdown file and extract story references.

    Epic files are markdown with a list of stories, typically formatted as:
        - [x] Story title (completed)
        - [ ] Story title (pending)
        - Story slug: UI-001

    We look for story slugs (e.g., UI-001, CORE-003) in the text. We also
    check epics/stories/ for matching .md files to get full story details.

    Returns a dict with:
        title: str           -- the epic title (first H1 heading)
        stories: list[dict]  -- list of story dicts with slug, title, status, path
    """
    if not epic_path.exists():
        log.error("Epic file not found: %s", epic_path)
        sys.exit(1)

    text = epic_path.read_text()
    lines = text.strip().split("\n")

    # Extract title from the first H1 heading
    title = "Unknown Epic"
    for line in lines:
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break

    # Find story slugs -- patterns like UI-001, CORE-003, TRAIN-012
    # These appear in the epic as checklist items or inline references.
    slug_pattern = re.compile(r"\b([A-Z]+-\d{3})\b")
    found_slugs: list[str] = []
    for line in lines:
        matches = slug_pattern.findall(line)
        for slug in matches:
            if slug not in found_slugs:
                found_slugs.append(slug)

    # For each slug, check if a story file exists in epics/stories/
    stories_dir = PROJECT_ROOT / "epics" / "stories"
    stories: list[dict[str, Any]] = []
    for slug in found_slugs:
        story_path = stories_dir / f"{slug}.md"
        story_info: dict[str, Any] = {
            "slug": slug,
            "title": slug,
            "status": "unknown",
            "path": str(story_path) if story_path.exists() else None,
        }
        # If the story file exists, parse it for title and status
        if story_path.exists():
            story_text = story_path.read_text()
            for sline in story_text.split("\n"):
                if sline.startswith("# "):
                    story_info["title"] = sline.lstrip("# ").strip()
                if sline.lower().startswith("**status:**"):
                    status_match = re.search(r"\*\*Status:\*\*\s*(.+)", sline)
                    if status_match:
                        story_info["status"] = status_match.group(1).strip()
        stories.append(story_info)

    # If no slugs were found in the epic file, try to parse checklist items
    # as ad-hoc stories (generate slugs from position).
    if not stories:
        checklist_pattern = re.compile(r"^-\s*\[(.)\]\s*(.+)$")
        for i, line in enumerate(lines):
            m = checklist_pattern.match(line.strip())
            if m:
                done = m.group(1).lower() == "x"
                story_title = m.group(2).strip()
                slug = f"ADHOC-{i:03d}"
                stories.append({
                    "slug": slug,
                    "title": story_title,
                    "status": "Completed" if done else "Pending",
                    "path": None,
                })

    return {"title": title, "stories": stories, "epic_path": str(epic_path)}


def load_story_content(story: dict[str, Any]) -> str:
    """Load the full text content of a story file.

    If the story has a file path (from epics/stories/), read it. Otherwise,
    return a minimal placeholder with just the slug and title.
    """
    if story.get("path") and Path(story["path"]).exists():
        return Path(story["path"]).read_text()
    return f"# {story['slug']} - {story['title']}\n\nNo story file found.\n"


# ---------------------------------------------------------------------------
# Spec resolution -- find the relevant spec for a story.
# ---------------------------------------------------------------------------

def find_spec_for_story(story_content: str) -> Optional[str]:
    """Attempt to find the relevant openspec capability spec for a story.

    Stories typically reference a spec via a path like:
        openspec/capabilities/core-ebm/spec.md

    Or via a REQ-* identifier like REQ-CORE-001 which maps to a capability
    directory name. We search the story text for these patterns.

    Returns the spec file content if found, or None.
    """
    # Look for explicit spec path references
    spec_path_pattern = re.compile(
        r"openspec/capabilities/([a-z0-9-]+)/spec\.md"
    )
    match = spec_path_pattern.search(story_content)
    if match:
        cap_name = match.group(1)
        spec_path = (
            PROJECT_ROOT / "openspec" / "capabilities" / cap_name / "spec.md"
        )
        if spec_path.exists():
            return spec_path.read_text()

    # Look for REQ-* identifiers and try to guess capability directory
    req_pattern = re.compile(r"REQ-([A-Z]+)")
    req_match = req_pattern.search(story_content)
    if req_match:
        # REQ-DOCUI -> documentation-ui, REQ-CORE -> core-ebm, etc.
        # This is a heuristic; we scan all capability dirs for matching REQs.
        caps_dir = PROJECT_ROOT / "openspec" / "capabilities"
        if caps_dir.exists():
            for cap_dir in caps_dir.iterdir():
                if cap_dir.is_dir():
                    spec_file = cap_dir / "spec.md"
                    if spec_file.exists():
                        spec_text = spec_file.read_text()
                        if req_match.group(0) in spec_text:
                            return spec_text

    return None


# ---------------------------------------------------------------------------
# Contract generation -- build a sprint contract YAML from story + spec.
# ---------------------------------------------------------------------------

def build_contract(
    story: dict[str, Any],
    story_content: str,
    spec_content: Optional[str],
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Build a sprint contract for the generator agent.

    The contract is a structured YAML document that tells the generator exactly
    what must be implemented and how it will be evaluated. It includes:
    - The story details (slug, title, acceptance criteria)
    - The spec requirements (REQ-* and SCENARIO-*) if available
    - The evaluation criteria and weights from config
    - Build/test commands to verify the work
    - The expected handoff artifact path

    The contract is saved to .harness/contracts/<slug>-<run_id>.yaml.
    """
    # Extract acceptance criteria from story content.
    # These are typically lines starting with "- [x]" or "- [ ]" under an
    # "Acceptance Criteria" or "Stories" heading.
    acceptance_criteria: list[str] = []
    in_criteria_section = False
    for line in story_content.split("\n"):
        lower = line.strip().lower()
        if "acceptance criteria" in lower or "stories" in lower:
            in_criteria_section = True
            continue
        if in_criteria_section:
            if line.strip().startswith("- "):
                # Strip checkbox markers
                criterion = re.sub(r"^\s*-\s*\[.\]\s*", "", line.strip())
                criterion = criterion.lstrip("- ").strip()
                if criterion:
                    acceptance_criteria.append(criterion)
            elif line.startswith("#"):
                # Next heading ends the section
                in_criteria_section = False

    # Extract REQ-* and SCENARIO-* identifiers from the spec
    req_ids: list[str] = []
    scenario_ids: list[str] = []
    if spec_content:
        req_ids = list(set(re.findall(r"REQ-[A-Z0-9-]+", spec_content)))
        scenario_ids = list(
            set(re.findall(r"SCENARIO-[A-Z0-9-]+", spec_content))
        )

    contract: dict[str, Any] = {
        "contract_id": f"{story['slug']}-{run_id[:8]}",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "story_slug": story["slug"],
        "story_title": story["title"],
        "acceptance_criteria": acceptance_criteria,
        "spec_requirements": req_ids,
        "spec_scenarios": scenario_ids,
        "evaluation_criteria": config.get("evaluation", {}).get("criteria", {}),
        "coverage_targets": config.get("coverage", {}),
        "build_commands": {
            "rust_build": "cargo build --workspace --exclude carnot-python",
            "rust_test": "cargo test --workspace --exclude carnot-python",
            "python_test": (
                "pytest tests/python --cov=python/carnot "
                "--cov-report=term-missing --cov-fail-under=100"
            ),
            "lint_rust": (
                "cargo clippy --workspace --exclude carnot-python -- -D warnings"
            ),
            "lint_python": "ruff check python/ tests/ && mypy python/carnot",
        },
        "handoff_path": (
            f".harness/handoffs/generator-{story['slug']}-{run_id[:8]}.yaml"
        ),
    }

    # Save the contract to disk
    contract_path = (
        PROJECT_ROOT
        / ".harness"
        / "contracts"
        / f"{story['slug']}-{run_id[:8]}.yaml"
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    with open(contract_path, "w") as f:
        yaml.dump(contract, f, default_flow_style=False, sort_keys=False)
    log.info("Contract written to %s", contract_path)

    return contract


# ---------------------------------------------------------------------------
# Agent invocation -- run the configured agent CLI as a subprocess.
# ---------------------------------------------------------------------------


def resolve_agent_type(config: dict[str, Any]) -> str:
    """Resolve the active agent provider from env or harness config."""
    harness_config = config.get("harness", {})
    raw_agent_type = (
        os.environ.get("HARNESS_AGENT_TYPE")
        or os.environ.get("AGENT_TYPE")
        or harness_config.get("agent_type")
        or "claude"
    )
    agent_type = str(raw_agent_type).lower()
    if agent_type not in SUPPORTED_AGENT_TYPES:
        supported = ", ".join(SUPPORTED_AGENT_TYPES)
        log.error(
            "Unsupported harness agent type '%s'. Supported values: %s",
            agent_type,
            supported,
        )
        sys.exit(1)
    return agent_type


def resolve_agent_bin(agent_type: str, config: dict[str, Any]) -> str:
    """Resolve the CLI executable for the selected agent provider."""
    harness_config = config.get("harness", {})
    agent_bins = harness_config.get("agent_bins", {})
    env_var = AGENT_BIN_ENV_VARS[agent_type]
    return os.environ.get(env_var, agent_bins.get(agent_type, agent_type))


def resolve_agent_model(
    agent_type: str, agent_config: dict[str, Any]
) -> Optional[str]:
    """Resolve the model configured for this role and provider."""
    provider_models = agent_config.get("models", {})
    return provider_models.get(agent_type) or agent_config.get("model")


def resolve_project_instruction_file(
    agent_type: str, config: dict[str, Any]
) -> Path:
    """Resolve the provider-specific project instruction file path."""
    harness_config = config.get("harness", {})
    configured_files = harness_config.get("project_instruction_files", {})
    instruction_files = dict(DEFAULT_PROJECT_INSTRUCTION_FILES)
    instruction_files.update(configured_files)
    return PROJECT_ROOT / instruction_files[agent_type]


def resolve_role_prompt_file(
    role: str,
    agent_config: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Resolve the prompt file for a harness role."""
    prompt_dir = Path(
        config.get("directories", {}).get("prompts", ".harness/prompts")
    )
    agent_prompt_ref = Path(agent_config.get("prompt", f"prompts/{role}.md"))
    candidates = [
        PROJECT_ROOT / ".harness" / agent_prompt_ref,
        PROJECT_ROOT / prompt_dir / agent_prompt_ref.name,
        PROJECT_ROOT / prompt_dir / f"{role}.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_optional_text(path: Path) -> Optional[str]:
    """Read a UTF-8 text file if it exists, otherwise return None."""
    if not path.exists():
        return None
    return path.read_text()


def compose_prompt(*sections: tuple[str, Optional[str]]) -> str:
    """Render named markdown sections into a single prompt payload."""
    rendered: list[str] = []
    for title, body in sections:
        if body:
            rendered.append(f"# {title}\n\n{body.strip()}")
    return "\n\n".join(rendered).strip()


def build_agent_invocation(
    role: str,
    prompt_text: str,
    agent_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the command line and prompt payload for the selected provider."""
    agent_type = resolve_agent_type(config)
    agent_bin = resolve_agent_bin(agent_type, config)
    agent_display = AGENT_DISPLAY_NAMES[agent_type]
    model = resolve_agent_model(agent_type, agent_config)
    max_turns = agent_config.get("max_turns")
    reasoning_effort = agent_config.get("reasoning_effort")

    role_prompt_file = resolve_role_prompt_file(role, agent_config, config)
    role_prompt_text = _read_optional_text(role_prompt_file)
    project_instruction_file = resolve_project_instruction_file(
        agent_type, config
    )
    project_instruction_text = None
    if agent_type in INLINE_PROJECT_INSTRUCTIONS:
        project_instruction_text = _read_optional_text(project_instruction_file)

    stdin_text: Optional[str] = prompt_text
    inline_prompt = prompt_text

    if agent_type != "claude":
        inline_prompt = compose_prompt(
            ("Project Workflow", project_instruction_text),
            ("Harness Role Instructions", role_prompt_text),
            ("Assignment", prompt_text),
        )

    if agent_type == "claude":
        cmd = [agent_bin, "-p", "--dangerously-skip-permissions"]
        if role_prompt_text:
            cmd.extend(["--append-system-prompt", role_prompt_text])
        if model:
            cmd.extend(["--model", model])
        if reasoning_effort:
            cmd.extend(["--effort", reasoning_effort])
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])
    elif agent_type == "gemini":
        cmd = [
            agent_bin,
            "--prompt",
            "Follow the instructions from stdin exactly.",
            "--yolo",
        ]
        if model:
            cmd.extend(["--model", model])
        stdin_text = inline_prompt
    elif agent_type == "codex":
        cmd = [
            agent_bin,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--color",
            "never",
            "--cd",
            str(PROJECT_ROOT),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append("-")
        stdin_text = inline_prompt
    else:
        cmd = [
            agent_bin,
            "run",
            "--dangerously-skip-permissions",
            "--dir",
            str(PROJECT_ROOT),
        ]
        if model:
            cmd.extend(["--model", model])
        if reasoning_effort:
            cmd.extend(["--variant", reasoning_effort])
        cmd.append(inline_prompt)
        stdin_text = None

    return {
        "agent_type": agent_type,
        "agent_bin": agent_bin,
        "agent_display": agent_display,
        "cmd": cmd,
        "stdin_text": stdin_text,
        "model": model,
        "max_turns": max_turns,
        "role_prompt_file": role_prompt_file,
        "role_prompt_text": role_prompt_text,
        "project_instruction_file": project_instruction_file,
        "project_instruction_text": project_instruction_text,
    }


def invoke_agent(
    role: str,
    prompt_text: str,
    agent_config: dict[str, Any],
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Invoke a configured agent CLI subprocess for a harness role."""
    invocation = build_agent_invocation(role, prompt_text, agent_config, config)
    cmd = invocation["cmd"]
    stdin_text = invocation["stdin_text"]
    model = invocation["model"]
    max_turns = invocation["max_turns"]
    agent_display = invocation["agent_display"]
    agent_type = invocation["agent_type"]

    if invocation["role_prompt_text"]:
        log.info(
            "Using %s role prompt from %s (%d chars)",
            role,
            invocation["role_prompt_file"],
            len(invocation["role_prompt_text"]),
        )
    else:
        log.warning(
            "No role prompt found at %s -- %s will run without "
            "role-specific instructions",
            invocation["role_prompt_file"],
            agent_display,
        )

    if invocation["project_instruction_text"]:
        log.info(
            "Inlining project workflow from %s (%d chars)",
            invocation["project_instruction_file"],
            len(invocation["project_instruction_text"]),
        )

    if dry_run:
        log.info(
            "[DRY RUN] Would invoke %s agent via %s with %d-char prompt",
            role,
            agent_display,
            len(prompt_text),
        )
        log.info("[DRY RUN] Command: %s", " ".join(cmd[:8]) + " ...")
        return {
            "stdout": "[DRY RUN] No output generated",
            "stderr": "",
            "returncode": 0,
            "duration_s": 0.0,
            "cost_usd": 0.0,
        }

    log.info(
        "Invoking %s agent via %s (provider=%s, model=%s, max_turns=%s)",
        role,
        agent_display,
        agent_type,
        model,
        max_turns,
    )

    timeout_hours = agent_config.get("session_limit_hours", 2)
    timeout_seconds = int(timeout_hours * 3600)

    start_time = datetime.datetime.now(datetime.timezone.utc)
    try:
        result = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        duration = (
            datetime.datetime.now(datetime.timezone.utc) - start_time
        ).total_seconds()
        log.error("%s agent timed out after %.0f seconds", role, duration)
        return {
            "stdout": "",
            "stderr": f"TIMEOUT after {duration:.0f}s",
            "returncode": -1,
            "duration_s": duration,
            "cost_usd": 0.0,
        }
    except FileNotFoundError:
        log.error(
            "%s not found. %s",
            invocation["agent_bin"],
            AGENT_INSTALL_HINTS[agent_type],
        )
        return {
            "stdout": "",
            "stderr": f"{agent_display} CLI not found",
            "returncode": -1,
            "duration_s": 0.0,
            "cost_usd": 0.0,
        }

    duration = (
        datetime.datetime.now(datetime.timezone.utc) - start_time
    ).total_seconds()
    cost_usd = _parse_cost_from_output(result.stdout + result.stderr)

    log.info(
        "%s agent finished in %.1fs (exit=%d, cost=$%.2f)",
        role,
        duration,
        result.returncode,
        cost_usd,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "duration_s": duration,
        "cost_usd": cost_usd,
    }


def _parse_cost_from_output(text: str) -> float:
    """Try to extract cost information from agent CLI output."""
    # Pattern: "Total cost: $1.23" or "Cost: $0.45"
    cost_pattern = re.compile(r"[Cc]ost[:\s]+\$([0-9]+\.?[0-9]*)")
    match = cost_pattern.search(text)
    if match:
        return float(match.group(1))

    # Pattern: token counts that we could estimate cost from
    # (not implemented yet -- would need model-specific pricing)
    return 0.0


# ---------------------------------------------------------------------------
# Handoff artifact handling -- read/write structured YAML handoffs.
# ---------------------------------------------------------------------------

def read_handoff(handoff_path: str) -> Optional[dict[str, Any]]:
    """Read a handoff artifact produced by the generator agent.

    The generator writes a YAML file at the path specified in the contract.
    This file contains a structured summary of what was done: files changed,
    tests added, specs updated, and any notes for the evaluator.

    If the file does not exist (generator crashed or forgot to write it),
    we return None and the evaluator will be told no handoff was produced.
    """
    full_path = PROJECT_ROOT / handoff_path
    if not full_path.exists():
        log.warning("No handoff artifact found at %s", full_path)
        return None
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def read_evaluation(evaluation_path: str) -> Optional[dict[str, Any]]:
    """Read an evaluation report produced by the evaluator agent.

    The evaluator writes a YAML file with its verdict (pass/fail/partial),
    per-criterion scores, and narrative critique. The orchestrator uses
    the verdict to decide whether to proceed or retry.

    Expected schema:
        verdict: "pass" | "fail" | "partial"
        scores: dict[str, float]   -- per-criterion scores (0.0 to 1.0)
        critique: str              -- free-text explanation
        blocking_issues: list[str] -- specific issues that must be fixed
    """
    full_path = PROJECT_ROOT / evaluation_path
    if not full_path.exists():
        log.warning("No evaluation report found at %s", full_path)
        return None
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Prompt construction -- build the full prompt for each agent role.
# ---------------------------------------------------------------------------

def build_generator_prompt(
    story: dict[str, Any],
    story_content: str,
    spec_content: Optional[str],
    contract: dict[str, Any],
    prior_critique: Optional[str] = None,
) -> str:
    """Build the full prompt for the generator agent.

    The generator gets:
    1. The story content (what to build)
    2. The spec content (requirements and scenarios to satisfy)
    3. The contract (evaluation criteria, coverage targets, build commands)
    4. Prior critique from the evaluator (on retries only)

    The prompt is structured so the generator knows exactly what is expected
    and where to write its handoff artifact when done.
    """
    sections = []

    sections.append("# Sprint Assignment\n")
    sections.append(f"You are implementing story **{story['slug']}**.\n")
    sections.append(
        "Read the story, spec, and contract below carefully. Implement "
        "everything required to satisfy the acceptance criteria and pass "
        "the evaluation criteria. When done, write your handoff artifact "
        f"to `{contract['handoff_path']}`.\n"
    )

    sections.append("## Story\n")
    sections.append(story_content)

    if spec_content:
        sections.append("\n## Relevant Spec\n")
        sections.append(spec_content)

    sections.append("\n## Sprint Contract\n")
    sections.append("```yaml")
    sections.append(yaml.dump(contract, default_flow_style=False, sort_keys=False))
    sections.append("```\n")

    if prior_critique:
        sections.append("\n## Prior Evaluator Critique (MUST ADDRESS)\n")
        sections.append(
            "The previous implementation attempt was rejected by the evaluator. "
            "You MUST address every blocking issue listed below. Do not repeat "
            "the same mistakes.\n"
        )
        sections.append(prior_critique)

    sections.append("\n## Handoff Instructions\n")
    sections.append(
        "When you have finished implementation, write a YAML handoff artifact "
        f"to `{contract['handoff_path']}` with this structure:\n"
    )
    sections.append(textwrap.dedent("""\
        ```yaml
        story_slug: <slug>
        status: "completed"
        files_changed:
          - path: "path/to/file"
            action: "created" | "modified" | "deleted"
            description: "What was changed and why"
        tests_added:
          - path: "path/to/test"
            covers: ["REQ-XXX", "SCENARIO-YYY"]
        specs_updated:
          - path: "openspec/capabilities/xxx/spec.md"
            changes: "What was updated"
        notes: "Any important context for the evaluator"
        ```
    """))

    return "\n".join(sections)


def build_evaluator_prompt(
    story: dict[str, Any],
    contract: dict[str, Any],
    handoff: Optional[dict[str, Any]],
    retry_number: int,
) -> str:
    """Build the full prompt for the evaluator agent.

    The evaluator gets:
    1. The contract (what was supposed to be done)
    2. The handoff artifact (what the generator claims was done)
    3. Instructions to independently verify everything

    The evaluator NEVER sees the generator's conversation or reasoning.
    It must verify by reading code, running tests, and checking specs.
    """
    evaluation_path = (
        f".harness/evaluations/eval-{story['slug']}-"
        f"{contract['contract_id'].split('-')[-1]}-r{retry_number}.yaml"
    )

    sections = []

    sections.append("# Evaluation Assignment\n")
    sections.append(
        f"You are independently evaluating story **{story['slug']}**. "
        "You did NOT implement this -- another agent did. Your job is to "
        "verify the work with skepticism.\n"
    )

    sections.append("## Sprint Contract (what was supposed to be done)\n")
    sections.append("```yaml")
    sections.append(yaml.dump(contract, default_flow_style=False, sort_keys=False))
    sections.append("```\n")

    if handoff:
        sections.append("## Generator Handoff (what the generator claims)\n")
        sections.append("```yaml")
        sections.append(
            yaml.dump(handoff, default_flow_style=False, sort_keys=False)
        )
        sections.append("```\n")
    else:
        sections.append("## Generator Handoff\n")
        sections.append(
            "**WARNING: No handoff artifact was produced.** The generator "
            "either crashed or failed to write its handoff. Evaluate based "
            "on what you find in the codebase.\n"
        )

    sections.append("## Evaluation Instructions\n")
    sections.append(textwrap.dedent("""\
        1. **Do NOT trust the handoff blindly.** Read the actual code, tests,
           and specs yourself.
        2. **Run the build and test commands** from the contract. Record
           whether they pass or fail.
        3. **Check every acceptance criterion** in the contract. Mark each
           as met or unmet with evidence.
        4. **Check spec traceability** -- every REQ-* and SCENARIO-* in the
           contract should have a corresponding test.
        5. **Check code quality** -- run lint and type checks.
        6. **Score each evaluation criterion** from 0.0 to 1.0.
        7. **Render a verdict**: "pass", "fail", or "partial".
           - "pass" means ALL criteria are met and ALL tests pass.
           - "fail" means blocking issues exist that prevent acceptance.
           - "partial" means mostly done but minor issues remain.
    """))

    sections.append(f"## Output\n")
    sections.append(
        f"Write your evaluation report to `{evaluation_path}` "
        "with this structure:\n"
    )
    sections.append(textwrap.dedent("""\
        ```yaml
        verdict: "pass" | "fail" | "partial"
        scores:
          spec_fidelity: 0.0-1.0
          functional_completeness: 0.0-1.0
          integration_correctness: 0.0-1.0
          code_quality: 0.0-1.0
          robustness: 0.0-1.0
        weighted_score: 0.0-1.0
        test_results:
          rust_build: "pass" | "fail"
          rust_test: "pass" | "fail"
          python_test: "pass" | "fail"
          lint: "pass" | "fail"
        acceptance_criteria:
          - criterion: "description"
            met: true | false
            evidence: "what you observed"
        critique: |
          Free-text explanation of your findings.
        blocking_issues:
          - "Specific issue that must be fixed"
        ```
    """))

    return "\n".join(sections), evaluation_path


# ---------------------------------------------------------------------------
# Traceability and ops document updates.
# ---------------------------------------------------------------------------

def update_traceability(story: dict[str, Any], evaluation: dict[str, Any]) -> None:
    """Update _bmad/traceability.md after a story passes evaluation.

    This is a lightweight update -- we append a note to the traceability
    matrix indicating that the story's requirements are now implemented.
    A full reconciliation would require parsing the markdown table, which
    is fragile; instead we append a dated note at the bottom.
    """
    trace_path = PROJECT_ROOT / "_bmad" / "traceability.md"
    if not trace_path.exists():
        log.warning("traceability.md not found at %s", trace_path)
        return

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    note = (
        f"\n\n<!-- Orchestrator update {timestamp} -->\n"
        f"<!-- Story {story['slug']} passed evaluation. "
        f"Weighted score: {evaluation.get('weighted_score', 'N/A')} -->\n"
    )

    with open(trace_path, "a") as f:
        f.write(note)
    log.info("Updated traceability.md for story %s", story["slug"])


def log_to_changelog(message: str) -> None:
    """Append an entry to ops/changelog.md.

    Every orchestrator action is logged here so there is a complete audit
    trail of what happened during the sprint. Entries are timestamped and
    prefixed with [orchestrator] to distinguish them from manual entries.
    """
    changelog_path = PROJECT_ROOT / "ops" / "changelog.md"
    changelog_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    entry = f"- **{timestamp}** [orchestrator] {message}\n"

    if changelog_path.exists():
        # Append after the first heading (preserve document structure)
        content = changelog_path.read_text()
        # Find the end of the first heading block and insert after it
        lines = content.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                break
        # Skip any blank lines after the heading
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1
        lines.insert(insert_idx, entry)
        changelog_path.write_text("\n".join(lines))
    else:
        changelog_path.write_text(f"# Changelog\n\n{entry}")

    log.info("Logged to changelog: %s", message)


def escalate_to_known_issues(
    story: dict[str, Any], evaluation: Optional[dict[str, Any]]
) -> None:
    """Add a story to ops/known-issues.md when it exhausts retries.

    This is the escalation path -- when the generator cannot produce work
    that passes evaluation after max_retries attempts, we log the story
    as a known issue so a human (or a future agent session) can pick it up.
    """
    issues_path = PROJECT_ROOT / "ops" / "known-issues.md"
    issues_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    critique = "No evaluation report available"
    if evaluation:
        critique = evaluation.get("critique", "No critique provided")
        blocking = evaluation.get("blocking_issues", [])
        if blocking:
            critique += "\n  Blocking issues:\n"
            for issue in blocking:
                critique += f"    - {issue}\n"

    entry = (
        f"\n## {story['slug']} - Failed Sprint ({timestamp})\n\n"
        f"**Story:** {story['title']}\n"
        f"**Status:** Exhausted retries, needs human review\n"
        f"**Last evaluation critique:**\n{critique}\n"
    )

    if issues_path.exists():
        with open(issues_path, "a") as f:
            f.write(entry)
    else:
        with open(issues_path, "w") as f:
            f.write(f"# Known Issues\n{entry}")

    log.info("Escalated story %s to known-issues.md", story["slug"])


def update_status(
    state: dict[str, Any], epic_info: dict[str, Any]
) -> None:
    """Update ops/status.md with current orchestration progress.

    This adds a section showing which stories completed, which failed,
    and what the overall sprint status is. Per the project workflow rules,
    we NEVER remove existing content -- only add new sections.
    """
    status_path = PROJECT_ROOT / "ops" / "status.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    completed = state.get("stories_completed", [])
    failed = state.get("stories_failed", [])
    total = len(epic_info.get("stories", []))

    section = (
        f"\n\n## Orchestration Run ({timestamp})\n\n"
        f"**Epic:** {epic_info['title']}\n"
        f"**Run ID:** {state.get('run_id', 'unknown')}\n"
        f"**Stories completed:** {len(completed)}/{total}\n"
        f"**Stories failed:** {len(failed)}/{total}\n"
        f"**Total cost:** ${state.get('cost_total_usd', 0.0):.2f}\n"
    )

    if completed:
        section += f"**Completed:** {', '.join(completed)}\n"
    if failed:
        section += f"**Failed (escalated):** {', '.join(failed)}\n"

    if status_path.exists():
        with open(status_path, "a") as f:
            f.write(section)
    else:
        with open(status_path, "w") as f:
            f.write(f"# Operational Status\n{section}")

    log.info("Updated ops/status.md")


# ---------------------------------------------------------------------------
# Main orchestration loop.
# ---------------------------------------------------------------------------

def run_sprint(
    epic_path: Path,
    target_story: Optional[str],
    max_retries: int,
    dry_run: bool,
) -> None:
    """Execute the BMAD sprint orchestration loop.

    This is the main entry point. It:
    1. Loads config and state
    2. Parses the epic to get the story queue
    3. Optionally filters to a single story (--story flag)
    4. For each story, runs the generate-evaluate-retry loop
    5. Updates traceability, changelog, and status on completion

    The loop is crash-recoverable: state is saved after every transition,
    so re-running the same command will skip already-completed stories.
    """
    # -----------------------------------------------------------------------
    # INITIALIZE: Load config, parse epic, set up state
    # -----------------------------------------------------------------------
    config = load_config()
    harness_config = config.get("harness", {})
    log.info("Harness agent type: %s", resolve_agent_type(config))

    # Use config default for max_retries if not overridden on CLI
    if max_retries is None:
        max_retries = harness_config.get("max_retries_per_sprint", 3)

    # Parse the epic to get the list of stories
    epic_info = parse_epic(epic_path)
    log.info(
        "Parsed epic '%s' with %d stories",
        epic_info["title"],
        len(epic_info["stories"]),
    )

    # Load or initialize orchestrator state
    state = load_state()

    # Check if we are resuming a previous run on the same epic
    if state.get("epic_file") == str(epic_path):
        log.info(
            "Resuming previous run %s (completed: %d, failed: %d)",
            state.get("run_id", "?"),
            len(state.get("stories_completed", [])),
            len(state.get("stories_failed", [])),
        )
        run_id = state["run_id"]
    else:
        # Fresh run -- initialize state
        run_id = str(uuid.uuid4())
        state = {
            "epic_file": str(epic_path),
            "run_id": run_id,
            "started_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "stories_completed": [],
            "stories_failed": [],
            "current_story": None,
            "current_retry": 0,
            "cost_total_usd": 0.0,
            "history": [],
        }
        save_state(state)

    # Build the execution queue -- skip already-completed and failed stories
    done_slugs = set(
        state.get("stories_completed", []) + state.get("stories_failed", [])
    )
    stories_queue = [
        s
        for s in epic_info["stories"]
        if s["slug"] not in done_slugs
        and s.get("status", "").lower() != "completed"
    ]

    # If --story is specified, filter to just that one
    if target_story:
        stories_queue = [s for s in stories_queue if s["slug"] == target_story]
        if not stories_queue:
            # Maybe the story exists but was already completed or not in epic
            all_slugs = [s["slug"] for s in epic_info["stories"]]
            if target_story in done_slugs:
                log.info("Story %s was already completed/failed.", target_story)
                return
            elif target_story not in all_slugs:
                log.error(
                    "Story %s not found in epic. Available: %s",
                    target_story,
                    ", ".join(all_slugs),
                )
                sys.exit(1)

    if not stories_queue:
        log.info("No stories to process. Sprint complete.")
        update_status(state, epic_info)
        return

    log.info(
        "Execution queue: %s",
        ", ".join(s["slug"] for s in stories_queue),
    )
    log_to_changelog(
        f"Sprint started for epic '{epic_info['title']}' "
        f"(run_id={run_id[:8]}, stories={len(stories_queue)})"
    )

    # -----------------------------------------------------------------------
    # LOOP: Process each story through the generate-evaluate-retry cycle
    # -----------------------------------------------------------------------
    for story in stories_queue:
        log.info("=" * 70)
        log.info("Starting story: %s - %s", story["slug"], story["title"])
        log.info("=" * 70)

        # Update state to reflect current story
        state["current_story"] = story["slug"]
        state["current_retry"] = 0
        save_state(state)

        # Load story content and find relevant spec
        story_content = load_story_content(story)
        spec_content = find_spec_for_story(story_content)

        # Build the sprint contract
        contract = build_contract(
            story, story_content, spec_content, config, run_id
        )
        log_to_changelog(f"Contract built for {story['slug']}")

        # Retry loop -- generator gets up to max_retries attempts
        passed = False
        prior_critique: Optional[str] = None
        last_evaluation: Optional[dict[str, Any]] = None

        for retry in range(max_retries + 1):
            attempt_label = (
                f"attempt {retry + 1}/{max_retries + 1}"
                if retry > 0
                else "initial attempt"
            )
            log.info("--- %s: %s ---", story["slug"], attempt_label)

            state["current_retry"] = retry
            save_state(state)

            # Step 1: Launch the Generator agent
            generator_config = config.get("agents", {}).get("generator", {})
            generator_prompt = build_generator_prompt(
                story, story_content, spec_content, contract, prior_critique
            )
            log_to_changelog(
                f"Generator invoked for {story['slug']} ({attempt_label})"
            )

            gen_result = invoke_agent(
                "generator",
                generator_prompt,
                generator_config,
                config,
                dry_run=dry_run,
            )

            # Track cost
            state["cost_total_usd"] = (
                state.get("cost_total_usd", 0.0) + gen_result["cost_usd"]
            )
            state["history"].append({
                "role": "generator",
                "story": story["slug"],
                "retry": retry,
                "duration_s": gen_result["duration_s"],
                "cost_usd": gen_result["cost_usd"],
                "returncode": gen_result["returncode"],
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            })
            save_state(state)

            if gen_result["returncode"] != 0 and not dry_run:
                log.error(
                    "Generator failed with exit code %d",
                    gen_result["returncode"],
                )
                log.error("stderr: %s", gen_result["stderr"][:500])
                # Don't retry on generator crash -- the evaluator won't
                # have anything to evaluate. Log and continue to evaluation
                # anyway so we get a formal "fail" verdict.

            # Step 2: Read the handoff artifact
            handoff = read_handoff(contract["handoff_path"])

            # Step 3: Launch the Evaluator agent
            evaluator_config = config.get("agents", {}).get("evaluator", {})
            eval_prompt, evaluation_path = build_evaluator_prompt(
                story, contract, handoff, retry
            )
            log_to_changelog(
                f"Evaluator invoked for {story['slug']} ({attempt_label})"
            )

            eval_result = invoke_agent(
                "evaluator",
                eval_prompt,
                evaluator_config,
                config,
                dry_run=dry_run,
            )

            # Track cost
            state["cost_total_usd"] = (
                state.get("cost_total_usd", 0.0) + eval_result["cost_usd"]
            )
            state["history"].append({
                "role": "evaluator",
                "story": story["slug"],
                "retry": retry,
                "duration_s": eval_result["duration_s"],
                "cost_usd": eval_result["cost_usd"],
                "returncode": eval_result["returncode"],
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            })
            save_state(state)

            # Step 4: Read the evaluation report
            evaluation = read_evaluation(evaluation_path)
            last_evaluation = evaluation

            if dry_run:
                log.info(
                    "[DRY RUN] Skipping evaluation check -- would read %s",
                    evaluation_path,
                )
                passed = True
                break

            if evaluation is None:
                log.warning(
                    "No evaluation report produced. Treating as fail."
                )
                prior_critique = (
                    "The evaluator did not produce an evaluation report. "
                    "Ensure your handoff artifact is correctly written and "
                    "all tests actually pass."
                )
                continue

            verdict = evaluation.get("verdict", "fail").lower()
            weighted_score = evaluation.get("weighted_score", 0.0)
            log.info(
                "Evaluation verdict: %s (weighted score: %s)",
                verdict,
                weighted_score,
            )

            # Step 5: Route based on verdict
            if verdict == "pass":
                log.info("Story %s PASSED evaluation!", story["slug"])
                passed = True
                break
            elif verdict == "partial":
                # Partial is treated as a pass if weighted score >= 0.8
                # (the 80% threshold is a pragmatic choice for MVP)
                if isinstance(weighted_score, (int, float)) and weighted_score >= 0.8:
                    log.info(
                        "Story %s scored %.2f (partial but acceptable)",
                        story["slug"],
                        weighted_score,
                    )
                    passed = True
                    break
                else:
                    log.info(
                        "Story %s scored %.2f (partial, below threshold, retrying)",
                        story["slug"],
                        weighted_score if isinstance(weighted_score, (int, float)) else 0.0,
                    )

            # Verdict is "fail" or low-scoring "partial" -- prepare critique
            # for the next generator attempt.
            critique_parts = []
            if evaluation.get("critique"):
                critique_parts.append(evaluation["critique"])
            blocking = evaluation.get("blocking_issues", [])
            if blocking:
                critique_parts.append("Blocking issues:")
                for issue in blocking:
                    critique_parts.append(f"  - {issue}")
            prior_critique = "\n".join(critique_parts) if critique_parts else (
                "The evaluator rejected the implementation but did not "
                "provide specific critique."
            )
            log.info(
                "Retrying with evaluator critique (%d chars)",
                len(prior_critique),
            )

        # Post-loop: update state based on outcome
        if passed:
            state["stories_completed"].append(story["slug"])
            state["current_story"] = None
            save_state(state)

            # Update traceability and changelog
            if last_evaluation:
                update_traceability(story, last_evaluation)
            log_to_changelog(
                f"Story {story['slug']} completed and passed evaluation"
            )
        else:
            # All retries exhausted
            state["stories_failed"].append(story["slug"])
            state["current_story"] = None
            save_state(state)

            escalate_to_known_issues(story, last_evaluation)
            log_to_changelog(
                f"Story {story['slug']} FAILED after {max_retries + 1} "
                f"attempts -- escalated to known-issues.md"
            )

    # -----------------------------------------------------------------------
    # RECONCILE: Final status updates
    # -----------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Sprint orchestration complete")
    log.info(
        "Completed: %d, Failed: %d, Total cost: $%.2f",
        len(state.get("stories_completed", [])),
        len(state.get("stories_failed", [])),
        state.get("cost_total_usd", 0.0),
    )
    log.info("=" * 70)

    update_status(state, epic_info)
    log_to_changelog(
        f"Sprint complete for '{epic_info['title']}' "
        f"(completed={len(state.get('stories_completed', []))}, "
        f"failed={len(state.get('stories_failed', []))})"
    )


# ---------------------------------------------------------------------------
# CLI argument parsing.
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the sprint orchestration loop.

    Usage examples:
        # Run all stories in an epic
        python3 scripts/orchestrate.py --epic epics/epic-1.md

        # Run a single story
        python3 scripts/orchestrate.py --epic epics/epic-1.md --story UI-001

        # Preview what would execute without running agents
        python3 scripts/orchestrate.py --epic epics/epic-1.md --dry-run

        # Override max retries (default is from .harness/config.yaml)
        python3 scripts/orchestrate.py --epic epics/epic-1.md --max-retries 5
    """
    parser = argparse.ArgumentParser(
        description=(
            "BMAD Sprint Orchestrator -- runs generator and evaluator agents "
            "on stories from an epic file, with retry logic and traceability."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --epic epics/epic-1.md
              %(prog)s --epic epics/epic-1.md --story CORE-001
              %(prog)s --epic epics/epic-1.md --dry-run
              %(prog)s --epic epics/epic-1.md --max-retries 5

            The orchestrator persists state to .harness/state.yaml for crash
            recovery. Re-running the same command will resume where it left off.

            Agent selection resolves in this order:
              1. --agent-type
              2. HARNESS_AGENT_TYPE
              3. AGENT_TYPE
              4. .harness/config.yaml

            Claude receives role prompts via --append-system-prompt. Other
            providers receive the role prompt inlined into the task payload.
        """),
    )

    parser.add_argument(
        "--epic",
        required=True,
        type=Path,
        help="Path to the epic markdown file (e.g., epics/epic-1.md)",
    )
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help=(
            "Run only this story slug (e.g., UI-001). "
            "If omitted, all pending stories in the epic are executed."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=(
            "Maximum number of retries per story before escalating. "
            "Defaults to the value in .harness/config.yaml (usually 3)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what would execute without actually invoking agents. "
            "Useful for previewing the execution plan."
        ),
    )
    parser.add_argument(
        "--agent-type",
        choices=SUPPORTED_AGENT_TYPES,
        default=None,
        help=(
            "Agent provider to use for harness roles. Overrides "
            "HARNESS_AGENT_TYPE/AGENT_TYPE and the default in "
            ".harness/config.yaml."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging for more detailed output.",
    )

    args = parser.parse_args()

    if args.agent_type:
        os.environ["HARNESS_AGENT_TYPE"] = args.agent_type

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve the epic path relative to the project root if it's not absolute
    epic_path = args.epic
    if not epic_path.is_absolute():
        epic_path = PROJECT_ROOT / epic_path

    log.info("BMAD Sprint Orchestrator starting")
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Epic: %s", epic_path)
    if args.agent_type:
        log.info("Agent type override: %s", args.agent_type)
    if args.story:
        log.info("Target story: %s", args.story)
    if args.dry_run:
        log.info("DRY RUN mode -- no agents will be invoked")

    run_sprint(
        epic_path=epic_path,
        target_story=args.story,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
