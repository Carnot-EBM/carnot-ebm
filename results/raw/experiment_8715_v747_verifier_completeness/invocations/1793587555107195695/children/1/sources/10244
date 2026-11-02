"""Root-clutter sweeper — keep the repo root free of agent/experiment scratch.

WHY. The conductor launches every experiment subagent with cwd=PROJECT_ROOT (see
research_conductor.py subprocess calls), so when an agent writes a one-off debug script
(probe.py, check_acc.py, test_foo.py, append_spec.py, ...) it lands in the REPO ROOT. Over
many milestones ~155 such files accumulated (archived 2026-06-09 to legacy/root-scratch-2026-06/).
The .gitignore `/*.py` guard stops them being COMMITTED, but they still accrete on disk as
untracked files. This sweeper is the disk-layer defense: it relocates untracked root scratch to a
gitignored quarantine dir and deletes regenerable build artifacts — mechanically, regardless of
which agent created them.

SAFETY (conservative by design):
  * DRY-RUN by default. Pass --apply to actually move/delete.
  * AGE GUARD: only touches files older than --min-age-min (default 120 min), so a subagent's
    in-flight scratch is never swept mid-run (same philosophy as the orphan-cleanup janitor).
  * NEVER touches TRACKED files (git ls-files), the ALLOWLIST of legit root files, dotfiles, or
    directories. Tracked non-allowlist files are only WARNED about (a human decides).
  * Untracked root *.py / scratch -> MOVED to .root-scratch-trash/<UTC-date>/ (reversible; the dir
    is gitignored). Regenerable build artifacts (main.aux/log/out, vivado*.{jou,log}, clockInfo.txt)
    -> DELETED (they are regenerated on every build).

Run on demand:  python3 scripts/root_clutter_sweep.py            # dry-run, shows what it would do
                python3 scripts/root_clutter_sweep.py --apply    # actually sweep
Wire into the 30-min janitor for full automation (see ops + the carnot-orphan-cleanup model).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRASH = REPO / ".root-scratch-trash"

# Files that LEGITIMATELY live at the repo root. Anything else at root is candidate clutter.
# (Dotfiles are always allowed; directories are never swept.)
ALLOWLIST = {
    # docs
    "README.md",
    "CLAUDE.md",
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "RELEASE_NOTES.md",
    "RELEASES.md",
    "MANIFEST.in",
    # multi-agent instruction files (each agent reads its own)
    "AGENTS.md",
    "CODEX.md",
    "GEMINI.md",
    "OPENCODE.md",
    # build / packaging / config
    "Cargo.toml",
    "Cargo.lock",
    "rustfmt.toml",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "Makefile",
    "docker-compose.yml",
    "Dockerfile.sandbox",
    # whitelisted python that a build may legitimately put at root
    "conftest.py",
    "setup.py",
    # research project files (CLAUDE.md Key Paths)
    "research-complete.yaml",
    "research-roadmap.yaml",
    # 2026-07-03: the conductor's in-progress next-milestone draft. UNTRACKED by
    # nature while the planner is mid-draft / activation is stuck-and-retrying
    # (it only becomes tracked once activation succeeds and it gets copied to
    # research-roadmap.yaml) -- which is exactly the state this sweeper's own
    # untracked-file criterion matches. Confirmed via /tmp/root-clutter-sweep.log
    # ("mv research-roadmap-next.yaml") that this sweeper silently relocated a
    # STUCK-BUT-FIXABLE roadmap draft mid-stall on at least two occasions (.475,
    # .476), each time discarding up to 2 real hours of planner compute and
    # forcing the conductor to fall back to "no research-roadmap-next.yaml --
    # launching planning agent" instead of the stall getting diagnosed and fixed.
    # This file is not scratch -- it is the SAME class of legitimate research
    # artifact as its already-allowlisted siblings above, just transiently
    # untracked. See CLAUDE.md "Pre-Staged Roadmap Convention".
    "research-roadmap-next.yaml",
    "research-program.md",
    "research-studying.md",
    "research-references.md",
    "research-hardware-wishlist.md",
}

# Regenerable build artifacts at root -> safe to DELETE (a build remakes them).
DELETE_PATTERNS = (
    "main.aux",
    "main.log",
    "main.out",
    "main.bbl",
    "main.blg",
    "main.toc",
    "main.fls",
    "main.fdb_latexmk",
    "main.synctex.gz",
    "clockInfo.txt",
)
DELETE_GLOBS = ("vivado*.jou", "vivado*.log", "webtalk*.jou", "webtalk*.log", "hs_err_pid*.log")


def _tracked_files() -> set[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except Exception:
        return set()
    # only root-level (no slash)
    return {line for line in out.splitlines() if "/" not in line}


def sweep(apply: bool, min_age_min: int) -> dict:
    tracked = _tracked_files()
    now = time.time()
    age_cut = min_age_min * 60
    moved, deleted, warned, skipped_young = [], [], [], []

    for entry in sorted(REPO.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue  # never sweep dirs or dotfiles
        name = entry.name
        if name in ALLOWLIST:
            continue
        # regenerable build artifacts -> delete
        is_delete = name in DELETE_PATTERNS or any(entry.match(g) for g in DELETE_GLOBS)
        is_scratch_py = name.endswith(".py")
        if not (is_delete or is_scratch_py):
            # some other unexpected root file: if tracked, a human should decide; if untracked,
            # quarantine it too (it is not on the allowlist and not a known artifact).
            if name in tracked:
                warned.append(name)
                continue
        if name in tracked:
            # a tracked non-allowlist root .py should not exist after the gitignore guard; warn only
            warned.append(name)
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age < age_cut:
            skipped_young.append(name)
            continue
        if is_delete:
            deleted.append(name)
            if apply:
                entry.unlink(missing_ok=True)
        else:  # untracked scratch .py (or other untracked non-artifact) -> quarantine
            moved.append(name)
            if apply:
                day = time.strftime("%Y-%m-%d", time.gmtime(now))
                dest = TRASH / day
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry), str(dest / name))

    return {
        "applied": apply,
        "min_age_min": min_age_min,
        "moved_to_trash": moved,
        "deleted_artifacts": deleted,
        "skipped_too_young": skipped_young,
        "warn_tracked_nonallowlist": warned,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep agent/experiment scratch out of the repo root.")
    ap.add_argument("--apply", action="store_true", help="actually move/delete (default: dry-run)")
    ap.add_argument(
        "--min-age-min",
        type=int,
        default=120,
        help="only sweep files older than this many minutes (protects in-flight scratch)",
    )
    args = ap.parse_args()

    res = sweep(apply=args.apply, min_age_min=args.min_age_min)
    mode = "APPLIED" if res["applied"] else "DRY-RUN (use --apply to act)"
    print(f"[root-clutter-sweep] {mode}  min_age={res['min_age_min']}min")
    print(f"  quarantine -> .root-scratch-trash/: {len(res['moved_to_trash'])} files")
    for n in res["moved_to_trash"][:40]:
        print(f"     mv  {n}")
    print(f"  delete (regenerable artifacts): {len(res['deleted_artifacts'])} files")
    for n in res["deleted_artifacts"][:40]:
        print(f"     rm  {n}")
    if res["skipped_too_young"]:
        print(
            f"  skipped (younger than {res['min_age_min']}min, possibly in-flight): "
            f"{len(res['skipped_too_young'])}"
        )
    if res["warn_tracked_nonallowlist"]:
        print(
            f"  WARN tracked non-allowlist root files (a human should relocate these): "
            f"{res['warn_tracked_nonallowlist']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
