"""SCENARIO-HARNESS-5935-RETIRED-WRITER-SUBSTRATE-DOES-NOT-RETURN (REQ-HARNESS-5935).

The exp3946 incident class: the ARTIFACT was corrected but the WRITER was not, so every
re-run recreated the illegal inference_substrate. Fixed 2026-07-27 for exp3946 and
2026-08-28 for seven sibling writers. This test walks each writer's AST for the retired
literal, so a comment that NAMES the string (every fixed writer has one) does not fire --
only a real Python string constant does, which is exactly what a reverted fix or a
copy-pasted writer would reintroduce.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The retired value the exp3946 incident class kept regenerating. See CLAUDE.md's
# Inference-Substrate Declaration Discipline for the legal set.
RETIRED_SUBSTRATE = "offline_arc_agi3_perception_planner_real_env_confirmed"


def _string_literals(path: Path) -> list[str]:
    """Every string constant in the file's AST. Comments are not in the AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_no_experiment_writer_emits_the_retired_substrate() -> None:
    # SCENARIO-HARNESS-5935-RETIRED-WRITER-SUBSTRATE-DOES-NOT-RETURN
    writers_dir = REPO / "scripts" / "experiments"
    assert writers_dir.is_dir(), "scripts/experiments/ missing -- cannot verify anything"
    offenders = []
    for path in sorted(writers_dir.glob("*.py")):
        try:
            literals = _string_literals(path)
        except SyntaxError:
            # A writer that does not parse cannot regenerate anything; not this
            # test's failure to report.
            continue
        if any(RETIRED_SUBSTRATE in lit for lit in literals):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "These writers carry the retired substrate "
        f"'{RETIRED_SUBSTRATE}' as a string constant, so a re-run would write an "
        f"artifact scripts/arc_artifact_lint.py rejects: {offenders}. Fix the writer "
        "to declare a legal substrate (see the exp3946 correction), never the baseline."
    )


def test_the_eight_fixed_writers_are_present_and_scanned() -> None:
    # SCENARIO-HARNESS-5935-RETIRED-WRITER-SUBSTRATE-DOES-NOT-RETURN: the scan above
    # is only meaningful while the incident's writers are actually in its population.
    expected = [
        "arc3_r11l_solve.py",
        "experiment_3946_r11l_first_solve.py",
        "experiment_3954_second_game_solve.py",
        "experiment_3964_r11l_incremental_l2.py",
        "experiment_3965_lp85_incremental_l2.py",
        "experiment_3966_third_game_first_solve.py",
        "experiment_3981_fourth_game_first_solve.py",
        "experiment_3993_fourth_game_verifier_pruned.py",
    ]
    writers_dir = REPO / "scripts" / "experiments"
    missing = [name for name in expected if not (writers_dir / name).is_file()]
    assert not missing, f"incident writers vanished from scripts/experiments/: {missing}"
