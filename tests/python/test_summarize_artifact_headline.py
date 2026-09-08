"""The summary must say when an artifact's result is not in its headline metrics.

exp7106 measured four arms over 720 events with pre-declared intervals excluding zero,
and its headline read as two completion flags set to 1. Read through this tool, as the
Reading-Results Discipline requires, a genuine positive looked like readiness. Headline
metrics are chosen from top-level scalars, so an effect size living in rows can never
appear among them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "summarize_artifact.py"


def _module():
    spec = importlib.util.spec_from_file_location("_summ", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_summ"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_completion_flags_with_measurement_rows_are_called_out() -> None:
    """REQ-PUBLISH-3701: the exp7106 shape, every headline a flag and the result in rows."""

    mod = _module()
    d = {
        "procedural_memory_value_ready_score": 1,
        "paired_delta_rows": [{"mean_delta": 0.46}, {"mean_delta": 0.56}],
        "confidence_interval_rows": [{"lower": 0.32}, {"lower": 0.42}],
    }
    heads = {"procedural_memory_value_ready_score": 1}
    lines = mod.result_is_not_in_the_headline(d, heads)
    assert lines, "a flag-only headline over measurement rows must be called out"
    assert "COMPLETION FLAG" in lines[0]
    assert "paired_delta_rows" in lines[1]
    assert "confidence_interval_rows" in lines[1]


def test_a_real_metric_alongside_a_flag_stays_quiet() -> None:
    """REQ-PUBLISH-3701: fire only when EVERY headline is a completion flag, else it is noise."""

    mod = _module()
    d = {
        "auroc": 0.91,
        "x_ready_score": 1,
        "paired_delta_rows": [{"mean_delta": 0.1}, {"mean_delta": 0.2}],
    }
    assert mod.result_is_not_in_the_headline(d, {"auroc": 0.91, "x_ready_score": 1}) == []


def test_flags_without_measurement_rows_stay_quiet() -> None:
    """REQ-PUBLISH-3701: a readiness artifact with nothing measured has nothing to point at."""

    mod = _module()
    d = {"x_ready_score": 1, "preconditions_checked": [{"check": "a"}]}
    assert mod.result_is_not_in_the_headline(d, {"x_ready_score": 1}) == []
    assert mod.result_is_not_in_the_headline({}, {}) == []


def test_empty_row_lists_do_not_count_as_a_result() -> None:
    """REQ-PUBLISH-3701: an empty rows list is plumbing, not a measurement to point at."""

    mod = _module()
    d = {"x_complete_score": 1, "paired_delta_rows": []}
    assert mod.result_is_not_in_the_headline(d, {"x_complete_score": 1}) == []


def test_the_line_reaches_the_rendered_summary(tmp_path: Path) -> None:
    """REQ-PUBLISH-3701: the warning must survive in the RENDERED output, not just the helper."""

    import json
    import subprocess

    art = tmp_path / "experiment_9999_headline_probe.json"
    art.write_text(
        json.dumps(
            {
                "experiment": 9999,
                "honest_verdict": "complete_positive_probe",
                "duration_s": 1.0,
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "probe_value_ready_score": 1,
                "paired_delta_rows": [
                    {"mean_delta": 0.46, "wins": 44},
                    {"mean_delta": 0.56, "wins": 59},
                ],
            }
        )
    )
    out = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), str(SCRIPT), str(art)],
        capture_output=True,
        text=True,
    ).stdout
    assert "COMPLETION FLAG" in out, out
    assert "paired_delta_rows" in out, out


def test_req_publish_3701_rows_are_found_by_SHAPE_not_by_name() -> None:
    """REQ-PUBLISH-3701: a differently-named result must still be found.

    The first version listed field names taken from one artifact and then stayed silent on
    exp7134, whose result lives in effective_sample_size_rows. A varying numeric column across
    two or more rows is what a comparison looks like in every vocabulary.
    """

    mod = _module()
    d = {
        "sampler_benchmark_complete_score": 1,
        "effective_sample_size_rows": [{"ess": 41.2}, {"ess": 88.7}],
    }
    assert mod.measurement_rows(d) == ["effective_sample_size_rows"]
    lines = mod.result_is_not_in_the_headline(d, {"sampler_benchmark_complete_score": 1})
    assert lines and "effective_sample_size_rows" in lines[1]


def test_req_publish_3701_constant_metadata_rows_are_not_a_measurement() -> None:
    """REQ-PUBLISH-3701: a column that never varies is metadata, not a comparison."""

    mod = _module()
    constant = {"x_ready_score": 1, "preconditions_checked": [{"cap_s": 300.0}, {"cap_s": 300.0}]}
    assert mod.measurement_rows(constant) == []
    single = {"x_ready_score": 1, "only_one_row": [{"ess": 41.2}]}
    assert mod.measurement_rows(single) == []


def test_req_publish_3701_a_varying_INTEGER_column_is_structure_not_measurement() -> None:
    """REQ-PUBLISH-3701: attempt numbers and indices vary too, and are not results.

    Measured on the real corpus: dropping the float requirement takes the fire rate from 114 to
    250 of 365 completion-flag-only artifacts, because varying integer columns are overwhelmingly
    indices and counts.
    """

    mod = _module()
    ints = {"x_ready_score": 1, "attempt_rows": [{"attempt": 1}, {"attempt": 2}]}
    assert mod.measurement_rows(ints) == []
    floats = {"x_ready_score": 1, "ess_rows": [{"ess": 1.0}, {"ess": 2.0}]}
    assert mod.measurement_rows(floats) == ["ess_rows"]
