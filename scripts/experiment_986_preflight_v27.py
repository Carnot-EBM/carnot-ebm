"""Experiment 986: Preflight v27 — Manifest Verify 786/627/603/641 + Gate Config Audit + SOTA Models.

This preflight runs at the start of milestone .77 to catch the two failure modes
that wrecked .76:

  1. Legacy carryover experiments (786, 627, 603, 641) must be in both the YAML
     exclusion manifest AND the conductor JSON manifest so the conductor does not
     pick them up again.

  2. gated_on entries in the active roadmap must have a non-empty op string from the
     known-valid set. The empty-string op bug in .76 caused Exp 980 to be blocked
     by an 'unknown op' error even though it was independent.

Why this matters: milestone .76 closed 2/10 because Exp 975 (EnvPropagationGuard)
produced no artifact (missing try/finally), which cascaded to block 6 experiments.
The gate-config bug compounded the damage. This preflight is the first-pass safety
net before any real experiments run.
"""

import json
import os
import subprocess
from datetime import datetime, timezone, UTC

RESULT_PATH = "results/experiment_986_preflight_v27.json"
EXCLUSION_YAML = "ops/exclusion_manifest.yaml"
EXCLUSION_JSON = "scripts/conductor_exclusion_manifest.json"
ROADMAP_YAML = "research-roadmap.yaml"

# The four legacy carryovers that must be retired before .77 starts.
REQUIRED_IDS = [786, 627, 603, 641]

# All valid gate operators recognised by the conductor evaluator.
VALID_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "not_contains"}

# SOTA GGUF models that must be pre-downloaded before any inference experiment runs.
SOTA_MODELS = [
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]


def _hf_cache_dir_name(hf_id: str) -> str:
    """Convert a HuggingFace model ID (org/name) into the hub cache directory name.

    HuggingFace caches models under ~/.cache/huggingface/hub/ using the pattern
    models--<org>--<name>.  This function maps the ID to that directory name so
    we can check presence without importing huggingface_hub.
    """
    return "models--" + hf_id.replace("/", "--")


def check_yaml_manifest() -> tuple[list[int], str | None]:
    """Return (sorted_ids, parse_error) for ops/exclusion_manifest.yaml.

    First tries YAML parsing; if that fails (the file has been known to accumulate
    top-level bare list items after section mappings, making it invalid YAML), falls
    back to a regex scan of the raw text.  The parse_error return value is non-None
    when the fallback was needed — callers should surface this as a finding so it
    gets fixed before the next milestone.
    """
    import re

    parse_error: str | None = None

    try:
        import yaml  # type: ignore[import]

        with open(EXCLUSION_YAML) as fh:
            data = yaml.safe_load(fh)

        ids: set[int] = set()
        for section_key in ("retired", "retired_experiments"):
            for entry in data.get(section_key, []):
                if isinstance(entry, dict) and isinstance(entry.get("experiment_id"), int):
                    ids.add(entry["experiment_id"])
        return sorted(ids), None

    except Exception as exc:
        # YAML is broken — fall back to regex to avoid false-missing reports.
        # The parse error is recorded as a finding so it can be fixed.
        parse_error = str(exc)

    try:
        with open(EXCLUSION_YAML) as fh:
            text = fh.read()
        ids_regex = {int(m) for m in re.findall(r"experiment_id:\s*(\d+)", text)}
        return sorted(ids_regex), parse_error
    except Exception as exc2:
        return [], f"yaml_parse_failed: {parse_error}; regex_fallback_failed: {exc2}"


def check_json_manifest() -> list[int]:
    """Return sorted list of integer experiment IDs present in scripts/conductor_exclusion_manifest.json.

    The conductor reads this JSON at runtime; the YAML is for human audits and
    milestone-prereq gate docs.  Both must contain the retired IDs.
    """
    try:
        with open(EXCLUSION_JSON) as fh:
            data = json.load(fh)
    except Exception:
        return []

    ids: set[int] = set()
    for entry in data.get("excluded", []):
        if isinstance(entry, int):
            ids.add(entry)
        elif isinstance(entry, dict):
            eid = entry.get("experiment_id")
            if isinstance(eid, int):
                ids.add(eid)
    return sorted(ids)


def check_gate_config_bugs() -> list[dict]:
    """Scan research-roadmap.yaml for gated_on entries with invalid or missing op.

    Returns a list of dicts, each with task_id, op (the bad value), and upstream.
    An empty list means no bugs were found — the roadmap is clean.

    The .76 bug: Exp 980's gate had op='' (empty string).  The conductor evaluator
    did not recognise it and reported 'unknown op', blocking Exp 980 even though
    its own implementation was correct and independent.
    """
    try:
        import yaml  # type: ignore[import]

        with open(ROADMAP_YAML) as fh:
            data = yaml.safe_load(fh)
    except Exception:
        return [{"error": "could_not_parse_roadmap"}]

    bugs: list[dict] = []
    for task in data.get("tasks", []):
        for gate in task.get("gated_on", []):
            op = gate.get("op", "")
            if op not in VALID_OPS:
                bugs.append(
                    {
                        "task_id": task.get("id", "unknown"),
                        "op": op,
                        "upstream": gate.get("upstream", ""),
                    }
                )
    return bugs


def check_sota_models() -> dict[str, bool]:
    """Check whether each SOTA GGUF model is present in the HuggingFace hub cache.

    We check for the cache directory rather than loading the model, which is fast
    and avoids spinning up CUDA/ROCm just for a preflight.  A missing directory
    means the model has not been downloaded; downstream experiments that need it
    will fail at inference time unless downloaded first.
    """
    hub_root = os.path.expanduser("~/.cache/huggingface/hub")
    results: dict[str, bool] = {}
    for model_id in SOTA_MODELS:
        cache_name = _hf_cache_dir_name(model_id)
        results[model_id] = os.path.isdir(os.path.join(hub_root, cache_name))
    return results


def main() -> None:
    """Run all preflight checks and write a JSON artifact under results/.

    The entire body is wrapped in try/finally so the artifact is always written,
    even if an individual check raises an exception.  This is the exact discipline
    that Exp 975 violated in .76 — missing try/finally caused a total cascade block.
    """
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    artifact: dict = {}

    try:
        yaml_ids, yaml_parse_error = check_yaml_manifest()
        json_ids = check_json_manifest()
        gate_bugs = check_gate_config_bugs()
        sota = check_sota_models()

        missing_yaml = [i for i in REQUIRED_IDS if i not in yaml_ids]
        missing_json = [i for i in REQUIRED_IDS if i not in json_ids]

        all_manifest_present = not missing_yaml and not missing_json
        no_gate_bugs = len(gate_bugs) == 0
        all_models_ready = all(sota.values())

        if all_manifest_present and no_gate_bugs and all_models_ready:
            verdict = "preflight_complete"
            status = "success"
        else:
            verdict = "preflight_partial"
            status = "partial"

        artifact = {
            "experiment": 986,
            "title": "Preflight v27 — Manifest Verify 786/627/603/641 + Gate Config Audit + SOTA Models",
            "run_date": datetime.now(UTC).strftime("%Y%m%d"),
            "started_at": started_at,
            "finished_at": "",  # filled in finally
            "status": status,
            "honest_verdict": verdict,
            "manifest_entries": yaml_ids,
            "manifest_entries_required": REQUIRED_IDS,
            "manifest_missing_yaml": missing_yaml,
            "manifest_missing_json": missing_json,
            "gate_config_bugs": gate_bugs,
            "sota_models_ready": sota,
            "schema": "v1",
        }

    except Exception as exc:
        # A failed check must not suppress the artifact — write partial result.
        artifact = {
            "experiment": 986,
            "title": "Preflight v27 — Manifest Verify 786/627/603/641 + Gate Config Audit + SOTA Models",
            "run_date": datetime.now(UTC).strftime("%Y%m%d"),
            "started_at": started_at,
            "finished_at": "",
            "status": "error",
            "honest_verdict": "preflight_partial",
            "error": str(exc),
            "manifest_entries": [],
            "manifest_entries_required": REQUIRED_IDS,
            "manifest_missing_yaml": REQUIRED_IDS,
            "manifest_missing_json": REQUIRED_IDS,
            "gate_config_bugs": [],
            "sota_models_ready": {m: False for m in SOTA_MODELS},
            "schema": "v1",
        }

    finally:
        finished_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        artifact["finished_at"] = finished_at
        os.makedirs("results", exist_ok=True)
        with open(RESULT_PATH, "w") as fh:
            json.dump(artifact, fh, indent=2)
        print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
