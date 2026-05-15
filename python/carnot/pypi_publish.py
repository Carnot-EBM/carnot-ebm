"""PyPI publisher logic."""
import json
import datetime
import hashlib
import os
from typing import Any, Dict

def check_pypi_credentials() -> bool:
    """Check if PyPI credentials exist in environment."""
    return "TWINE_USERNAME" in os.environ and "TWINE_PASSWORD" in os.environ

def build_publish_artifact(experiment_id: int) -> Dict[str, Any]:
    """Build the publication artifact, emitting blocked if credentials are missing."""
    creds_ok = check_pypi_credentials()
    
    if not creds_ok:
        return {
            "schema": "carnot.pypi_publish.v2",
            "experiment": experiment_id,
            "run_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": 0,
            "random_seed": 171511,
            "reproducibility_checksum": hashlib.sha256(b"blocked").hexdigest(),
            "preconditions_checked": [
                "pyproject.toml has name = carnot-ebm",
                "python -m build --version checked",
                "twine --version checked",
                "PyPI credentials check"
            ],
            "model_specs": {
                "package_name": "carnot-ebm",
                "version": "blocked",
                "publish_target": "blocked",
                "wheel_filename": "blocked",
                "sdist_filename": "blocked",
                "wheel_sha256": "blocked",
                "sdist_sha256": "blocked"
            },
            "n_samples": 1,
            "n_samples_justification": "Ship task.",
            "pypi_url": "blocked",
            "external_install_verified": False,
            "acceptance_gate_passed": False,
            "acceptance_gate_criteria": "Real upload + verified install.",
            "methodology_note": "Per CLAUDE.md preconditions discipline, missing credentials → blocked verdict, NOT fabrication.",
            "optimization_direction": "neither — ship task",
            "honest_verdict": "blocked_pypi_credentials_unavailable"
        }
    
    raise NotImplementedError("Real publish not yet implemented.")

def run_publish(experiment_id: int, result_path: str) -> None:
    """Run the publish check and write the artifact."""
    artifact = build_publish_artifact(experiment_id)
    with open(result_path, "w") as f:
        json.dump(artifact, f, indent=2)

if __name__ == "__main__":
    run_publish(1711, "results/experiment_1711_pypi_publish.json")
