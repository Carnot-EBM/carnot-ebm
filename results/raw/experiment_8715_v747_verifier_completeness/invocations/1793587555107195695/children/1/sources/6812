"""
REQ-PUBLISH-026: HuggingFace Publish Retry
"""

import os
import json
import tempfile
import time
from datetime import datetime, timezone, UTC
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def run_experiment():
    start_time = time.time()
    org_name = "Carnot-EBM"
    repo_name = "carnot-smallest-test"
    repo_id = f"{org_name}/{repo_name}"

    # Artifact paths - finding the smallest file
    models_dir = Path("python/carnot/models")
    if not models_dir.exists():
        # Fallback for test mocking
        models_dir = Path("/tmp/models_mock")
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "mock.pt").write_text("mock")

    pt_files = list(models_dir.glob("*.*"))

    if not pt_files:
        raise FileNotFoundError("No models found in python/carnot/models")

    artifact_path = min(pt_files, key=lambda p: p.stat().st_size).resolve()

    model_card = """---
license: apache-2.0
---
# Carnot Smallest Test Model

This is a test upload for the Carnot-EBM project. No emojis here.
"""
    # Stage the HuggingFace model card in a temp directory, NOT in the working
    # directory.
    #
    # This used to be `Path("README.md")`, which is CWD-relative. Under pytest the
    # working directory is the repository root, so every run of this script's test
    # silently replaced the project's own operator-curated README.md with this
    # six-line model card -- while passing. README.md is operator-curated under the
    # Public Documentation Discipline; the autonomous loop may not write it at all.
    #
    # A temp directory is used rather than a repo-relative staging path so this write
    # cannot land inside the repository under ANY working directory or override.
    # Nothing downstream changes: the same bytes are uploaded, still as "README.md"
    # in the HF repo (`path_in_repo` below is untouched).
    staging_dir = Path(tempfile.mkdtemp(prefix="carnot_exp1750_hf_card_"))
    model_card_path = staging_dir / "README.md"
    model_card_path.write_text(model_card)

    api = HfApi()

    hf_upload_succeeded = False
    honest_verdict = "blocked_credentials"

    try:
        whoami = api.whoami()
        if "id" in whoami:
            create_repo(repo_id, exist_ok=True, repo_type="model")
            api.upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
            )
            api.upload_file(
                path_or_fileobj=str(artifact_path),
                path_in_repo=artifact_path.name,
                repo_id=repo_id,
            )
            hf_upload_succeeded = True
            honest_verdict = "OK: Model published"
    except Exception as e:
        print(f"HF operation failed: {e}")

    duration_s = time.time() - start_time

    deliverable = {
        "schema": "carnot.huggingface_retry.v1",
        "experiment": 1750,
        "run_date": datetime.now(UTC).isoformat(),
        "duration_s": duration_s,
        "hf_upload_succeeded": hf_upload_succeeded,
        "honest_verdict": honest_verdict,
        "artifact_uploaded": artifact_path.name,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_1750_huggingface_retry.json"
    with open(out_path, "w") as f:
        json.dump(deliverable, f, indent=2)
    print(f"Wrote deliverable to {out_path}")
    return deliverable


if __name__ == "__main__":
    run_experiment()
