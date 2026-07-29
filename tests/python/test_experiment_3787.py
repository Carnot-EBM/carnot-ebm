# Path-resolution traceability: the repo-root/sys.path resolution in this file traces to
# REQ-ARC-WMTE-6043 (centralised output-path resolution). That is the ONLY behaviour in this
# file covered by that requirement -- the GAP-3/GAP-4 assertions below predate spec
# traceability and are recorded as pre-existing debt in ops/known-issues.md, not claimed here.
import os
import sys
import json
from unittest.mock import patch
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
PROJECT_ROOT = str(repo_root())
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import experiment_3787_p1_discrete_search_adjudication_v3_retry as exp


def test_experiment_3787_blocked_no_cuda(tmp_path):
    with patch("torch.cuda.is_available", return_value=False):
        exp.main()

    res_path = os.path.join(
        PROJECT_ROOT, "results", "experiment_3787_p1_discrete_search_adjudication_v3_retry.json"
    )
    assert os.path.exists(res_path)
    with open(res_path) as f:
        data = json.load(f)

    assert data["honest_verdict"] == "blocked_cuda_unavailable"
    assert data["preconditions_checked"] is False


def test_experiment_3787_blocked_no_free_gpu(tmp_path):
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", return_value=(0, 24 * 1024**3)),
    ):
        exp.main()

    res_path = os.path.join(
        PROJECT_ROOT, "results", "experiment_3787_p1_discrete_search_adjudication_v3_retry.json"
    )
    assert os.path.exists(res_path)
    with open(res_path) as f:
        data = json.load(f)

    assert data["honest_verdict"] == "blocked_no_free_gpu"
    assert data["handoff_to_operator"] is True
    assert data["preconditions_checked"] is False
