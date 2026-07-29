import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock
import torch  # Import torch here to avoid memory leak false positive during first test
from carnot.paths import repo_root

# Ensure the script is importable
# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
PROJECT_ROOT = str(repo_root())
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import experiment_3735_bounded_train_chunk2_resume as exp_script

# REQ-EBT-3734-STABLE-TRAIN


def test_format_gsm8k_row():
    row = {"question": "Q", "answer": "A"}
    res = exp_script.format_gsm8k_row(row)
    assert res == "Question: Q\nAnswer: A"


def test_tokenize_empty():
    import numpy as np

    res = exp_script.tokenize_texts_to_blocks([], 10)
    assert res.shape == (0, 11)


def test_run_blocked_no_checkpoint(tmp_path):
    with patch.object(exp_script, "PROJECT_ROOT", str(tmp_path)):
        res = exp_script.run_experiment(mock=True)
        assert res["honest_verdict"] == "blocked_no_checkpoint"
        assert res["cumulative_steps_trained"] == 0


def test_run_experiment_mock(tmp_path):
    # Setup dummy checkpoint files
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = results_dir / "experiment_3734_checkpoint.pt"
    json_path = results_dir / "experiment_3734_fix_harness_and_bounded_train_chunk1.json"

    import torch

    # Save a fake torch state
    ebt, ar = exp_script.build_tiny_models(
        dim=16, n_layers=1, n_heads=2, ffn_dim_multiplier=4.0, batch_size=2, block_size=16
    )
    ebt_opt = torch.optim.AdamW(ebt.parameters())
    ar_opt = torch.optim.AdamW(ar.parameters())
    torch.save(
        {
            "ebt_state": ebt.state_dict(),
            "ar_state": ar.state_dict(),
            "ebt_opt": ebt_opt.state_dict(),
            "ar_opt": ar_opt.state_dict(),
            "steps": 2,
        },
        ckpt_path,
    )

    with open(json_path, "w") as f:
        json.dump({"cumulative_steps_trained": 2}, f)

    with patch.object(exp_script, "PROJECT_ROOT", str(tmp_path)):
        # Run mock experiment
        res = exp_script.run_experiment(max_steps=2, mock=True)
        assert (
            "complete: ebt_train_resumed_total_" in res["honest_verdict"]
            or "diverged" in res["honest_verdict"]
        )

        # also test check_preconditions directly
        pre = exp_script.check_preconditions()
        assert pre["checkpoint_present"] is True

    import gc

    del ebt, ar, ebt_opt, ar_opt
    gc.collect()


def test_check_preconditions_no_cuda():
    with patch.dict("sys.modules", {"torch": None}):
        pre = exp_script.check_preconditions()
        assert pre["cuda"] is False


def test_check_preconditions_import_error():
    real_find_spec = exp_script.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "carnot.phase3.ebt_upstream":
            raise Exception("foo")
        return real_find_spec(name)

    with patch("importlib.util.find_spec", side_effect=fake_find_spec):
        pre = exp_script.check_preconditions()
        assert pre["ebt_vendored"] is False


def test_check_preconditions_json_corrupt(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / "experiment_3734_checkpoint.pt"
    json_path = results_dir / "experiment_3734_fix_harness_and_bounded_train_chunk1.json"
    ckpt_path.touch()
    json_path.write_text("corrupt json")
    with patch.object(exp_script, "PROJECT_ROOT", str(tmp_path)):
        pre = exp_script.check_preconditions()
        assert pre["checkpoint_present"] is False


def test_main():
    with patch.object(exp_script, "run_experiment") as mock_run:
        exp_script.main()
        mock_run.assert_called_once_with(max_steps=200, mock=False)


def teardown_module(module):
    import gc

    gc.collect()
