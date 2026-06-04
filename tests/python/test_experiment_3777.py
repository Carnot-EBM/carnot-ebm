import os
import sys
import json
import pytest
import tempfile
import importlib.util as _ilu
from unittest.mock import patch, MagicMock

# Setup dynamic import so it works with patch
PROJECT_ROOT = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
spec = _ilu.spec_from_file_location("exp3777", os.path.join(PROJECT_ROOT, "scripts", "experiment_3777_p1_discrete_search_adjudication_v3.py"))
exp3777 = _ilu.module_from_spec(spec)
sys.modules["exp3777"] = exp3777
spec.loader.exec_module(exp3777)

@patch("exp3777.torch.save")
@patch("exp3777.pb.build_corpus")
@patch("exp3777.pb.corpus_to_blocks")
@patch("exp3777.sc.build_models")
@patch("exp3777.pb.train_models")
@patch("exp3777.sc.fit_decoder")
@patch("exp3777.pb.ar_greedy")
@patch("exp3777.pb.ar_selfconsistency")
@patch("exp3777.pb.ebt_generate")
@patch("exp3777.sc.ebt_descent_generate")
@patch("exp3777.ebt_beam_generate")
def test_experiment_3777_main(
    mock_ebt_beam, mock_ebt_descent, mock_ebt_gen, mock_ar_sc, mock_ar_g,
    mock_fit_dec, mock_train, mock_build, mock_to_blocks, mock_corpus, mock_torch_save, tmp_path
):
    # Setup mocks
    mock_corpus.side_effect = [[("1+2", "003")], [("2+3", "005")]]
    mock_to_blocks.return_value = MagicMock(shape=(2, 48))
    mock_build.return_value = (MagicMock(), MagicMock())
    mock_train.return_value = False
    import torch.nn as nn
    mock_fit_dec.return_value = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 258))
    
    true_ans = exp3777.enc("005")
    mock_ar_g.return_value = (true_ans, None)
    mock_ar_sc.return_value = (true_ans, 1)
    mock_ebt_beam.return_value = (true_ans, 1)
    mock_ebt_gen.return_value = (exp3777.enc("000"), None)
    mock_ebt_descent.return_value = (exp3777.enc("000"), None)

    with patch.object(exp3777, "PROJECT_ROOT", str(tmp_path)):
        os.makedirs(os.path.join(tmp_path, "results"), exist_ok=True)
        # Patch sleep or time to speed up if needed, though there's no sleep
        # The script calls ar_greedy inside a loop over items.
        exp3777.main()

        out_file = os.path.join(tmp_path, "results", "experiment_3777_p1_discrete_search_adjudication_v3.json")
        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            data = json.load(f)
        
        assert data["positive_control_passed"] is True
        assert data["ar_best"] == 1.0
        assert data["ebt_best"] == 1.0
        assert data["adjudication"] == "decode_artifact_bounded"
        assert data["n_train"] == 40000
