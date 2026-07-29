# Path-resolution traceability: the repo-root/sys.path resolution in this file traces to
# REQ-ARC-WMTE-6043 (centralised output-path resolution). That is the ONLY behaviour in this
# file covered by that requirement -- the GAP-3/GAP-4 assertions below predate spec
# traceability and are recorded as pre-existing debt in ops/known-issues.md, not claimed here.
import pytest
import torch
import sys
import os
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
PROJECT_ROOT = str(repo_root())
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import importlib.util as _ilu


def _load(name, fname):
    spec = _ilu.spec_from_file_location(name, os.path.join(PROJECT_ROOT, "scripts", fname))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ebt_beam_generate():
    exp = _load("exp3777", "experiment_3777_p1_discrete_search_adjudication_v3.py")

    assert hasattr(exp, "main")
    assert hasattr(exp, "ebt_beam_generate")

    class MockEBT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = torch.nn.Embedding(exp.VOCAB, 16)

        def forward(self, orig, pred):
            vocab_size = orig.shape[0]
            seq_len = orig.shape[1]
            return torch.zeros((vocab_size, seq_len, 1))

    ebt = MockEBT()
    device = torch.device("cpu")
    pid = [1, 2, 3]
    ans_len = 2
    best_ids, nf = exp.ebt_beam_generate(ebt, pid, ans_len, device, beam=2, topk=2)

    assert len(best_ids) == ans_len
    assert nf > 0
