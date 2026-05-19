import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
from experiment_2503_arxiv_readiness import check_arxiv_readiness

def test_check_arxiv_readiness():
    res = check_arxiv_readiness()
    assert res["arxiv_ready"] is False
    assert res["auroc_adversarially_verified"] is True

if __name__ == "__main__":
    test_check_arxiv_readiness()
    print("Test passed!")
