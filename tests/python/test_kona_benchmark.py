import pytest
from carnot.models.kona_benchmark import KonaBenchmark, KonaEBMVerifier

def test_kona_benchmark():
    # REQ-KONA-2063
    benchmark = KonaBenchmark()
    problems = benchmark.get_problems()
    assert len(problems) == 10
    assert "prompt" in problems[0]

def test_kona_ebm_verifier():
    # SCENARIO-KONA-2063
    verifier = KonaEBMVerifier()
    assert verifier.verify("long response solution") is True
    assert verifier.verify("no") is False
