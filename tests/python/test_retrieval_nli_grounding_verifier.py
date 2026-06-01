from carnot.verify.retrieval_nli_grounding_verifier import RetrievalNLIGroundingVerifier

def test_split_into_claims():
    verifier = RetrievalNLIGroundingVerifier()
    assert verifier.split_into_claims("H1. H2!") == ["H1", "H2"]
    assert verifier.split_into_claims("No punctuation") == ["No punctuation"]
    assert verifier.split_into_claims("") == [""]

def test_compute_entailment_proxy():
    verifier = RetrievalNLIGroundingVerifier()
    # 'H' indicates unentailed (energy=1.0)
    assert verifier.compute_entailment_proxy("H1", "context") == 1.0
    # 'R' indicates entailed (energy=0.0)
    assert verifier.compute_entailment_proxy("R1", "context") == 0.0

def test_verify():
    verifier = RetrievalNLIGroundingVerifier()
    # "H1. R1." has 2 claims, one unentailed -> energy 0.5
    assert verifier.verify("H1. R1.", "context") == 0.5
    assert verifier.verify("R1. R2.", "context") == 0.0
    assert verifier.verify("H1. H2.", "context") == 1.0
    assert verifier.verify("", "context") == 0.0
    # test punctuation without valid claims if any
    assert verifier.verify(".", "context") == 0.0

