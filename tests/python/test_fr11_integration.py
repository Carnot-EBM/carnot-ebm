import pytest
from carnot.pipeline.fr11_integration import FR11IntegrationPipeline

def test_fr11_pipeline_initialization():
    pipeline = FR11IntegrationPipeline()
    assert pipeline.tier4_model is not None
    assert "default_constraint" in pipeline.constraint_weights
    
def test_fr11_pipeline_run():
    pipeline = FR11IntegrationPipeline()
    
    # Run a few times to trigger Tier 4 adaptation
    for i in range(5):
        res = pipeline.run(
            query="test_query",
            partial_response="part",
            full_response="full",
            label="incorrect"
        )
        
    # By the 5th run (since region is consistently 2.0 based on query length mock in our pipeline, wait it was hardcoded 2.0),
    # Tier 4 should have triggered and adapted.
    assert "tier4_to_tier1_feedback" in res
