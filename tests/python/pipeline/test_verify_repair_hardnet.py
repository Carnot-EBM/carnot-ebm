"""Tests for VerifyRepairPipeline with HardNet integration.

Spec references: REQ-HARDNET-2087
"""

import pytest
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_hardnet_in_verify_repair_pipeline_satisfied():
    """Verify that the pipeline uses HardNetLayer and correctly evaluates satisfied bounds."""
    pipeline = VerifyRepairPipeline(use_hardnet=True)
    
    constraint = ConstraintResult(
        constraint_type="bound",
        description="Value must be between 0 and 10",
        metadata={
            "lower_bound": 0.0,
            "upper_bound": 10.0,
            "value": 5.0
        }
    )
    
    result = pipeline._evaluate_constraints([constraint])
    
    assert result.verified is True
    assert len(result.violations) == 0
    assert result.energy == 0.0
    
def test_hardnet_in_verify_repair_pipeline_violated():
    """Verify that the pipeline uses HardNetLayer and correctly evaluates violated bounds."""
    pipeline = VerifyRepairPipeline(use_hardnet=True)
    
    constraint = ConstraintResult(
        constraint_type="bound",
        description="Value must be between 0 and 10",
        metadata={
            "lower_bound": 0.0,
            "upper_bound": 10.0,
            "value": 15.0
        }
    )
    
    result = pipeline._evaluate_constraints([constraint])
    
    assert result.verified is False
    assert len(result.violations) == 1
    assert result.energy == 5.0  # 15.0 - 10.0 = 5.0
    
def test_hardnet_in_verify_repair_pipeline_violated_lower():
    """Verify that the pipeline uses HardNetLayer and correctly evaluates violated lower bounds."""
    pipeline = VerifyRepairPipeline(use_hardnet=True)
    
    constraint = ConstraintResult(
        constraint_type="bound",
        description="Value must be greater than 0",
        metadata={
            "lower_bound": 0.0,
            "value": -3.0
        }
    )
    
    result = pipeline._evaluate_constraints([constraint])
    
    assert result.verified is False
    assert len(result.violations) == 1
    assert result.energy == 3.0  # 0.0 - (-3.0) = 3.0
