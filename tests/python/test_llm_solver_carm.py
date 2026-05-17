"""Tests for CARM integration in LLM solver.

References: REQ-CARM-1772-1
"""

from unittest.mock import patch, MagicMock
from carnot.inference.llm_solver import LLMSolverConfig, iterative_refine_code, iterative_refine_with_properties

def test_iterative_refine_code_carm():
    """Test that CARM context is prepended in iterative_refine_code."""
    config = LLMSolverConfig(use_carm=True)
    task_description = "Calculate the sum of two integers"
    test_cases = [((1, 2), 3)]
    
    with patch("carnot.pipeline.carm.CARM") as MockCARM:
        mock_carm_instance = MockCARM.return_value
        mock_carm_instance.retrieve_domains.return_value = ["arithmetic"]
        mock_carm_instance.retrieve_constraint_types.return_value = ["sum"]
        
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            import openai
            mock_client = MagicMock()
            openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="```python\ndef test(a, b):\n    return a + b\n```"))]
            )
            
            result = iterative_refine_code(config, task_description, "test", test_cases)
            
            mock_carm_instance.retrieve_domains.assert_called_with(task_description)
            mock_carm_instance.retrieve_constraint_types.assert_called_with(task_description)
            
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert any("CARM Context" in msg["content"] and "arithmetic" in msg["content"] for msg in messages)
            assert result.final_verified is True

def test_iterative_refine_with_properties_carm():
    """Test that CARM context is prepended in iterative_refine_with_properties."""
    config = LLMSolverConfig(use_carm=True)
    task_description = "Calculate the sum of two integers"
    test_cases = [((1, 2), 3)]
    
    with patch("carnot.pipeline.carm.CARM") as MockCARM:
        mock_carm_instance = MockCARM.return_value
        mock_carm_instance.retrieve_domains.return_value = ["arithmetic"]
        mock_carm_instance.retrieve_constraint_types.return_value = ["sum"]
        
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            import openai
            mock_client = MagicMock()
            openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="```python\ndef test(a, b):\n    return a + b\n```"))]
            )
            
            properties = [{
                "name": "comm",
                "gen_args": lambda rng: (rng.randint(1, 10), rng.randint(1, 10)),
                "check": lambda res, a, b: res == a + b
            }]
            
            result = iterative_refine_with_properties(config, task_description, "test", test_cases, properties, property_samples=2)
            
            mock_carm_instance.retrieve_domains.assert_called_with(task_description)
            
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert any("CARM Context" in msg["content"] and "arithmetic" in msg["content"] for msg in messages)
            assert result.final_verified is True
