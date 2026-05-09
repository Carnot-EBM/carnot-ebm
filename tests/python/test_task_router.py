import pytest
from carnot.pipeline.task_router import calculate_character_entropy, EntropyTaskRouter

def test_calculate_character_entropy():
    # Empty string
    assert calculate_character_entropy("") == 0.0
    
    # Single character
    assert calculate_character_entropy("aaaa") == 0.0
    
    # Multiple characters
    entropy = calculate_character_entropy("ab")
    assert entropy == pytest.approx(1.0)

def test_entropy_task_router():
    # REQ-PIPELINE-1625: Entropy-Based Task Router
    router = EntropyTaskRouter(threshold=4.15)
    assert router.route("a") == "ebm_verifier"
    assert router.route("abcdefghijklmnopqrstuvwxyz" * 10) == "base_llm"

def test_entropy_task_router_scenario():
    # SCENARIO-PIPELINE-1625: Entropy Router Routes GSM8K to EBM Verifier
    router = EntropyTaskRouter(threshold=4.15)
    
    math_question = "If John has 5 apples and gives 2 away, how many does he have left?"
    qa_question = "Can you explain the socio-economic impacts of the Industrial Revolution in 19th century Europe, detailing the shift from agrarian societies to urban industrialized centers, the rise of the working class, and the subsequent changes in living conditions, labor laws, and political ideologies such as the emergence of socialism and trade unionism?"
    
    assert router.route(math_question) == "ebm_verifier"
    assert router.route(qa_question) == "base_llm"
