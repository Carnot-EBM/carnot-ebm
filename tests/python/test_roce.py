import os
import pytest
from carnot.pipeline.roce import DynamicConstraint, PromptConstraintExtractor

def test_extract_must_contain_req_extract_055_2():
    # SCENARIO-EXTRACT-094
    prompt = "Your response must include the word 'summary'"
    extractor = PromptConstraintExtractor()
    constraints = extractor.extract_from_prompt(prompt)
    assert len(constraints) == 1
    assert constraints[0].instruction_type == "must_contain"
    assert constraints[0].metadata["term"] == "summary"
    assert constraints[0].check("Here is a summary.") == True
    assert constraints[0].check("This is the end.") == False

def test_extract_10_instruction_types_scenario_extract_095():
    # REQ-EXTRACT-055-2, REQ-EXTRACT-055-4
    extractor = PromptConstraintExtractor()
    
    # 1. must_contain
    c = extractor.extract_from_prompt("must include the word 'hello'")[0]
    assert c.check("hello world") == True
    assert c.check("bye world") == False

    # 2. must_not_contain
    c = extractor.extract_from_prompt("must not include the word 'bad'")[0]
    assert c.check("good stuff") == True
    assert c.check("bad stuff") == False

    # 3. format_json
    c = extractor.extract_from_prompt("format as json")[0]
    assert c.check('{"a": 1}') == True
    assert c.check("not json") == False

    # 4. format_list
    c = extractor.extract_from_prompt("format as list")[0]
    assert c.check("- item 1\n- item 2") == True
    assert c.check("1. item\n2. item") == True
    assert c.check("just text") == False

    # 5. max_words
    c = extractor.extract_from_prompt("maximum of 3 words")[0]
    assert c.check("one two") == True
    assert c.check("one two three four") == False

    # 6. min_words
    c = extractor.extract_from_prompt("minimum of 2 words")[0]
    assert c.check("one two") == True
    assert c.check("one") == False

    # 7. numeric_range
    c = extractor.extract_from_prompt("between 10 and 20")[0]
    assert c.check("I have 15 apples") == True
    assert c.check("I have 25 apples") == False
    assert c.check("I have no apples") == False

    # 8. starts_with
    c = extractor.extract_from_prompt("starts with 'Start'")[0]
    assert c.check("Start doing things") == True
    assert c.check("Do not start") == False

    # 9. ends_with
    c = extractor.extract_from_prompt("ends with 'End'")[0]
    assert c.check("This is the End") == True
    assert c.check("This is not") == False

    # 10. no_repetition
    c = extractor.extract_from_prompt("no repetition")[0]
    assert c.check("unique words here") == True
    assert c.check("repeat repeat words") == False

def test_check_response_req_extract_055_5():
    # REQ-EXTRACT-055-5
    extractor = PromptConstraintExtractor()
    constraints = extractor.extract_from_prompt("must include the word 'hello' and maximum of 2 words")
    
    # Violates max_words, passes must_contain
    violated = extractor.check_response("hello my friend", constraints)
    assert len(violated) == 1
    assert violated[0].instruction_type == "max_words"
    
    # Violates both
    violated_both = extractor.check_response("greetings my friend", constraints)
    assert len(violated_both) == 2
    
    # Violates neither
    violated_none = extractor.check_response("hello friend", constraints)
    assert len(violated_none) == 0

def test_ci_mode_and_live_mode_req_extract_055_4(monkeypatch):
    # REQ-EXTRACT-055-4, REQ-EXTRACT-055-3
    def mock_generate_fn(prompt):
        return [DynamicConstraint("mock_type", "Mock generated")]
    
    extractor = PromptConstraintExtractor(generate_fn=mock_generate_fn)
    
    # Default (no CARNOT_FORCE_LIVE), mock should not be called
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    constraints = extractor.extract_from_prompt("must include the word 'test'")
    assert len(constraints) == 1
    assert constraints[0].instruction_type == "must_contain"
    
    # With CARNOT_FORCE_LIVE=1, mock should be called
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    constraints_live = extractor.extract_from_prompt("must include the word 'test'")
    assert len(constraints_live) == 2
    assert constraints_live[1].instruction_type == "mock_type"

def test_default_check_unknown_type():
    constraint = DynamicConstraint("unknown_type", "unknown")
    assert constraint.check("anything") == True
