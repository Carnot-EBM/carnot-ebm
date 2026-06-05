import pytest
import os
import tempfile
import json
from carnot.research.experiment_3841 import ResearchRefresh3841

def test_check_section_intact():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Some content\n## .353 additions\nMore content")
        tmp_name = f.name
    try:
        refresh = ResearchRefresh3841(references_file=tmp_name)
        assert refresh.check_section_intact() is True
    finally:
        os.remove(tmp_name)

def test_check_section_not_intact():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Some content\nMore content")
        tmp_name = f.name
    try:
        refresh = ResearchRefresh3841(references_file=tmp_name)
        assert refresh.check_section_intact() is False
    finally:
        os.remove(tmp_name)

def test_append_papers():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Some content\n## .353 additions\n")
        tmp_name = f.name
    try:
        refresh = ResearchRefresh3841(references_file=tmp_name)
        count = refresh.append_papers()
        assert count == 5
        
        with open(tmp_name, 'r') as f2:
            content = f2.read()
        assert "arXiv:2605.30914" in content
        
        # Test duplicate append
        count2 = refresh.append_papers()
        assert count2 == 0
    finally:
        os.remove(tmp_name)

def test_append_papers_fails_when_not_intact():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Some content\n")
        tmp_name = f.name
    try:
        refresh = ResearchRefresh3841(references_file=tmp_name)
        with pytest.raises(ValueError):
            refresh.append_papers()
    finally:
        os.remove(tmp_name)

def test_generate_artifact():
    with tempfile.TemporaryDirectory() as temp_dir:
        references_file = os.path.join(temp_dir, "research-references.md")
        output_json = os.path.join(temp_dir, "results.json")
        with open(references_file, 'w') as f:
            f.write("## .353 additions\n")
            
        refresh = ResearchRefresh3841(references_file=references_file)
        artifact = refresh.generate_artifact(output_path=output_json)
        
        assert artifact["honest_verdict"] == "complete: external_research_refresh_353_section_intact_references_appended_numbers_as_reported"
        assert artifact["section_intact"] is True
        assert artifact["n_references_appended"] == 5
        assert len(artifact["references_filed"]) == 5
        
        with open(output_json, 'r') as f:
            saved_artifact = json.load(f)
        assert saved_artifact == artifact

def test_generate_artifact_blocked():
    with tempfile.TemporaryDirectory() as temp_dir:
        references_file = os.path.join(temp_dir, "missing.md")
        output_json = os.path.join(temp_dir, "results.json")
            
        refresh = ResearchRefresh3841(references_file=references_file)
        artifact = refresh.generate_artifact(output_path=output_json)
        
        assert artifact["honest_verdict"] == "blocked_research-references.md"
        assert artifact["section_intact"] is False
        assert artifact["n_references_appended"] == 0
        assert artifact["references_filed"] == []
