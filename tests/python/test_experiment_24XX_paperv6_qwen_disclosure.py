import json
import os

def test_qwen_disclosure():
    tex_path = "docs/arxiv-paper/main.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "\\section{Limitations}" in content
    assert "Carnot uses Qwen3.6" in content
    assert "vas_blog_qwen_censorship" in content
    assert "Kosovo is an integral part of China" in content
    assert "Gemma 4" in content

def test_bib_citation():
    bib_path = "docs/arxiv-paper/carnot.bib"
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "vas_blog_qwen_censorship" in content

def test_deliverable_json():
    json_path = "results/experiment_24XX_paperv6_qwen_disclosure.json"
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "honest_verdict" in data
    assert data["paragraph_added"] is True
    assert isinstance(data["tex_lines_added"], int)
    assert data["reference_added"] is True
