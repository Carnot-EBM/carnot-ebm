import os
import json
import time
import urllib.request
import urllib.error
import glob

def check_pypi():
    """Check if carnot-ebm is available on PyPI."""
    reachable = False
    try:
        urllib.request.urlopen("https://pypi.org/pypi/carnot-ebm/json", timeout=5)
        reachable = True
    except Exception:
        pass

    published = False
    version = None
    if reachable:
        try:
            resp = urllib.request.urlopen("https://pypi.org/pypi/carnot-ebm/json", timeout=5)
            data = json.loads(resp.read())
            if "info" in data and "version" in data["info"]:
                published = True
                version = data["info"]["version"]
        except Exception:
            pass
    return reachable, published, version

def check_hf():
    """Check if the HuggingFace Carnot-EBM mirror is up."""
    reachable = False
    try:
        req = urllib.request.Request("https://huggingface.co/Carnot-EBM", headers={'User-Agent': 'Mozilla/5.0'})
        urllib.request.urlopen(req, timeout=5)
        reachable = True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            reachable = True
    except Exception:
        pass

    mirror_up = False
    if reachable:
        try:
            req = urllib.request.Request("https://huggingface.co/api/models?search=carnot-ebm&limit=5", headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            if len(data) > 0:
                mirror_up = True
        except Exception:
            pass
    return reachable, mirror_up

def evaluate_gate(repo_root="."):
    """Evaluate all Phase 1 ship gate criteria."""
    start_time = time.time()
    pypi_reachable, pypi_published, pypi_version = check_pypi()
    hf_reachable, hf_mirror_up = check_hf()
    
    mcp_docs_present = len(glob.glob(os.path.join(repo_root, "docs/*mcp*"))) > 0
    cli_docs_present = len(glob.glob(os.path.join(repo_root, "docs/*cli*"))) > 0
    
    ci_workflow_count = len(glob.glob(os.path.join(repo_root, ".github/workflows/*.yml")))
    external_reproducer_exists = ci_workflow_count > 0 and os.path.exists(os.path.join(repo_root, "ops/test-results.md"))

    missing_criteria = []
    if not pypi_published: missing_criteria.append("PyPI package carnot-ebm not published")
    if not hf_mirror_up: missing_criteria.append("HuggingFace Carnot-EBM mirror has no models")
    if not mcp_docs_present: missing_criteria.append("MCP docs missing in docs/")
    if not cli_docs_present: missing_criteria.append("CLI docs missing in docs/")
    if not external_reproducer_exists: missing_criteria.append("External reproducer missing (CI workflow + ops/test-results.md)")

    phase1_ship_gate_met = len(missing_criteria) == 0

    end_time = time.time()

    return {
        "honest_verdict": "Phase 1 ship gate evaluation complete.",
        "phase1_ship_gate_met": phase1_ship_gate_met,
        "pypi_published": pypi_published,
        "pypi_version": pypi_version,
        "hf_mirror_up": hf_mirror_up,
        "mcp_docs_present": mcp_docs_present,
        "cli_docs_present": cli_docs_present,
        "external_reproducer_exists": external_reproducer_exists,
        "missing_criteria": missing_criteria,
        "duration_s": int(end_time - start_time),
        "preconditions_checked": {
            "pypi_reachable": pypi_reachable,
            "hf_reachable": hf_reachable
        }
    }
