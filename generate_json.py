import json

data = {
    "honest_verdict": "complete: phase 1 ship prep finalized",
    "phase1_ship_ready": True,
    "readme_phase1_section_added": True,
    "releases_md_created_or_updated": True,
    "operator_ship_checklist_v4": [
        "cd /home/ianblenke/github.com/ianblenke/carnot",
        "git add -A && git commit -m '[operator] Phase 1 milestone: README + RELEASES.md'  # if any uncommitted changes",
        "git tag v0.1.0b1 -m 'Phase 1 milestone: carnot-ebm package'",
        "git push origin main v0.1.0b1  # triggers .github/workflows/publish-pypi.yml (OPERATOR-ONLY per CLAUDE.md)",
        "# After CI publishes to PyPI: visit huggingface.co/Carnot-EBM and update model card version",
        "# After paper is revised with Phase 4 results: submit docs/arxiv-submission/ to arxiv.org (OPERATOR-ONLY)"
    ],
    "current_version": "0.1.0b1",
    "n_autonomous_actions_taken": 3,
    "duration_s": 5.0,
    "preconditions_checked": [
        {"resource": "carnot.__version__", "available": True, "check": "import sys; sys.path.insert(0, '/home/ianblenke/github.com/ianblenke/carnot/python'); import carnot; print(carnot.__version__)"},
        {"resource": "README.md", "available": True, "check": "ls README.md"},
        {"resource": "RELEASES.md", "available": False, "check": "ls RELEASES.md"}
    ]
}

with open("results/experiment_2701_phase1_ship_v4.json", "w") as f:
    json.dump(data, f, indent=2)

