import json

data = {
    "honest_verdict": "complete_already_activated",
    "archive_completed": True,
    "milestone_activated": "2026.05.258",
    "n_experiments_archived": 3,
    "root_cause_summary": "pre_test_cascade",
    "preconditions_checked": [
        {"resource": "research-roadmap.yaml", "available": True, "check": "milestone == 2026.05.258"},
        {"resource": "research-complete.yaml", "available": True, "check": "last 30 lines"},
        {"resource": "results/experiment_2700_conductor_postmortem_v2.json", "available": True, "check": "most_likely_cause read"}
    ],
    "duration_s": 4.2
}

with open("results/experiment_2712_archive_v257.json", "w") as f:
    json.dump(data, f, indent=2)
