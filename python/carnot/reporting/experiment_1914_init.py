import json

def generate_init(output_path: str) -> None:
    """
    Generate the milestone initialization artifact for .195.
    
    This function satisfies REQ-REPORT-195, creating the initialization 
    artifact indicating the transition from milestone .194 to .195.
    """
    data = {
        "schema": "carnot.init.v1",
        "experiment": 1914,
        "status_updated": True,
        "honest_verdict": "complete: initialized .195"
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
