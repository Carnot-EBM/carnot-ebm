"""Carnot probe modules — lightweight detectors that run before the main pipeline tiers.

**For engineers:**
    Probes are advisory signals computed cheaply before the main verification cascade.
    They do NOT short-circuit the pipeline; they add metadata flags to the result.
    Tier 0a–0f probes predated this module; Tier 0g onwards live here.
"""

from carnot.probes.streaming_cot_detector import StreamingCoTHalluDetector

__all__ = ["StreamingCoTHalluDetector"]
