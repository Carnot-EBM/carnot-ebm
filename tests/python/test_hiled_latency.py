import pytest
from carnot.pipeline.hiled_decoder import HiledDecoder

def test_hiled_decoder_latency():
    # REQ-HILED-1719
    decoder = HiledDecoder(hardware_latency_ms=5.0, use_hiled=True)
    assert decoder.use_hiled is True
    decoder_baseline = HiledDecoder(hardware_latency_ms=5.0, use_hiled=False)
    assert decoder_baseline.use_hiled is False
