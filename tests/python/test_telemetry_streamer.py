import pytest
import time
from unittest.mock import MagicMock
from carnot.training.telemetry_streamer import TelemetryStreamer
from carnot.pipeline.verify_repair import VerificationResult

def test_telemetry_streamer_record_and_process():
    """Test that the streamer queues and processes items asynchronously.
    
    Spec: REQ-LEARN-102
    """
    streamer = TelemetryStreamer(max_size=10)
    streamer.start()
    
    res1 = MagicMock(spec=VerificationResult)
    res2 = MagicMock(spec=VerificationResult)
    
    assert streamer.record(res1) is True
    assert streamer.record(res2) is True
    
    # Allow background thread to process
    time.sleep(0.5)
    streamer.stop()
    
    assert len(streamer.results) == 2
    assert streamer.results[0] == res1
    assert streamer.results[1] == res2

def test_telemetry_streamer_queue_full():
    """Test non-blocking behavior when queue is full."""
    streamer = TelemetryStreamer(max_size=1)
    # Do NOT start the streamer so the queue fills up immediately
    
    res1 = MagicMock(spec=VerificationResult)
    res2 = MagicMock(spec=VerificationResult)
    
    assert streamer.record(res1) is True
    assert streamer.record(res2) is False  # Queue should be full
