import queue
import threading
from typing import List
from carnot.pipeline.verify_repair import VerificationResult

class TelemetryStreamer:
    """Asynchronous telemetry streamer for violation feedback.
    
    Spec: REQ-LEARN-102
    """
    
    def __init__(self, max_size: int = 10000):
        self._queue: queue.Queue[VerificationResult] = queue.Queue(maxsize=max_size)
        self._results: List[VerificationResult] = []
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        
    def start(self) -> None:
        """Start the background telemetry worker."""
        self._stop_event.clear()
        self._worker_thread.start()
        
    def stop(self) -> None:
        """Stop the background worker and process remaining items."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join()
            
    def record(self, result: VerificationResult) -> bool:
        """Record a verification result in the non-blocking queue.
        
        Args:
            result: The VerificationResult instance to record.
            
        Returns:
            True if the item was successfully queued, False if the queue was full.
        """
        try:
            self._queue.put_nowait(result)
            return True
        except queue.Full:
            return False
            
    def _worker_loop(self) -> None:
        """Background worker loop to process telemetry items."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                # Wait briefly for an item
                item = self._queue.get(timeout=0.1)
                self._process_item(item)
                self._queue.task_done()
            except queue.Empty:
                continue
                
    def _process_item(self, item: VerificationResult) -> None:
        """Process a single result off the queue."""
        self._results.append(item)
        
    @property
    def results(self) -> List[VerificationResult]:
        """Get the accumulated results."""
        return self._results
