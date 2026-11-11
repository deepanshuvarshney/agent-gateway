"""Circuit breaker with a half-open trial."""

import threading
import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_seconds: float) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.state = "closed"
        self.opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if self.state == "open":
                if now - self.opened_at >= self.reset_seconds:
                    self.state = "half_open"
                    return True
                return False
            if self.state == "half_open":
                # A trial is already in flight.
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = time.monotonic()
