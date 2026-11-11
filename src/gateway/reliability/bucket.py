"""Token bucket, one per provider."""

import threading
import time


class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        if rate < 0 or capacity <= 0:
            raise ValueError("token bucket needs a positive capacity and non-negative rate")
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.updated = now
            if self.tokens < amount:
                return False
            self.tokens -= amount
            return True
