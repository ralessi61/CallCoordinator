"""Rate limiter for calls coordinator."""

import asyncio
import threading
import time


class RateLimiter:
    """
    A token bucket rate limiter.
    Idea taken from assistenza_clienti.py
    Tokens refill at rate per second up to burst; a call takes one token; if the bucket is empty the caller must wait tokens / rate at seconds
    """

    def __init__(self, rate: float, burst: int = 1) -> None:
        if rate <= 0:
            raise ValueError("rate must be greater than zero")
        if burst < 1:
            raise ValueError("burst must be at least 1")
        self._rate = rate
        self._burst = float(burst)
        self._tokens = float(burst)
        self._updated = time.perf_counter()
        self._lock = threading.Lock()
        # the thread strategy runs several event loops, asyncio.Lock belongs to 1 loop

    def _reserve(self) -> float:
        """Take one token and return how long the caller must wait for it."""

        with self._lock:
            # update the number of tokens based on the time since the last update
            now = time.perf_counter()
            # take the minimum of the burst and the number of tokens that have accumulated since the last update
            # perf_counter() high-resolution clock value in seconds, intended for measuring elapsed time
            self._tokens = min(self._burst, self._tokens + (now - self._updated) * self._rate)
            self._updated = now
            self._tokens -= 1.0
            return max(0.0, -self._tokens / self._rate)

    async def acquire(self) -> float:
        """Wait until a call may start. Returns the seconds waited."""

        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)
        return wait
