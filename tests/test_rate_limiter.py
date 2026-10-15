"""Tests for the RateLimiter class."""

import time
import unittest

from calls_coordinator.rate_limiter import RateLimiter
from tests.helpers import async_test


class RateLimiterTests(unittest.TestCase):
    @async_test
    async def test_six_acquisitions_at_rate_50_take_at_least_100_ms(self) -> None:
        """Test that six acquisitions at a rate of 50 per second take at least 100 ms (5 * 20 ms) to complete."""
        
        limiter = RateLimiter(rate=50, burst=1)

        started = time.perf_counter()
        for _ in range(6):
            await limiter.acquire()
        elapsed = time.perf_counter() - started
        # The first acquisition should be immediate, and the next five should take at least 20 ms each, for a total of at least 100 ms.
        self.assertGreaterEqual(elapsed, 5 * 0.020)

    def test_non_positive_rate_raises_value_error(self) -> None:
        """Test that a non-positive rate raises a ValueError."""

        for rate in (0, -1):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    RateLimiter(rate=rate)