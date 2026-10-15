"""Three simulated providers: no network, no API key.

* ``fast``: quick and reliable.10–30 ms, never fails;
* ``slow``: variable latency, sometimes times out.0.2–1 sec; wprobable timeout_rate that raises ProviderTimeoutError
* ``strict``: a tight quota; calls over the limit get a 429 (``RateLimitError``).
               at most max_calls per window seconds (list of timestamps under a threading.Lock); beyond that, RateLimitError
some ideas takes from assistenza_clienti.py, which is a simple example of a token bucket rate limiter.
"""

import asyncio
import random
import threading
import time
from typing import Optional

from ..exceptions import ProviderTimeoutError, RateLimitError
from ..models import LlmRequest,LlmResponse, TokenUsage, estimate_tokens
from .base import BaseProvider


class SimulatedProvider(BaseProvider):
    """Shared behavior: wait for a random latency, then build the answer."""

    def __init__(self, latency: tuple = (0.01, 0.03), seed: Optional[int] = None) -> None:
        # tuple of two floats, the minimum and maximum latency in seconds for the simulated provider
        self._latency = latency
        self._random = random.Random(seed)

    
    async def _wait(self) -> float:
        """Wait for a random latency and return the delay in seconds."""

        delay = self._random.uniform(*self._latency)
        await asyncio.sleep(delay)
        return delay
    

    def _answer(self, request: LlmRequest, latency: float) -> LlmResponse:
        """Build a simulated answer for the request."""

        content = f"simulated {request.task_type.value} for document {request.document_id}"
        prompt_tokens = estimate_tokens(request.prompt)
        completion_tokens = estimate_tokens(content)
        return LlmResponse(
            request_id=request.request_id,
            provider=self.name,
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_seconds=latency,
        )


class FastProvider(SimulatedProvider):
    """Quick and always successful."""

    name = "fast"
    """
    FastProvider inherits from SimulatedProvider, which inherits from BaseProvider:
    When Python defines FastProvider, it sees name = "fast" in FastProvider body (provider_name = attrs.get("name", ""))  
    """

    async def complete(self, request: LlmRequest) -> LlmResponse:
        """Wait for a random latency delay and return the answer."""

        return self._answer(request, await self._wait())


class SlowProvider(SimulatedProvider):
    """Variable latency; a fraction of calls end in a timeout."""

    name = "slow"

    def __init__(
        self,
        latency: tuple = (0.2, 1.0),
        timeout_rate: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(latency, seed) # calls the parent class initialize
        self._timeout_rate = timeout_rate

    async def complete(self, request: LlmRequest) -> LlmResponse:
        elapsed = await self._wait()
        if self._random.random() < self._timeout_rate:
            raise ProviderTimeoutError(f"no answer after {elapsed:.2f}s")
        # Above 'if' simulates an occasional timeout:
        # self._random.random() generates a random number from 0.0 up to, but not including, 1.0.
        # If that number is less than self._timeout_rate, the condition is true. For example, if timeout_rate is 0.15, this happens about 15% of the time.
        # The provider then raises ProviderTimeoutError, with a message containing the elapsed time formatted to two decimal places, such as "no answer after 0.47s"
        return self._answer(request, elapsed)


class StrictProvider(SimulatedProvider):
    """Accepts ``max_calls`` per ``window`` seconds, then answers 429.

    The counter is protected by a ``threading.Lock`` so the provider can be
    shared by the threading strategy.
    """

    name = "strict"

    def __init__(
        self,
        max_calls: int = 5,
        window: float = 1.0,
        latency: tuple = (0.01, 0.03),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(latency, seed)
        self._max_calls = max_calls
        self._window = window
        self._stamps: list = []
        self._lock = threading.Lock()

    def _take_slot(self) -> None:
        """Take a slot for the call, or raise RateLimitError if the limit is exceeded."""

        now = time.perf_counter()
        with self._lock:
            self._stamps = [t for t in self._stamps if now - t < self._window]
            if len(self._stamps) >= self._max_calls:
                raise RateLimitError("rate limit exceeded")
            self._stamps.append(now)

    
    async def complete(self, request: LlmRequest) -> LlmResponse:
        """Take a slot, wait for a random latency delay, and return the answer."""

        self._take_slot()
        return self._answer(request, await self._wait())
