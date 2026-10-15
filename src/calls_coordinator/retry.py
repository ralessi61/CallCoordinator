"""Retry policy with backoff."""

from dataclasses import dataclass
# loads the dataclass decorator from the standard-library module dataclasses
# @dataclass saves you from writing the constructor.
from .exceptions import InvalidRequestError, ProviderTimeoutError, TransientError
import asyncio

@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 30.0

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, TransientError)

    def delay_for(self, attempt: int) -> float:
        return min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))


async def call_with_retry(operation, policy, on_retry=None, sleep=asyncio.sleep):
    """
    Call an asynchronous operation with retries according to the given policy.
    If that call fails with a transient error it waits and tries again, until it succeeds or the policy says to stop.
    """
    attempt = 1
    while True:
        try:
            return await operation(), attempt
        except Exception as error:
            if attempt >= policy.max_attempts or not policy.is_retryable(error):
                raise
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, error, delay)
            await sleep(delay)
            attempt += 1   