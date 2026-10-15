"""Tests for the retry policy. BEFORE the retry.py implementation, all fails """

import unittest

from calls_coordinator.exceptions import (
    InvalidRequestError,
    ProviderTimeoutError,
    RateLimitError,
)
from calls_coordinator.retry import RetryPolicy, call_with_retry

from tests.helpers import async_mock, async_test


class RetryPolicyTests(unittest.TestCase):
    """Tests for the RetryPolicy class."""

    def test_transient_errors_are_retryable(self) -> None:
        """Check that transient errors are considered retryable by the RetryPolicy."""

        policy = RetryPolicy()
        self.assertTrue(policy.is_retryable(ProviderTimeoutError("slow")))
        self.assertTrue(policy.is_retryable(RateLimitError("quota")))

    def test_permanent_errors_are_not_retryable(self) -> None:
        """Check that permanent errors are not considered retryable by the RetryPolicy."""

        self.assertFalse(RetryPolicy().is_retryable(InvalidRequestError("bad")))

    def test_unknown_exceptions_are_not_retryable(self) -> None:
        """Check that unknown exceptions are not considered retryable by the RetryPolicy."""

        self.assertFalse(RetryPolicy().is_retryable(ValueError("bug")))

    def test_delay_grows_exponentially(self) -> None:
        """Check that the delay grows exponentially with each retry attempt."""

        policy = RetryPolicy(base_delay=1.0, max_delay=100.0)
        delays = [policy.delay_for(n) for n in (1, 2, 3)]
        self.assertEqual(delays, [1.0, 2.0, 4.0])

    def test_delay_is_capped(self) -> None:
        """Check that the delay is capped at max_delay."""

        policy = RetryPolicy(base_delay=1.0, max_delay=5.0)
        self.assertEqual(policy.delay_for(10), 5.0)

class CallWithRetryTests(unittest.TestCase):
    """
    Tests for the call_with_retry function.
    @async_test lets the regular unittest runner execute this asynchronous test.
    for test_success_on_first_attempt:
    operation = async_mock(return_value="ok") creates an awaitable mock that returns "ok".
    sleep = async_mock() creates an awaitable mock to stand in for the retry delay.
    await call_with_retry(...) runs the operation with the default retry policy and captures its result and attempt count.
    assertEqual((result, attempts), ("ok", 1)) checks that the result is "ok" and it took one attempt.
    sleep.assert_not_called() checks that no delay was needed—so no retry occurred.    
    """

    @async_test
    async def test_success_on_first_attempt(self) -> None:
        """Check that call_with_retry returns the result immediately if the operation succeeds on the first attempt."""
        
        operation = async_mock(return_value="ok")
        sleep = async_mock()
        result, attempts = await call_with_retry(operation, RetryPolicy(), sleep=sleep)
        self.assertEqual((result, attempts), ("ok", 1))
        sleep.assert_not_called()

    @async_test
    async def test_transient_error_is_retried_then_succeeds(self) -> None:
        """
        Check that call_with_retry retries the operation if it raises a transient error, and returns the result if it eventually succeeds.
        with sleep=sleep so tests pass a mock and never really wait 
        """
        
        operation = async_mock(side_effect=[ProviderTimeoutError("t"), "ok"])
        sleep = async_mock()
        policy = RetryPolicy(max_attempts=3)
        result, attempts = await call_with_retry(operation, policy, sleep=sleep)
        self.assertEqual((result, attempts), ("ok", 2))
        self.assertEqual(sleep.call_count, 1)

    @async_test
    async def test_permanent_error_is_not_retried(self) -> None:
        """Check that call_with_retry does not retry the operation if it raises a permanent error, and raises the error immediately. """

        operation = async_mock(side_effect=InvalidRequestError("bad"))
        sleep = async_mock()
        with self.assertRaises(InvalidRequestError):
            await call_with_retry(operation, RetryPolicy(max_attempts=5), sleep=sleep)
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()

    @async_test
    async def test_gives_up_after_max_attempts(self) -> None:
        """Check that call_with_retry gives up after the maximum number of attempts and raises the last transient error."""

        operation = async_mock(side_effect=ProviderTimeoutError("t"))
        sleep = async_mock()
        with self.assertRaises(ProviderTimeoutError):
            await call_with_retry(operation, RetryPolicy(max_attempts=3), sleep=sleep)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @async_test
    async def test_on_retry_callback_is_called(self) -> None:
        """Check that the on_retry callback is called with the correct arguments each time a retry occurs."""

        operation = async_mock(side_effect=[ProviderTimeoutError("t"), "ok"])
        seen = []
        await call_with_retry(
            operation,
            RetryPolicy(),
            on_retry=lambda attempt, exc, delay: seen.append((attempt, type(exc))),
            sleep=async_mock(),
        )
        self.assertEqual(seen, [(1, ProviderTimeoutError)])


