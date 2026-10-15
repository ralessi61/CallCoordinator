"""Test the decorators."""

import logging
import unittest

from calls_coordinator.decorators import logged, retry, timing
from calls_coordinator.exceptions import ProviderTimeoutError
from calls_coordinator.retry import RetryPolicy

from tests.helpers import async_mock, async_test

QUICK = RetryPolicy(max_attempts=3, base_delay=0.0)


async def sample(value: int) -> int:
    """Sample docstring."""
    return value


class MetadataTests(unittest.TestCase):
    def test_wraps_preserves_name_and_doc(self) -> None:
        """Test if decorator has the name and docstring samples"""

        for decorator in (timing, logged, retry(QUICK)):
            wrapped = decorator(sample)
            self.assertEqual(wrapped.__name__, "sample")
            self.assertEqual(wrapped.__doc__, "Sample docstring.")

    def test_decorators_compose_in_any_order(self) -> None:
        """Stacking retry, timing, and logged in either order keeps the name 'sample'."""

        first = retry(QUICK)(timing(logged(sample)))
        second = logged(timing(retry(QUICK)(sample)))
        self.assertEqual(first.__name__, second.__name__, "sample")


class BehaviourTests(unittest.TestCase):

    @async_test
    async def test_retry_decorator_repeats_and_counts(self) -> None:
        """A failing call is repeated until it succeeds, and ``last_attempts`` counts them."""

        operation = async_mock(side_effect=[ProviderTimeoutError("t"), ProviderTimeoutError("t"), 5])
        operation.__name__ = "operation"
        wrapped = retry(QUICK)(operation)
        self.assertEqual(await wrapped(), 5)
        self.assertEqual(wrapped.last_attempts, 3)

    @async_test
    async def test_timing_records_duration(self) -> None:
        """``last_duration`` is zero or more after a successful call."""

        wrapped = timing(sample)
        self.assertEqual(await wrapped(3), 3)
        self.assertGreaterEqual(wrapped.last_duration, 0.0)

    @async_test
    async def test_logged_does_not_write_arguments(self) -> None:
        """Debug logs name the function and never include its arguments."""

        wrapped = logged(sample)
        with self.assertLogs("calls_coordinator.decorators", level=logging.DEBUG) as logs:
            await wrapped(12345)
        self.assertNotIn("12345", "\n".join(logs.output))

    @async_test
    async def test_logged_reports_failure_type_only(self) -> None:
        """A failure log names the exception type and omits the message."""

        async def failing(secret: str) -> None:
            """Raise a timeout whose message contains a secret."""

            raise ProviderTimeoutError("contains sensible-value")

        with self.assertLogs("calls_coordinator.decorators", level=logging.WARNING) as logs:
            with self.assertRaises(ProviderTimeoutError):
                await logged(failing)("sensible-value")
        output = "\n".join(logs.output)
        self.assertIn("ProviderTimeoutError", output)
        self.assertNotIn("sensible-value", output)
