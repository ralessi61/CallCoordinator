"""Test helpers built on unittest.mock, functools and asyncio.run."""

import asyncio
import functools
from unittest.mock import MagicMock


def async_test(test_method):
    """A decorator for test methods that run async code. It wraps the test method in asyncio.run() so that it can be called from a synchronous test runner.
    The test method must be an async function that takes self as the first argument."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        return asyncio.run(test_method(self, *args, **kwargs))

    return wrapper


def async_mock(side_effect=None, return_value=None):
    """A MagicMock whose calls return a coroutine, so it can be awaited.

    ``side_effect`` may be:
      * a list: one item per call, in order (exceptions are raised);
      * an exception (class or instance): raised on every call;
      * a function: called with the arguments; it may return a value, raise,
        or be an ``async def``.
    Without ``side_effect`` the mock returns ``return_value``.

    async_mock(...) creates a mock function that can be awaited, while letting a test specify what each call should return or raise. 
    Inside it, call is the asynchronous behavior assigned to the mock.

    In other words:
    When the test awaits the mock, call chooses an outcome:

    If side_effect is a list, next(items) takes the next item, so successive calls can have different outcomes. For example, [1, 2] returns 1, then 2.
    If side_effect is an exception instance or exception class, it sets that as the outcome, which will be raised below.
    If side_effect is callable, it calls it with the mock’s arguments. If the result is awaitable (such as a coroutine), it awaits it, then returns its result directly. 
    Exceptions raised by that callable propagate.
    Otherwise, it returns the configured return_value.
    After picking an outcome from the list or exception branch, it checks whether that outcome is an exception instance or exception class. 
    If so, raise outcome raises it; otherwise it returns the outcome.

    In short:
    it supports a sequence of outcomes, a repeated exception, a callback, or a default return value.
    """
    mock = MagicMock()
    items = iter(side_effect) if isinstance(side_effect, list) else None

    async def call(*args, **kwargs):
        if items is not None:
            outcome = next(items)
        elif isinstance(side_effect, BaseException) or (
            isinstance(side_effect, type) and issubclass(side_effect, BaseException)
        ):
            outcome = side_effect
        elif callable(side_effect):
            outcome = side_effect(*args, **kwargs)
            if hasattr(outcome, "__await__"):
                outcome = await outcome
            return outcome
        else:
            return return_value
        if isinstance(outcome, BaseException) or (
            isinstance(outcome, type) and issubclass(outcome, BaseException)
        ):
            raise outcome
        return outcome

    mock.side_effect = call
    return mock
