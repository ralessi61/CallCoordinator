"""
Decorators for coroutine functions: retry, timing, logging.
reference decoratori_metaclassi.py
"""
import functools
import logging
import time
from typing import Any, Awaitable, Callable, Optional, TypeAlias, TypeVar
from .retry import RetryPolicy, call_with_retry

AsyncFunc = TypeVar("AsyncFunc", bound=Callable[..., Awaitable[Any]])
OnRetry: TypeAlias = Callable[..., None]
logger = logging.getLogger(__name__)


def retry(
        policy: RetryPolicy, on_retry: Optional[OnRetry] = None
) -> Callable[[AsyncFunc], AsyncFunc]:
    """Build a decorator that retries a coroutine with ``policy``.

    Transient failures are repeated until the call succeeds or the policy
    stops. The wrapper keeps the attempts used by the last call in
    ``last_attempts``. When ``on_retry`` is set, it is called before each wait.
    """

    def decorator(func: AsyncFunc) -> AsyncFunc:
        """Retry ``func`` when a call fails with a transient error."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Run one call and store its attempt count in ``last_attempts``."""

            retries = 0  # local inside wrapper

            def count_retry(attempt: int, error: Exception, delay: float) -> None:
                """Count this retry and forward it to ``on_retry``."""

                nonlocal retries
                retries += 1
                if on_retry is not None:
                    on_retry(attempt, error, delay)

            try:
                result, _ = await call_with_retry(
                    lambda: func(*args, **kwargs), policy, count_retry
                )
            finally:
                wrapper.last_attempts = retries + 1
            return result

        wrapper.last_attempts = 0
        return wrapper

    return decorator


def timing(func: AsyncFunc) -> AsyncFunc:
    """Measure each call; the last duration is kept in ``last_duration``."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Time one call and store the elapsed seconds in ``last_duration``."""

        started = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            wrapper.last_duration = time.perf_counter() - started

    wrapper.last_duration = 0.0
    return wrapper


def logged(func: AsyncFunc) -> AsyncFunc:
    """Log start, success or failure of each call."""

    name = getattr(func, "__name__", "call")  # mocks and partials may have no name

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Log the function name at start, then success or the exception type."""

        logger.debug("calling %s", name)
        try:
            result = await func(*args, **kwargs)
        except Exception as error:
            logger.warning("%s failed: %s", name, type(error).__name__)
            raise
        logger.debug("%s succeeded", name)
        return result

    return wrapper
