"""Exceptions: transient errors can be retried, permanent errors cannot."""


class CoordinatorError(Exception):
    """Base class for every error raised by this library."""


class PermanentError(CoordinatorError):
    """Repeating the call is useless."""


class TransientError(CoordinatorError):
    """Repeating the call may work."""


class InvalidRequestError(PermanentError):
    """The provider rejected the request content."""


class UnknownProviderError(PermanentError):
    """The request names a provider that is not registered."""


class ProviderTimeoutError(TransientError):
    """The provider did not answer in time."""


class RateLimitError(TransientError):
    """The provider refused the call because of its quota."""
