import unittest
from calls_coordinator.exceptions import (
    CoordinatorError,
    PermanentError,
    RateLimitError,
    TransientError,
)


class TestExceptions(unittest.TestCase):
    def test_rate_limit_is_transient(self):
        error = RateLimitError("quota")
        # Check that RateLimitError is a TransientError and not a PermanentError
        self.assertEqual(isinstance(error, TransientError), True)
        self.assertEqual(isinstance(error, PermanentError), False)


    def test_rate_limit_message(self):
        error = RateLimitError("quota")
        self.assertEqual(str(error), "quota")


    def test_rate_limit_inheritance(self):
        error = RateLimitError("quota")
        self.assertTrue(issubclass(RateLimitError, TransientError))
        self.assertFalse(issubclass(RateLimitError, PermanentError))


    def test_transient_error_inheritance(self):
        # Check that TransientError is a subclass of Exception and not a subclass of PermanentError
        # so for others tests
        self.assertTrue(issubclass(TransientError, Exception))
        self.assertFalse(issubclass(TransientError, PermanentError))


    def test_permanent_error_inheritance(self):
        self.assertTrue(issubclass(PermanentError, Exception))
        self.assertFalse(issubclass(PermanentError, TransientError))


    def test_coordinator_error_inheritance(self):
        self.assertTrue(issubclass(TransientError, Exception))
        self.assertTrue(issubclass(PermanentError, Exception))
        self.assertTrue(issubclass(TransientError, CoordinatorError))
        self.assertTrue(issubclass(PermanentError, CoordinatorError))


    def test_coordinator_error_message(self):
        error = CoordinatorError("coordinator error")
        self.assertEqual(str(error), "coordinator error")


    def test_permanent_error_message(self):
        error = PermanentError("permanent error")
        self.assertEqual(str(error), "permanent error")


    def test_transient_error_message(self):
        error = TransientError("transient error")
        self.assertEqual(str(error), "transient error")

    def test_invalid_request_error_inheritance(self):
        # Check that InvalidRequestError is a subclass of PermanentError and not a subclass of TransientError
        from calls_coordinator.exceptions import InvalidRequestError
        self.assertTrue(issubclass(InvalidRequestError, PermanentError))
        self.assertFalse(issubclass(InvalidRequestError, TransientError))

        