import unittest

from pydantic import ValidationError  # type: ignore[import-not-found]

from calls_coordinator.models import (
    CallResult,
    CallStatus,
    LlmRequest,
    LlmResponse,
    TaskType,
    TokenUsage,
    parse_requests,
    validate_batch,
)


# **changes collects every keyword argument into one dictionary named changes
def make_request(**changes):
    """ Local tests helper, return a valid LlmRequest, built from a dictionary with optional changes to the default values."""
    data = dict(
        request_id="r1",
        document_id="d1",
        task_type="summary",
        prompt="hello",
        provider="fast",
    )
    data.update(changes)
    # above writes over those entries with the keyword arguments passed to make_reques
    return LlmRequest(**data)


class RequestValidationTests(unittest.TestCase):
    """Tests for the LlmRequest model and related functions."""

    def test_valid_request(self) -> None:
        """Check that a valid LlmRequest can be created."""

        request = make_request()
        self.assertEqual(request.task_type, TaskType.SUMMARY)
        self.assertEqual(request.provider, "fast")

    def test_blank_prompt_is_rejected_with_clear_message(self) -> None:
        """Check that a blank prompt raises a ValidationError with a clear message."""

        with self.assertRaises(ValidationError) as ctx:
            make_request(prompt="   ")
        self.assertIn("prompt is empty", str(ctx.exception))

    def test_temperature_out_of_range(self) -> None:
        """Check that a temperature outside the range 0-2 raises a ValidationError with a clear message."""

        #the test pass if make_request(...) raises ValidationError
        with self.assertRaises(ValidationError) as ctx:
            make_request(temperature=3)
        #then tests if the exception message includes the text "between 0 and 2"
        #not a good approach, better codes mainframe style for easy frontend translation
        self.assertIn("between 0 and 2", str(ctx.exception))

    def test_provider_name_is_lowercased(self) -> None:
        """Check that the provider name is lowercased by the model validator."""

        self.assertEqual(make_request(provider="FAST").provider, "fast")

    def test_duplicate_ids_in_batch(self) -> None:
        """Check that validate_batch raises a ValueError if there are duplicate request_ids in the batch. """

        with self.assertRaises(ValueError) as ctx:
            validate_batch([make_request(), make_request()])
        self.assertIn("duplicate request_id 'r1'", str(ctx.exception))

    def test_parse_requests_reports_every_bad_item(self) -> None:
        """Check that parse_requests raises a ValueError with a message that includes every bad item in the list."""

        raw = [
            dict(request_id="a", document_id="d", task_type="summary", prompt="ok"),
            dict(request_id="b", document_id="d", task_type="nope", prompt="ok"),
            dict(request_id="c", document_id="d", task_type="summary", prompt=" "),
        ]
        with self.assertRaises(ValueError) as ctx:
            parse_requests(raw)
        message = str(ctx.exception)
        self.assertIn("item 1", message)
        self.assertIn("item 2", message)

    def test_parse_requests_needs_a_list(self) -> None:
        """Check that parse_requests raises a ValueError if the input is not a list."""
        
        with self.assertRaises(ValueError):
            parse_requests({"not": "a list"})


class ResponseValidationTests(unittest.TestCase):
    """"Tests for the LlmResponse and TokenUsage models."""

    def test_token_total_must_match(self) -> None:
        """Check that TokenUsage raises a ValidationError if total_tokens does not equal prompt_tokens + completion_tokens."""

        with self.assertRaises(ValidationError) as ctx:
            TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=9)
        # row 75 of models.py, assertIn it checks that the validation error’s text includes
        self.assertIn("prompt + completion is 7", str(ctx.exception))

    def test_empty_answer_is_rejected(self) -> None:
        """Check that LlmResponse raises a ValidationError if the content is empty or whitespace."""

        with self.assertRaises(ValidationError):
            LlmResponse(
                request_id="r1", provider="fast", content=" ",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                latency_seconds=0.1,
            )

    def test_valid_response(self) -> None:
        """Check that a valid LlmResponse can be created."""

        response = LlmResponse(
            request_id="r1",
            provider="fast",
            content="answer",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_seconds=0.1,
        )
        self.assertEqual(response.content, "answer")

    @staticmethod
    def _result(**changes):
        """ Local tests helper, return a valid CallResult, built from a dictionary with optional changes to the default values."""

        data = dict(
            request_id="r1", document_id="d1", task_type=TaskType.SUMMARY,
            status=CallStatus.FAILED, attempts=1, duration_seconds=0.1,
            error_type="ProviderTimeoutError", error_message="slow",
        )
        data.update(changes)
        return CallResult(**data)

    def test_failed_result_needs_error(self) -> None:
        """Check that a failed CallResult must have an error_type , error_message optional."""

        with self.assertRaises(ValidationError):
            self._result(error_type=None)

    def test_success_result_has_response_and_no_error(self) -> None:
        """A successful CallResult requires a response and must not have an error_type."""

        response = LlmResponse(
            request_id="r1", provider="fast", content="answer", latency_seconds=0.1,
        )
        result = self._result(
            status=CallStatus.SUCCESS, response=response, error_type=None,
        )
        self.assertEqual(result.status, CallStatus.SUCCESS)
        self.assertIs(result.response, response)
        self.assertIsNone(result.error_type)

    def test_success_without_response_is_rejected(self) -> None:
        """A successful CallResult without a response is invalid."""

        with self.assertRaises(ValidationError):
            self._result(status=CallStatus.SUCCESS, error_type=None)

    # test_ lets unittest discover it automatically
    def test_valid_failed_result(self) -> None:
        """ 
        Check that a valid failed CallResult can be created.
        self._result() calls the helper, which builds a CallResult 
        with status=CallStatus.FAILED and an error type/message.
        .status reads the result’s status.
        assertEqual(..., CallStatus.FAILED) passes if the status is FAILED; 
        otherwise the test fails.
        """
        self.assertEqual(self._result().status, CallStatus.FAILED)
