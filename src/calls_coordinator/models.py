"""Pydantic models for requests, responses and results."""

import math
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError, model_validator


# TaskType is an enumeration whose values are strings
class TaskType(str, Enum):
    """The three calls made for every document."""

    CLASSIFICATION = "classification"
    SUMMARY = "summary"
    EXTRACTION = "extraction"


class CallStatus(str, Enum):
    """The status of an http call to an LLM."""

    SUCCESS = "success"
    FAILED = "failed"


# The three dots are the Python value Ellipsis. In Field(...) they mean the field is required: it has no default.
class LlmRequest(BaseModel):
    """The request to the LLM."""
    request_id: str = Field(min_length=1, description="The unique identifier for the request.")
    document_id: str = Field(min_length=1, description="The unique identifier for the document.")
    task_type: TaskType = Field(..., description="The type of task to be performed.")
    prompt: str = Field(..., description="The prompt to be sent to the LLM.")
    provider: str = Field(..., description="The provider to be used for the request.")
    temperature: Optional[float] = Field(0.7, description="The temperature to be used for the LLM request.")

    @model_validator(mode="before")
    @classmethod
    def normalise_provider(cls, data):
        """Normalise the provider name to lower case."""

        if isinstance(data, dict) and isinstance(data.get("provider"), str):
            data["provider"] = data["provider"].lower()
        return data

    # here no @classmethod because we want to validate the instance, not the class
    @model_validator(mode="after")
    def check_content(self):
        """Check that the prompt and provider are not empty and that the temperature is in range."""

        if not self.prompt.strip():
            raise ValueError("prompt is empty")
        if not self.provider.strip():
            raise ValueError("provider name is empty")
        if self.temperature is None or not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"LLM call temperature {self.temperature} is out of range: must be between 0 and 2"
            )
        return self

    # ge=0 rejects a negative integer; ge=0.0 rejects a negative float.


class TokenUsage(BaseModel):
    """The token usage for a request."""

    prompt_tokens: int = Field(ge=0, description="The number of tokens in the prompt.")
    completion_tokens: int = Field(ge=0, description="The number of tokens in the completion.")
    total_tokens: int = Field(ge=0, description="The total number of tokens used.")

    @model_validator(mode="after")
    def total_matches_parts(self) -> "TokenUsage":
        """Check that the total tokens is equal to the sum of prompt and completion tokens."""

        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"total_tokens is {self.total_tokens} but prompt + completion is {expected}"
            )
        return self


class LlmResponse(BaseModel):
    """The response from the LLM."""

    request_id: str = Field(..., description="The unique identifier for the request.")
    provider: str = Field(..., description="The provider that was used for the request.")
    content: str = Field(..., description="The content of the response from the LLM.")
    # Optional[str] means the value may be a string or None. None inside Field is the default
    usage: Optional[TokenUsage] = Field(None, description="The token usage for the request.")
    latency_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def content_not_empty(self) -> "LlmResponse":
        """Check that the response content is not empty."""

        if not self.content.strip():
            raise ValueError("content is empty")
        return self


class CallResult(BaseModel):
    """The result of a call to the LLM."""

    request_id: str = Field(..., description="The unique identifier for the request.")
    document_id: str = Field(..., description="The unique identifier for the document.")
    task_type: TaskType = Field(..., description="The type of task that was performed.")
    status: CallStatus = Field(..., description="The status of the call to the LLM.")
    attempts: int = Field(ge=1, description="The number of attempts made to call the LLM.")
    duration_seconds: float = Field(ge=0, description="The duration of the call in seconds.")
    response: Optional[LlmResponse] = Field(None, description="The provider response after the call.")
    error_type: Optional[str] = Field(None, description="The type of error if the call failed.")
    error_message: Optional[str] = None
    transient: Optional[bool] = Field(None, description="Whether the error is transient or not.")

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> "CallResult":
        if self.status is CallStatus.SUCCESS:
            if self.response is None or self.error_type is not None:
                raise ValueError("a successful result needs a response and no error")
        elif self.response is not None or not self.error_type:
            raise ValueError("a failed result needs an error_type and no response")
        return self


class BatchMetrics(BaseModel):
    """Metrics for a batch of calls to the LLM."""

    total: int = Field(ge=0, description="The total number of calls made.")
    successful_: int = Field(ge=0, description="The number of successful calls.")
    failed: int = Field(ge=0, description="The number of failed calls.")
    total_retries: int = 0
    wall_seconds: float = Field(ge=0.0, description="The total wall time in seconds for the batch of calls.")
    avg_call_seconds: float = 0.0
    max_call_seconds: float = 0.0
    calls_per_second: float = 0.0
    estimated_tokens: int = 0
    errors_by_type: Dict[str, int] = Field(default_factory=dict)
    # default_factory=dict tells Pydantic to call dict() when the field is omitted, so the stored value is {}


class BatchResult(BaseModel):
    """The result of a batch of calls to the LLM."""

    results: List[CallResult] = Field(..., description="The results of the calls in the batch.")
    metrics: BatchMetrics = Field(..., description="The metrics for the batch of calls.")

    # class methods
    def by_request_id(self) -> Dict[str, CallResult]:
        """Map each request_id to its result."""

        return {r.request_id: r for r in self.results}

    def by_document(self) -> Dict[str, List[CallResult]]:
        """Group results by document_id of CallResult. One document has one result per task."""

        grouped: Dict[str, List[CallResult]] = {}
        for result in self.results:
            grouped.setdefault(result.document_id, []).append(result)
        return grouped

    def failures(self) -> List[CallResult]:
        """Return only the results whose status is FAILED."""

        return [r for r in self.results if r.status is CallStatus.FAILED]


def validate_batch(requests: List[LlmRequest]) -> None:
    """
    Validate a batch of requests to the LLM.
    Raises a ValueError if any request_id is duplicated in the batch.
    """

    seen: Dict[str, int] = {}
    for index, request in enumerate(requests):
        if request.request_id in seen:
            raise ValueError(
                f"duplicate request_id {request.request_id!r} at positions "
                f"{seen[request.request_id]} and {index}: ids must be unique in a batch"
            )
        seen[request.request_id] = index


# estimate_tokens is a public function of the module calls_coordinator.models
def estimate_tokens(text: str) -> int:
    """The estimated number of tokens for a given prompt, using a simple heuristic of 1 token per 4 characters."""

    return max(1, math.ceil(len(text) / 4))


def _first_message(error: ValueError) -> str:
    """Return the first error message from a Pydantic validation error.

    A plain ``ValueError`` is returned as text. A ``ValidationError`` is
    formatted as ``field.path: message``, for example ``prompt: Field required``.
    """

    # ValidationError is a ValueError with an errors() method. isinstance
    # tells the type checker that errors() exists, so it is not None.
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return f"{where}: {first['msg']}"
    return str(error)


def parse_requests(raw: Any) -> List[LlmRequest]:
    """
    Parse a JSON list of requests to the LLM into a list of LlmRequest objects.
    """

    if not isinstance(raw, list):
        raise ValueError("requests must be a JSON list of objects")
    requests: List[LlmRequest] = []
    problems: List[str] = []
    for position, item in enumerate(raw):
        try:
            requests.append(LlmRequest.model_validate(item))
        except ValueError as error:
            problems.append(f"item {position}: {_first_message(error)}")
    if problems:
        raise ValueError("invalid requests:\n" + "\n".join(problems))
    validate_batch(requests)
    return requests
