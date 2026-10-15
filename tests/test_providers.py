import unittest

from calls_coordinator.exceptions import (
    ProviderTimeoutError,
    RateLimitError,
    UnknownProviderError,
)
from calls_coordinator.models import LlmRequest, LlmResponse
from calls_coordinator.providers.base import ProviderMeta
from calls_coordinator.providers.base import (
    BaseProvider,
    create_provider,
    registered_providers,
)
from calls_coordinator.providers.simulated import (
    FastProvider,
    SlowProvider,
    StrictProvider,
)

from tests.helpers import async_test


def request(i: int = 1) -> LlmRequest:
    """ Helper function to create a sample LLMRequest for testing purposes."""

    return LlmRequest(
        request_id=f"r{i}",
        document_id="d",
        task_type="summary",
        prompt="bonjour",
        provider="fast",
    )


class RegistryTests(unittest.TestCase):
    """ Tests for the provider registry and automatic registration of subclasses."""

    def test_simulated_providers_are_registered_automatically(self) -> None:
        """ Ensure that the simulated providers are registered in the correct order."""

        self.assertEqual(registered_providers()[:3], ["fast", "slow", "strict"])

    def test_new_subclass_registers_itself(self) -> None:
        """ 
        Ensure that a new subclass of BaseProvider registers itself in the registry.
        pragma: exclude that line from coverage reports—usually because the line is intentionally unreachable or impractical to test
        """
        
        class EchoProvider(BaseProvider):
            name = "echo-test"

            async def complete(self, request):  # pragma: no cover - never called
                raise NotImplementedError

        try:
            self.assertIn("echo-test", registered_providers())
            self.assertIsInstance(create_provider("echo-test"), EchoProvider)
        finally:
            # Clean up the registry to avoid side effects on other tests
            del ProviderMeta.registry["echo-test"]

    
    def test_duplicate_name_is_rejected(self) -> None:
        """ Ensure that attempting to register a provider with a duplicate name raises a TypeError."""

        with self.assertRaises(TypeError):
            class Clash(BaseProvider):
                name = "fast"

                async def complete(self, request):  # pragma: no cover
                    raise NotImplementedError

    
    def test_unknown_provider(self) -> None:
        """ Ensure that requesting a provider that is not registered raises an UnknownProviderError."""

        with self.assertRaises(UnknownProviderError) as ctx:
            create_provider("missing")
        self.assertIn("registered:", str(ctx.exception))

    
    
    def test_abstract_class_without_name_is_not_registered(self) -> None:
        """ Ensure that an abstract subclass of BaseProvider without a name is not registered."""

        before = list(registered_providers())

        class Helper(BaseProvider):
            async def complete(self, request):  # pragma: no cover
                raise NotImplementedError

        self.assertEqual(registered_providers(), before)


class SimulatedBehaviourTests(unittest.TestCase):
    """ Tests for the simulated provider behaviors, including latency, timeouts, and rate limiting."""
    
    @async_test
    async def test_fast_provider_answers_for_the_right_request(self) -> None:
        """ 
        Ensure that the FastProvider returns a response with the correct request_id.
        @async_test is a decorator that allows the test method to be run as an asynchronous test case.
        """
        
        response = await FastProvider(latency=(0, 0)).complete(request(7))
        self.assertIsInstance(response, LlmResponse)
        self.assertEqual(response.request_id, "r7")


    @async_test
    async def test_slow_provider_can_time_out(self) -> None:
        """ Ensure that the SlowProvider can raise a ProviderTimeoutError when the timeout rate is set to 1.0 (always time out)."""
        provider = SlowProvider(latency=(0, 0), timeout_rate=1.0)
        with self.assertRaises(ProviderTimeoutError):
            await provider.complete(request())

    
    @async_test
    async def test_slow_provider_can_succeed(self) -> None:
        """ Ensure that the SlowProvider can successfully complete a request when the timeout rate is set to 0.0 (never time out)."""
        
        provider = SlowProvider(latency=(0, 0), timeout_rate=0.0)
        self.assertEqual((await provider.complete(request())).provider, "slow")

    @async_test
    async def test_strict_provider_rate_limits(self) -> None:
        """ Ensure that the StrictProvider enforces rate limiting correctly. """

        provider = StrictProvider(max_calls=2, window=5.0, latency=(0, 0))
        await provider.complete(request(1))
        await provider.complete(request(2))
        with self.assertRaises(RateLimitError):
            await provider.complete(request(3))
