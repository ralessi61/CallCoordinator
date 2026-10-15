"""
Common provider interface with automatic registration through a metaclass.
Developed starting from concepts in decoratori_metaclassi.py
"""

from typing import Any, Dict, List, Type

from ..exceptions import UnknownProviderError
from ..models import LlmRequest, LlmResponse


class ProviderMeta(type):
    """
    Metaclass that registers every provider class the moment it is defined.
    Is a class that controls how other classes are created. In Python, most classes are created by the built-in metaclass type
    A class that declares a non-empty ``name`` is added to ``registry``. Adding
    a provider therefore means writing one class: no manual registry to edit and
    no change to the coordinator.
    registry:
    Dict[...] a dictionary type.
    str its keys are strings, such as "openai".
    Type["BaseProvider"] says its values are classes that are BaseProvider or subclasses—not provider instances.
    "BaseProvider" is in quotes because BaseProvider is defined later in the file. 
    """

    registry: Dict[str, Type["BaseProvider"]] = {}

    def __new__(mcs, name: str, bases: tuple, attrs: Dict[str, Any]) -> type:
        """Create a new provider class and register it if it has a non-empty ``name``."""

        new_class = super().__new__(mcs, name, bases, attrs)
        provider_name = attrs.get("name", "")
        if provider_name:  # base or intermediate classes have no name: not registered
            if provider_name in ProviderMeta.registry:
                raise TypeError(f"provider name {provider_name!r} is already registered")
            ProviderMeta.registry[provider_name] = new_class  # type: ignore[assignment]
        return new_class


class BaseProvider(metaclass=ProviderMeta):
    """
    Every provider sets a ``name`` and implements ``complete``.
    Python reaches this line, it asks ProviderMeta to create BaseProvider. 
    The metaclass’s __new__ method runs, but this base class has name = "", so it is not added to the provider registry. 
    Provider subclasses that define a non-empty name are registered automatically.
    """

    name = ""

    async def complete(self, request: LlmRequest) -> LlmResponse:
        """Answer one request.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
            TransientError: If the failure is temporary and the call may succeed if retried.
            PermanentError: If the request cannot succeed, even if retried.
        """
        raise NotImplementedError


def create_provider(name: str) -> BaseProvider:
    """Instantiate a registered provider by name.

    Raises:
        UnknownProviderError: if the name is not registered.
    """

    if name not in ProviderMeta.registry:
        known = ", ".join(sorted(ProviderMeta.registry)) or "none"
        raise UnknownProviderError(f"unknown provider {name!r} (registered: {known})")
    return ProviderMeta.registry[name]()


def registered_providers() -> List[str]:
    """Names of all registered providers."""

    return sorted(ProviderMeta.registry)
