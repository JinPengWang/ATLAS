"""Registry module for algorithms and problems.

Provides decorator-based auto-registration so that new algorithms/problems
are available simply by importing their modules.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type


class _Registry:
    """Generic registry mapping string names to classes."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._registry: Dict[str, Type] = {}

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------
    def __call__(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> Callable:
        """Return a decorator that registers *cls* under *name* (and any *aliases*).

        Args:
            name: The primary lookup key (usually lowercase, e.g. ``'pso'``).
            aliases: Optional list of additional lookup keys (e.g. ``['pso_nfe']``).

        Returns:
            A class decorator.
        """

        def decorator(cls: Type) -> Type:
            keys = [name.lower()]
            if aliases:
                keys.extend([a.lower() for a in aliases])

            for key in keys:
                if key in self._registry and self._registry[key] is not cls:
                    # Allow idempotent re-registration of the same class
                    raise ValueError(
                        f"{self._kind} '{key}' is already registered "
                        f"({self._registry[key].__name__}). "
                        f"Cannot register {cls.__name__}."
                    )
                self._registry[key] = cls
            return cls

        return decorator


    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get(self, name: str) -> Type:
        """Retrieve a registered class by *name*.

        Args:
            name: Lookup key (case-insensitive).

        Returns:
            The registered class.

        Raises:
            KeyError: If *name* is not found.
        """
        key = name.lower()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise KeyError(
                f"{self._kind} '{key}' not found. Available: {available}"
            )
        return self._registry[key]

    def list_all(self) -> list:
        """Return a sorted list of all registered names."""
        return sorted(self._registry)

    def items(self):
        """Return (name, cls) pairs."""
        return list(self._registry.items())

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._registry


# ---- Global singleton registries ------------------------------------
AlgorithmRegistry = _Registry("Algorithm")
ProblemRegistry = _Registry("Problem")


# Convenience aliases
register_algorithm = AlgorithmRegistry
register_problem = ProblemRegistry


def get_algorithm(name: str) -> Type:
    """Get an algorithm class by name.

    Args:
        name: Algorithm identifier (case-insensitive).

    Returns:
        The algorithm class.

    Raises:
        KeyError: If not registered.
    """
    return AlgorithmRegistry.get(name)


def list_algorithms() -> list:
    """Return sorted list of registered algorithm names."""
    return AlgorithmRegistry.list_all()


def get_problem(name: str) -> Type:
    """Get a problem class by name.

    Args:
        name: Problem identifier (case-insensitive).

    Returns:
        The problem class.

    Raises:
        KeyError: If not registered.
    """
    return ProblemRegistry.get(name)


def list_problems() -> list:
    """Return sorted list of registered problem names."""
    return ProblemRegistry.list_all()
