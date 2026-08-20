"""Experiment labels."""

from __future__ import annotations

from enum import Enum


class Configuration(str, Enum):
    """Supported LimberCloud spectrum configurations."""

    SINGLE = "Single"
    DOUBLE = "Double"
    TRIPLE = "Triple"

    @classmethod
    def parse(cls, value: str | "Configuration") -> "Configuration":
        """Return a configuration from a case-insensitive value."""

        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for configuration in cls:
            if normalized in {
                configuration.name.lower(),
                configuration.value.lower(),
            }:
                return configuration

        choices = ", ".join(configuration.value for configuration in cls)
        raise ValueError(f"Unknown configuration {value!r}; expected one of: {choices}")
