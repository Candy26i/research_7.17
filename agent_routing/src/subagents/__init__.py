"""Subagent package.

Schema objects are loaded lazily so lightweight utilities such as JSONL prompt
export do not require pydantic at import time.
"""

_SCHEMA_EXPORTS = {
    "AgentKind",
    "ExtractorOutput",
    "ReasonerOutput",
    "VerifierOutput",
    "SCHEMA_REGISTRY",
}


def __getattr__(name):
    if name in _SCHEMA_EXPORTS:
        from . import schemas
        return getattr(schemas, name)
    raise AttributeError(name)


__all__ = sorted(_SCHEMA_EXPORTS)
