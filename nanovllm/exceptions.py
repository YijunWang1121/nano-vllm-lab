class NanoVLLMError(Exception):
    """Base for engine-raised errors that fail a single request/step rather
    than the whole process."""


class RequestTooLargeError(NanoVLLMError):
    """Raised by Scheduler.add() when a request can never be served by this
    engine's configured limits, no matter how the KV-cache pool drains."""


class EngineOOMError(NanoVLLMError):
    """Raised by ModelRunner.run() when a real CUDA OOM occurs during a
    forward pass or sampling call; caught by LLMEngine's step methods."""
