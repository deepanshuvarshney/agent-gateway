"""Errors returned to clients in an OpenAI-style envelope."""


class GatewayError(Exception):
    """A request failed in a way the client can act on."""

    def __init__(self, message: str, status_code: int, error_type: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code

    def to_body(self) -> dict[str, dict[str, str]]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        }


class ProviderError(Exception):
    """An upstream model call failed."""

    def __init__(self, message: str, status_code: int, retryable: bool) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
