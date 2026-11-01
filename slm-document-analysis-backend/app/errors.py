"""Shared API error type.

Every error returned by the API has the same JSON shape::

    {"error": {"code": "<machine_readable_code>", "message": "<human readable>"}}
"""


class ApiError(Exception):
    """An error that should be returned to the client with a specific HTTP status."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
