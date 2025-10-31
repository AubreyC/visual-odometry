class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


class ProcessingError(RuntimeError):
    """Raised when data processing fails."""

    pass
