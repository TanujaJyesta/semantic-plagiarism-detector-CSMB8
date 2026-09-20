"""Expected service errors that the HTTP layer can present safely."""


class ConfigurationError(RuntimeError):
    """A required local configuration value is absent or invalid."""


class ExternalServiceError(RuntimeError):
    """A provider request failed after provider-specific handling."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class DocumentProcessingError(ValueError):
    """The submitted file cannot be safely converted into meaningful text."""


class ModelUnavailableError(RuntimeError):
    """The local SBERT model could not be initialized."""
