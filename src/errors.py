class TextOriginError(Exception):
    """Base class for expected application failures."""


class ModelUnavailableError(TextOriginError):
    pass


class ExternalServiceError(TextOriginError):
    pass


class InferenceCapacityError(TextOriginError):
    pass
