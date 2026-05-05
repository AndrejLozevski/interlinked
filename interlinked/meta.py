import sys
import logging

log = logging.getLogger(__name__)


class Error(SystemExit):
    def __init__(self, message, *args, error=None):
        if error is not None:
            name = error.__name__ if isinstance(error, type) else str(error)
            message = f"{name}: {message}"
        log.error(message, *args)
        super().__init__(1)

class Exception(Exception):
    def __init__(self, message, *args, exception=None):
        if exception is not None:
            name = exception.__name__ if isinstance(exception, type) else str(exception)
            message = f"{name}: {message}"
        log.error(message, *args)
        super().__init__()

