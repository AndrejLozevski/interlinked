import sys
import logging

log = logging.getLogger(__name__)


MSG_SUCCESS = "Script completed!"
MSG_ERROR   = "Error occured!"
MSG_WARNING = "Warning triggered!"

class Success():
    def __init__(self, message=MSG_SUCCESS, *args):
        log.info(message, *args)
        return sys.exit(0)

class Error():
    def __init__(self, message=MSG_ERROR, *args, error=None):
        if error is not None:
            name = error.__name__ if isinstance(error, type) else str(error)
            message = f"{name}: {message}"
        log.error(message, *args)
        return sys.exit(1)

class Warning():
    def __init__(self, message=MSG_WARNING, *args, warning=None):
        if warning is not None:
            name = warning.__name__ if isinstance(warning, type) else str(warning)
            message = f"{name}: {message}"
        log.warning(message, *args)
        return

