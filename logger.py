import logging
import logging.config

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[%(asctime)s] %(name)s - %(levelname)s : %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {"": {"level": "INFO", "handlers": ["console"]}},
}

logging.config.dictConfig(logging_config)


def get_logger(name):
    return logging.getLogger(name)
