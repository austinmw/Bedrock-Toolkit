"""Logger Manager Module"""

from typing import Literal, Optional

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


class LoggerManager:
    _instance: Optional["LoggerManager"] = None
    _configured: bool = False
    _current_log_level: LogLevel = "INFO"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._configured = False
            cls._instance._current_log_level = "INFO"
        return cls._instance

    def configure_logger(self, log_level: LogLevel = "INFO", console_width: int = 160):
        console = Console(width=console_width)
        if not self._configured:
            logger.remove()
            logger.add(
                RichHandler(console=console, markup=False),
                level=log_level,
                format="<level>{level}</level>: <level>{message}</level>",
            )
            logger.info(f"Logger configured with level: {log_level}")
            self._configured = True
        elif log_level != self._current_log_level:
            logger.remove()
            logger.add(
                RichHandler(console=console, markup=False),
                level=log_level,
                format="<level>{level}</level>: <level>{message}</level>",
            )
            logger.info(f"Logger log level updated to: {log_level}")
        else:
            logger.debug(
                "Logger is already configured with the same log level. Skipping reconfiguration."
            )

        self._current_log_level = log_level

    @staticmethod
    def get_logger():
        if LoggerManager._instance is None:
            LoggerManager._instance = LoggerManager()
        return logger
