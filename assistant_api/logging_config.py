import logging
import re
from pathlib import Path

try:
    from .paths import LOG_FILE_PATH
except ImportError:
    from paths import LOG_FILE_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

TELEGRAM_BOT_TOKEN_PATTERN = re.compile(r"/bot(\d+):([A-Za-z0-9_-]+)")
OPENAI_KEY_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]+)")


class RedactingFilter(logging.Filter):
    """Redacts secrets from log messages before they reach handlers."""

    @staticmethod
    def _sanitize(value: str) -> str:
        sanitized = TELEGRAM_BOT_TOKEN_PATTERN.sub(r"/bot\1:<redacted>", value)
        sanitized = OPENAI_KEY_PATTERN.sub("<redacted-openai-key>", sanitized)
        return sanitized

    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize the fully formatted message so args-based loggers like httpx
        # do not leak secrets through deferred formatting.
        record.msg = self._sanitize(record.getMessage())
        record.args = ()

        return True


def setup_logging(debug: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()

    if getattr(setup_logging, "_configured", False):
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    formatter = logging.Formatter(LOG_FORMAT)
    redacting_filter = RedactingFilter()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redacting_filter)
    root_logger.addHandler(console_handler)

    file_path = Path(log_file) if log_file else LOG_FILE_PATH
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redacting_filter)
        root_logger.addHandler(file_handler)
    except OSError:
        root_logger.warning("Не удалось настроить файловое логирование: %s", file_path)

    setup_logging._configured = True
