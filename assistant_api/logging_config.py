import logging
from pathlib import Path

from paths import LOG_FILE_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(debug: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()

    if getattr(setup_logging, "_configured", False):
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    formatter = logging.Formatter(LOG_FORMAT)
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_path = Path(log_file) if log_file else LOG_FILE_PATH
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError:
        root_logger.warning("Не удалось настроить файловое логирование: %s", file_path)

    setup_logging._configured = True
