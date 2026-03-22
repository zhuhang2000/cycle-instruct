import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str | None = None,
    *,
    log_dir: str | Path = "logs",
    file_prefix: str | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """创建一个同时输出到文件和控制台的通用 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if not log_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = file_prefix or name
        log_file = str(Path(log_dir) / f"{prefix}_{ts}.log")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info(f"日志文件: {log_path}")
    return logger
