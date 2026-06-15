from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

__all__ = ["logger"]
