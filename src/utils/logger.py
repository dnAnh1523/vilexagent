import sys
from loguru import logger

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - {message}",
    level="DEBUG"
)

logger.add(
    "E:/vilexagent/logs/vilexagent.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

logger.add(
    "logs/ragas_tokens.log",
    rotation="5 MB",
    level="DEBUG",
    filter=lambda record: "RAGAS LLM tokens" in record["message"]
)