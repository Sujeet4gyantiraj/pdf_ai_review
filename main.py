import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes.route import router
from db_files.db import init_db, close_pool

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "level":     "INFO",
            "stream":    "ext://sys.stdout",
        },
        "file": {
            "class":     "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level":     "DEBUG",
            "filename":  "app.log",
            "maxBytes":  10 * 1024 * 1024,
            "backupCount": 5,
            "encoding":  "utf-8",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — DB pool open on startup, closed on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    await init_db()      # creates pdf_requests table if not exists, opens pool
    yield
    await close_pool()   # gracefully closes all DB connections
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)
app.include_router(router)