"""
s_db.py

PostgreSQL database layer for request logging.

Table: pdf_requests
  id                SERIAL PRIMARY KEY
  request_id        TEXT NOT NULL          -- 8-char UUID prefix from the request
  pdf_name          TEXT NOT NULL          -- original uploaded filename
  pdf_size_bytes    INTEGER                -- raw file size in bytes
  total_pages       INTEGER                -- total pages in the PDF
  pages_analysed    INTEGER                -- pages actually processed
  input_tokens      INTEGER                -- total prompt tokens across all LLM calls
  output_tokens     INTEGER                -- total completion tokens across all LLM calls
  total_tokens      INTEGER                -- input + output
  completion_time_s NUMERIC(10,3)          -- wall-clock seconds for the full request
  endpoint          TEXT                   -- which endpoint was called
  status            TEXT                   -- 'success' | 'error' | 'blank_pdf'
  error_message     TEXT                   -- populated on error, NULL otherwise
  created_at        TIMESTAMPTZ DEFAULT NOW()

Install dependency:
  pip install asyncpg

Set in .env:
  DATABASE_URL=postgresql://user:password@localhost:5432/dbname
"""

import os
import logging
import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Module-level connection pool — created once on first use
_pool: asyncpg.Pool | None = None


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Add it to your .env file: "
                "DATABASE_URL=postgresql://user:password@host:5432/dbname"
            )
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("[db] Connection pool created")
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("[db] Connection pool closed")


# ---------------------------------------------------------------------------
# Schema initialisation — run once at startup
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pdf_requests (
    id                SERIAL PRIMARY KEY,
    request_id        TEXT        NOT NULL,
    pdf_name          TEXT        NOT NULL,
    pdf_size_bytes    INTEGER,
    total_pages       INTEGER,
    pages_analysed    INTEGER,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    total_tokens      INTEGER,
    completion_time_s NUMERIC(10, 3),
    endpoint          TEXT,
    status            TEXT,
    error_message     TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_requests_created_at
    ON pdf_requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_pdf_requests_request_id
    ON pdf_requests (request_id);
"""


async def init_db() -> None:
    """
    Create the pdf_requests table and indexes if they don't exist.
    Safe to call every startup — uses CREATE IF NOT EXISTS.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("[db] Schema initialised (pdf_requests table ready)")
    except Exception as e:
        logger.error(f"[db] Schema init failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Insert a completed request record
# ---------------------------------------------------------------------------

async def log_request(
    request_id:        str,
    pdf_name:          str,
    pdf_size_bytes:    int  = 0,
    total_pages:       int  = 0,
    pages_analysed:    int  = 0,
    input_tokens:      int  = 0,
    output_tokens:     int  = 0,
    completion_time_s: float = 0.0,
    endpoint:          str  = "/analyze",
    status:            str  = "success",
    error_message:     str | None = None,
) -> None:
    """
    Insert one row into pdf_requests.
    Failures are logged but never raised — DB errors must not break the API response.
    """
    total_tokens = input_tokens + output_tokens
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pdf_requests (
                    request_id, pdf_name, pdf_size_bytes,
                    total_pages, pages_analysed,
                    input_tokens, output_tokens, total_tokens,
                    completion_time_s, endpoint, status, error_message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                )
                """,
                request_id,
                pdf_name,
                pdf_size_bytes,
                total_pages,
                pages_analysed,
                input_tokens,
                output_tokens,
                total_tokens,
                round(completion_time_s, 3),
                endpoint,
                status,
                error_message,
            )
        logger.info(
            f"[db] logged request_id={request_id} "
            f"pdf='{pdf_name}' "
            f"tokens={input_tokens}in/{output_tokens}out/{total_tokens}total "
            f"time={completion_time_s:.2f}s status={status}"
        )
    except Exception as e:
        # Never let a DB error propagate to the caller
        logger.error(f"[db] log_request failed for request_id={request_id}: {e}")