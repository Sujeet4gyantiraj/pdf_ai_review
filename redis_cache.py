"""
redis_cache.py — async Redis caching helpers for PDF AI endpoints.

Cache key: SHA-256 of raw PDF bytes + endpoint tag (+ analysis_type where relevant).
TTL:       Configurable via REDIS_CACHE_TTL_SECONDS (default 86400 = 24 h).
"""

import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (from environment)
# ---------------------------------------------------------------------------
REDIS_URL              = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_TTL        = int(os.environ.get("REDIS_CACHE_TTL_SECONDS", 86400))  # 24 h
REDIS_ENABLED          = os.environ.get("REDIS_ENABLED", "true").lower() == "true"

_redis_client: aioredis.Redis | None = None


# ---------------------------------------------------------------------------
# Lifecycle helpers (call from FastAPI lifespan)
# ---------------------------------------------------------------------------

async def init_redis() -> None:
    """Connect to Redis. Logs and disables caching on failure."""
    global _redis_client, REDIS_ENABLED
    if not REDIS_ENABLED:
        logger.info("Redis caching disabled via REDIS_ENABLED=false")
        return
    try:
        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        _redis_client = client
        logger.info(f"Redis connected: {REDIS_URL} (TTL={REDIS_CACHE_TTL}s)")
    except Exception as e:
        logger.warning(f"Redis unavailable — caching disabled: {e}")
        REDIS_ENABLED = False


async def close_redis() -> None:
    """Close the Redis connection gracefully."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed.")


# ---------------------------------------------------------------------------
# Public cache helpers
# ---------------------------------------------------------------------------

def make_cache_key(content_hash: str, endpoint: str, **extra) -> str:
    """Build a namespaced cache key."""
    parts = ["pdf_ai", endpoint, content_hash]
    for k, v in sorted(extra.items()):
        parts.append(f"{k}:{v}")
    return ":".join(parts)


async def get_cached(key: str) -> Any | None:
    """
    Return parsed JSON value from cache, or None on miss / error.
    """
    if not REDIS_ENABLED or _redis_client is None:
        return None
    try:
        raw = await _redis_client.get(key)
        if raw is None:
            return None
        logger.debug(f"Cache HIT  key={key}")
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Cache GET error (key={key}): {e}")
        return None


async def set_cached(key: str, value: Any, ttl: int = REDIS_CACHE_TTL) -> None:
    """
    Serialise value to JSON and store in Redis with TTL.
    """
    if not REDIS_ENABLED or _redis_client is None:
        return
    try:
        await _redis_client.setex(key, ttl, json.dumps(value))
        logger.debug(f"Cache SET  key={key}  ttl={ttl}s")
    except Exception as e:
        logger.warning(f"Cache SET error (key={key}): {e}")
