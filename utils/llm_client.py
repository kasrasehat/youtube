"""LLM client helpers using LangChain ChatOpenAI with optional prompt caching."""

from __future__ import annotations

import json
import logging
import os
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# Load environment variables from .env file
def load_environment() -> bool:
    """Load environment variables with robust path resolution."""
    script_dir = Path(__file__).parent

    # Possible .env file locations (in order of preference)
    possible_env_paths = [
        script_dir.parent / ".env",  # Parent directory (main project)
        script_dir / ".env",  # Same directory as script
        Path.cwd() / ".env",  # Current working directory
    ]

    env_loaded = False
    try:
        from dotenv import load_dotenv as _load
    except Exception:
        _load = None

    for env_path in possible_env_paths:
        if env_path.exists() and _load is not None:
            try:
                result = _load(env_path, override=True)
                if result:
                    logger.info(f"Loaded .env from: {env_path}")
                    env_loaded = True
                    break
                else:
                    logger.warning(f".env file found at {env_path} but no variables loaded")
            except Exception as e:
                logger.warning(f"Error loading .env from {env_path}: {e}")
        else:
            logger.debug(f".env not found at: {env_path}")

    if not env_loaded:
        logger.warning("No .env file found. Please ensure .env exists with required variables")
        logger.info("Expected .env locations:")
        for path in possible_env_paths:
            logger.info(f"   - {path}")

    return env_loaded


# Initialize environment loading
load_environment()

PROMPT_CACHE_SHARDS = int(os.getenv("PROMPT_CACHE_SHARDS", "1"))
PROMPT_CACHE_RETENTION_RAW = os.getenv("PROMPT_CACHE_RETENTION", "0")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")


def _parse_duration_to_seconds(value: str) -> int:
    """Parse a simple duration string into seconds.

    Supported formats:
    - Integer seconds: "0", "300", "86400"
    - Suffixed: "30s", "15m", "24h", "7d"

    Returns 0 for empty/invalid inputs.
    """
    raw = (value or "").strip().lower()
    if not raw:
        return 0

    # Plain integer seconds
    if raw.isdigit():
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    unit = raw[-1]
    num = raw[:-1]
    if unit in units and num.replace(".", "", 1).isdigit():
        try:
            seconds = float(num) * units[unit]
            return max(0, int(seconds))
        except Exception:
            return 0

    return 0


PROMPT_CACHE_RETENTION_SECONDS = _parse_duration_to_seconds(PROMPT_CACHE_RETENTION_RAW)


def _stable_json_dumps(payload: Dict[str, Any]) -> str:
    """Serialize a dict into stable, compact JSON for deterministic prompts."""
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _crc32_u32(text: str) -> int:
    """Compute a non-negative CRC32 for cache sharding."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _build_prompt_cache_key(prefix_version: str, model: str, shard: int) -> str:
    """Build a prompt cache key suitable for the Responses API cache."""
    return f"{prefix_version}:{model}:shard-{shard}"


def _extract_passage(user_payload: Dict[str, Any]) -> str:
    """Extract the main passage to send as the user message."""
    for key in ("transcript", "passage", "text", "content"):
        value = user_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return _stable_json_dumps(user_payload)


def invoke_llm(
    *,
    system_prompt: str,
    user_payload: Dict[str, Any],
    model_name: str,
    prompt_cache_retention: Optional[int] = None,
) -> str:
    """Invoke a chat model with a JSON user payload and optional prompt caching."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    alias_map = {
        "gpt40": "gpt-4o",
        "gpt-40": "gpt-4o",
        "gpt4o": "gpt-4o",
        "gpt4o-mini": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt5-low": "gpt-5",
        "gpt5-medium": "gpt-5",
        "gpt5-high": "gpt-5",
        "gpt5-nano": "gpt-5-nano",
        "gpt-5-nano": "gpt-5-nano",
        "gpt5-1": "gpt-5-1",
    }
    chosen_model = alias_map.get(model_name, model_name)

    passage = _extract_passage(user_payload)
    runtime_json = _stable_json_dumps({"passage": passage})
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"the passage is:\n{passage}"),
    ]

    retention = PROMPT_CACHE_RETENTION_SECONDS if prompt_cache_retention is None else prompt_cache_retention
    shard = 0
    if PROMPT_CACHE_SHARDS > 1:
        shard = _crc32_u32(runtime_json) % PROMPT_CACHE_SHARDS

    prompt_cache_key = _build_prompt_cache_key(
        prefix_version=PROMPT_VERSION,
        model=chosen_model,
        shard=shard,
    )

    llm_kwargs: Dict[str, Any] = {
        "model": chosen_model,
        "temperature": 0,
        "api_key": openai_api_key,
        "use_responses_api": True,
        "request_timeout": 60,
        "max_retries": 2,
    }

    if retention and retention > 0:
        llm_kwargs["prompt_cache_retention"] = retention

    if chosen_model in ["gpt-5", "gpt-5-1"]:
        llm_kwargs["reasoning"] = {"effort": "low"}

    model = ChatOpenAI(**llm_kwargs)

    if retention and retention > 0:
        response = model.invoke(
            messages,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=retention,
        )
    else:
        response = model.invoke(messages)

    return getattr(response, "content", str(response))

