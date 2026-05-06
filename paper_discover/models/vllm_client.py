"""
OpenAI-compatible async client for a local vLLM server.
Supports guided JSON decoding via vLLM's extra_body['guided_json'].
Falls back to a cloud provider if the local server is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import yaml
from openai import AsyncOpenAI, APIConnectionError

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/models.yaml")


def _load_config() -> dict:
    path = os.environ.get("PAPER_DISCOVER_MODELS_CONFIG", _CONFIG_PATH)
    with open(path) as f:
        return yaml.safe_load(f)


class LLMClient:
    """
    Thin async wrapper. All calls use the local vLLM server unless unreachable,
    in which case the cloud fallback (if enabled) is tried once.
    """

    def __init__(self) -> None:
        cfg = _load_config()
        local = cfg["local"]
        self._local_url = os.environ.get("PAPER_DISCOVER_VLLM_URL", local["base_url"])
        self._local_key = local.get("api_key", "token-local")
        self._judge_model = local["judge_model"]
        self._judge_max_tokens = local["judge_max_tokens"]
        self._judge_temperature = local["judge_temperature"]
        self._planner_model = local["planner_model"]
        self._planner_max_tokens = local["planner_max_tokens"]
        self._planner_temperature = local["planner_temperature"]

        self._fallback = cfg.get("cloud_fallback", {})
        self._client: AsyncOpenAI | None = None
        self._fallback_client: AsyncOpenAI | None = None

    def _local(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self._local_url, api_key=self._local_key
            )
        return self._client

    def _cloud(self) -> AsyncOpenAI | None:
        if not self._fallback.get("enabled"):
            return None
        if self._fallback_client is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            self._fallback_client = AsyncOpenAI(
                base_url=self._fallback["base_url"], api_key=api_key
            )
        return self._fallback_client

    async def _complete(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        json_schema: dict | None = None,
    ) -> str:
        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["guided_json"] = json_schema

        try:
            resp = await self._local().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra or None,
            )
            return resp.choices[0].message.content or ""
        except APIConnectionError:
            logger.warning("Local vLLM unreachable; trying cloud fallback")
            cloud = self._cloud()
            if cloud is None:
                raise
            resp = await cloud.chat.completions.create(
                model=self._fallback["model"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

    async def judge(
        self,
        messages: list[dict],
        json_schema: dict,
        retries: int = 1,
    ) -> dict:
        """Run the judge model, return parsed JSON. Retries once on parse error."""
        for attempt in range(retries + 1):
            raw = await self._complete(
                messages,
                self._judge_model,
                self._judge_max_tokens,
                self._judge_temperature,
                json_schema=json_schema,
            )
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == retries:
                    raise
                logger.warning("Judge parse error on attempt %d; retrying", attempt + 1)
        raise RuntimeError("unreachable")

    async def plan(self, messages: list[dict]) -> str:
        """Run the planner model, return raw text."""
        return await self._complete(
            messages,
            self._planner_model,
            self._planner_max_tokens,
            self._planner_temperature,
        )

    async def plan_json(self, messages: list[dict], json_schema: dict) -> dict:
        """Run the planner model expecting structured JSON output."""
        raw = await self._complete(
            messages,
            self._planner_model,
            self._planner_max_tokens,
            self._planner_temperature,
            json_schema=json_schema,
        )
        return json.loads(raw)


# Module-level singleton so the connection pool is reused.
_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
