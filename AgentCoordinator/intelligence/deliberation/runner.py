"""Provider adapter for isolated evidence-debate LLM invocations."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from ..contracts import DebateAgentProfile


class DebateRunner(ABC):
    @abstractmethod
    async def invoke(
        self,
        profile: DebateAgentProfile,
        phase: str,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def model_name(self, profile: DebateAgentProfile) -> str:
        return profile.model_route


class OpenAICompatibleDebateRunner(DebateRunner):
    """Resolve named existing engine profiles and parse a strict JSON envelope."""

    ROUTE_FIELDS: Dict[str, Tuple[str, str, str]] = {
        "query": ("QUERY_ENGINE_API_KEY", "QUERY_ENGINE_MODEL_NAME", "QUERY_ENGINE_BASE_URL"),
        "media": ("MEDIA_ENGINE_API_KEY", "MEDIA_ENGINE_MODEL_NAME", "MEDIA_ENGINE_BASE_URL"),
        "insight": ("INSIGHT_ENGINE_API_KEY", "INSIGHT_ENGINE_MODEL_NAME", "INSIGHT_ENGINE_BASE_URL"),
        "report": ("REPORT_ENGINE_API_KEY", "REPORT_ENGINE_MODEL_NAME", "REPORT_ENGINE_BASE_URL"),
        "mindspider": ("MINDSPIDER_API_KEY", "MINDSPIDER_MODEL_NAME", "MINDSPIDER_BASE_URL"),
        "forum": ("FORUM_HOST_API_KEY", "FORUM_HOST_MODEL_NAME", "FORUM_HOST_BASE_URL"),
    }

    def __init__(self, settings: Any):
        self.settings = settings
        self._clients: Dict[str, Any] = {}

    async def invoke(
        self,
        profile: DebateAgentProfile,
        phase: str,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = self._client(profile.model_route)
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        response = await asyncio.to_thread(
            client.invoke,
            system_prompt,
            user_prompt,
            temperature=profile.temperature,
        )
        return parse_json_object(response)

    def model_name(self, profile: DebateAgentProfile) -> str:
        fields = self.ROUTE_FIELDS.get(profile.model_route, self.ROUTE_FIELDS["query"])
        return str(getattr(self.settings, fields[1], None) or profile.model_route)

    def _client(self, route: str):
        route = route if route in self.ROUTE_FIELDS else "query"
        if route in self._clients:
            return self._clients[route]
        key_field, model_field, base_field = self.ROUTE_FIELDS[route]
        api_key = getattr(self.settings, key_field, None)
        model = getattr(self.settings, model_field, None)
        base_url = getattr(self.settings, base_field, None)
        if not api_key or not model:
            raise RuntimeError(f"Debate model route '{route}' is not configured")
        from QueryEngine.llms import LLMClient

        self._clients[route] = LLMClient(api_key=api_key, model_name=model, base_url=base_url)
        return self._clients[route]


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Debate agent did not return a JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Debate agent JSON envelope must be an object")
    return parsed
