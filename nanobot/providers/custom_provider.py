"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

from typing import Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default", reasoning_effort: str | None = None, enable_thinking: bool | None = None, thinking_budget: int | None = None, presence_penalty: float | None = None):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.presence_penalty = presence_penalty
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None, enable_thinking: bool | None = None,
                   thinking_budget: int | None = None, presence_penalty: float | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        
        # Use provided reasoning_effort or fall back to instance default
        effort = reasoning_effort or self.reasoning_effort
        # Use provided enable_thinking or fall back to instance default
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        if thinking:
            budget = thinking_budget if thinking_budget is not None else self.thinking_budget
            extra_body: dict[str, Any] = {"enable_thinking": True}
            if budget is not None:
                extra_body["thinking_budget"] = budget
            # Add reasoning effort to extra_body if specified
            if effort:
                extra_body["reasoning"] = {"effort": effort}
            kwargs["extra_body"] = extra_body
        elif effort:
            # If only reasoning_effort is set (without thinking), still send it via extra_body
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}
        # Use provided presence_penalty or fall back to instance default
        penalty = presence_penalty if presence_penalty is not None else self.presence_penalty
        if penalty is not None:
            kwargs["presence_penalty"] = penalty

        logger.info(f"CustomProvider chat: model={kwargs['model']}, reasoning_effort={effort}, enable_thinking={thinking}, thinking_budget={budget if thinking else None}, presence_penalty={penalty}")
        logger.debug(f"CustomProvider chat details: extra_body={kwargs.get('extra_body')}")
        
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        
        # Log thinking info if available
        reasoning_content = getattr(msg, "reasoning_content", None)
        usage = getattr(response, "usage", None)
        
        if reasoning_content:
            logger.debug(f"Thinking content received: {len(reasoning_content)} chars")
        if usage:
            # Log token usage breakdown for thinking models
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            
            # Check if thinking tokens are reported separately
            completion_tokens_details = getattr(usage, "completion_tokens_details", None)
            if completion_tokens_details:
                reasoning_tokens = getattr(completion_tokens_details, "reasoning_tokens", 0)
                if reasoning_tokens > 0:
                    logger.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, reasoning={reasoning_tokens}, total={total_tokens}")
                else:
                    logger.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            else:
                logger.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=reasoning_content or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

