import json
import os
import time
from typing import Any


def _extract_json_block(text: str) -> str:
    """Extract the outer JSON object from plain or fenced model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return a JSON object.")
    return text[start : end + 1]


class GemmaHandler:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Configure a Gemini client using supplied or environment credentials."""
        self.api_key = api_key or os.getenv("GEMMA_API_KEY")
        self.model = model or os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
        self.thinking_level = os.getenv("GEMMA_THINKING_LEVEL", "high")

        if not self.api_key:
            raise ValueError("Missing GEMMA_API_KEY in environment.")

        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._client = genai.Client(api_key=self.api_key)

    def _generate_text_and_metadata(
        self, prompt: str, max_retries: int = 4
    ) -> tuple[str, dict[str, Any]]:
        """Send a prompt to the configured model with retry on 503/429/temporary errors."""
        last_exception: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                from google.genai import types  # type: ignore

                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self.thinking_level
                        )
                    ),
                )
                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Model response did not include text.")

                usage = getattr(response, "usage_metadata", None)
                metadata: dict[str, Any] = {}
                if usage is not None:
                    metadata = {
                        "prompt_tokens": getattr(usage, "prompt_token_count", None),
                        "candidates_tokens": getattr(usage, "candidates_token_count", None),
                        "total_tokens": getattr(usage, "total_token_count", None),
                    }
                return text, metadata
            except Exception as exc:
                last_exception = exc
                err_str = str(exc)
                # Check for high demand / transient errors
                if attempt < max_retries and any(
                    code in err_str
                    for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "demand")
                ):
                    wait_time = 2**attempt
                    print(
                        f"    [API Busy: 503/429] Retrying in {wait_time}s (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise exc

        raise last_exception or RuntimeError("Failed to generate content after retries.")

    def solve(self, prompt: str) -> dict[str, Any]:
        """Generate and parse a structured task solution with a single model call."""
        text, metadata = self._generate_text_and_metadata(prompt)
        raw_json = _extract_json_block(text)
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict) and metadata:
            parsed["_usage_metadata"] = metadata
        return parsed
