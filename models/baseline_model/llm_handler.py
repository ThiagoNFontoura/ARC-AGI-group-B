import json
import os
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
        self.model = model or os.getenv("GEMMA_MODEL", "gemma-3-27b-it")

        if not self.api_key:
            raise ValueError("Missing GEMMA_API_KEY in environment.")

        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._client = genai.Client(api_key=self.api_key)

    def _generate_text(self, prompt: str) -> str:
        """Send a prompt to the configured model and return its text response."""
        response = self._client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Model response did not include text.")
        return text

    def solve(self, prompt: str) -> dict[str, Any]:
        """Generate and parse a structured task solution with a single model call."""
        text = self._generate_text(prompt)
        raw_json = _extract_json_block(text)
        return json.loads(raw_json)
