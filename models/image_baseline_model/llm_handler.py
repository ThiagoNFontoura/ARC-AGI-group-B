import json
import os
from pathlib import Path
from typing import Any


def _extract_json_block(text: str) -> str:
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
        response = self._client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Model response did not include text.")
        return text

    def _parse_response(self, text: str) -> dict[str, Any]:
        raw_json = _extract_json_block(text)
        result = json.loads(raw_json)
        if not isinstance(result, dict):
            raise ValueError("Model response must be a JSON object.")
        return result

    def solve(self, prompt: str) -> dict[str, Any]:
        first_text = self._generate_text(prompt)

        try:
            return self._parse_response(first_text)
        except Exception:
            retry_prompt = (
                "Your previous response was invalid. "
                "Return only valid JSON that matches the required schema.\\n\\n"
                + prompt
            )
            second_text = self._generate_text(retry_prompt)
            return self._parse_response(second_text)

    def solve_with_images(
        self,
        prompt: str,
        labeled_image_paths: list[tuple[str, Path]],
    ) -> dict[str, Any]:
        from google.genai import types  # type: ignore

        contents: list[Any] = [prompt]
        for label, image_path in labeled_image_paths:
            contents.append(label)
            contents.append(
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png")
            )

        first_response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        first_text = getattr(first_response, "text", None)
        if not first_text:
            raise ValueError("Model response did not include text.")

        try:
            return self._parse_response(first_text)
        except Exception:
            retry_contents = [
                "Your previous response was invalid. Return only valid JSON that matches the required schema."
            ] + contents
            retry_response = self._client.models.generate_content(
                model=self.model,
                contents=retry_contents,
            )
            retry_text = getattr(retry_response, "text", None)
            if not retry_text:
                raise ValueError("Model retry response did not include text.")
            return self._parse_response(retry_text)
