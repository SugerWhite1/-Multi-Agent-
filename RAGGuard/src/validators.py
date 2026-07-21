"""LLM output validation with JSON parse, Pydantic schema, and auto-fix.

Four-layer defense:
  1. JSON format recovery (direct parse → markdown block extraction)
  2. Auto-fix common LLM quirks (casing, whitespace, bool strings)
  3. Pydantic strict schema validation
  4. Cross-field consistency (model_validator in state.py)
"""

import re
import json
from typing import Type
from pydantic import BaseModel, ValidationError


def _safe_json_parse(content: str) -> dict:
    """Parse JSON from LLM output, with markdown code block fallback."""
    content = content.strip()

    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { or [ and parse from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = content.find(start_char)
        if start >= 0:
            end = content.rfind(end_char)
            if end > start:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass

    raise json.JSONDecodeError("No valid JSON found in LLM output", content, 0)


# Auto-fix rules: each normalizer handles a specific LLM output quirk
# (wrong case, leading/trailing whitespace, bool-as-string, enum value variants).
# Applied before Pydantic validation so the schema sees clean data.
_AUTO_FIXES = {
    "nli_status": lambda v: (
        v.strip().upper()
        if isinstance(v, str) else v
    ),
    "hallucination_type": lambda v: v.strip() if isinstance(v, str) else v,
    "severity": lambda v: (
        v.strip().capitalize()
        if isinstance(v, str) else v
    ),
    "is_hallucination": lambda v: (
        v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
    ),
    "claim_type": lambda v: v.strip().lower() if isinstance(v, str) else v,
    "claim_text": lambda v: v.strip() if isinstance(v, str) else v,
    "reasoning": lambda v: v.strip()[:300] if isinstance(v, str) else str(v)[:300],
    "detail": lambda v: v.strip()[:500] if isinstance(v, str) else str(v)[:500],
}


def _auto_fix(data: dict, model_cls: Type[BaseModel]) -> dict:
    """Apply auto-fix normalizers to common LLM output quirks.

    Handles: casing, whitespace, bool-as-string, extra newlines in enum values.
    Does NOT fill in missing required fields — those should trigger ValidationError.
    """
    if not isinstance(data, dict):
        return data

    fixed = {}
    for key, value in data.items():
        fixer = _AUTO_FIXES.get(key)
        if fixer:
            try:
                fixed[key] = fixer(value)
            except Exception:
                fixed[key] = value
        else:
            fixed[key] = value

    return fixed


class LLMOutputValidator:
    """Validate LLM JSON output against a Pydantic model.

    Usage:
        try:
            result = LLMOutputValidator.parse(content, NLIVerdict)
        except LLMValidationError as e:
            # e.details has structured error info
            fallback(...)
    """

    @staticmethod
    def parse_json(content: str) -> dict:
        """Layer 1: Recover JSON from LLM output text."""
        return _safe_json_parse(content)

    @staticmethod
    def auto_fix(data: dict, model_cls: Type[BaseModel]) -> dict:
        """Layer 2: Normalize common LLM output quirks."""
        return _auto_fix(data, model_cls)

    @classmethod
    def validate(cls, data: dict, model_cls: Type[BaseModel]) -> BaseModel:
        """Layer 3: Strict Pydantic schema validation."""
        return model_cls.model_validate(data)

    @classmethod
    def parse(cls, content: str, model_cls: Type[BaseModel]) -> BaseModel:
        """Full pipeline: JSON parse → auto-fix → Pydantic validate.

        Raises:
            json.JSONDecodeError: If content contains no valid JSON.
            ValidationError: If data fails Pydantic schema after auto-fix.
        """
        data = cls.parse_json(content)
        data = cls.auto_fix(data, model_cls)
        return cls.validate(data, model_cls)

