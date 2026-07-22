"""Reliable structured generation against unreliable models.

`with_structured_output` assumes the provider honours a JSON schema. Free-tier
and small local models frequently do not: they wrap JSON in prose, emit
markdown fences, use single quotes, or trail a comma. Rather than let a node
crash on that, this module escalates:

    1. ask nicely, with the schema in the prompt
    2. salvage JSON from whatever came back
    3. show the model its own validation errors and ask again
    4. fall back to a caller-supplied default

Step 3 is the interesting one — a validation error is a far better repair
signal than "try harder", because it names the exact field that failed.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from sentinel.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str | None:
    """Pull the most plausible JSON object out of a model response."""
    if not text:
        return None
    if m := _FENCE.search(text):
        text = m.group(1)
    # Scan for a balanced top-level object rather than regexing braces, which
    # breaks on any nested structure.
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _repair(raw: str) -> str:
    """Fix the malformations small models actually produce."""
    raw = raw.replace("'", '"') if raw.count('"') < raw.count("'") else raw
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)  # trailing commas
    raw = re.sub(r"\bNone\b", "null", raw)
    raw = re.sub(r"\bTrue\b", "true", raw)
    raw = re.sub(r"\bFalse\b", "false", raw)
    return raw


def _schema_hint(schema: type[BaseModel]) -> str:
    js = schema.model_json_schema()
    return (
        f"Respond with a single JSON object matching this schema. "
        f"No prose, no markdown fences, no explanation outside the JSON.\n\n"
        f"{json.dumps(js, indent=2)}"
    )


async def generate_structured(
    model: BaseChatModel,
    schema: type[T],
    prompt: str,
    *,
    system: str = "",
    max_attempts: int = 3,
    default: T | None = None,
) -> T:
    """Generate an instance of `schema`, repairing and retrying as needed."""
    messages = [
        SystemMessage(f"{system}\n\n{_schema_hint(schema)}".strip()),
        HumanMessage(prompt),
    ]
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = await model.ainvoke(messages)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            log.warning("structured.call_failed", attempt=attempt, error=last_error)
            continue

        text = str(response.content or "")
        candidate = extract_json(text)
        if candidate is None:
            last_error = "no JSON object found in the response"
        else:
            for variant in (candidate, _repair(candidate)):
                try:
                    return schema.model_validate_json(variant)
                except ValidationError as e:
                    last_error = _format_errors(e)
                except json.JSONDecodeError as e:
                    last_error = f"invalid JSON: {e}"

        log.warning("structured.retry", schema=schema.__name__, attempt=attempt, error=last_error)
        if attempt < max_attempts:
            messages.append(response)
            messages.append(
                HumanMessage(
                    f"That response was rejected: {last_error}\n\n"
                    f"Return ONLY a corrected JSON object. Fix exactly the fields named above."
                )
            )

    if default is not None:
        log.error("structured.gave_up", schema=schema.__name__, error=last_error)
        return default
    raise ValueError(
        f"could not produce a valid {schema.__name__} after {max_attempts} attempts: {last_error}"
    )


def _format_errors(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        parts.append(f"field '{loc}': {err['msg']}")
    return "; ".join(parts)
