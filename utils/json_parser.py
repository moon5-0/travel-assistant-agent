"""统一提取模型响应文本，并对常见 JSON 格式问题进行容错解析。"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Iterable, Optional


logger = logging.getLogger(__name__)


def _strip_markdown_fence(text: str) -> str:
    """移除包裹整个响应的 Markdown 代码块标记。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _iter_json_objects(text: str) -> Iterable[str]:
    """依次提取文本中大括号平衡的 JSON/Python 字典候选片段。"""
    start: Optional[int] = None
    depth = 0
    quote: Optional[str] = None
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                quote = None
                escaped = False
            continue

        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if char in ('"', "'"):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if quote is not None:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                yield text[start:index + 1]
                start = None


def _escape_newlines_in_strings(text: str) -> str:
    """只转义双引号字符串内部的换行、回车和制表符。"""
    result = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            result.append(char)
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string and char in ("\n", "\r", "\t"):
            result.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
        else:
            result.append(char)

    return "".join(result)


def _clean_json_candidate(candidate: str) -> str:
    """组合应用安全的 JSON 清理规则。"""
    cleaned = _escape_newlines_in_strings(candidate)
    # 删除 JSON 不允许的控制字符，但保留结构性空白。
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)
    # 删除对象或数组末尾多余的逗号。
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def _parse_candidate(candidate: str) -> dict:
    """按标准 JSON、组合清理、Python 字面量和可选 JSON5 依次解析。"""
    errors = []

    for value in (candidate, _clean_json_candidate(candidate)):
        try:
            result = json.loads(value)
            if not isinstance(result, dict):
                raise ValueError("JSON root must be an object")
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(exc)

    # ast.literal_eval 仅解析 Python 字面量，可安全兼容单引号和尾部逗号。
    try:
        result = ast.literal_eval(candidate)
        if not isinstance(result, dict):
            raise ValueError("Parsed value root must be a dictionary")
        return result
    except (SyntaxError, ValueError) as exc:
        errors.append(exc)

    try:
        import json5

        result = json5.loads(candidate)
        if not isinstance(result, dict):
            raise ValueError("JSON5 root must be an object")
        return result
    except ImportError:
        pass
    except Exception as exc:
        errors.append(exc)

    raise ValueError(f"Unable to parse JSON object: {errors[-1] if errors else 'unknown error'}")


def robust_json_parse(text: Any, fallback=None) -> dict:
    """将可能带解释文本或常见格式问题的模型输出解析为字典。"""
    if isinstance(text, dict):
        return text
    if text is None or not str(text).strip():
        if fallback is not None:
            return fallback
        raise ValueError("Empty text provided")

    cleaned_text = _strip_markdown_fence(str(text))
    candidates = list(_iter_json_objects(cleaned_text))
    if not candidates:
        if fallback is not None:
            logger.warning("No JSON object found in response, using fallback")
            return fallback
        raise ValueError("No JSON object found in response")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return _parse_candidate(candidate)
        except ValueError as exc:
            last_error = exc

    logger.error(
        "All JSON parsing attempts failed. Response sample: %s",
        cleaned_text[:200],
    )
    if fallback is not None:
        logger.warning("Using fallback value")
        return fallback
    raise ValueError(f"Failed to parse JSON response: {last_error}")


def _content_to_text(content: Any) -> str:
    """将常见 content 结构转换为文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def extract_json_from_response(response, field_name: str = "content") -> str:
    """从字符串、字典或模型响应对象中提取文本。"""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return _content_to_text(response.get(field_name))

    if hasattr(response, "text"):
        text = _content_to_text(getattr(response, "text"))
        if text:
            return text
    if hasattr(response, field_name):
        return _content_to_text(getattr(response, field_name))
    return str(response) if response is not None else ""


def _merge_stream_text(current: str, new: str) -> str:
    """同时兼容累计快照流和增量文本流。"""
    if not new:
        return current
    if not current:
        return new
    if new.startswith(current):
        return new
    if current.startswith(new):
        return current

    max_overlap = min(len(current), len(new))
    for size in range(max_overlap, 0, -1):
        if current.endswith(new[:size]):
            return current + new[size:]
    return current + new


async def extract_json_from_async_response(response, field_name: str = "content") -> str:
    """从异步流或普通模型响应中提取完整文本。"""
    if not hasattr(response, "__aiter__"):
        return extract_json_from_response(response, field_name)

    text = ""
    async for chunk in response:
        chunk_text = extract_json_from_response(chunk, field_name)
        text = _merge_stream_text(text, chunk_text)
    return text
