"""调试日志的入参格式化工具。

给每个方法打「入口 + 入参」日志时使用，负责：
  - 密钥/口令脱敏（按参数名或字典键命中 key/secret/token/password）
  - 图片（bytes）与长文本截断
"""
from __future__ import annotations

from collections.abc import Mapping

_MAX_LEN = 200
_SECRET_HINTS = ("key", "secret", "token", "password")


def _is_secret(name: str) -> bool:
    lowered = name.lower().replace("_", "").replace("-", "")
    return any(hint in lowered for hint in _SECRET_HINTS)


def _fmt_val(value, max_len: int = _MAX_LEN) -> str:
    if isinstance(value, bytes):
        return f"<bytes {len(value)}B>"
    if isinstance(value, Mapping):
        parts = []
        for k, v in list(value.items())[:20]:
            ks = str(k)
            if _is_secret(ks):
                parts.append(f"{ks!r}: ***")
            else:
                parts.append(f"{ks!r}: {_fmt_val(v, 80)}")
        suffix = ", ..." if len(value) > 20 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    try:
        s = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def fmt_args(**values) -> str:
    """把 name=value 渲染成适合打印的参数字符串；密钥脱敏、长值截断。"""
    parts = []
    for name, value in values.items():
        if _is_secret(name):
            parts.append(f"{name}=***")
        else:
            parts.append(f"{name}={_fmt_val(value)}")
    return ", ".join(parts)
