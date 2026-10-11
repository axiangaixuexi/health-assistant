from backend._log import fmt_args
from typing import Any

from langgraph.config import get_stream_writer


def emit_custom(payload: dict) -> None:
    print(f"--- [streaming] emit_custom({fmt_args(payload=payload)}) ---", flush=True)
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is not None:
        writer(payload)


def emit_card(card_type: str, data: Any) -> None:
    print(f"--- [streaming] emit_card({fmt_args(card_type=card_type, data=data)}) ---", flush=True)
    emit_custom({"type": "card", "payload": {"type": card_type, "data": data}})


def emit_text(content: str) -> None:
    print(f"--- [streaming] emit_text({fmt_args(content=content)}) ---", flush=True)
    if content:
        emit_custom({"type": "text", "content": content})