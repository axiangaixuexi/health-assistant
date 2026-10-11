"""
主 LangGraph 应用的 checkpointer 工厂。

checkpointer 为图提供服务端记忆：每个 super-step 都
持久化在 ``thread_id`` 下，这样下一次请求只需发送新的
用户消息，并且基于 ``interrupt()`` 的追问也能恢复。

后端（通过 ``CHECKPOINTER`` 环境变量选择）：
  memory（默认）— ``MemorySaver``；进程级，重启即丢失
  sqlite           — ``AsyncSqliteSaver``，持久化在 ``CHECKPOINT_DB_PATH``
"""
from __future__ import annotations


from backend._log import fmt_args
import asyncio
import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

DEFAULT_CHECKPOINT_DB_PATH = "checkpoints.sqlite"
_BACKEND_ALIASES = {
    "": "memory",
    "memory": "memory",
    "inmemory": "memory",
    "in-memory": "memory",
    "memorysaver": "memory",
    "sqlite": "sqlite",
    "aiosqlite": "sqlite",
}


class CheckpointerConfigurationError(ValueError):
    """当 checkpointer 环境变量不可用时抛出。"""


def resolve_checkpointer_backend(env: dict[str, str] | None = None) -> str:
    """把 ``CHECKPOINTER`` 归一化为 ``memory`` / ``sqlite`` 之一。"""
    print(f"--- [checkpointer] resolve_checkpointer_backend({fmt_args(env=env)}) ---", flush=True)
    values = os.environ if env is None else env
    raw = values.get("CHECKPOINTER", "").strip().lower().replace("_", "")
    backend = _BACKEND_ALIASES.get(raw)
    if backend is None:
        supported = "memory, sqlite"
        raise CheckpointerConfigurationError(
            f"不支持的 CHECKPOINTER '{raw}'。支持的后端：{supported}"
        )
    return backend


def resolve_checkpoint_db_path(env: dict[str, str] | None = None) -> str:
    """返回 ``sqlite`` 后端使用的 SQLite 文件路径。"""
    print(f"--- [checkpointer] resolve_checkpoint_db_path({fmt_args(env=env)}) ---", flush=True)
    values = os.environ if env is None else env
    return values.get("CHECKPOINT_DB_PATH", "").strip() or DEFAULT_CHECKPOINT_DB_PATH


def create_checkpointer(env: dict[str, str] | None = None) -> BaseCheckpointSaver:
    """构建由环境变量选定的 checkpointer。"""
    print(f"--- [checkpointer] create_checkpointer({fmt_args(env=env)}) ---", flush=True)
    backend = resolve_checkpointer_backend(env)
    if backend == "memory":
        return MemorySaver()

    try:
        asyncio.get_running_loop()
    except RuntimeError as exc:  # pragma: no cover - 取决于导入上下文
        raise CheckpointerConfigurationError(
            "CHECKPOINTER=sqlite 需要一个正在运行的事件循环；请从运行中的服务器"
            "（uvicorn）构建图，或改用 CHECKPOINTER=memory。"
        ) from exc

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = resolve_checkpoint_db_path(env)
    connection = aiosqlite.connect(db_path, check_same_thread=False)
    connection.daemon = True
    return AsyncSqliteSaver(connection)
