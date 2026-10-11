"""
专科智能体之间类似 Swarm 的交接（handoff）。

为什么不在工具里直接使用 ``Command(goto=..., graph=Command.PARENT)``？
五个专科智能体中有四个是普通的节点*函数*，它们内部调用
``create_react_agent(...).ainvoke(...)``。那个内部智能体有自己独立的
Pregel 运行，所以在它的某个工具里抛出的 ``Command`` 只能到达那个
内部图——永远到不了主图。因此这里采用更稳健的模式：

  1. ``transfer_to_<agent>`` 是一个普通工具。它返回一个可识别的 JSON
     载荷 ``{"handoff": "<agent_id>"}`` 作为其 ``ToolMessage``。
  2. 在 ``ainvoke`` 返回后，节点函数调用 :func:`apply_handoff`，
     扫描新产生的消息以查找该载荷。
  3. 命中后节点返回 ``Command(goto=<target node>, update={...})``，从而
     主图将*同一*用户轮次重新分发给正确的智能体。

clinic 智能体是一个被挂载的子图，所以它通过
``Command(graph=Command.PARENT, ...)`` 进行交接——见 ``apply_handoff(parent=True)``。

``MainAgentState`` 中的 ``handoff_count`` 将其限制为每轮最多
``MAX_HANDOFFS_PER_TURN`` 跳，这样两个智能体就不会无限地互相推来推去。
``router_node`` 在每一轮开始时将其重置。
"""
from __future__ import annotations

from backend._log import fmt_args
import json
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

# 智能体 id（保存在 ``MainAgentState['active_agent']`` 中的值）→ 图节点。
AGENT_TO_NODE: dict[str, str] = {
    "clinic_agent": "clinic_node",
    "insurance_agent": "insurance_node",
    "report_agent": "report_node",
    "pharmacy_agent": "pharmacy_node",
    "advisor_agent": "advisor_node",
}

# 智能体 id → (工具名, 该专科智能体的职责)。
_HANDOFF_SPECS: dict[str, tuple[str, str]] = {
    "clinic_agent": (
        "transfer_to_clinic",
        "预问诊助手：用户描述身体不适或症状，想知道该看哪个科、是否需要就医。",
    ),
    "insurance_agent": (
        "transfer_to_insurance",
        "医保助手：医保余额、报销比例与政策、消费明细、缴费记录、异地就医备案。",
    ),
    "report_agent": (
        "transfer_to_report",
        "报告解读助手：化验单、体检报告、检验指标数值的解读。",
    ),
    "pharmacy_agent": (
        "transfer_to_pharmacy",
        "用药咨询助手：药品说明书、用法用量、副作用、药物相互作用、附近药店、OTC 推荐。",
    ),
    "advisor_agent": (
        "transfer_to_advisor",
        "健康顾问：通用健康科普、饮食作息建议、以及与上述专科都无关的闲聊问候。",
    ),
}

# 单个用户轮次内最多允许这么多次交接。
MAX_HANDOFFS_PER_TURN = 1

HANDOFF_TOOL_NAMES: frozenset[str] = frozenset(
    name for name, _ in _HANDOFF_SPECS.values()
)

# 工具名 → 转接期间 main.py 显示的展示标签。
HANDOFF_TOOL_LABELS: dict[str, str] = {
    "transfer_to_clinic": "正在转接预问诊助手…",
    "transfer_to_insurance": "正在转接医保助手…",
    "transfer_to_report": "正在转接报告解读助手…",
    "transfer_to_pharmacy": "正在转接用药咨询助手…",
    "transfer_to_advisor": "正在转接健康顾问…",
}


def _build_handoff_tool(agent_id: str) -> BaseTool:
    """为某个目标专科智能体创建 ``transfer_to_<agent>`` 工具。"""
    print(f"--- [handoff] _build_handoff_tool({fmt_args(agent_id=agent_id)}) ---", flush=True)
    tool_name, responsibility = _HANDOFF_SPECS[agent_id]
    description = (
        f"把当前对话转接给{responsibility}\n"
        "当用户这一轮的请求明显属于该专家的职责范围、而不属于你自己时调用本工具。"
        "调用后不要再尝试回答用户的问题，接手的专家会回答。"
    )

    @tool(tool_name, description=description)
    def _handoff(reason: str = "") -> str:
        """把对话转交给另一个专科智能体。"""
        print(f"--- [Handoff] → {agent_id} ({reason or 'no reason given'}) ---", flush=True)
        return json.dumps({"handoff": agent_id, "reason": reason}, ensure_ascii=False)

    return _handoff


def get_handoff_tools(exclude: str) -> list[BaseTool]:
    """为除 ``exclude``（调用方自身）之外的每个智能体生成 handoff 工具。"""
    print(f"--- [handoff] get_handoff_tools({fmt_args(exclude=exclude)}) ---", flush=True)
    return [
        _build_handoff_tool(agent_id)
        for agent_id in _HANDOFF_SPECS
        if agent_id != exclude
    ]


def handoff_prompt_section(exclude: str) -> str:
    """告诉智能体何时需要转接的系统提示词段落。"""
    print(f"--- [handoff] handoff_prompt_section({fmt_args(exclude=exclude)}) ---", flush=True)
    lines = [
        f"- {_HANDOFF_SPECS[agent_id][0]} → {_HANDOFF_SPECS[agent_id][1]}"
        for agent_id in _HANDOFF_SPECS
        if agent_id != exclude
    ]
    targets = "\n".join(lines)
    return f"""

【转接其他专家】
如果用户这一轮的问题明显不属于你的职责范围，请调用对应的转接工具，把对话交给更合适的专家：
{targets}

转接规则：
- 只有当用户的需求清楚地属于别人时才转接；不确定就自己先回答。
- 调用转接工具后，只回复一句不超过 20 字的过渡语，不要再回答用户的问题——接手的专家会回答。
- 一轮对话最多转接一次。"""


# ─── 检测 ────────────────────────────────────────────────────────────────

def handoff_target(message: BaseMessage) -> str | None:
    """当 ``message`` 是 handoff ToolMessage 时，返回目标智能体 id。"""
    print(f"--- [handoff] handoff_target({fmt_args(message=message)}) ---", flush=True)
    if not isinstance(message, ToolMessage):
        return None
    if getattr(message, "name", None) not in HANDOFF_TOOL_NAMES:
        return None
    content = message.content
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    target = payload.get("handoff")
    return target if target in AGENT_TO_NODE else None


def split_handoff(
    new_messages: Sequence[BaseMessage],
) -> tuple[str | None, list[BaseMessage]]:
    """
    在第一次 handoff 处拆分子智能体的新消息。

    返回 ``(target_agent_id | None, messages_to_keep)``。从请求转接的那条
    ``AIMessage`` 开始往后的内容都会被丢弃：工具调用、
    它的 ``ToolMessage`` 以及子智能体的过渡性收尾句。这样
    能保持消息历史内部一致（没有孤立的 ``tool_calls``），
    并让用户自己的问题成为目标智能体看到的最后一条消息。
    """
    print(f"--- [handoff] split_handoff({fmt_args(new_messages=new_messages)}) ---", flush=True)
    messages = list(new_messages)
    for index, message in enumerate(messages):
        target = handoff_target(message)
        if target is None:
            continue
        cut = index
        call_id = getattr(message, "tool_call_id", None)
        for back in range(index - 1, -1, -1):
            candidate = messages[back]
            if isinstance(candidate, AIMessage) and any(
                call.get("id") == call_id for call in (candidate.tool_calls or [])
            ):
                cut = back
                break
        return target, messages[:cut]
    return None, messages


def apply_handoff(
    state: dict,
    new_messages: Sequence[BaseMessage],
    *,
    parent: bool = False,
    prefix_messages: Sequence[BaseMessage] | None = None,
) -> Command | dict:
    """
    把子智能体的输出转换为普通状态更新或 ``Command``。

    ``prefix_messages`` 是节点在运行子智能体之前产生的消息
    （clinic 追问问答）；它们总是被交给父级。
    ``parent=True`` 用于 clinic 子图，其节点必须通过
    ``Command(graph=Command.PARENT, ...)`` 显式指向主图。
    """
    print(f"--- [handoff] apply_handoff({fmt_args(state=state, new_messages=new_messages, parent=parent, prefix_messages=prefix_messages)}) ---", flush=True)
    prefix = list(prefix_messages or [])
    target, kept = split_handoff(new_messages)

    if target is None:
        return {"messages": [*prefix, *new_messages]}

    honoured = int(state.get("handoff_count", 0) or 0)
    if honoured >= MAX_HANDOFFS_PER_TURN:
        # 本轮已转过一次——保留子智能体自己的回复
        # （包含工具调用，这样历史仍然有效），然后在这里停止。
        print(
            f"--- [Handoff] Ignoring a second handoff to {target} this turn ---",
            flush=True,
        )
        return {"messages": [*prefix, *new_messages]}

    print(f"--- [Handoff] Routing to {AGENT_TO_NODE[target]} ---", flush=True)
    update = {
        "messages": [*prefix, *kept],
        "active_agent": target,
        "next_agent": target,
        "handoff_count": honoured + 1,
    }
    if parent:
        return Command(
            graph=Command.PARENT,
            goto=AGENT_TO_NODE[target],
            update=update,
        )
    return Command(goto=AGENT_TO_NODE[target], update=update)


# 具有 handoff 能力的智能体可以跳转到的每个节点，用于 ``add_node(destinations=)``。
def handoff_destinations(exclude: str) -> tuple[str, ...]:
    """智能体 ``exclude`` 可以通过 handoff 到达的图节点。"""
    print(f"--- [handoff] handoff_destinations({fmt_args(exclude=exclude)}) ---", flush=True)
    return tuple(
        AGENT_TO_NODE[agent_id] for agent_id in _HANDOFF_SPECS if agent_id != exclude
    )
