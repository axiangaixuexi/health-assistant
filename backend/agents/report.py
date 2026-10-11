from backend._log import fmt_args
import json

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from backend.agents.handoff import handoff_prompt_section, get_handoff_tools, apply_handoff
from backend.agents.llm import get_chat_llm
from backend.agents.state import MainAgentState
from backend.agents.streaming import emit_card
from backend.rag.knowledge_base import get_knowledge_base
from backend.skills import get_agent_tools, load_skill

AGENT_ID="report_agent"

_CARD_SKILL="lab_interpreter"
_CARD_TYPE="report_analysis"

REPORT_SYSTEM_PROMPT = """你是一位专业的检验报告解读助手。用户会向你描述或粘贴化验单数据，你需要：

【解读流程】
1. **指标识别**：先找出用户提到的所有检验指标和数值。
2. **结构化解读**：调用 lab_interpreter 技能对所有数值进行精确对比和分类（偏高/偏低/正常）。
3. **通俗解释**：用老百姓能听懂的语言解释偏高/偏低代表什么临床意义。
4. **生活建议**：针对异常项，给出饮食、作息、复查等方面的具体建议。
5. **免责声明**：最后必须注明"本解读仅供参考，具体诊疗请面诊专业医生"。

【格式要求】
- 使用清晰的列表格式
- 对正常指标简要说明，重点详细解释异常项
- 如果所有指标正常，给予肯定和保持健康的建议

【可用技能工具】
- lab_interpreter：将用户的化验数值列表结构化解读，自动比对参考范围，输出 HIGH/LOW/NORMAL 分类
- load_skill：按名称加载任意注册技能

【注意】
- 严禁凭报告数据直接下诊断，只陈述指标含义和建议
- 保持客观严谨，不夸大也不轻描淡写"""


def _emit_report_cards(new_messages: list[BaseMessage]) -> None:
    """把本轮的 lab_interpreter 结果重放为前端卡片。"""
    print(f"--- [report] _emit_report_cards({fmt_args(new_messages=new_messages)}) ---", flush=True)
    for message in new_messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) != _CARD_SKILL:
            continue
        content = message.content
        if not isinstance(content, str):
            continue
        try:
            emit_card(_CARD_TYPE, json.loads(content))
        except json.JSONDecodeError as exc:
            print(f"--- [Report] Unparsable {_CARD_SKILL} output: {exc} ---", flush=True)


async def report_node(state: MainAgentState) -> Command | dict:
    """
    Report agent：使用 lab_interpreter skill 进行结构化异常判读，
    并结合 RAG 获取临床背景来解读检验报告。当子 agent 将本轮转交给
    另一位专家时返回 ``Command``。
    """
    print(f"--- [report] report_node({fmt_args(state=state)}) ---", flush=True)
    llm = get_chat_llm("precise")
    user_info = state.get("user_info", {})
    age = user_info.get("age", "")
    elder_mode = user_info.get("elder_mode", False)

    extra = ""
    if age:
        extra += f"\n用户年龄：{age}，请结合年龄特点解读。"
    if elder_mode:
        extra += "\n请使用简单易懂的语言，避免复杂的医学术语。"

    system = REPORT_SYSTEM_PROMPT + extra + handoff_prompt_section(AGENT_ID)

    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    if last_user_msg:
        kb = get_knowledge_base()
        docs = await kb.aretrieve(last_user_msg, k=5)
        rag_context = kb.format_context(docs)
        if rag_context:
            system += f"\n\n## 参考检验范围\n以下为相关检验指标的标准参考范围，请以此为依据进行解读：\n\n{rag_context}"

    skill_tools = get_agent_tools(tags=["report"])

    agent = create_react_agent(
        llm,
        tools=[load_skill, *skill_tools, *get_handoff_tools(AGENT_ID)],
        prompt=SystemMessage(content=system),
    )
    sub_result = await agent.ainvoke({"messages": state["messages"]})
    original_count = len(state["messages"])
    new_messages = list(sub_result["messages"][original_count:])
    _emit_report_cards(new_messages)
    return apply_handoff(state, new_messages)