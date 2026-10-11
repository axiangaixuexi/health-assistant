"""
Clinic Agent —— 一个编译好的 LangGraph 子图，以 ``clinic_node`` 形式挂载。

诊所线是唯一一个真正具备多步状态机的 agent。它持有
私有状态 ``ClinicState``，并作为子图挂载在主图
``agents/graph.py`` 中：

    emergency_gate ──CRITICAL──────────────────────────────────────────► END
          │                                                              ▲
          │ 足够安全，可进行问诊                                     │
          ▼                                                              │
    extract_symptoms ──► check_sufficiency ──信息足够──► conclude ─────────┘
          ▲                     │
          │                     │ 缺少必需信息
          │                     ▼
          └── await_answer ◄── ask_followup        (interrupt() + resume)

节点职责：
  emergency_gate    纯规则匹配（不经过 LLM）。直接调用 ``EmergencyTriageSkill``，
                    确保红色警报关卡永远不会被一个决定不调用该工具的
                    LLM 跳过。CRITICAL 会带着原文安全提示直接短路整个子图。
  extract_symptoms  LLM 通过 ``with_structured_output`` 输出 ``SymptomFacts``；
                    并在各轮次间合并进 ``ClinicState['collected']``。
  check_sufficiency 对必填字段做确定性检查。
  ask_followup      LLM 只写一个温和的追问。
  await_answer      ``interrupt()`` —— 挂起整个图，直到同一 ``thread_id`` 上的
                    下一次请求携带用户答案将其恢复，答案随后回流进
                    extract_symptoms。
  conclude          ReAct agent（clinic skills + RAG）撰写分诊文本，
                    然后以 ``clinic_recommendation`` 卡片形式把结构化建议
                    流式推送给客户端。

与父图的契约：子图返回 ``{"messages": [...]}``，其中只包含本轮产生的消息
（见 ``_replace``）。
"""
from __future__ import annotations

from backend._log import fmt_args
import operator
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from backend.agents.handoff import handoff_prompt_section, get_handoff_tools, apply_handoff
from backend.agents.llm import get_chat_llm
from backend.agents.streaming import emit_custom
from backend.rag.knowledge_base import get_knowledge_base
from backend.skills import get_agent_tools, load_skill
from backend.skills.emergency_triage.skill import EmergencyTriageSkill

AGENT_ID = "clinic_agent"

CLINIC_SYSTEM_PROMPT = """你是大健康App中的"AI预问诊助手"，态度温和、专业。

你的任务是通过友好的对话帮助用户描述清楚症状，然后给出合理的分诊建议。

【对话要求】
1. **禁止下绝对诊断**：不允许说"您得了XX病"，只能说"从症状来看，可能需要就诊XX科"。
2. **安全第一**：如果用户提到剧烈胸痛、呼吸困难、突发意识丧失等急症，必须立刻提示"请立即拨打 120！"
3. **信息已收集完毕**：症状要点已由系统结构化整理，请直接据此给出结论，不要再反复追问。
4. **输出分诊建议**：以清晰的格式给出建议，包括：可能的科室、就诊优先级（紧急/较快/可择期）、初步注意事项。
5. **语气**：温和、关怀、专业，像一个有经验的家庭医生。

【可用技能（Skills）调用规则】
- **symptom_scorer**：根据已收集的症状信息调用，获取严重程度分数（0-100）和分诊颜色，用于辅助分诊建议。
- **health_calculator**：用户询问BMI、体重、热量时调用，需提供身高、体重、年龄、性别。
- **risk_assessor**：用户有多项慢性病风险因素时调用，评估心血管和糖尿病10年风险。
- **medication_calculator**：用户询问按体重/年龄计算药物剂量时调用。
- **lab_interpreter**：用户同时提供化验数值时调用，对指标进行结构化判读。

急症安全评估已由系统在进入本节点前完成，无需再调用 emergency_triage。
请直接开始回答，不要重复系统提示内容。"""

EXTRACT_SYSTEM_PROMPT = """你是一个医疗信息抽取器。请阅读预问诊对话，抽取结构化的症状要点。

规则：
- 只抽取用户明确说过的内容，没有提到的字段一律留空字符串或空数组，禁止推测或编造。
- duration 用原话表达（如"三天"、"两周"、"今天早上开始"）。
- severity 归一化为"轻微"/"中等"/"严重"之一。
字段说明（除 associated_symptoms 是数组外，其余都是字符串；用户没提到的留空字符串）：
- chief_complaint：用户最主要的不适，例如 咳嗽。只要用户描述了任何不适，本字段必须填写，不得留空。
- location：不适的具体部位，例如 右下腹。
- duration：症状持续多久，保留用户原话。
- severity：轻微 / 中等 / 严重。
- associated_symptoms：伴随症状数组。
- onset：起病方式，例如 突然 / 逐渐加重。
- triggers：诱因或加重、缓解因素。
- 只输出一个 json 对象（字段名与给定结构一致），不要输出任何解释、前后缀或 Markdown 代码块。
- 不要输出诊断结论。"""

FOLLOWUP_SYSTEM_PROMPT = """你是一位温和的预问诊护士。请针对缺失的信息，向用户提出**一个**问题。

规则：
- 只问一个问题，不超过 40 个字，不要编号，不要罗列多个问题。
- 语气关怀、口语化，可以先用半句话共情再提问。
- 不要给出诊断、用药或就诊建议。
- 不要重复已经问过的问题。"""

RECOMMENDATION_SYSTEM_PROMPT = """请把下面这段预问诊结论压缩成结构化分诊建议。

规则：
- department 给出 1-3 个最贴切的科室名称（如"呼吸内科"）。
- urgency 只能是 emergency（需立即急诊）/ soon（建议尽快就诊）/ routine（可择期就诊）之一。
- summary 用一句话概括建议，不超过 60 个字。
- notes 给出 1-4 条就诊前注意事项。
- 只输出一个 json 对象（含 department、urgency、summary、notes），不要输出任何解释或代码块。
- 只依据给定文本，不要新增结论。"""

def _structured_output(llm, schema):
    """用 json_mode 取得结构化输出（兼容 DeepSeek 等 OpenAI 兼容端点）。

    with_structured_output() 的默认路径在本项目配置的端点上不可用：
      1. langchain-openai 1.x 在 strict 未显式关闭时会把结构化输出升级为
         response_format={"type": "json_schema"}，DeepSeek 不支持，直接返回
         400 "This response_format type is unavailable now"。
      2. method="function_calling" 会强制 tool_choice={"type": "function", ...}，
         DeepSeek 的 Thinking 模式同样拒绝，返回
         400 "Thinking mode does not support this tool_choice"。
    json_mode 走 response_format={"type": "json_object"}，两种模式下都可用；
    代价是提示词里必须出现 json 字样（见 EXTRACT_SYSTEM_PROMPT /
    RECOMMENDATION_SYSTEM_PROMPT 的最后一行）。
    """
    print(f"--- [clinic] _structured_output({fmt_args(llm=llm, schema=schema)}) ---", flush=True)
    return llm.with_structured_output(schema, method="json_mode")

# 在允许 agent 给出结论前，问诊必须收集到的字段。
REQUIRED_FIELDS: tuple[str, ...] = ("chief_complaint", "duration", "severity")

# 安全阀：每轮追问用户的次数不得超过该值。
MAX_FOLLOWUPS = 3

_FIELD_LABELS = {
    "chief_complaint": "主要不适（主诉）",
    "location": "不适的具体部位",
    "duration": "症状持续了多久",
    "severity": "严重程度（轻微/中等/严重）",
    "associated_symptoms": "是否伴随其他症状",
    "onset": "症状是怎么开始的",
    "triggers": "有没有诱因或加重/缓解因素",
}

_URGENCY_TO_SEVERITY = {
    "emergency": "high",
    "soon": "medium",
    "routine": "low",
}


# ─── 结构化模型 ───────────────────────────────────────────────────────

class SymptomFacts(BaseModel):
    """从问诊对话中抽取的结构化症状画像。"""

    chief_complaint: str = Field(default="", description="主诉，用户最主要的不适")
    location: str = Field(default="", description="不适部位")
    duration: str = Field(default="", description="症状持续时间，保留用户原话")
    severity: str = Field(default="", description="严重程度：轻微 / 中等 / 严重")
    associated_symptoms: list[str] = Field(default_factory=list, description="伴随症状列表")
    onset: str = Field(default="", description="起病方式，如突发 / 逐渐加重")
    triggers: str = Field(default="", description="诱因或加重、缓解因素")


class TriageRecommendation(BaseModel):
    """由前端诊所卡片渲染的结构化分诊结果。"""

    department: list[str] = Field(default_factory=list, description="推荐就诊科室")
    urgency: Literal["emergency", "soon", "routine"] = Field(
        default="routine", description="就诊优先级"
    )
    summary: str = Field(default="", description="一句话分诊建议")
    notes: list[str] = Field(default_factory=list, description="就诊前注意事项")


# ─── 子图状态 ───────────────────────────────────────────────────────────

def _replace(_current: Any, incoming: Any) -> Any:
    """后写覆盖的 reducer。"""
    print(f"--- [clinic] _replace({fmt_args(_current=_current, incoming=incoming)}) ---", flush=True)
    return incoming


def _merge_facts(current: dict | None, incoming: dict | None) -> dict:
    """合并新抽取的症状要点，绝不用空值覆盖已知值。"""
    print(f"--- [clinic] _merge_facts({fmt_args(current=current, incoming=incoming)}) ---", flush=True)
    merged = dict(current or {})
    for key, value in (incoming or {}).items():
        if value in ("", [], None):
            continue
        merged[key] = value
    return merged


class ClinicState(TypedDict, total=False):
    """
    诊所子图的私有状态。

    ``messages`` 被特意设计为后写覆盖通道：进入时它保存主图传递下来的
    对话历史，而终端节点只保留本轮产生的消息覆盖它。父图的
    ``operator.add`` reducer 随后精确追加这些消息，从而维持应用其余部分
    所期望的 ``{"messages": new_messages}`` 契约。
    """

    messages: Annotated[Sequence[BaseMessage], _replace]
    user_info: dict
    # 镜像自 MainAgentState，使 conclude() 在逃逸到父图前能遵守每轮
    # 的 handoff 预算。
    handoff_count: int
    # 本轮内产生的消息（追问问答 + 最终分诊回答）。
    turn_messages: Annotated[list[BaseMessage], operator.add]
    # 跨抽取轮次合并的症状要点。
    collected: Annotated[dict, _merge_facts]
    emergency_level: str
    missing_fields: list[str]
    is_sufficient: bool
    followup_count: int
    pending_question: str
    recommendation: dict


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

# 为提升本模块可读性而保留的别名；实现与其他 agent 共享于 agents/streaming.py。
_emit = emit_custom


def _transcript(state: ClinicState) -> list[BaseMessage]:
    """入站历史加上本轮产生的所有内容，按顺序排列。"""
    print(f"--- [clinic] _transcript({fmt_args(state=state)}) ---", flush=True)
    return [*state.get("messages", []), *state.get("turn_messages", [])]


def _message_text(message: BaseMessage) -> str:
    print(f"--- [clinic] _message_text({fmt_args(message=message)}) ---", flush=True)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def _latest_user_text(state: ClinicState) -> str:
    print(f"--- [clinic] _latest_user_text({fmt_args(state=state)}) ---", flush=True)
    for message in reversed(_transcript(state)):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def _format_transcript(state: ClinicState) -> str:
    print(f"--- [clinic] _format_transcript({fmt_args(state=state)}) ---", flush=True)
    lines = []
    for message in _transcript(state):
        if isinstance(message, HumanMessage):
            lines.append(f"用户：{_message_text(message)}")
        elif isinstance(message, AIMessage):
            text = _message_text(message)
            if text:
                lines.append(f"助手：{text}")
    return "\n".join(lines)


def _user_age(state: ClinicState) -> int | None:
    print(f"--- [clinic] _user_age({fmt_args(state=state)}) ---", flush=True)
    age = state.get("user_info", {}).get("age")
    return age if isinstance(age, int) else None


def _format_collected(collected: dict) -> str:
    print(f"--- [clinic] _format_collected({fmt_args(collected=collected)}) ---", flush=True)
    lines = []
    for key, label in _FIELD_LABELS.items():
        value = collected.get(key)
        if not value:
            continue
        rendered = "、".join(value) if isinstance(value, list) else str(value)
        lines.append(f"- {label}：{rendered}")
    return "\n".join(lines) or "- （暂无结构化要点）"


def _missing_fields(collected: dict) -> list[str]:
    print(f"--- [clinic] _missing_fields({fmt_args(collected=collected)}) ---", flush=True)
    return [field for field in REQUIRED_FIELDS if not collected.get(field)]


def _clinic_card(recommendation: dict) -> dict:
    """构建前端渲染的 ``clinic_recommendation`` 卡片载荷。"""
    print(f"--- [clinic] _clinic_card({fmt_args(recommendation=recommendation)}) ---", flush=True)
    urgency = recommendation.get("urgency", "routine")
    return {
        "type": "card",
        "payload": {
            "type": "clinic_recommendation",
            "data": {
                "summary": recommendation.get("summary", ""),
                "departments": recommendation.get("department", []),
                "severity": _URGENCY_TO_SEVERITY.get(urgency, "low"),
                "urgency": urgency,
                "notes": recommendation.get("notes", []),
            },
        },
    }


# ─── 节点 ────────────────────────────────────────────────────────────────────

async def emergency_gate(state: ClinicState) -> dict:
    """
    代码级红色警报关卡。直接运行 ``emergency_triage`` skill ——
    绝不经过 LLM —— 从而保证 CRITICAL 情况始终会短路整个流程。
    """
    print(f"--- [clinic] emergency_gate({fmt_args(state=state)}) ---", flush=True)
    result = EmergencyTriageSkill().run(
        symptoms_text=_latest_user_text(state),
        age=_user_age(state),
    )
    if result.level != "CRITICAL":
        return {"emergency_level": result.level}

    flags = "、".join(result.triggered_flags)
    safety_text = result.safety_message
    if flags:
        safety_text += f"\n\n识别到的危险信号：{flags}。"
    safety_text += f"\n\n{result.disclaimer}"

    recommendation = {
        "department": ["急诊科"],
        "urgency": "emergency",
        "summary": f"检测到危急征象（{flags or '危急症状'}），请立即拨打120或前往最近急诊。",
        "notes": ["不要自行驾车前往医院", "保持电话畅通并有人陪同", "记录症状开始时间供医生参考"],
    }

    _emit({"type": "text", "content": safety_text})
    _emit(_clinic_card(recommendation))

    answer = AIMessage(content=safety_text)
    return {
        "emergency_level": result.level,
        "turn_messages": [answer],
        "messages": [answer],
        "recommendation": recommendation,
    }


def route_after_gate(state: ClinicState) -> str:
    """CRITICAL 会立即结束子图；其他情况则开始问诊。"""
    print(f"--- [clinic] route_after_gate({fmt_args(state=state)}) ---", flush=True)
    return END if state.get("emergency_level") == "CRITICAL" else "extract_symptoms"


async def extract_symptoms(state: ClinicState) -> dict:
    """从对话记录中抽取结构化症状要点并合并。"""
    print(f"--- [clinic] extract_symptoms({fmt_args(state=state)}) ---", flush=True)
    transcript = _format_transcript(state)
    if not transcript:
        return {}

    llm = get_chat_llm("precise", streaming=False)
    try:
        facts = await _structured_output(llm, SymptomFacts).ainvoke(
            [
                SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=f"预问诊对话：\n{transcript}"),
            ]
        )
    except Exception as exc:  # pragma: no cover - 依赖 provider/网络
        print(f"--- [Clinic] symptom extraction failed: {exc} ---", flush=True)
        return {}

    if isinstance(facts, SymptomFacts):
        return {"collected": facts.model_dump()}
    if isinstance(facts, dict):
        return {"collected": facts}
    return {}


async def check_sufficiency(state: ClinicState) -> dict:
    """确定性关卡：是否所有必填字段都已具备（或重试次数已耗尽）？"""
    print(f"--- [clinic] check_sufficiency({fmt_args(state=state)}) ---", flush=True)
    collected = state.get("collected", {})
    missing = _missing_fields(collected)
    exhausted = state.get("followup_count", 0) >= MAX_FOLLOWUPS
    return {
        "missing_fields": missing,
        "is_sufficient": not missing or exhausted,
    }


def route_after_sufficiency(state: ClinicState) -> str:
    """当问诊已收集到所需信息时，路由到 ``conclude``。"""
    print(f"--- [clinic] route_after_sufficiency({fmt_args(state=state)}) ---", flush=True)
    return "conclude" if state.get("is_sufficient") else "ask_followup"


async def ask_followup(state: ClinicState) -> dict:
    """针对缺失信息，只写一个温和的追问。"""
    print(f"--- [clinic] ask_followup({fmt_args(state=state)}) ---", flush=True)
    missing = state.get("missing_fields") or list(REQUIRED_FIELDS)
    wanted = "、".join(_FIELD_LABELS.get(field, field) for field in missing)
    question = f"方便再多说一点吗？想了解一下您的{wanted}。"

    llm = get_chat_llm("balanced", streaming=False)
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"已收集到的信息：\n{_format_collected(state.get('collected', {}))}\n\n"
                        f"还缺少：{wanted}\n\n"
                        f"对话记录：\n{_format_transcript(state)}"
                    )
                ),
            ]
        )
        text = _message_text(response).strip()
        if text:
            question = text
    except Exception as exc:  # pragma: no cover - 依赖 provider/网络
        print(f"--- [Clinic] follow-up generation failed: {exc} ---", flush=True)

    return {
        "pending_question": question,
        "turn_messages": [AIMessage(content=question)],
    }


async def await_answer(state: ClinicState) -> dict:
    """
    挂起整个图，直到用户作答。

    ``interrupt()`` 会把待处理的问题持久化到 checkpoint；同一 ``thread_id``
    上的下一次请求会以 ``Command(resume=<answer>)`` 恢复，答案则通过
    ``extract_symptoms`` 重新进入问诊流程。
    """
    print(f"--- [clinic] await_answer({fmt_args(state=state)}) ---", flush=True)
    question = state.get("pending_question", "")
    answer = interrupt(question)
    return {
        "turn_messages": [HumanMessage(content=str(answer))],
        "followup_count": state.get("followup_count", 0) + 1,
        "pending_question": "",
    }


async def _build_system_prompt(state: ClinicState) -> str:
    print(f"--- [clinic] _build_system_prompt({fmt_args(state=state)}) ---", flush=True)
    user_info = state.get("user_info", {})
    name = user_info.get("name", "您")
    age = user_info.get("age", "")
    history = user_info.get("medical_history", "无")
    lang_note = "请使用极度通俗易懂的语言。" if user_info.get("elder_mode") else ""

    system = f"""{CLINIC_SYSTEM_PROMPT}

【当前用户】姓名：{name}  年龄：{age}  既往病史：{history}
{lang_note}

【系统已结构化的症状要点】
{_format_collected(state.get("collected", {}))}
【系统急症安全评估】{state.get("emergency_level", "NON_URGENT")}"""

    query = _latest_user_text(state) or state.get("collected", {}).get("chief_complaint", "")
    if query:
        kb = get_knowledge_base()
        docs = await kb.amulti_query_retrieve(
            [query, f"{query} 疾病 症状 诊断", f"{query} 就诊科室 治疗"],
            k=5,
        )
        rag_context = kb.format_context(docs)
        if rag_context:
            system += (
                "\n\n## 参考临床知识\n"
                "以下为相关疾病知识，用于辅助分诊判断（不要直接引用，结合症状灵活使用）：\n\n"
                f"{rag_context}"
            )
    return system


async def _summarize_recommendation(triage_text: str, state: ClinicState) -> dict:
    """从 ReAct agent 的回答中提炼出结构化分诊卡片。"""
    print(f"--- [clinic] _summarize_recommendation({fmt_args(triage_text=triage_text, state=state)}) ---", flush=True)
    llm = get_chat_llm("precise", streaming=False)
    try:
        result = await _structured_output(llm, TriageRecommendation).ainvoke(
            [
                SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"症状要点：\n{_format_collected(state.get('collected', {}))}\n\n"
                        f"预问诊结论：\n{triage_text}"
                    )
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - 依赖 provider/网络
        print(f"--- [Clinic] recommendation summary failed: {exc} ---", flush=True)
        return {}

    if isinstance(result, TriageRecommendation):
        return result.model_dump()
    return result if isinstance(result, dict) else {}


async def conclude(state: ClinicState) -> Command | dict:
    """
    ReAct 分诊回答 + 结构化 ``clinic_recommendation`` 卡片。

    诊所是一个挂载的子图，因此这里的 handoff 通过
    ``Command(graph=Command.PARENT, ...)`` 离开，而不是普通节点 ``Command``。
    """
    print(f"--- [clinic] conclude({fmt_args(state=state)}) ---", flush=True)
    system = await _build_system_prompt(state) + handoff_prompt_section(AGENT_ID)
    skill_tools = get_agent_tools(tags=["clinic"])
    agent = create_react_agent(
        get_chat_llm("balanced"),
        tools=[load_skill, *skill_tools, *get_handoff_tools(AGENT_ID)],
        prompt=SystemMessage(content=system),
    )

    transcript = _transcript(state)
    sub_result = await agent.ainvoke({"messages": transcript})
    produced = list(sub_result["messages"][len(transcript):])

    handoff = apply_handoff(
        state,
        produced,
        parent=True,
        prefix_messages=state.get("turn_messages", []),
    )
    if isinstance(handoff, Command):
        return handoff

    triage_text = next(
        (
            _message_text(message)
            for message in reversed(produced)
            if isinstance(message, AIMessage) and _message_text(message).strip()
        ),
        "",
    )

    update: dict = {
        "turn_messages": produced,
        "messages": [*state.get("turn_messages", []), *produced],
    }
    if triage_text:
        recommendation = await _summarize_recommendation(triage_text, state)
        if recommendation:
            update["recommendation"] = recommendation
            _emit(_clinic_card(recommendation))
    return update


# ─── 子图组装 ────────────────────────────────────────────────────────

def build_clinic_graph() -> StateGraph:
    print("--- [clinic] build_clinic_graph() ---", flush=True)
    workflow = StateGraph(ClinicState)

    workflow.add_node("emergency_gate", emergency_gate)
    workflow.add_node("extract_symptoms", extract_symptoms)
    workflow.add_node("check_sufficiency", check_sufficiency)
    workflow.add_node("ask_followup", ask_followup)
    workflow.add_node("await_answer", await_answer)
    workflow.add_node("conclude", conclude)

    workflow.add_edge(START, "emergency_gate")
    workflow.add_conditional_edges(
        "emergency_gate",
        route_after_gate,
        {"extract_symptoms": "extract_symptoms", END: END},
    )
    workflow.add_edge("extract_symptoms", "check_sufficiency")
    workflow.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {"ask_followup": "ask_followup", "conclude": "conclude"},
    )
    workflow.add_edge("ask_followup", "await_answer")
    workflow.add_edge("await_answer", "extract_symptoms")
    workflow.add_edge("conclude", END)

    return workflow.compile()


# 编译好的子图，在主图中挂载为 `clinic_node`。
clinic_node = build_clinic_graph()
