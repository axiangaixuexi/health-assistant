from backend._log import fmt_args
import json
from datetime import date

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from backend.agents.handoff import get_handoff_tools, handoff_prompt_section, apply_handoff
from backend.agents.llm import get_chat_llm
from backend.agents.state import MainAgentState
from backend.agents.streaming import emit_card
from backend.rag.knowledge_base import get_knowledge_base

# ─── 模拟数据 ────────────────────────────────────────────────────────────────

_MOCK_USER = {
    "name": "李明",
    "id_last4": "3821",
    "region": "北京市",
    "insurance_type": "城镇职工医保",
    "company": "北京科技有限公司",
    "registered_since": "2018-03-01",
}

_MOCK_BALANCE = {
    "personal_account": 3_248.56,   # 个人账户余额（元）
    "medical_savings": 12_800.00,   # 统筹账户可用（元）
    "last_month_deposit": 320.00,   # 上月个人缴费入账
    "updated_at": str(date.today()),
}

_MOCK_CONSUMPTION = [
    {"date": "2025-02-10", "hospital": "北京协和医院", "department": "内科门诊",
     "amount": 285.50, "self_pay": 95.50, "reimbursed": 190.00, "category": "普通门诊"},
    {"date": "2025-01-28", "hospital": "北京安贞医院", "department": "药房",
     "amount": 168.00, "self_pay": 50.40, "reimbursed": 117.60, "category": "处方药品"},
    {"date": "2025-01-15", "hospital": "朝阳区社区卫生中心", "department": "全科",
     "amount": 48.00, "self_pay": 14.40, "reimbursed": 33.60, "category": "社区门诊"},
    {"date": "2024-12-22", "hospital": "北京儿童医院", "department": "检验科",
     "amount": 320.00, "self_pay": 96.00, "reimbursed": 224.00, "category": "检查检验"},
    {"date": "2024-11-05", "hospital": "北京同仁医院", "department": "眼科门诊",
     "amount": 580.00, "self_pay": 174.00, "reimbursed": 406.00, "category": "专科门诊"},
]

_MOCK_PAYMENTS = [
    {"year_month": "2026-02", "individual": 320.00, "employer": 1_580.00,
     "total": 1_900.00, "status": "已入账"},
    {"year_month": "2026-01", "individual": 320.00, "employer": 1_580.00,
     "total": 1_900.00, "status": "已入账"},
    {"year_month": "2025-12", "individual": 318.00, "employer": 1_568.00,
     "total": 1_886.00, "status": "已入账"},
    {"year_month": "2025-11", "individual": 318.00, "employer": 1_568.00,
     "total": 1_886.00, "status": "已入账"},
    {"year_month": "2025-10", "individual": 318.00, "employer": 1_568.00,
     "total": 1_886.00, "status": "已入账"},
    {"year_month": "2025-09", "individual": 318.00, "employer": 1_568.00,
     "total": 1_886.00, "status": "已入账"},
]

_MOCK_CROSS_REGION = {
    "status": "已备案",
    "filed_date": "2025-08-10",
    "city": "广州市",
    "province": "广东省",
    "valid_until": "2026-08-09",
    "designated_hospitals": [
        {"name": "中山大学附属第一医院", "level": "三甲", "type": "综合"},
        {"name": "广州市第一人民医院",  "level": "三甲", "type": "综合"},
        {"name": "南方医科大学南方医院", "level": "三甲", "type": "综合"},
    ],
    "reimbursement_rate": "门诊60%，住院75%（与本地同等待遇）",
    "how_to_use": "持身份证及医保电子凭证在定点医院直接就诊结算，无需垫付。",
}

@tool
def get_insurance_balance() -> str:
    """查询用户的医保账户余额及参保信息（个人账户余额、统筹账户等）。"""
    print("--- [insurance] get_insurance_balance() ---", flush=True)
    result = {
        "tool": "insurance_balance",
        "user": _MOCK_USER,
        **_MOCK_BALANCE,
    }
    emit_card("insurance_balance",result)
    return json.dumps(result,ensure_ascii=False)

@tool
def get_consumption_records(months: int = 3) -> str:
    """查询用户近期医保消费明细（就诊报销记录）。months 为最近几个月，默认3个月。"""
    print(f"--- [insurance] get_consumption_records({fmt_args(months=months)}) ---", flush=True)
    result = {
        "tool": "insurance_expenses",
        "user": _MOCK_USER,
        "period_months": months,
        "records": _MOCK_CONSUMPTION,
        "total_amount": sum(r["amount"] for r in _MOCK_CONSUMPTION),
        "total_self_pay": sum(r["self_pay"] for r in _MOCK_CONSUMPTION),
        "total_reimbursed": sum(r["reimbursed"] for r in _MOCK_CONSUMPTION),
    }
    emit_card("insurance_expenses", result)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_payment_records(months: int = 6) -> str:
    """查询用户的医保缴费记录，包含单位缴费和个人缴费情况。months 默认6个月。"""
    print(f"--- [insurance] get_payment_records({fmt_args(months=months)}) ---", flush=True)
    result = {
        "tool": "insurance_payments",
        "user": _MOCK_USER,
        "period_months": months,
        "records": _MOCK_PAYMENTS[:months],
        "annual_total_individual": sum(r["individual"] for r in _MOCK_PAYMENTS[:months]),
        "annual_total_employer": sum(r["employer"] for r in _MOCK_PAYMENTS[:months]),
    }
    emit_card("insurance_payments", result)
    return json.dumps(result, ensure_ascii=False)

@tool
def get_cross_region_info() -> str:
    """查询用户的异地就医备案状态及定点医院信息。"""
    print("--- [insurance] get_cross_region_info() ---", flush=True)
    result = {
        "tool": "insurance_cross_region",
        "user": _MOCK_USER,
        **_MOCK_CROSS_REGION,
    }
    emit_card("insurance_cross_region", result)
    return json.dumps(result, ensure_ascii=False)

@tool
async def search_insurance_policy(query: str) -> str:
    """搜索医疗保险政策文件，获取报销规定、覆盖范围、申请流程等政策信息。
    适用于用户询问政策性问题，如：报销比例、门慢/门特政策、家庭共济、大病保险等。"""
    print(f"--- [insurance] search_insurance_policy({fmt_args(query=query)}) ---", flush=True)
    kb = get_knowledge_base()
    docs = await kb.aretrieve(query, k=5)
    context = kb.format_context(docs)
    return context if context else "未找到相关医保政策信息，建议拨打当地医保服务热线12393咨询。"


INSURANCE_SYSTEM_PROMPT = """你是一个专业的医保政策咨询助手，熟悉中国基本医疗保险制度。

【职责范围】
- 解答医保报销政策（门诊、住院、门慢、大病等）
- 调用工具查询用户的账户余额、消费明细、缴费记录、异地就医信息
- 说明常见的医保办理流程（异地就医备案、转诊、家庭共济等）

【工具使用规则】
- 当用户询问"余额"/"账户"时，调用 get_insurance_balance
- 当用户询问"消费"/"报销"/"明细"/"看病记录"时，调用 get_consumption_records
- 当用户询问"缴费"/"扣款"/"单位缴费"时，调用 get_payment_records
- 当用户询问"异地"/"外地就医"/"备案"时，调用 get_cross_region_info
- 当用户询问政策问题（报销比例、门慢政策、大病保险、家庭共济、申请流程等）时，调用 search_insurance_policy

【回答格式】
工具调用完成后，简洁总结关键数据（1-3句）。数字已由卡片展示，不用大量复述数字。"""

AGENT_ID="insurance_agent"

_INSURANCE_TOOLS=[
    get_insurance_balance,
    get_consumption_records,
    get_payment_records,
    get_cross_region_info,
    search_insurance_policy
]


def _build_insurance_agent(llm, extra_context: str = ""):
    """构建一个限定在医保工具加 handoff 工具范围内的 ReAct 子 agent。"""
    print(f"--- [insurance] _build_insurance_agent({fmt_args(llm=llm, extra_context=extra_context)}) ---", flush=True)
    prompt = INSURANCE_SYSTEM_PROMPT + extra_context + handoff_prompt_section(AGENT_ID)
    return create_react_agent(
        llm,
        tools=[*_INSURANCE_TOOLS, *get_handoff_tools(AGENT_ID)],
        prompt=SystemMessage(content=prompt),
    )


async def insurance_node(state: MainAgentState) -> Command | dict:
    """
    Insurance agent：使用 ReAct 子 agent 调用医保工具并回答问题。
    从子图结果中提取消息以兼容 MainAgentState，
    或当子 agent 将本轮交给另一位专家时返回 ``Command``。
    """
    print(f"--- [insurance] insurance_node({fmt_args(state=state)}) ---", flush=True)
    llm = get_chat_llm("balanced")
    user_info = state.get("user_info", {})

    extra_context = ""
    region = user_info.get("region", "")
    elder_mode = user_info.get("elder_mode", False)
    if region:
        extra_context += f"\n\n用户所在地区：{region}，尽量结合当地政策作答。"
    if elder_mode:
        extra_context += "\n请使用通俗易懂的语言，避免复杂的政策术语。"

    agent = _build_insurance_agent(llm, extra_context)

    sub_result = await agent.ainvoke({"messages": state["messages"]})
    # 只提取新消息（除原始输入消息以外的所有消息）
    original_count = len(state["messages"])
    new_messages = sub_result["messages"][original_count:]
    return apply_handoff(state, new_messages)