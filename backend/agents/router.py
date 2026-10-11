from backend._log import fmt_args
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.llm import get_chat_llm
from backend.agents.state import MainAgentState


ROUTER_SYSTEM_PROMPT = """你是大健康 App 的「智能意图路由器」，负责把用户的最新提问分发给最合适的专家智能体。

【身份】
- 你只做「分类与调度」，不回答任何健康问题，也不下诊断、不给建议。
- 你只会收到用户最新一条提问，请仅依据它判断，不要臆测历史上下文。

【候选专家】从下面 5 个中严格选择 1 个：
1. "clinic_agent" - 用户描述了身体不适、症状，想知道该看什么科、是否需要就医，或可能得了什么病。
2. "insurance_agent" - 用户询问医保余额、报销政策、消费明细、缴费记录、异地就医等医保事务。
3. "report_agent" - 用户让你看化验单、体检报告，或解读某项检查指标。
4. "pharmacy_agent" - 用户询问药品信息（用法用量、副作用、说明书）、药物相互作用、找药店，或按症状推荐非处方药。
5. "advisor_agent" - 其它通用健康科普、饮食作息建议，以及不属于以上任何一类的闲聊或问候。

【输出要求】
- 只输出一个分类 ID 字符串，例如：clinic_agent
- 不要输出任何解释、标点、引号、换行或 Markdown 代码块。

【示例】（输入 → 输出）
- "我最近一直咳嗽，晚上更厉害" → clinic_agent
- "我的医保卡里还剩多少钱" → insurance_agent
- "帮我看看血常规报告，血红蛋白偏低" → report_agent
- "布洛芬和头孢能一起吃吗" → pharmacy_agent
- "每天喝多少水比较合适" → advisor_agent
- "你好" → advisor_agent"""

EXIT_PHRASES = frozenset(
    {
        "退出",
        "结束",
        "结束问诊",
        "退出问诊",
        "不看了",
        "不用了",
        "我要退出",
        "取消",
        "退出诊室",
        "退出模式",
        "退出功能",
    }
)

_EXIT_BOUNDARY = " \t　，,。.！!？?；;、~～…·"

def is_exit_request(text: str) -> bool:
    print(f"--- [router] is_exit_request({fmt_args(text=text)}) ---", flush=True)
    normalized = text.strip().strip(_EXIT_BOUNDARY)
    if not normalized:
        return False
    if normalized in EXIT_PHRASES:
        return True
    for phrase in EXIT_PHRASES:
        if not normalized.startswith(phrase):
            continue
        remainder = normalized[len(phrase):]
        if remainder and remainder[0] in _EXIT_BOUNDARY:
            return True
    return False


async def router_node(state: MainAgentState) -> dict:
    print("--- [Router] Entering router_node ---", flush=True)

    last_message=state["messages"][-1]
    user_text=last_message.content if hasattr(last_message, "content") else str(last_message)

    active_agent=state.get("active_agent","")
    if is_exit_request(user_text):
        print("--- [Router] User requested exit, routing back to advisor_agent ---", flush=True)
        return {
            "next_agent": "advisor_agent",
            "active_agent": "advisor_agent",
            "handoff_count": 0,
        }

    if active_agent and active_agent not in ("advisor_agent", ""):
        print(f"--- [Router] Short-circuit: already in {active_agent}, bypassing LLM classification ---", flush=True)
        return {"next_agent": active_agent, "handoff_count": 0}

    # 3. 常规 LLM 意图分类
    llm = get_chat_llm("router", streaming=False)
    try:
        print("--- [Router] Invoking LLM... ---", flush=True)
        response = await llm.ainvoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_text)
        ])
        raw_content_lower = response.content.lower()
        # 移除所有空白（空格、换行、制表符），以处理 LLM 的格式怪癖（例如 'clin\nic_agent'）
        raw_clean = "".join(raw_content_lower.split())

        # 健壮的子串匹配
        if "clinic_agent" in raw_clean:
            next_agent = "clinic_agent"
        elif "insurance_agent" in raw_clean:
            next_agent = "insurance_agent"
        elif "report_agent" in raw_clean:
            next_agent = "report_agent"
        elif "pharmacy_agent" in raw_clean:
            next_agent = "pharmacy_agent"
        else:
            next_agent = "advisor_agent"

        print(f"--- [Router] Matched next_agent: '{next_agent}' ---", flush=True)

    except Exception as e:
        import traceback
        print(f"--- [Router] LLM error: {e} ---", flush=True)
        print(f"--- [Router] Traceback:\n{traceback.format_exc()} ---", flush=True)
        next_agent = "advisor_agent"

    print(f"--- [Router] Dispatching to: {next_agent} ---", flush=True)

    # 将决策存入 active_agent，以便后续轮次持续生效
    return {
        "next_agent": next_agent,
        "active_agent": next_agent,
        "handoff_count": 0,
    }
