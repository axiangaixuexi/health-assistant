from backend._log import fmt_args
from contextlib import asynccontextmanager
import inspect
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import Any
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

from backend.agents.router import is_exit_request
from backend.agents.vision import recognize_image, compose_agent_message, SCAN_TYPE_TO_AGENT, VisionInputError, \
    normalize_scan_type, validate_image_upload

load_dotenv(Path(__file__).resolve().parent / ".env")
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from backend.agents.handoff import HANDOFF_TOOL_LABELS
from backend.agents.llm import LLMConfigurationError, resolve_model_settings

from backend.observability import configure_observability


_observability_runtime = configure_observability()
if _observability_runtime.provider != "none":
    print(
        f"--- [Observability] {_observability_runtime.provider} tracing enabled ---",
        flush=True,
    )


async def _close_checkpointer() -> None:
    """在关闭时释放 checkpointer 的后端连接（sqlite）。"""
    print("--- [main] _close_checkpointer() ---", flush=True)
    connection = getattr(getattr(_master_app, "checkpointer", None), "conn", None)
    close = getattr(connection, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # pragma: no cover - 关闭时尽力而为
        print(f"--- [API] Checkpointer close failed: {exc} ---", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"--- [main] lifespan({fmt_args(_app=_app)}) ---", flush=True)
    yield
    await _close_checkpointer()
    _observability_runtime.shutdown()


app = FastAPI(title="大健康 AI 后端", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Vite 开发服务器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 延迟加载 graph 以加快启动速度
_master_app = None

def get_master_app():
    print("--- [main] get_master_app() ---", flush=True)
    global _master_app
    if _master_app is None:
        from backend.agents.graph import master_app
        _master_app = master_app
    return _master_app


# --- 工具名称映射 (Tool / Node display names) ---
_NODE_LABELS = {
    "router": "分析您的需求",
    "clinic_node": "进入预问诊模块",
    "insurance_node": "进入医保咨询模块",
    "report_node": "进入报告解读模块",
    "advisor_node": "进入健康顾问模块",
    "pharmacy_node": "进入用药咨询模块",
    # clinic 子图内部节点
    "emergency_gate": "正在进行急症安全筛查…",
    "extract_symptoms": "正在整理症状要点…",
    "check_sufficiency": "正在检查信息是否完整…",
    "ask_followup": "正在准备追问…",
    "conclude": "正在生成分诊建议…",
}

# --- 技能工具 → 前端展示标签 ---
_SKILL_LABELS: dict[str, str] = {
    # 医保工具
    "get_insurance_balance":    "正在查询医保余额…",
    "get_consumption_records":  "正在查询消费明细…",
    "get_payment_records":      "正在查询缴费记录…",
    "get_cross_region_info":    "正在查询异地就医信息…",
    "search_insurance_policy":  "正在检索医保政策知识库…",
    # 用药工具
    "search_drug_info":         "正在查询药品信息…",
    "check_drug_interaction":   "正在检查药物相互作用…",
    "find_nearby_pharmacy":     "正在查找附近药店…",
    "get_otc_recommendation":   "正在检索用药建议…",
    # 技能
    "load_skill":               "正在加载技能模块…",
    "emergency_triage":         "正在进行急症安全评估…",
    "symptom_scorer":           "正在评估症状严重程度…",
    "health_calculator":        "正在计算健康指标…",
    "lab_interpreter":          "正在解读化验指标…",
    "risk_assessor":            "正在评估慢性病风险…",
    "medication_calculator":    "正在计算用药剂量…",
    # agent 间交接（handoff）
    **HANDOFF_TOOL_LABELS,
}

_MODE_TO_AGENT = {
    "clinic": "clinic_agent",
    "insurance": "insurance_agent",
    "report": "report_agent",
    "pharmacy": "pharmacy_agent",
    "general": "advisor_agent",
    "dashboard": "advisor_agent",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class UserInfo(BaseModel):
    name: str = "用户"
    age: int | None = None
    medical_history: str = "无"
    elder_mode: bool = False
    region: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_info: UserInfo | None = None
    chat_mode: str = "general"
    # 服务端会话 id。提供后由 checkpointer 保存历史，
    # 仅把最新一条用户消息送入 graph。
    thread_id: str | None = None


def _sse_payload(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_lc_messages(messages: list[ChatMessage]) -> list[HumanMessage | AIMessage]:
    print(f"--- [main] _build_lc_messages({fmt_args(messages=messages)}) ---", flush=True)
    return [
        HumanMessage(content=msg.content)
        if msg.role == "user"
        else AIMessage(content=msg.content)
        for msg in messages
        if msg.role in ("user", "assistant") and msg.content.strip()
    ]


def _build_initial_state(
    messages: list[ChatMessage],
    user_info: UserInfo | None,
    active_agent: str,
) -> dict:
    print(f"--- [main] _build_initial_state({fmt_args(messages=messages, user_info=user_info, active_agent=active_agent)}) ---", flush=True)
    user_info_dict = user_info.model_dump() if user_info else {}
    return {
        "messages": _build_lc_messages(messages)[-10:],
        "user_info": user_info_dict,
        "next_agent": "",
        "active_agent": active_agent,
        "handoff_count": 0,
    }


def _parse_user_info_json(raw: str) -> UserInfo:
    print(f"--- [main] _parse_user_info_json({fmt_args(raw=raw)}) ---", flush=True)
    try:
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError("user_info 必须是一个对象")
        return UserInfo(**data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"user_info 格式错误: {exc}") from exc


def _parse_messages_json(raw: str) -> list[ChatMessage]:
    print(f"--- [main] _parse_messages_json({fmt_args(raw=raw)}) ---", flush=True)
    try:
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            raise ValueError("messages 必须是一个列表")
        return [ChatMessage(**item) for item in data if isinstance(item, dict)]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"messages 格式错误: {exc}") from exc


# agent 可通过 agents.streaming 直接推给客户端的 custom 流负载。
# 它们必须复用前端已识别的 SSE 事件类型——这个白名单是唯一对事件类型
# 做把关的地方。
_STREAMABLE_CUSTOM_TYPES = {"card", "text"}


def _latest_user_text(messages: list[ChatMessage]) -> str:
    print(f"--- [main] _latest_user_text({fmt_args(messages=messages)}) ---", flush=True)
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content
    return ""


def _split_stream_chunk(chunk: Any) -> tuple[str, Any] | None:
    """
    将根级 `astream` chunk 解码为 (stream_mode, payload)。

    当 `subgraphs=True` 时，根 chunk 为 `(namespace, mode, payload)`；节点级
    `on_chain_stream` 事件携带的是普通 dict，这里直接忽略。
    """
    print(f"--- [main] _split_stream_chunk({fmt_args(chunk=chunk)}) ---", flush=True)
    if not isinstance(chunk, tuple):
        return None
    if len(chunk) == 3:
        return chunk[1], chunk[2]
    if len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]
    return None


def _interrupt_question(value: Any) -> str:
    print(f"--- [main] _interrupt_question({fmt_args(value=value)}) ---", flush=True)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("question", "content", "text"):
            if isinstance(value.get(key), str):
                return value[key]
    return json.dumps(value, ensure_ascii=False, default=str)


async def _resolve_graph_input(
    config: dict,
    turn_state: dict,
    full_state: dict,
    user_text: str,
) -> Any:
    """
    决定本次请求要送入 graph 的输入。

    - 有挂起的 interrupt 且为退出语 -> None，即“丢弃该线程”
    - 线程上有挂起的 interrupt -> 用用户回答恢复该线程
    - 已知线程              -> 仅最新一条消息
    - 未知线程              -> 请求携带的完整历史
    """
    print(f"--- [main] _resolve_graph_input({fmt_args(config=config, turn_state=turn_state, full_state=full_state, user_text=user_text)}) ---", flush=True)
    master_app = get_master_app()
    try:
        snapshot = await master_app.aget_state(config)
    except Exception as exc:  # pragma: no cover - 取决于 checkpointer 后端
        print(f"--- [API] Unable to read thread state: {exc} ---", flush=True)
        return full_state

    pending = [
        interrupt
        for task in getattr(snapshot, "tasks", ())
        for interrupt in getattr(task, "interrupts", ())
    ]
    if pending:
        if is_exit_request(user_text):
            # 绝不要把退出语当作追问的回答吞掉。
            print("--- [API] Exit phrase during an interrupt, dropping thread ---", flush=True)
            return None
        print("--- [API] Resuming a pending interrupt ---", flush=True)
        return Command(resume=user_text)

    if getattr(snapshot, "values", None):
        return turn_state
    return full_state


async def _stream_agent_events(graph_input: Any, config: dict | None = None):
    print(f"--- [main] _stream_agent_events({fmt_args(graph_input=graph_input, config=config)}) ---", flush=True)
    seen_interrupts: set[str] = set()
    try:
        print("--- [API] Starting event stream ---", flush=True)
        master_app = get_master_app()
        async for event in master_app.astream_events(
            graph_input,
            config=config,
            version="v2",
            stream_mode=["updates", "custom"],
            subgraphs=True,
        ):
            kind = event["event"]
            node_name = event.get("name", "")

            if kind not in (
                "on_chat_model_stream",
                "on_chat_model_start",
                "on_chat_model_end",
                "on_chain_stream",
            ):
                print(f"--- [Event] {kind} | Node: {node_name} ---", flush=True)

            # 1. LLM 正在流式输出文本 token
            if kind == "on_chat_model_stream":
                chunk_content = event["data"]["chunk"].content
                if chunk_content:
                    yield _sse_payload({"type": "text", "content": chunk_content})

            # 1b. 根级流 chunk：agent 自身（agents/streaming.py）推送的
            #     custom 卡片/文本事件，以及 interrupt。
            elif kind == "on_chain_stream":
                decoded = _split_stream_chunk(event.get("data", {}).get("chunk"))
                if decoded is None:
                    continue
                stream_mode, stream_payload = decoded
                if stream_mode == "custom":
                    if (
                        isinstance(stream_payload, dict)
                        and stream_payload.get("type") in _STREAMABLE_CUSTOM_TYPES
                    ):
                        yield _sse_payload(stream_payload)
                elif stream_mode == "updates" and isinstance(stream_payload, dict):
                    for item in stream_payload.get("__interrupt__", ()) or ():
                        marker = str(getattr(item, "id", "") or id(item))
                        if marker in seen_interrupts:
                            continue
                        seen_interrupts.add(marker)
                        question = _interrupt_question(getattr(item, "value", item))
                        yield _sse_payload({"type": "text", "content": question})
                        yield _sse_payload({"type": "interrupt", "content": question})

            # 2. 某个 graph 节点开始执行（用于在 UI 显示 agent 状态）
            elif kind == "on_chain_start":
                label = _NODE_LABELS.get(node_name)
                if label:
                    yield _sse_payload({"type": "node_start", "node": node_name, "content": label})

            # 3. 某个 graph 节点执行完毕
            elif kind == "on_chain_end":
                if node_name in _NODE_LABELS:
                    yield _sse_payload({"type": "node_end", "node": node_name})

            # 4. 工具 / 技能调用
            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                label = _SKILL_LABELS.get(tool_name, f"正在调用：{tool_name}")
                yield _sse_payload({"type": "tool_start", "tool": tool_name, "content": label})

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                yield _sse_payload({"type": "tool_end", "tool": tool_name})
                # `card` 事件并非在此生成：每个产生卡片的工具
                #（或其 agent 节点）会自行通过 agents.streaming 写入负载，
                # 并经由上面的 `custom` 流到达。

        print("--- [API] Event stream finished successfully ---", flush=True)
        yield 'data: {"type": "finish"}\n\n'

    except Exception as e:
        print(f"--- [API] Event stream error: {e} ---", flush=True)
        yield _sse_payload({"type": "error", "content": str(e)})


@app.get("/")
async def root():
    print("--- [main] root() ---", flush=True)
    return {"message": "大健康 AI 后端 v0.4 (LangGraph)"}


@app.get("/health")
async def health():
    print("--- [main] health() ---", flush=True)
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    print(f"--- [main] chat({fmt_args(request=request)}) ---", flush=True)
    try:
        resolve_model_settings()
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=f"大模型配置错误：{exc}") from exc

    active_agent = _MODE_TO_AGENT.get(request.chat_mode, "advisor_agent")
    thread_id = (request.thread_id or "").strip() or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    full_state = _build_initial_state(request.messages, request.user_info, active_agent)
    turn_state = _build_initial_state(request.messages[-1:], request.user_info, active_agent)
    user_text = _latest_user_text(request.messages)

    async def event_generator():
        active_thread_id = thread_id
        active_config = config
        graph_input = await _resolve_graph_input(config, turn_state, full_state, user_text)
        if graph_input is None:
            # 用户在问诊中途离开：放弃这个挂起的线程，让 router 在
            # 干净的会话中看到退出语。
            active_thread_id = uuid4().hex
            active_config = {"configurable": {"thread_id": active_thread_id}}
            graph_input = full_state
        yield _sse_payload({"type": "session", "thread_id": active_thread_id})
        async for payload in _stream_agent_events(graph_input, active_config):
            yield payload

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/vision-chat")
async def vision_chat(
    file: UploadFile = File(...),
    scan_type: str = Form(...),
    user_info: str = Form("{}"),
    messages: str = Form("[]"),
    thread_id: str = Form(""),
):
    print(f"--- [main] vision_chat({fmt_args(file=file, scan_type=scan_type, user_info=user_info, messages=messages, thread_id=thread_id)}) ---", flush=True)
    normalized_scan_type = normalize_scan_type(scan_type)
    parsed_user_info = _parse_user_info_json(user_info)
    parsed_messages = _parse_messages_json(messages)

    image_bytes = await file.read()
    try:
        validate_image_upload(file.content_type, len(image_bytes))
    except VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_thread_id = thread_id.strip() or uuid4().hex
    config = {"configurable": {"thread_id": resolved_thread_id}}

    async def event_generator():
        try:
            yield _sse_payload({"type": "session", "thread_id": resolved_thread_id})
            # 图片上传永远不是退出语，因此该线程始终保留。
            yield _sse_payload({
                "type": "tool_start",
                "tool": "vision_model",
                "content": "正在识别图片内容…",
            })
            vision_text = await recognize_image(
                image_bytes,
                file.content_type or "",
                normalized_scan_type,
            )
            yield _sse_payload({"type": "tool_end", "tool": "vision_model"})

            injected_message = compose_agent_message(normalized_scan_type, vision_text)
            latest_message = ChatMessage(role="user", content=injected_message)
            vision_messages = [*parsed_messages[-9:], latest_message]
            active_agent = SCAN_TYPE_TO_AGENT[normalized_scan_type]
            full_state = _build_initial_state(vision_messages, parsed_user_info, active_agent)
            turn_state = _build_initial_state([latest_message], parsed_user_info, active_agent)
            graph_input = await _resolve_graph_input(
                config, turn_state, full_state, injected_message
            )
            async for payload in _stream_agent_events(graph_input or full_state, config):
                yield payload

        except VisionInputError as exc:
            yield _sse_payload({"type": "error", "content": str(exc)})
        except Exception as exc:
            print(f"--- [Vision] Event stream error: {exc} ---", flush=True)
            yield _sse_payload({"type": "error", "content": "图片识别失败，请稍后再试"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
