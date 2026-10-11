from backend._log import fmt_args
import os
from collections.abc import Mapping
from dataclasses import field, dataclass
from typing import Literal, Any

from langchain_openai import ChatOpenAI

ModelPurpose=Literal["chat","vision"]

class LLMConfigurationError(ValueError):
    """当模型提供方的环境变量不完整时抛出。"""


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    api_key: str= field(repr=False)
    model: str
    base_url: str | None
    extra_body: Mapping[str,Any] | None = None


@dataclass(frozen=True)
class _ProviderProfile:
    api_key_envs: tuple[str, ...]
    model_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...] = ()
    vision_model_envs: tuple[str, ...] = ()
    default_base_url: str | None = None
    default_model: str | None = None
    requires_base_url: bool = False
    extra_body: Mapping[str, Any] | None = None


_PROVIDERS: dict[str, _ProviderProfile] = {
    "ark": _ProviderProfile(
        api_key_envs=("ARK_API_KEY",),
        model_envs=("ARK_MODEL_ID",),
        base_url_envs=("ARK_BASE_URL",),
        vision_model_envs=("ARK_VISION_MODEL_ID",),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-seed-1-6-flash-250828",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "openai": _ProviderProfile(
        api_key_envs=("OPENAI_API_KEY",),
        model_envs=("OPENAI_MODEL",),
        base_url_envs=("OPENAI_BASE_URL",),
        vision_model_envs=("OPENAI_VISION_MODEL",),
    ),
    "deepseek": _ProviderProfile(
        api_key_envs=("DEEPSEEK_API_KEY",),
        model_envs=("DEEPSEEK_MODEL",),
        base_url_envs=("DEEPSEEK_BASE_URL",),
        vision_model_envs=("DEEPSEEK_VISION_MODEL",),
        default_base_url="https://api.deepseek.com",
    ),
    "qwen": _ProviderProfile(
        api_key_envs=("DASHSCOPE_API_KEY",),
        model_envs=("QWEN_MODEL", "DASHSCOPE_MODEL"),
        base_url_envs=("DASHSCOPE_BASE_URL",),
        vision_model_envs=("QWEN_VISION_MODEL",),
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "zhipu": _ProviderProfile(
        api_key_envs=("ZAI_API_KEY", "ZHIPU_API_KEY"),
        model_envs=("ZHIPU_MODEL", "ZAI_MODEL"),
        base_url_envs=("ZHIPU_BASE_URL", "ZAI_BASE_URL"),
        vision_model_envs=("ZHIPU_VISION_MODEL", "ZAI_VISION_MODEL"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
    ),
    "custom": _ProviderProfile(
        api_key_envs=(),
        model_envs=(),
        requires_base_url=True,
    ),
}

_PROVIDER_ALIASES = {
    "volcengine": "ark",
    "doubao": "ark",
    "dashscope": "qwen",
    "tongyi": "qwen",
    "glm": "zhipu",
    "zai": "zhipu",
    "compatible": "custom",
    "openai-compatible": "custom",
}

_PRESETS: dict[str, float] = {
    "router": 0.0,
    "precise": 0.2,
    "balanced": 0.4,
    "fast": 0.7,
}

def _canonical_provider(raw_provider: str) -> str:
    print(f"--- [llm] _canonical_provider({fmt_args(raw_provider=raw_provider)}) ---", flush=True)
    normalized = raw_provider.strip().lower().replace("_", "-") or "ark"
    provider = _PROVIDER_ALIASES.get(normalized, normalized)
    if provider not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS)
        raise LLMConfigurationError(
            f"不支持的 LLM_PROVIDER '{raw_provider}'。支持的提供方：{supported}"
        )
    return provider


def _first_value(env: Mapping[str,str],names: tuple[str,...]) ->str:
    print(f"--- [llm] _first_value({fmt_args(env=env, names=names)}) ---", flush=True)
    for name in names:
        value=env.get(name,"").strip()
        if value:
            return value
    return ""


def resolve_model_settings(
    env: Mapping[str, str] | None = None,
    *,
    purpose: ModelPurpose = "chat",
) -> ModelSettings:
    print(f"--- [llm] resolve_model_settings({fmt_args(env=env, purpose=purpose)}) ---", flush=True)
    if purpose not in ("chat", "vision"):
        raise LLMConfigurationError("Model purpose 必须是 'chat' 或 'vision'")

    values = os.environ if env is None else env
    chat_provider = _canonical_provider(values.get("LLM_PROVIDER", "ark"))
    vision_provider_raw = values.get("VISION_PROVIDER", "")
    provider = (
        _canonical_provider(vision_provider_raw)
        if purpose == "vision" and vision_provider_raw.strip()
        else chat_provider
    )
    profile = _PROVIDERS[provider]
    inherits_chat_settings = purpose == "chat" or provider == chat_provider

    api_key_names = ("LLM_API_KEY", *profile.api_key_envs)
    model_names = ("LLM_MODEL", *profile.model_envs)
    base_url_names = ("LLM_BASE_URL", *profile.base_url_envs)

    if purpose == "vision":
        api_key_names = (
            "VISION_API_KEY",
            *(("LLM_API_KEY",) if inherits_chat_settings else ()),
            *profile.api_key_envs,
        )
        model_names = (
            "VISION_MODEL",
            *profile.vision_model_envs,
            *(("LLM_MODEL",) if inherits_chat_settings else ()),
            *profile.model_envs,
        )
        base_url_names = (
            "VISION_BASE_URL",
            *(("LLM_BASE_URL",) if inherits_chat_settings else ()),
            *profile.base_url_envs,
        )

    api_key = _first_value(values, api_key_names)
    if not api_key:
        raise LLMConfigurationError(
            f"缺少模型 API key。请配置以下之一：{', '.join(api_key_names)}"
        )

    model = _first_value(values, model_names) or profile.default_model or ""
    if not model:
        raise LLMConfigurationError(
            f"缺少模型名称。请配置以下之一：{', '.join(model_names)}"
        )

    base_url = _first_value(values, base_url_names) or profile.default_base_url
    if profile.requires_base_url and not base_url:
        raise LLMConfigurationError(
            f"缺少兼容 API 的 base URL。请配置以下之一：{', '.join(base_url_names)}"
        )

    return ModelSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=profile.extra_body,
    )



def get_chat_llm(
        preset: str="balanced",
        *,
        streaming: bool = True,
        temperature: float | None=None,
) -> ChatOpenAI:
    print(f"--- [llm] get_chat_llm({fmt_args(preset=preset, streaming=streaming, temperature=temperature)}) ---", flush=True)
    settings=resolve_model_settings()
    resolved_temp=temperature if temperature is not None else _PRESETS.get(preset,0.4)

    kwargs: dict[str,Any]={
        "api_key": settings.api_key,
        "model": settings.model,
        "temperature": resolved_temp,
        "streaming": streaming,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.extra_body:
        kwargs["extra_body"] = settings.extra_body

    return ChatOpenAI(**kwargs)

