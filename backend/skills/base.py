"""
BaseSkill —— 每个技能都必须实现的抽象契约。

设计原则（来自 prompt-engineering-patterns 与 langchain-architecture 最佳实践）：
  - 使用 Pydantic I/O 模式，提供经过校验、对 LLM 友好的工具调用
  - 带 confidence 评分的结构化输出
  - 每个技能都可自我描述（name、description、tags），使 agent 能够
    根据用户意图决定是否加载该技能
"""


from backend._log import fmt_args
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class SkillInput(BaseModel):
    pass


class SkillOutput(BaseModel):
    skill_name: str
    success: bool
    confidence: float = 1.0
    disclaimer: str=""


class BaseSkill(ABC):
    """
    注册表中每个技能的抽象基类。

    子类必须声明：
      name        —— 注册表使用的唯一 snake_case 标识符
      description —— 展示给 LLM 作为工具提示的单行描述
      tags        —— 类别（例如 ['clinic', 'pharmacy']），控制哪些
                     agent 可以自动加载该技能
    """
    name: str
    description: str
    tags: list[str] = []

    @abstractmethod
    def run(self, **kwargs: Any) -> SkillOutput:
        """同步执行入口。"""
        print(f"--- [skills.base] BaseSkill.run({fmt_args(kwargs=kwargs)}) ---", flush=True)
        ...

    async def arun(self, **kwargs: Any) -> SkillOutput:
        """
        异步执行入口。
        默认实现包装 run() —— 需要真正异步工作时请重写。
        """
        print(f"--- [skills.base] BaseSkill.arun({fmt_args(kwargs=kwargs)}) ---", flush=True)
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(**kwargs))

    def to_tool_description(self) -> str:
        """返回作为工具 description 传给 LLM 的字符串。"""
        print("--- [skills.base] BaseSkill.to_tool_description() ---", flush=True)
        return f"[Skill: {self.name}] {self.description}"
