"""
技能注册表 —— 自动发现、加载并包装为 LangChain 工具。

工作原理：
  1. 导入时扫描 backend/skills/*/SKILL.md，发现已安装的技能
  2. 读取 frontmatter（name、description、tags），但不执行技能代码
  3. 首次使用时惰性导入技能类（避免启动时的重量级导入）
  4. 将每个技能包装为 LangChain @tool，以便 ReAct agent 可以调用

Agent 集成：
  from backend.skills import get_agent_tools
  tools = get_agent_tools(tags=["clinic"])   # 返回 list[BaseTool]

  # 或者让 agent 在运行时按名称调用任意技能：
  from backend.skills import load_skill_tool
  # 将 load_skill_tool 添加到 agent 的工具列表
"""
from __future__ import annotations

from backend._log import fmt_args

import importlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from backend._log import fmt_args
from backend.skills.base import BaseSkill

_SKILLS_DIR = Path(__file__).parent

# ─── SKILL.md frontmatter 解析器 ──────────────────────────────────────────────

def _parse_frontmatter(skill_md: Path) -> dict:
    """
    从 SKILL.md 文件中解析类似 YAML 的 frontmatter 块。
    返回一个至少包含 {name, description, tags} 的字典。
    """
    print(f"--- [skills] _parse_frontmatter({fmt_args(skill_md=skill_md)}) ---", flush=True)
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}

    fm: dict = {}
    block = match.group(1)

    # 简单的逐行 YAML 解析器（无外部依赖）
    current_key = None
    current_multiline: list[str] = []

    for line in block.splitlines():
        if line.startswith("  ") and current_key and current_multiline is not None:
            current_multiline.append(line.strip())
            continue

        if current_key and current_multiline:
            fm[current_key] = " ".join(current_multiline).strip()
            current_key = None
            current_multiline = []

        if ": " in line or line.endswith(":"):
            key, _, val = line.partition(": ")
            key = key.strip()
            val = val.strip()

            if val == "|":
                current_key = key
                current_multiline = []
            elif val.startswith("[") and val.endswith("]"):
                # 内联列表：[a, b, c]
                items = [i.strip() for i in val[1:-1].split(",")]
                fm[key] = items
            else:
                fm[key] = val

    if current_key and current_multiline:
        fm[current_key] = " ".join(current_multiline).strip()

    # 确保 tags 始终是列表
    if "tags" not in fm:
        fm["tags"] = []
    elif isinstance(fm["tags"], str):
        fm["tags"] = [t.strip() for t in fm["tags"].split(",")]

    return fm


# ─── 注册表 ─────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    通过扫描 SKILL.md 文件自动发现所有技能。
    提供惰性加载和 LangChain 工具包装。
    """

    def __init__(self) -> None:
        print("--- [skills] SkillRegistry.__init__() ---", flush=True)
        self._meta: dict[str, dict] = {}       # name → frontmatter 字典
        self._instances: dict[str, BaseSkill] = {}  # name → 技能实例（惰性）
        self._scan()

    def _scan(self) -> None:
        print("--- [skills] SkillRegistry._scan() ---", flush=True)
        for skill_md in sorted(_SKILLS_DIR.glob("*/SKILL.md")):
            meta = _parse_frontmatter(skill_md)
            if "name" not in meta:
                continue
            meta["_dir"] = skill_md.parent
            self._meta[meta["name"]] = meta

    def list_skills(self, tags: list[str] | None = None) -> list[dict]:
        """返回所有已注册技能的元数据，可按 tag 过滤。"""
        print(f"--- [skills] SkillRegistry.list_skills({fmt_args(tags=tags)}) ---", flush=True)
        results = []
        for meta in self._meta.values():
            if tags is None:
                results.append(meta)
            elif any(t in meta.get("tags", []) for t in tags):
                results.append(meta)
        return results

    def _load_instance(self, name: str) -> BaseSkill:
        """从技能子目录中惰性导入并实例化技能类。"""
        print(f"--- [skills] SkillRegistry._load_instance({fmt_args(name=name)}) ---", flush=True)
        if name in self._instances:
            return self._instances[name]

        meta = self._meta.get(name)
        if meta is None:
            raise KeyError(f"注册表中未找到技能 '{name}'")

        skill_dir = meta["_dir"]
        # 从技能子目录导入 skill.py
        module_name = f"backend.skills.{skill_dir.name}.skill"
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ImportError(f"无法加载技能 '{name}'：{e}") from e

        # 在模块中查找 BaseSkill 子类
        skill_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseSkill)
                and attr is not BaseSkill
            ):
                skill_cls = attr
                break

        if skill_cls is None:
            raise ImportError(f"在 {module_name} 中未找到 BaseSkill 子类")

        instance = skill_cls()
        self._instances[name] = instance
        return instance

    def get_tool(self, name: str):
        """
        返回包装指定技能的 LangChain @tool 函数。
        该工具接受一个关键字参数的 JSON 字符串。
        """
        print(f"--- [skills] SkillRegistry.get_tool({fmt_args(name=name)}) ---", flush=True)
        meta = self._meta[name]
        description = meta.get("description", f"Skill: {name}").replace("\n", " ").strip()

        registry_ref = self  # 捕获引用供闭包使用

        @tool(name, description=description)
        async def _skill_tool(params_json: str = "{}") -> str:
            """执行一个已注册的技能。params_json 是关键字参数的 JSON 对象。"""
            try:
                kwargs: dict[str, Any] = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError:
                kwargs = {}

            skill_instance = registry_ref._load_instance(name)
            result = await skill_instance.arun(**kwargs)
            return result.model_dump_json(ensure_ascii=False)

        return _skill_tool

    def get_agent_tools(self, tags: list[str]) -> list:
        """
        返回所有 tags 与所请求的 agent tags 有交集的技能的 LangChain 工具，
        可直接传给 create_react_agent()。
        """
        print(f"--- [skills] SkillRegistry.get_agent_tools({fmt_args(tags=tags)}) ---", flush=True)
        tools = []
        for meta in self._meta.values():
            skill_tags = meta.get("tags", [])
            if any(t in skill_tags for t in tags):
                tools.append(self.get_tool(meta["name"]))
        return tools


# ─── 单例 ────────────────────────────────────────────────────────────────

_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """返回进程级单例技能注册表。"""
    print("--- [skills] get_registry() ---", flush=True)
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def get_agent_tools(tags: list[str]) -> list:
    """
    便捷快捷方式：获取指定 agent tags 对应的 LangChain 工具。

    在 agent 节点中的用法：
        from backend.skills import get_agent_tools
        skill_tools = get_agent_tools(["clinic"])
        agent = create_react_agent(llm, tools=[*core_tools, *skill_tools], ...)
    """
    print(f"--- [skills] get_agent_tools({fmt_args(tags=tags)}) ---", flush=True)
    return get_registry().get_agent_tools(tags)


# ─── 运行时 load_skill 工具 ──────────────────────────────────────────────────
# 该工具可以添加到任意 agent 的工具列表中，使 agent 能够在运行时
# 按名称发现并调用技能——无需注册表预先加载所有技能。

@tool
async def load_skill(skill_name: str, params_json: str = "{}") -> str:
    """
    按名称加载并执行任意已注册技能。

    参数：
        skill_name:  技能名称（例如 'health_calculator'、'symptom_scorer'）。
        params_json: 包含技能输入参数的 JSON 对象字符串。

    返回一个 JSON 字符串，内容为技能的结构化输出，其中包含必须
    展示给用户的 'disclaimer' 字段。

    可用技能：
      health_calculator   —— BMI、理想体重、每日热量需求
      symptom_scorer      —— 症状严重程度评分 + 分诊颜色
      lab_interpreter     —— 化验值异常检测 + 临床说明
      risk_assessor       —— 10 年 CVD 与糖尿病风险（Framingham + FINDRISC）
      emergency_triage    —— 危及生命症状的快速安全关卡
      medication_calculator —— 按体重/年龄计算剂量
    """
    print(f"--- [skills] load_skill({fmt_args(skill_name=skill_name, params_json=params_json)}) ---", flush=True)
    registry = get_registry()
    try:
        instance = registry._load_instance(skill_name)
    except KeyError:
        available = [m["name"] for m in registry.list_skills()]
        return json.dumps({
            "success": False,
            "error": f"未找到技能 '{skill_name}'。",
            "available_skills": available,
        }, ensure_ascii=False)

    try:
        kwargs: dict[str, Any] = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError:
        return json.dumps({"success": False, "error": "params_json 不是有效的 JSON"})

    result = await instance.arun(**kwargs)
    return result.model_dump_json(ensure_ascii=False)
