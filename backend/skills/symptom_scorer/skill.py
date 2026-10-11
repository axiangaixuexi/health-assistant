"""
Skill: symptom_scorer

对用户报告的症状应用结构化、基于规则的评分量表，
生成严重程度评分（0–100）和分诊颜色。

评分维度（来自临床分诊文献）：
  1. 急症标志     —— 危及生命的关键词 → 立即 RED
  2. 症状数量     —— 并发症状越多 → 评分越高
  3. 持续时间     —— 持续时间越长 → 升级
  4. 疼痛评分     —— 用户自评疼痛程度
  5. 生命体征线索 —— 发热程度、血压提及

使用方：clinic_agent
"""
from __future__ import annotations

from backend._log import fmt_args
import re
from typing import Literal

from pydantic import BaseModel, Field

from backend.skills.base import BaseSkill, SkillOutput


# ─── I/O 模式 ──────────────────────────────────────────────────────────────

class SymptomScorerInput(BaseModel):
    symptoms_text: str = Field(
        description="用户描述的症状原文，如：'发烧38.5度，头痛，四肢乏力，已经两天了'"
    )
    pain_scale: int | None = Field(
        default=None, ge=0, le=10,
        description="疼痛评分 0-10（NRS量表），用户自评，可选"
    )
    duration_days: float | None = Field(
        default=None, ge=0,
        description="症状持续天数（可选，若文本中已含天数则自动解析）"
    )
    age: int | None = Field(default=None, description="患者年龄（影响风险权重）")


class SymptomScorerOutput(SkillOutput):
    score: int                            # 0–100
    triage_color: Literal["RED", "YELLOW", "GREEN"]
    triage_label: str                     # 紧急 / 较快就诊 / 可择期
    triggered_flags: list[str]            # 识别出的红色预警症状
    recommended_action: str
    estimated_wait: str                   # 建议就诊等待时间


# ─── 评分引擎 ───────────────────────────────────────────────────────────

# 红色预警 → 立即急救
_RED_FLAGS = [
    (r"剧烈胸痛|胸口剧痛|心绞痛", "胸痛（急性心梗征兆）"),
    (r"呼吸困难|喘不过气|窒息", "呼吸困难"),
    (r"意识丧失|昏迷|晕倒|不省人事", "意识障碍"),
    (r"大量出血|喷血|血流不止", "大出血"),
    (r"突发剧烈头痛|雷击样头痛", "雷击样头痛（蛛网膜下腔出血征兆）"),
    (r"口眼歪斜|半身不遂|突然说不出话", "卒中症状"),
    (r"高烧.*4[0-9]|4[0-9].*度", "超高热（≥40℃）"),
    (r"过敏.*休克|血压.*下降.*过敏|喉头水肿", "过敏性休克"),
    (r"儿童.*抽搐|婴儿.*发烧.*抽", "儿童热性惊厥"),
]

# 黄色预警 → 数小时至 1 天内就医
_YELLOW_FLAGS = [
    (r"持续.*发烧|高烧.*3[89]|3[89].*度", "持续高热"),
    (r"持续.*呕吐|止不住.*吐|吐血", "顽固性呕吐"),
    (r"胸闷|心悸|心跳.*快", "心脏症状"),
    (r"腹痛.*剧烈|肚子.*痛.*不能动", "剧烈腹痛"),
    (r"耳鸣.*听力|视力.*模糊|眼睛.*突然", "感觉器官急症"),
    (r"骨折|扭伤.*严重|关节.*脱位", "骨骼损伤"),
]

# 持续时间升级阈值（天数 → 评分加成）
_DURATION_SCORE = [(1, 0), (3, 10), (7, 20), (14, 30), (30, 40)]

_FEVER_RE = re.compile(r"(\d{2}(?:\.\d)?)[\s]*(?:度|℃|°C)")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:天|日|周|个月)")


def _parse_fever(text: str) -> float | None:
    print(f"--- [symptom_scorer] _parse_fever({fmt_args(text=text)}) ---", flush=True)
    m = _FEVER_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _parse_duration(text: str) -> float | None:
    print(f"--- [symptom_scorer] _parse_duration({fmt_args(text=text)}) ---", flush=True)
    m = _DURATION_RE.search(text)
    if m:
        val = float(m.group(1))
        if "周" in text[m.start():m.end() + 1]:
            val *= 7
        if "个月" in text[m.start():m.end() + 3]:
            val *= 30
        return val
    return None


class SymptomScorerSkill(BaseSkill):
    name = "symptom_scorer"
    description = (
        "对用户描述的症状进行结构化评分（0-100）和分诊定色（红/黄/绿）。"
        "输出包括触发的红色预警症状、推荐就诊优先级和建议等待时长。"
        "用于预问诊中辅助分诊决策。"
    )
    tags = ["clinic"]

    def run(self, **kwargs) -> SymptomScorerOutput:  # type: ignore[override]
        print(f"--- [symptom_scorer] SymptomScorerSkill.run({fmt_args(kwargs=kwargs)}) ---", flush=True)
        data = SymptomScorerInput(**kwargs)
        text = data.symptoms_text

        score = 0
        flags: list[str] = []

        # 1. 红色预警扫描
        for pattern, label in _RED_FLAGS:
            if re.search(pattern, text):
                flags.append(label)
                score += 60  # 任一红色预警 → 评分非常高

        # 2. 黄色预警扫描
        for pattern, label in _YELLOW_FLAGS:
            if re.search(pattern, text) and label not in flags:
                flags.append(label)
                score += 20

        # 3. 发热温度
        fever = _parse_fever(text)
        if fever:
            if fever >= 40:
                score += 40
            elif fever >= 39:
                score += 25
            elif fever >= 38.5:
                score += 15
            elif fever >= 38:
                score += 8

        # 4. 持续时间
        duration = data.duration_days or _parse_duration(text)
        if duration is not None:
            for threshold, add in reversed(_DURATION_SCORE):
                if duration >= threshold:
                    score += add
                    break

        # 5. 疼痛评分
        pain = data.pain_scale
        if pain is not None:
            if pain >= 8:
                score += 30
            elif pain >= 6:
                score += 15
            elif pain >= 4:
                score += 8

        # 6. 年龄修正
        age = data.age
        if age is not None:
            if age < 3 or age > 75:
                score += 15
            elif age < 12 or age > 65:
                score += 8

        score = min(score, 100)

        # 分诊颜色
        if score >= 70 or any("急性心梗" in f or "卒中" in f or "休克" in f for f in flags):
            color = "RED"
            label = "紧急——请立即拨打120或前往急诊"
            action = "立即拨打120急救电话或由人陪同前往最近急诊室，请勿自行驾车。"
            wait = "立即"
        elif score >= 35:
            color = "YELLOW"
            label = "较快就诊——建议今日内就医"
            action = "建议在4-8小时内前往医院门诊或急诊就诊，如症状加重请立即就医。"
            wait = "4-8小时内"
        else:
            color = "GREEN"
            label = "可择期就诊——症状较轻"
            action = "症状目前较轻，可先居家观察，若持续超过3天或加重应及时就医。"
            wait = "1-3天内（可先观察）"

        return SymptomScorerOutput(
            skill_name=self.name,
            success=True,
            confidence=0.75,
            disclaimer="本评分基于规则引擎，仅供参考，不替代临床医生诊断。",
            score=score,
            triage_color=color,
            triage_label=label,
            triggered_flags=flags,
            recommended_action=action,
            estimated_wait=wait,
        )
