---
name: risk_assessor
description: |
  当用户想了解自己长期的心血管疾病（CVD）或 2 型糖尿病风险时，使用本技能。由
  “我心脏病风险高吗”“我会得糖尿病吗”“帮我评估健康风险”等问题触发，或在了解到
  用户存在多项风险因素（吸烟、高血压、肥胖、家族史）之后调用。用于 clinic_agent 和
  advisor_agent。调用本技能前可先用 health_calculator 获取 BMI。
tags: [clinic, advisor]
version: "1.0"
---

# 风险评估技能（Risk Assessor Skill）

使用两个经过验证的临床工具评估用户 10 年疾病风险：

| 模型 | 目标 | 依据 |
|-------|--------|-------|
| 简化 Framingham | 心血管疾病（CVD） | 12 因子加权评分 |
| FINDRISC | 2 型糖尿病 | 芬兰 8 项验证量表 |

## 风险等级

| 等级 | CVD 10年 | 糖尿病 10年 | 颜色 |
|-------|----------|---------------|--------|
| 低风险 | <5% | <1% | 🟢 GREEN |
| 中等风险 | 5–15% | 1–4% | 🟡 YELLOW |
| 高风险 | 15–30% | ~17% | 🟠 ORANGE |
| 极高风险 | >30% | >33% | 🔴 RED |

## 必需输入

- `age`、`gender`
- `is_smoker`（布尔值）

## 可选但强烈建议提供

- `systolic_bp` —— 收缩压，单位 mmHg
- `bmi` —— 若未知可通过 `health_calculator` 技能获取
- `has_diabetes`、`has_hypertension`、`bp_medication`
- `family_history_cvd`、`family_history_diabetes`
- `physical_activity`、`fruit_veg_daily`
- `tc_mmol`、`hdl_mmol` —— 用于更精确的 CVD 评分

## Agent 使用说明

- 列出 `key_modifiable_factors` —— 这些是用户最高杠杆的干预点
- 将 `recommendations` 呈现为可执行的清单
- 对于高风险或极高风险：强烈建议转诊专科（心内科 / 内分泌科）
- 始终展示 `disclaimer` —— 风险模型是人群层面的估算
