---
name: health_calculator
description: |
  当用户提供了身高、体重、年龄和性别，并询问 BMI、理想体重、每日热量需求（TDEE）
  或腰围身高比时，使用本技能。当用户问“我是不是超重了？”“我每天该吃多少热量？”
  “我的健康体重范围是多少？”等问题时也会触发。适用于 advisor_agent 和 clinic_agent。
tags: [advisor, clinic]
version: "1.0"
---

# 健康计算器技能（Health Calculator Skill）

根据用户提供的测量数据，用经过验证的临床公式计算身体指标。所有计算都是统计学估算，
必须随附内置的免责声明。

## 能力

| 指标 | 公式 | 输出 |
|--------|---------|--------|
| BMI | 体重(kg) / 身高(m)² | 数值 + WHO/亚洲分类 |
| 理想体重 | Devine 公式 | kg + 与当前体重的差值 |
| BMR | Mifflin-St Jeor 公式 | 静息状态下的 kcal/天 |
| TDEE | BMR × 活动系数 | 按生活方式调整后的 kcal/天 |
| 腰围身高比 | 腰围(cm) / 身高(cm) | 比值 + 心血管风险分级 |

## 必需输入

- `height_cm` —— 身高（厘米，50–300）
- `weight_kg` —— 体重（千克，1–500）
- `age` —— 年龄（岁）
- `gender` —— `"male"` 或 `"female"`

## 可选输入

- `activity_level` —— 取 `sedentary | light | moderate | active | very_active` 之一
- `waist_cm` —— 腰围（用于腰围身高比的心血管风险评估，会增加 `whr_risk` 字段）

## 输出说明

- BMI 分类使用**亚洲切点**（超重 ≥24.0，肥胖 ≥28.0）
- 始终随结果一起展示 `disclaimer` 字段
- `weight_diff_kg` 为正数表示需要减重；为负数表示需要增重
