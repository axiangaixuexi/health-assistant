---
name: medication_calculator
description: |
  当用户询问用药剂量、根据体重或年龄该吃多少药，或当 pharmacy_agent / clinic_agent
  需要核实安全剂量范围时，使用本技能。由“我体重60公斤该吃多少剂量”“儿童用量怎么算”
  “这个药的最大剂量是多少”“按体重算应该用多少”等说法触发。支持常见药物类别的
  按体重与按年龄给药。始终返回免责声明，并建议向执业药师或医生确认。
tags: [pharmacy, clinic]
version: "1.0"
---

# 用药剂量计算技能（Medication Calculator Skill）

使用标准临床用药参考（如 BNF、UpToDate 指南）计算常见药物类别的按体重和按年龄给药剂量。

## 支持的药物类别

| 类别 | 示例 | 给药方式 |
|-------|---------|---------------|
| 解热镇痛 NSAIDs | 布洛芬, 对乙酰氨基酚 | 按体重 (mg/kg) |
| 抗生素 Antibiotics | 阿莫西林, 头孢克洛 | 按体重 (mg/kg/day) |
| 抗组胺 Antihistamines | 氯苯那敏, 氯雷他定 | 按年龄/体重 |
| 消化道 GI | 奥美拉唑, 多潘立酮 | 成人固定剂量 / 儿童按体重调整 |
| 降压 Antihypertensives | 氨氯地平, 依那普利 | 固定剂量滴定（请遵医嘱） |

## 必需输入

- `drug_name` —— 药品通用名（如 布洛芬、阿莫西林）
- `weight_kg` —— 患者体重
- `age_years` —— 患者年龄

## 可选输入

- `indication` —— 用药原因（可能选择不同的给药方案）
- `renal_impairment` —— `"mild" | "moderate" | "severe"`，用于给出剂量调整说明
- `is_pregnant` —— 触发妊娠安全分级

## 输出说明

- 剂量以范围（低–高）加频次的形式展示
- **处方药**：输出中包含强制性的“需凭处方”警示
- **儿科**：始终按每公斤计算剂量；强制封顶最大剂量
- 如适用，始终展示 `disclaimer` 和 `max_dose_warning`
- 建议在发药前与执业药师确认

## 安全规则

- 绝不超过既定的每日最大剂量上限
- 当 `renal_impairment == "severe"`：输出减量说明并转诊医生
- 妊娠：展示 FDA/CFDA 分级与安全说明
- 当年龄 < 2 岁：明确说明“需在医生指导下使用”
