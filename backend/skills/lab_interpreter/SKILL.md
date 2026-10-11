---
name: lab_interpreter
description: |
  当用户分享化验数值（血常规、血生化、甲功、凝血等），并询问其含义、是否正常或
  该如何处理时，使用本技能。由“帮我看看化验单”“我的血红蛋白偏低是什么意思”
  “这个指标正常吗”等说法触发。主要用于 report_agent；当症状与化验结果同时出现时，
  也供 clinic_agent 使用。
tags: [report, clinic]
version: "1.0"
---

# 化验解读技能（Lab Interpreter Skill）

将数值型检验结果与循证参考范围进行比对，把每项分类为 HIGH / LOW / NORMAL，
并为每项异常结果提供通俗易懂的临床意义说明。

## 支持的检验项目组

| 项目组 | 项目 |
|-------|-------|
| 血常规 (CBC) | WBC, RBC, HGB, PLT, HCT, MCV |
| 血糖 | GLU (空腹血糖), HBA1C |
| 肾功能 | CREA, BUN, UA |
| 肝功能 | ALT, AST |
| 血脂 | TC, TG, HDL, LDL |
| 甲状腺 | TSH, FT3, FT4 |
| 凝血 | PT, APTT, INR |

## 必需输入

- `items` —— `{name, value}` 对象列表，例如：
  ```json
  [{"name": "WBC", "value": 11.5}, {"name": "HGB", "value": 102}]
  ```

## 可选输入

- `patient_age` —— 调整儿科阈值
- `patient_gender` —— 调整 HGB、RBC、UA 的参考范围

## 输出说明

- `summary` 字段给出一句话的分诊结论
- 无法识别的项目名返回 `status: "UNKNOWN"` —— 不要编造参考范围
- 始终展示 `disclaimer`：参考范围因仪器和实验室而异
- 对于危急异常值（如 WBC > 30、HGB < 60），建议**立即**就医
