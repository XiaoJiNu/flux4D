---
name: paper-repro
description: 系统化复现论文的技能与提示词模板，输出复现方案、模块拆解、风险与验证清单。
metadata:
  short-description: 论文复现模板与流程
---

# 论文复现 Skill

## 适用场景
- 需要复现 CV/ML 论文并生成可执行的实现方案或任务拆解
- 需要一个可复用的提示词模板，适用于 Codex/Claude/Gemini

## 使用方式
1) 让用户填写 `references/prompt_template.md` 中的输入清单。
2) 若关键项缺失，先提出不超过 8 个问题补齐。
3) 按模板中的“输出格式”生成方案；所有假设要标注。
4) 如需落地实现，再基于方案生成文件级 TODO 或代码变更。

## 资源
- `references/prompt_template.md`: 通用提示词模板（含输出格式）
- `references/flux4d_example.md`: Flux4D 示例输入（按本仓库信息填充）

## 质量要求
- 不编造：任何未在论文/用户资料中明确的细节必须标注为假设
- 可执行：给出数据路径、训练/评估步骤、最小可行版本
- 可验证：列出复现风险、依赖、验证方式与失败回滚思路
