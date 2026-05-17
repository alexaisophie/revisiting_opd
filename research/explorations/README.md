# OPD 论文探索总入口

最后更新：2026-05-17

这里是围绕本地 `revisiting_opd` 项目推进论文探索的总入口。

## 当前决策

把已有的 **TrustOPD** 计划和前面调研得到的 **Outcome/Uncertainty-Gated Local Support OPD** 合并为一个主线，不拆成两个第一方向。

主线：

1. **TrustOPD / Gated Local Support**：`01_trustopd_gated_support/`

保留两个副线：

2. **Adaptive Prefix OPD**：`02_adaptive_prefix_opd/`
3. **Critical-State Agentic OPD**：`03_critical_state_agentic_opd/`

副线不是当前优先实现目标。只有当主线的 metric-only 诊断、smoke 训练和至少一个 baseline 对比跑通后，才进入副线。

## 推荐阅读顺序

1. `00_coordination/README.md`
2. `01_trustopd_gated_support/TRACKING.md`
3. `../opd_citations/opd_followup_research_report.md`
4. `../../docs/algo/trustopd_research_plan_zh.md`

## 本地资源

文献 PDF：

- `../opd_citations/papers/`

文献调研总结：

- `../opd_citations/opd_followup_research_report.md`

模型：

- 学生模型：`/root/autodl-tmp/Qwen2.5-1.5B-Instruct`
- 教师模型：`/root/autodl-tmp/OpenThinker3-7B`

数据：

- 训练 smoke：`data/math_opd/train_smoke.parquet`
- 训练主集：`data/math_opd/train.parquet`
- 评测 smoke：`data/eval_math/eval_smoke.parquet`
- 完整评测：`data/eval_math/math_500.parquet`、`gsm8k.parquet`、`aime_2024.parquet`、`aime_2025.parquet`

## 后续执行规则

每次继续推进时：

1. 先打开 `00_coordination/README.md`。
2. 再打开 `01_trustopd_gated_support/TRACKING.md`。
3. 从当前阶段的第一个未完成任务开始。
4. 每完成一次代码改动或实验，都更新对应 `TRACKING.md` 的任务状态和实验日志。
5. 不要在 metric-only 诊断跑通前启动大规模训练或 8 卡规划。

