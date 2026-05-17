# Adaptive Prefix OPD 跟踪文档

最后更新：2026-05-17

## 当前状态

副线方向。不要在 TrustOPD 主线的 metric-only 诊断跑通前实现。

## 一句话方向

用 TrustOPD 的诊断信号决定每条 rollout 中哪些 prefix/tail token 需要 teacher supervision，从而降低长推理 OPD 的 teacher-logit 计算成本。

## 和主线的关系

这个方向当前不是独立第一篇论文，而是 TrustOPD 的计算效率扩展。

只有满足以下条件时，才考虑把它独立成一篇：

- TrustOPD 诊断显示有大量 token 的 teacher signal 低价值；
- 全 token Teacher-TopK 在长 response 上计算成本明显成为瓶颈；
- 自适应 prefix budget 能在明显减少 teacher-supervised token 的同时保持性能。

## 最近相关工作

- Fast and Effective OPD from Reasoning Prefixes：固定或调度式 prefix supervision。
- Revisiting OPD：对 response token 做 local support matching。
- TrustOPD 主线：按 token/prefix 可信度打分。

本方向的新意必须是：

- 自适应 budget allocation，而不是固定 prefix length；
- 证明被选中的 prefix/tail token 对应 teacher reliability、rollout outcome 或训练稳定性；
- 明确统计 teacher-logit token 成本。

## 候选方法

为每条 rollout 分配 teacher-supervision budget：

```text
budget_i = base_budget + extra_budget * risk_i
```

其中 `risk_i` 可以来自：

- 早期 prefix divergence；
- teacher/student entropy gap；
- sampled token 在 teacher 分布下的 rank；
- repetition 或 truncation risk；
- 同 prompt 多条 rollout 的 verifier outcome。

token 选择方式：

- prefix-only：监督前 `budget_i` 个 response token；
- prefix + spikes：监督早期 prefix 和后续高风险位置；
- progressive schedule：训练早期短 prefix，后期按诊断扩展。

## Phase A：只做分析

任务：

- [ ] 从 TrustOPD metric-only artifacts 计算累计 KL/trust mass 随 position 的分布。
- [ ] 估计前 10/20/40/60% token 捕获了多少信号。
- [ ] 对比 correct vs incorrect rollout。
- [ ] 对比 fixed-prefix selection 和 top-trust token selection。

验收：

- [ ] 得到清晰表格或图，说明本地数学数据上的信号是否集中在 prefix。

## Phase B：Budgeted Training

任务：

- [ ] 实现 fixed prefix budget baseline。
- [ ] 实现基于早期诊断的 adaptive budget。
- [ ] 记录 teacher-token savings。
- [ ] 对比 all-token Teacher-TopK、TrustOPD soft 和 adaptive prefix。

验收：

- [ ] 在 smoke 或小规模训练上减少 30-50% teacher-supervised token，且性能接近；
- [ ] 或者在相同 token budget 下更稳定。

## 当前不做的事

- 不单独写 prefix 训练代码。
- 不先跑 prefix baseline。
- 不把该方向作为第一篇论文主线。

## Active Log

| 日期 | 步骤 | 结果 | 下一步 |
| --- | --- | --- | --- |
| 2026-05-17 | 创建中文跟踪文档 | 副线，依赖 TrustOPD metric-only artifacts | 等待主线 Phase 0 |

