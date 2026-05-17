# Critical-State Agentic OPD 跟踪文档

最后更新：2026-05-17

## 当前状态

副线方向。先完成数学任务上的 TrustOPD 主线，再考虑 agentic 扩展。

## 一句话方向

在长程 agentic 任务中，OPD 不应均匀监督所有生成 token，而应重点监督动作边界、工具调用、状态转移和最终答案/动作提交位置。

## 和主线的关系

这个方向复用 TrustOPD 的 trust score，但加入结构化 agentic 特征：

- action boundary；
- tool call position；
- environment step index；
- invalid action risk；
- final answer/action commitment point。

它适合作为 TrustOPD 的扩展实验或第二篇工作，不适合作为当前第一优先级。

## 最近相关工作

- Revisiting OPD：覆盖 math + agentic benchmarks，但没有深入隔离 critical state。
- MAD-OPD：指出 agentic OPD 有多步错误累积问题，但采用多教师 debate，成本较高。
- Multi-Rollout OPD：用同 prompt 多条 rollout 的成功/失败信息构造更有用的 teacher signal。
- GUI self-distillation：说明 OPD/OPSD 已经扩展到 action/grounding 场景。

本方向的新意必须是：

- 单教师、低成本的 critical-state selection；
- 证明 action-relevant token 与 reasoning filler token 应接受不同监督；
- 提供 step-level 诊断，而不仅是最终成功率。

## 前置条件

- [ ] ALFWorld 依赖安装完成。
- [ ] `alfworld-download -f` 完成，或确认已有 ALFWorld 数据路径。
- [ ] 当前 SDPA 环境下能跑一个 ALFWorld smoke。
- [ ] 明确 OpenThinker3-7B 是否作为非专用 teacher baseline；如果不合适，需要找 ALFWorld-specific teacher。

## 候选特征

| 特征 | 直觉 |
| --- | --- |
| action boundary mask | 真正影响环境转移的是动作字符串 |
| tool/action token span | teacher supervision 应集中在可执行决策上 |
| step index | 后期 step 更容易 drift |
| invalid action detector | 对纠错价值高 |
| divergence spike before action | 动作前 teacher/student 分歧可能预示错误 |
| repetition loop marker | 避免强化 wait/repeat 等退化行为 |

## Phase A：Instrumentation

任务：

- [ ] 将生成 response 切分为 reasoning/action spans。
- [ ] 标记 action boundary token。
- [ ] 记录 boundary 附近 KL/trust score。
- [ ] 对比 successful vs failed episodes。

验收：

- [ ] 诊断结果显示 action token 是否具有不同 entropy/divergence/trust profile。

## Phase B：Critical-State Weighting

任务：

- [ ] 给 action-boundary token 增加 multiplier。
- [ ] 对 repetition filler 降权。
- [ ] 对比 uniform Teacher-TopK、TrustOPD、TrustOPD + critical-state multiplier。

验收：

- [ ] 小规模 ALFWorld success 不下降。
- [ ] KL 更集中在 action-relevant token 附近。

## 当前不做的事

- 不先装/调 ALFWorld。
- 不下载新的 agent teacher。
- 不把该方向作为当前第一篇论文主线。

## Active Log

| 日期 | 步骤 | 结果 | 下一步 |
| --- | --- | --- | --- |
| 2026-05-17 | 创建中文跟踪文档 | 副线，等待数学 TrustOPD 稳定 | 暂停，先做主线 Phase 0 |

