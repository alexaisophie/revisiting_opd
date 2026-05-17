# OPD 探索协调文档

最后更新：2026-05-17

这个目录是当前 OPD 论文探索的“总控台”。后续 Codex 会话应从这里开始，确认当前主线、阶段、未完成任务和推进顺序。

## 方向决策

当前不拆分两个第一方向，而是合并为一个主线：

**TrustOPD：基于结果、置信度和前缀风险的 Local Support OPD 选择/加权方法。**

合并理由：

- 现有 `docs/algo/trustopd_research_plan_zh.md` 的核心命题是：dense teacher supervision 并不总是可靠，需要按 token/prefix 位置选择或加权。
- 前面新调研得到的 “Outcome/Uncertainty-Gated Local Support OPD” 也是同一个命题，只是补充了 outcome correctness、top-K overlap、teacher rank、stop-gradient TopK、entropy-aware KL 等外部文献压力。
- 如果拆成两个第一方向，会重复实现同一套 token 级指标、重复实验 baseline，也会削弱论文叙事。
- 更稳的组织方式是：一个主线方法，多个清晰 ablation。

## 轨道划分

| 轨道 | 作用 | 是否主线 | 当前优先级 |
| --- | --- | --- | --- |
| `01_trustopd_gated_support` | 主方法：按 token/prefix 可信度选择或加权 Teacher-TopK KL | 是 | 最高 |
| `02_adaptive_prefix_opd` | 计算效率副线：自适应决定哪些 prefix/tail 需要 teacher logits | 否 | 主线诊断后 |
| `03_critical_state_agentic_opd` | agentic 副线：围绕动作边界、工具调用、状态转移加权 | 否 | 数学任务稳定后 |

## 当前本地状态

环境：

- conda 环境：`/root/autodl-tmp/conda_envs/opd-bw`
- 当前代码已改为默认使用 PyTorch SDPA，不强依赖 `flash-attn`。
- 本地单卡是 RTX PRO 6000 Blackwell，约 96GB 显存，适合 1.5B student + 7B teacher 的小规模实验。

模型：

- Student：`/root/autodl-tmp/Qwen2.5-1.5B-Instruct`
- Teacher：`/root/autodl-tmp/OpenThinker3-7B`

数据：

- `data/math_opd/train_smoke.parquet`
- `data/math_opd/train.parquet`
- `data/eval_math/eval_smoke.parquet`
- `data/eval_math/eval_all.parquet`

文献：

- PDF：`research/opd_citations/papers/`
- 调研报告：`research/opd_citations/opd_followup_research_report.md`
- 原 TrustOPD 中文计划：`docs/algo/trustopd_research_plan_zh.md`

## 从当前状态推进的总计划

### 第 0 步：不要马上改 loss

先做 `metric_only`。只计算并记录 TrustOPD 指标，不改变训练目标。

原因：

- 当前最重要的问题不是先证明涨点，而是确认哪些 token/prefix 上 teacher signal 看起来可靠。
- 如果没有诊断图，后面 soft/hard gating 的系数会变成拍脑袋。
- 这一步也能验证本地数据、模型、vLLM/SDPA、teacher logprob 路径是否跑通。

### 第 1 步：跑通 smoke 训练链路

目标：

- `train_smoke.parquet` 能启动。
- actor/ref logprob 能正常产生。
- 不出现 vocab 越界、NaN、显存 OOM。
- 能落盘 JSONL/CSV 诊断结果。

### 第 2 步：实现 soft/hard gating

在 metric-only 指标可信后，再把 trust score 接入 KL loss：

- `soft`：按权重缩放 token KL。
- `hard`：保留 trust score 最高的一部分有效 response token。

### 第 3 步：加入 outcome gating

等 loss 接入稳定后，再把 verifier outcome 接进来：

- 错误 rollout：更强 teacher supervision。
- 正确 rollout：弱化 teacher supervision，避免过度模仿 teacher 风格或引入噪声。

### 第 4 步：单卡小规模证据

用 1.5B student + 7B teacher 跑：

- sampled-token OPD
- Teacher-TopK
- TrustOPD metric-only
- TrustOPD soft
- TrustOPD hard
- TrustOPD + outcome gating

评测：

- 先 `eval_smoke`
- 再 `MATH-500` / `GSM8K`
- 最后看 `AIME 2024/2025`

### 第 5 步：决定是否上 8 卡

只有满足以下任一条件才值得上 8 卡：

- TrustOPD 准确率不低于 Teacher-TopK，同时使用更少有效 KL token。
- TrustOPD 在 response length、truncation、repetition、KL spikes 上明显更稳。
- TrustOPD 的诊断指标能预测哪些样本/位置值得蒸馏，形成明确论文图。

## 后续会话执行规则

每次继续时：

1. 打开 `../01_trustopd_gated_support/TRACKING.md`。
2. 找到当前阶段第一个未勾选任务。
3. 实现最小改动。
4. 跑最小验证。
5. 把命令、结果、问题、下一步写回 tracking 文档。

