# OPD 后续文献调研与论文方向报告

日期：2026-05-17

种子论文：

**Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes**，arXiv:2603.25562

本地资源：

- PDF：`research/opd_citations/papers/`
- arXiv 源文件：`research/opd_citations/sources/`
- 本地代码：`/root/autodl-tmp/revisiting_opd`
- 本地 student：`/root/autodl-tmp/Qwen2.5-1.5B-Instruct`
- 本地 teacher：`/root/autodl-tmp/OpenThinker3-7B`
- 本地数据：
  - `data/math_opd/train.parquet`
  - `data/math_opd/train_smoke.parquet`
  - `data/math_opd/test.parquet`
  - `data/eval_math/aime_2024.parquet`
  - `data/eval_math/aime_2025.parquet`
  - `data/eval_math/math_500.parquet`
  - `data/eval_math/gsm8k.parquet`
  - `data/eval_math/eval_all.parquet`

## 引用状态

这篇种子论文很新：arXiv v1 提交于 2026-03-26，v2 更新于 2026-04-27。引用索引还不完整。

Semantic Scholar 在本次会话中通过代理仍返回 HTTP 429，因此这里不把 Semantic Scholar citation count 作为依据。已确认的直接引用来自下载下来的 arXiv 源文件。

### 已确认直接引用

1. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**，arXiv:2604.13016

   - 本地 PDF：`research/opd_citations/papers/2604.13016_rethinking_opd.pdf`
   - 证据：`research/opd_citations/sources/2604.13016/src/opd.bbl` 中包含 `arXiv preprint arXiv:2603.25562`。
   - 主题：从现象、机制和 recipe 角度重新审视 OPD，强调 teacher/student thinking pattern 兼容性、teacher 新能力和 top-token alignment。

2. **The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes**，arXiv:2605.11182

   - 本地 PDF：`research/opd_citations/papers/2605.11182_many_faces_opd.pdf`
   - 证据：`research/opd_citations/sources/2605.11182/src/references.bib` 中包含种子论文，`eprint={2603.25562}`。
   - 主题：系统总结 OPD/OPSD 的成败条件，指出 prefix-induced mismatch、TopK reverse-KL bias、OPSD privileged-information aggregation 等问题。

### 未确认直接引用，但高度相关的同期/后续工作

3. **Multi-Rollout On-Policy Distillation via Peer Successes and Failures**，arXiv:2605.12652

   - 本地 PDF：`research/opd_citations/papers/2605.12652_multi_rollout_opd.pdf`
   - 在本地源文件中未检索到 `2603.25562` 或种子论文标题。
   - 相关性：利用同一 prompt 下多条 student rollout 的成功/失败信息，构造更有信息量的 teacher signal。

4. **Fast and Effective On-Policy Distillation from Reasoning Prefixes**

   - 本地 PDF：`research/opd_citations/papers/openreview_fast_effective_opd_reasoning_prefixes.pdf`
   - 相关性：从计算效率角度切入，认为 OPD 的有效信号可以集中在 reasoning prefix。

5. **Entropy-Aware On-Policy Distillation of Language Models**，arXiv:2603.07079

   - 本地 PDF：`research/opd_citations/papers/2603.07079_entropy_aware_opd.pdf`
   - 相关性：指出纯 reverse KL 在 teacher entropy 高的位置可能过度 mode-seeking，建议在高熵位置引入 forward KL。

6. **On-Policy Context Distillation for Language Models**，arXiv:2602.12275

   - 本地 PDF：`research/opd_citations/papers/2602.12275_on_policy_context_distillation.pdf`
   - 相关性：使用 context-conditioned teacher，在 student rollout 上做 reverse KL。

7. **Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why**，arXiv:2605.10889

   - 本地 PDF：`research/opd_citations/papers/2605.10889_unmasking_opd.pdf`
   - 相关性：提出 training-free token-level diagnostic，判断 teacher/context gradient 是否和成功方向一致。

8. **OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning**，arXiv:2605.12400

   - 本地 PDF：`research/opd_citations/papers/2605.12400_ogls_sd.pdf`
   - 相关性：用 outcome correctness 校准 dense token guidance。

9. **Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding**，arXiv:2605.00642

   - 本地 PDF：`research/opd_citations/papers/2605.00642_gui_sd.pdf`
   - 相关性：说明 OPD/OPSD 正在扩展到 GUI grounding、action selection 等非纯数学场景。

10. **TIP / Demystifying OPD / REOPOLD / MAD-OPD**

   - 本地 PDF 已下载到 `research/opd_citations/papers/`。
   - 这些工作分别覆盖 entropy-divergence token selection、长度膨胀/截断/重复、policy optimization 解释、多教师 agentic OPD 等方向。

## 已被占据的研究空间

### 种子论文已经覆盖

- sampled-token OPD 的一 token 信号很脆弱。
- Teacher top-K local support matching 能提供更强的局部分布监督。
- special-token masking 和 top-p rollout sampling 能提升稳定性。
- 在数学和 math+agentic multi-task 上验证。

### Rethinking OPD 已经覆盖

- OPD 能否有效取决于 teacher/student thinking pattern 是否兼容。
- teacher 需要提供 student 没有的新能力，否则 OPD 可能没有收益。
- 高概率 token 对齐、shared top-token support 很关键。
- off-policy cold start 和 teacher-aligned prompt selection 可以缓解失败。
- 对长 horizon OPD 的可扩展性持谨慎态度。

### Many Faces 已经覆盖

- OPD/OPSD 不是 universally useful。
- prefix-induced teacher-student mismatch 是重要失败源。
- TopK reverse-KL 本身也可能引入 biased gradients。
- stop-gradient TopK、RLVR-adapted teacher、SFT-stabilized student 是重要 stabilizer。

### Multi-Rollout OPD 已经覆盖

- 同 prompt 多条 rollout 中的成功/失败 peer evidence 有价值。
- 不应把每条 rollout 完全独立看待。

### Prefix OPD 已经覆盖

- 长 reasoning response 的全量 OPD 成本高。
- 有效监督可能集中在 prefix。
- prefix scheduling 可以换取计算效率。

### Entropy-Aware OPD 已经覆盖

- reverse KL 在高 teacher entropy 位置可能过度 mode-seeking。
- 高熵位置混入 forward KL 能保留多样性并提高稳定性。

### Unmasking OPD 已经覆盖

- teacher/context choice 需要 token-level 诊断。
- distillation 对错误 rollout 可能更有价值。
- 不存在通用 teacher/context recipe。

## 仍然值得切入的空白

最有希望的空白不是“再做一个 TopK OPD 变体”，而是：

> OPD 需要一个 adaptive、outcome-aware、compute-aware 的 gate，决定 teacher local support matching 应该在什么位置、以什么强度使用。

这个问题把几条最新线索连起来：

- Revisiting OPD：Teacher-TopK local support 是强 baseline。
- Rethinking OPD：teacher/student compatibility 不保证。
- Many Faces：TopK reverse-KL 也可能不稳定。
- Entropy-Aware OPD：不同 entropy 区域应该用不同 divergence 或权重。
- Unmasking OPD：teacher guidance 在错误/不确定状态更可能有用。
- Prefix OPD：长 rollout 需要控制 teacher-token 成本。
- Multi-Rollout / OGLS：outcome 和 peer signal 能校准 dense supervision。

## 推荐主方向

### TrustOPD：结果感知与不确定性感知的 Local Support OPD

核心想法：

只在 teacher signal 可能有用的位置使用或强化 Teacher-TopK local support matching。gate/weight 可以来自：

- rollout verifier outcome：正确 / 错误；
- teacher entropy；
- student entropy；
- teacher-student top-K overlap；
- sampled token 在 teacher 分布中的 rank；
- teacher-student logprob gap；
- token position / prefix depth；
- repetition / length risk。

可能的训练规则：

- 错误 rollout：更强 teacher guidance，因为更需要纠错。
- 正确 rollout：降权 teacher guidance，避免过度模仿 teacher 风格或引入噪声。
- teacher entropy 高：降低 reverse-KL 权重，或混入 forward KL。
- top-K overlap 低：降低信任，或先只记录诊断。
- late prefix / repetition 风险高：降低或重定向监督。

为什么这个方向更适合发 paper：

- 它不是简单复现 Revisiting OPD，而是在回答“Teacher-TopK 什么时候该用”。
- 它正面回应 Rethinking、Many Faces、Unmasking、Entropy-Aware 等后续批评。
- 它能在当前单卡 96GB 环境先做小规模验证。
- 如果有效，可以自然扩展到 8 卡主实验。

## 备选方向

### Adaptive Prefix OPD

问题：

全量 OPD 在长 reasoning response 上成本高。是否可以根据诊断信号动态决定监督前多少 token 或哪些 prefix/tail 区域？

适合作为：

- TrustOPD 的 compute-efficiency extension；
- 如果主线主要收益体现在节省 teacher-token 计算，也可以转成独立方向。

风险：

- Fast Prefix OPD 已经占据“prefix 够用”的基本叙事。
- 新意必须落在 adaptive budget，而不是 fixed prefix。

### Teacher-Student Compatibility Diagnostics

问题：

训练前能否用少量 rollout 预测这个 teacher 是否适合这个 student/task？

信号：

- top-K overlap；
- teacher rank；
- entropy gap；
- logprob gap；
- special-token mismatch；
- short rollout outcome。

风险：

- Unmasking OPD 已经接近这个方向。
- 如果要做，需要强调“诊断 -> 选择 objective -> 实际训练收益”的闭环。

### Critical-State Agentic OPD

问题：

在 agentic 任务中，teacher supervision 是否应该集中在动作边界、工具调用和状态转移 token？

适合作为：

- TrustOPD 主线稳定后的 agentic extension。

风险：

- ALFWorld 环境和 teacher 选择会增加工程复杂度。
- 当前第一阶段应先聚焦数学任务。

## 当前推进计划

### 阶段 0：基础验证

目标：

- 确认当前 SDPA 环境、student/teacher、本地 smoke 数据可以进入训练链路。

任务：

- [ ] 跑 base import / trainer import。
- [ ] 用 `train_smoke.parquet` 启动最小训练。
- [ ] 确认 actor/ref model path 正确。
- [ ] 确认 vocab mismatch 不导致 top-K gather 越界。

### 阶段 1：Metric-Only TrustOPD

目标：

- 不改 loss，只记录 trust features。

任务：

- [ ] 加 `trustopd_enable`、`trustopd_mode=metric_only` 配置。
- [ ] 计算 student entropy、teacher entropy、gap/divergence、prefix depth。
- [ ] 输出 JSONL/CSV 诊断文件。
- [ ] 按 token position bucket 统计特征。

验收：

- [ ] 无 NaN/Inf。
- [ ] 不同 token/prefix 区域有明显分布差异。
- [ ] 能画出至少两类图：feature by prefix depth、correct vs incorrect 或 high/low gap 对比。

### 阶段 2：Soft/Hard Gating

目标：

- 将 trust score 接入 KL loss。

任务：

- [ ] soft 权重：`KL *= trust_weight`。
- [ ] hard mask：只保留 top-ratio token。
- [ ] 对比 Teacher-TopK、entropy-only、TrustOPD soft/hard。

验收：

- [ ] smoke 训练稳定。
- [ ] TrustOPD 能控制 effective KL token ratio。
- [ ] loss 曲线无明显异常。

### 阶段 3：Outcome Gating

目标：

- 把 verifier outcome 用于指导哪里蒸馏。

任务：

- [ ] 先分析 correct / incorrect rollout 的 feature 分布。
- [ ] 再尝试 incorrect rollout 强蒸馏、correct rollout 降权。

验收：

- [ ] 诊断支持“错误 rollout 上 teacher guidance 更有价值”或发现反例。
- [ ] 至少形成一张 paper-worthy 诊断图。

### 阶段 4：单卡小规模对比

目标：

- 判断是否值得上 8 卡。

实验：

- base student；
- sampled-token OPD；
- Teacher-TopK；
- TrustOPD soft；
- TrustOPD hard；
- TrustOPD + outcome gating。

评测：

- `eval_smoke`；
- `MATH-500`；
- `GSM8K`；
- `AIME 2024/2025`。

Go 条件：

- TrustOPD 性能不低于 Teacher-TopK，且 token 使用更少；
- 或稳定性显著更好；
- 或诊断信号非常清晰，足以支撑分析型论文。

## 当前最推荐的下一步

不要马上上大训练。下一步应该是：

1. 在代码里新增 TrustOPD metric-only 配置；
2. 只接入指标计算，不改 loss；
3. 用 `train_smoke.parquet` 跑 1-5 step；
4. 落盘诊断 JSONL；
5. 根据诊断决定 soft/hard gating 的第一版公式。

