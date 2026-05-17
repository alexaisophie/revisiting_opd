# TrustOPD 研究计划

最后更新：2026-05-16

本文档是将 Revisiting OPD 代码库转变为论文导向项目的工作计划。请将其作为后续 Codex 会话的权威来源：选择下一个未勾选的任务，实现它，运行最小的相关验证，并更新本文件。

## 1. 决策

主要方向：

**TrustOPD：面向稳定在线策略蒸馏的置信度感知 Token 与前缀选择**

核心主张：

> 密集的教师监督并非总是有用的。当教师 top-K 监督主要应用于信号具有信息量和可靠性的 token/前缀位置时，OPD 的性能会得到提升。

次要方向：

**面向长程智能体的关键状态 OPD（Critical-State OPD）**

这应被视为 TrustOPD 的扩展，而非独立的首篇论文。它关注智能体任务中的动作边界、工具调用、状态转移以及答案提交点。

## 2. 为什么要做这件事

Revisiting OPD 通过将单 token 对数比率监督替换为教师 top-K 局部支持匹配，修复了采样 token OPD 的问题。但它并未完全回答哪些位置应该接受蒸馏。

近期相关工作缩小了剩余差距：

- Revisiting OPD：教师 top-K 截断 reverse-KL、top-p  rollout、特殊 token 掩码。<https://arxiv.org/abs/2603.25562>
- Rethinking OPD：OPD 需要兼容的教师/学生思维模式以及真正新的教师能力；深层学生前缀可能使教师指导不可靠。<https://arxiv.org/abs/2604.13016>
- TIP：重要的 OPD token 来自高学生熵以及低熵/高发散区域。<https://arxiv.org/abs/2604.14084>
- Demystifying OPD：长时 on-policy rollout 可能导致长度膨胀、截断崩溃和重复饱和。<https://arxiv.org/abs/2604.08527>
- Fast OPD from Reasoning Prefixes：OPD 信号可能集中在推理前缀中，从而实现提前终止。<https://arxiv.org/abs/2602.15260>
- REOPOLD：OPD 可以被解释为策略优化，并通过松弛/选择性教师奖励来稳定。<https://arxiv.org/abs/2603.11137>
- MAD-OPD：智能体 OPD 遭受多步误差累积；多教师辩论是一种昂贵的解决方案。<https://arxiv.org/abs/2605.01347>

定位：

- 优于固定教师 top-K，因为它动态选择 token/前缀位置。
- 比仅基于熵的掩码更广泛，因为它包含教师-学生分歧和前缀可靠性。
- 与 MAD-OPD 不同，因为它保持低成本的单教师设置。
- 与 StableOPD 不同，因为它将长度/重复风险转化为 token 级监督门控。

## 3. 当前硬件快照

当前机器：

- GPU：1 x NVIDIA RTX PRO 6000 Blackwell Server Edition，约 98 GB 显存。
- CPU：Intel Xeon Platinum 8470Q，208 逻辑核心。
- 内存：约 1 TiB。
- 磁盘：`/root/autodl-tmp` 在暂存数学数据后剩余约 23 GB 可用空间。

影响：

- 该 GPU 足以进行实现测试以及中小规模 OPD 实验。
- 磁盘是眼前的瓶颈。完整的 7B 学生模型加 7B 教师模型检查点可能无法容纳，除非模型存储在外部或清理了空间。
- 初始实验应使用较短的最大响应长度和小规模子集。
- 最终论文结果应在 8 卡机器上运行。

## 4. 本地资产

`/root/autodl-tmp` 下的本地模型清单：

| 角色 | 路径 | 大小 | 架构 | 备注 |
| --- | --- | ---: | --- | --- |
| 学生模型 | `/root/autodl-tmp/Qwen2.5-1.5B-Instruct` | 约 2.9 GB | `Qwen2ForCausalLM`，bf16 | Qwen2.5-1.5B-Instruct，约 15.4 亿参数。适合 Stage 0/1 单 GPU 实验。 |
| 教师模型 | `/root/autodl-tmp/OpenThinker3-7B` | 约 15 GB | `Qwen2ForCausalLM`，bf16 | 在 OpenThoughts3-1.2M 上从 Qwen2.5-7B-Instruct 微调而来。适合 Stage 0/1 的数学/推理教师模型。 |

重要兼容性说明：

- 两个模型均使用 Qwen2 架构和 32K `max_position_embeddings`。
- 学生模型配置：词表大小 `151936`，隐藏层大小 `1536`，28 层，12 个注意力头，2 个 KV 头。
- 教师模型配置：词表大小 `152064`，隐藏层大小 `3584`，28 层，28 个注意力头，4 个 KV 头。
- `vocab.json` 哈希匹配，但 `tokenizer.json` 和 `merges.txt` 不同。
- 学生模型分词器配置中 `eos_token=<|im_end|>`；教师模型分词器配置中 `eos_token=<|endoftext|>`。两个生成配置均使用 EOS ID `[151645, 151643]`。
- 因为教师词表大于学生词表，教师 top-K 索引原则上可能超出学生词表范围。对于本地学生/教师模型对，初始冒烟测试应优先使用 `kl_topk_source=actor`，或者实现应在按教师索引收集学生 logits 之前，显式将支持限制在共享词表内。

本地数据清单：

- 原始数学文件暂存在 `data/math_raw/` 下。
- 处理后的数学 OPD 文件暂存在 `data/math_opd/` 和 `data/eval_math/` 下。
- `examples/data_preprocess/prepare_trustopd_math_data.py` 是从 `data/math_raw/` 重建处理文件的可复现本地脚本。
- `examples/data_preprocess/verl_agent_math.py` 仍仅准备 AIME 2024 评估数据；对于本项目，请使用上面的 TrustOPD 脚本。
- `examples/data_preprocess/prepare_multitask_data.py` 可以创建 ALFWorld 占位符加数学 parquet 数据，并且现在可以使用 `data/math_opd/train.parquet` 作为数学输入。

当前已有的原始数学数据：

| 数据集 | 原始路径 | 原始行数 | 备注 |
| --- | --- | ---: | --- |
| DAPO-Math | `data/math_raw/dapo-math-17k.parquet` | 1,791,700 | 从 `BytedTsinghua-SIA/DAPO-Math-17k` 下载；用作训练源。尽管仓库名为 17k，当前 parquet 实际有 179 万行。 |
| AIME 2024 | `data/math_raw/aime-2024.parquet` | 960 | 从 `BytedTsinghua-SIA/AIME-2024` 下载；这是 pass@32 扩展版本，预处理时去重为 30 道独立题目。 |
| AIME 2025 | `data/math_raw/aime-2025.parquet` | 30 | 从 `MathArena/aime_2025` 下载。 |
| MATH-500 | `data/math_raw/math-500.jsonl` | 500 | 从 `HuggingFaceH4/MATH-500` 下载。 |
| GSM8K test | `data/math_raw/gsm8k-test.parquet` | 1,319 | 从 `openai/gsm8k`，`main/test` 下载。 |

当前已有的处理后数学文件：

| 角色 | 路径 | 行数 | 备注 |
| --- | --- | ---: | --- |
| 训练集 | `data/math_opd/train.parquet` | 20,000 | 从 DAPO-Math 中采样的带随机种子样本，用于 Stage 0/1 单 GPU 实验。 |
| 训练冒烟 | `data/math_opd/train_smoke.parquet` | 512 | 快速实现/调试运行。 |
| 评估合集 | `data/math_opd/test.parquet` | 1,879 | 与 `data/eval_math/eval_all.parquet` 内容相同；可以作为 `data.val_files` 传入。 |
| AIME 2024 评估 | `data/eval_math/aime_2024.parquet` | 30 | 对 960 行原始文件去重后的独立 AIME 2024 题目。 |
| AIME 2025 评估 | `data/eval_math/aime_2025.parquet` | 30 | MathArena AIME 2025。 |
| MATH-500 评估 | `data/eval_math/math_500.parquet` | 500 | 标准 MATH-500 测试集。 |
| GSM8K 评估 | `data/eval_math/gsm8k.parquet` | 1,319 | GSM8K 测试集，`####` 后提取最终答案。 |
| 评估合集 | `data/eval_math/eval_all.parquet` | 1,879 | AIME 2024 + AIME 2025 + MATH-500 + GSM8K。 |
| 评估冒烟 | `data/eval_math/eval_smoke.parquet` | 128 | 快速验证/调试子集。 |

所有处理后的数学文件均使用相同的六列：`data_source`、`ability`、`reward_model`、`prompt`、`extra_info` 和 `env_kwargs`。`env_kwargs` 包含 `task_type=math`、`question`、`ground_truth` 和 `data_source`，因此相同文件可同时用于单任务数学和未来多任务设置。

重建命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/data_preprocess/prepare_trustopd_math_data.py \
  --train_limit 20000 \
  --train_smoke_limit 512 \
  --eval_smoke_limit 128
```

首次设置的备注：

- 在该环境中，`datasets.load_dataset(...)` 直到安装了 `socksio` 后才正常工作，因为当前代理通过 `httpx` 使用 SOCKS。已运行 `pip install socksio`。
- 通过 `hf-mirror.com` 的直接下载对所有必需的数学文件均有效。

下载/准备检查清单：

- [x] 准备符合仓库格式的小型数学训练 parquet（`prompt`、`reward_model`、`env_kwargs`）。当前文件：`data/math_opd/train.parquet`。
- [x] 准备 AIME 2024、AIME 2025、MATH-500 和 GSM8K 的评估 parquet。当前文件位于 `data/eval_math/` 下。
- [ ] 对于 ALFWorld 实验，安装 ALFWorld 依赖并运行 `alfworld-download -f`。
- [ ] 对于多任务复现，获取 ALFWorld 专用教师检查点，或明确记录 OpenThinker3-7B 被用作非专业化教师基线。
- [ ] 对于最终的 8 卡实验，获取更大的学生检查点，如 Qwen2.5-7B-Instruct 或 Qwen3 4B/8B 模型，放在外部或更大的磁盘上。除非清理了空间，否则不要将其下载到当前 50 GB 的 `/root/autodl-tmp` 卷中。
- [ ] 将模型和数据集缓存放在显式路径下，并避免在 Stage 0/1 期间生成频繁的检查点。

## 5. 方法概要

对于每个响应 token 位置 `t`，计算：

```text
student_entropy(t)
teacher_entropy(t)
teacher_student_divergence(t)
prefix_depth(t)
prefix_drift(t)
length_or_repetition_risk(t)
```

初始信任分数：

```text
trust_score(t) =
    a * normalized_student_entropy(t)
  + b * normalized_teacher_student_divergence(t)
  - c * normalized_teacher_entropy(t)
  - d * normalized_prefix_depth(t)
  - e * normalized_length_or_repetition_risk(t)
```

初始实现应同时支持：

- 硬掩码（hard mask）：按 `trust_score` 保留 top-ratio 位置；
- 软权重（soft weight）：将 token 级 KL 乘以经过裁剪/缩放的 `trust_score`。

训练目标：

```text
L_TrustOPD =
  mean_t trust_weight(t) * KL_topK(student || teacher at selected support)
```

使用现有的内存高效 top-K reverse-KL 作为基础损失。TrustOPD 应在损失聚合前改变 token 选择和 token 权重，而非改变整个训练器设计。

## 6. 实现计划

关键现有文件：

- `examples/opd/opd_math_qwen2.5-7b_it.sh`：单任务 Teacher-TopK 配方。
- `examples/opd/opd_multitask_qwen2.5-7b_it.sh`：数学 + ALFWorld 多任务配方。
- `verl/trainer/ppo/ray_trainer_multitask.py`：训练器端的熵掩码和 actor/ref logit 路由。
- `verl/trainer/ppo/core_algos.py`：内存高效的 KL 计算。
- `verl/workers/actor/dp_actor.py`：actor/ref logits、top-K 索引、logsumexp、特殊 token 掩码。

计划中的配置标志：

```text
+actor_rollout_ref.actor.trustopd_enable=True
+actor_rollout_ref.actor.trustopd_mode=soft       # soft | hard
+actor_rollout_ref.actor.trustopd_keep_ratio=0.3
+actor_rollout_ref.actor.trustopd_min_weight=0.0
+actor_rollout_ref.actor.trustopd_max_weight=1.0
+actor_rollout_ref.actor.trustopd_entropy_coef=1.0
+actor_rollout_ref.actor.trustopd_divergence_coef=1.0
+actor_rollout_ref.actor.trustopd_teacher_entropy_coef=0.5
+actor_rollout_ref.actor.trustopd_depth_coef=0.5
+actor_rollout_ref.actor.trustopd_length_risk_coef=0.0
```

推荐的本地冒烟测试覆盖参数：

```text
actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-1.5B-Instruct
actor_rollout_ref.ref.model.path=/root/autodl-tmp/OpenThinker3-7B
+actor_rollout_ref.actor.kl_topk_source=actor
data.max_response_length=2048
trainer.save_freq=-1
```

实现检查清单：

- [ ] 在训练器端添加 `trustopd_weight` 或 `trustopd_mask` 的计算。
- [ ] 从现有的 `entropys` 计算学生熵。
- [ ] 从可用的参考 top-K logits 计算教师熵。
- [ ] 从收集的 actor/ref logits 计算教师-学生分歧。
- [ ] 添加按有效响应长度归一化的前缀深度特征。
- [ ] 添加可选的重复/长度风险特征。
- [ ] 在 actor KL 损失聚合中应用 `trustopd_weight`。
- [ ] 记录保留比例、平均权重、按深度分层的分歧、以及按深度分层的 KL。
- [ ] 为 TrustOPD 数学冒烟测试添加一个小型配置/脚本变体。
- [ ] 为 TrustOPD 多任务冒烟测试添加一个小型配置/脚本变体。
- [ ] 为 top-K 索引收集添加共享词表安全检查，或者对词表不匹配的本地实验要求 `kl_topk_source=actor`。

验证检查清单：

- [ ] 单批次前向/后向冒烟测试通过。
- [ ] `trustopd_enable=False` 与原始 Teacher-TopK 行为匹配。
- [ ] 硬掩码在非填充响应 token 上保持配置的 token 比例。
- [ ] 软权重是有限的且有界的。
- [ ] 在微型运行中，至少 100 个更新步骤内 KL 损失是有限的。
- [ ] 指标包括长度、截断比例、重复代理指标和 token 保留比例。
- [ ] 本地学生/教师模型对运行时不出现越界 top-K 收集错误。

## 7. 实验计划

### Stage 0：当前机器上的实现诊断

目标：在追逐分数之前，使 TrustOPD 可测量。

任务：

- [ ] 在微型数学子集上运行原始 Teacher-TopK。
- [ ] 在微型数学子集上运行软权重（soft）TrustOPD。
- [ ] 在微型数学子集上运行硬 top-ratio 掩码（hard）TrustOPD。
- [ ] 比较损失曲线、保留比例、按前缀深度分层的 KL、响应长度。

验收标准：

- KL 或策略损失中无 NaN/Inf。
- Token 权重/掩码被记录且可解释。
- 当 TrustOPD 禁用时，原始行为被保留。

### Stage 1：单 GPU 小规模证据

目标：在有限算力下证明该方法能改善稳定性或效率。

建议设置：

- 学生模型：本地 `/root/autodl-tmp/Qwen2.5-1.5B-Instruct`。
- 教师模型：本地 `/root/autodl-tmp/OpenThinker3-7B`。
- 响应长度：从 2048 或 4096 开始。
- 数据集：MATH-500、AIME 2024/2025 子集。

基线：

- 采样 token OPD；
- Revisiting OPD Teacher-TopK；
- 仅熵掩码；
- TIP 风格熵 + 分歧；
- TrustOPD soft；
- TrustOPD hard。

指标：

- 准确率/通过率；
- 平均响应长度；
- 截断比例；
- 重复率/代理指标；
- 按前缀深度分层的 KL 方差；
- Token 保留比例；
- 训练吞吐量和峰值内存。

验收标准：

- TrustOPD 使用的有效 token 少于全 token Teacher-TopK。
- TrustOPD 在准确率上至少具有竞争力。
- TrustOPD 至少降低一项不稳定指标：截断、重复或 KL 尖峰。

### Stage 2：当前机器上的智能体扩展

目标：证明前缀/token 可靠性在长程设置中很重要。

建议设置：

- 首选 ALFWorld，因为代码库已支持它。
- 单 GPU 运行保持适中的响应长度。
- 跟踪步骤级的成功和失败模式。

额外的关键状态特征：

- 动作边界；
- 工具调用位置；
- 最终答案/动作位置；
- 步骤索引/深度；
- 无效动作前的分歧尖峰。

验收标准：

- TrustOPD 在小规模运行中不降低 ALFWorld 成功率。
- 关键状态加权使 KL 更集中在动作相关 token 周围。

### Stage 3：8 卡主结果

目标：产出论文级结果。

建议设置：

- 学生模型：Qwen2.5-7B-Instruct 或 Qwen3-4B/8B，取决于模型可用性。
- 教师模型：OpenThinker3-7B 或更强的数学教师模型。
- 任务：AIME 2024、AIME 2025、MATH-500、GSM8K、ALFWorld。
- 响应长度：包含长响应设置以暴露长度膨胀/截断问题。
- 随机种子：理想情况下 3 个；如果算力紧张，2 个种子加详细的稳定性曲线。

论文表格：

- 主结果表：准确率/成功率和成本。
- 稳定性表：长度、截断、重复、KL 尖峰。
- 消融表：仅熵、仅分歧、仅深度、熵+分歧、完整 TrustOPD。
- 效率表：保留 token 比例、峰值内存、吞吐量。

验收标准：

- TrustOPD 在主指标上击败或匹配 Teacher-TopK。
- TrustOPD 在长响应上显示出更明显的稳定性提升。
- 消融实验支持前缀深度或教师置信度作为 TIP 之外的真实贡献。

## 8. 论文大纲

工作标题：

**TrustOPD：面向稳定在线策略蒸馏的置信度感知 Token-前缀选择**

摘要主张：

OPD 受益于密集的 token 级教师反馈，但密集监督也包含低价值或有害的位置。我们提出了一种置信度感知的 token-前缀选择框架，结合学生不确定性、教师-学生分歧、教师不确定性和前缀深度风险，来决定教师 top-K 蒸馏应在何处应用。

章节：

1. 引言：密集监督并非自动可靠。
2. 背景：采样 token OPD、Teacher-TopK、OPD 失效模式。
3. 分析：token 重要性和前缀可靠性。
4. 方法：TrustOPD 评分、硬/软选择、KL 加权。
5. 实验：数学和智能体任务。
6. 消融：评分组件、硬 vs 软、top-K/top-p 支持。
7. 讨论：何时蒸馏、何时信任 RL/过程信号、局限性。

核心图表：

- 熵 vs 分歧 token 象限图。
- 按前缀深度分层的教师-学生分歧。
- 训练期间的响应长度/截断/重复情况。
- TrustOPD 选择前后的 KL 质量分布。

## 9. 风险与缓解措施

风险：TIP 已经覆盖了熵 + 分歧。

缓解措施：

- 将前缀可靠性/深度风险作为核心贡献。
- 包含 TIP 未重点关注的稳定性指标：长度膨胀、截断、重复。
- 展示深度重要的智能体扩展或长程设置。

风险：信任分数变成启发式大杂烩。

缓解措施：

- 从三个可解释的项开始：学生熵、分歧、教师熵/深度。
- 使用清晰的消融实验。
- 在学习评分器之前，优先使用简单归一化和固定系数。

风险：单 GPU 实验规模太小，缺乏说服力。

缓解措施：

- 使用单 GPU 进行实现、诊断和绘图。
- 将 8 卡运行保留用于最终结论。

风险：磁盘空间阻碍模型实验。

缓解措施：

- 如果可用，使用外部挂载的模型路径。
- 从小模型和短响应开始。
- 在 Stage 0/1 期间不要生成大量检查点。

风险：本地学生模型和教师模型使用不同的分词器元数据和词表大小。

缓解措施：

- 本地冒烟测试从 `kl_topk_source=actor` 开始。
- 如果使用教师来源的 top-K，添加显式的共享词表过滤器。
- 在分词器不匹配影响 KL 的比较实验中，保持启用特殊 token 掩码。

## 10. 下一个 Codex 任务

推荐的下一步行动：

- [ ] 添加 TrustOPD 仅指标模式：计算并记录信任分数，但不改变损失。
- [ ] 添加本地模型兼容性检查，报告词表不匹配和选定的 `kl_topk_source`。
- [ ] 准备一个微型数学训练/评估 parquet 集用于冒烟测试。
- [ ] 运行一个微型 Teacher-TopK 基线并保存诊断指标。
- [ ] 启用软权重并与仅指标日志进行比较。
- [ ] 添加硬掩码模式。
- [ ] 在 `examples/opd/` 下为 TrustOPD 数学和多任务运行创建脚本。
- [ ] 添加一个简短的 README 章节，链接本计划并解释新标志。
