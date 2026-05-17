# TrustOPD / Gated Local Support 跟踪文档

最后更新：2026-05-17

## 一句话方向

OPD 不应该在所有 token 上无条件使用 teacher dense supervision，而应该主要在 teacher signal 可靠、具有纠错价值、且值得计算成本的位置使用 Teacher-TopK local support matching。

## 工作标题

主标题：

**TrustOPD：面向稳定 On-Policy Distillation 的结果感知与不确定性感知 Local Support Matching**

备选标题：

**When Should the Teacher Speak? Gated Local Support Matching for On-Policy Distillation**

## 核心假设

Teacher-TopK local support matching 比 sampled-token OPD 更稳定，但“所有 token 都做 dense KL”仍会包含低价值甚至有害的位置。基于 outcome、uncertainty、teacher-student disagreement 和 prefix depth 的简单 gate/weight，可以：

- 提高训练稳定性；
- 减少有效 KL token；
- 保持或提升下游准确率；
- 解释什么时候 teacher guidance 有用，什么时候会伤害。

## 新颖性边界

这个方向必须和最近工作清楚区分：

- 相对 Revisiting OPD：不是均匀使用 Teacher-TopK，而是选择/加权 token 位置。
- 相对 TIP：不只是 entropy/divergence token selection，还加入 teacher reliability、prefix depth、outcome correctness 和稳定性指标。
- 相对 Many Faces：吸收其对 TopK reverse-KL bias/collapse 的提醒，但目标是给出可跑的 gated recipe。
- 相对 Unmasking OPD：不只做诊断分数，还把诊断信号用于训练，并验证诊断和训练收益的关系。
- 相对 Prefix OPD：不是固定只看早期 prefix，而是允许选择任意高价值 token/prefix 区域。

## 方法设计

基础 loss：

- 从 repo 现有 Teacher-TopK local support / memory-efficient KL 实现开始。
- `trustopd_enable=False` 时必须保持原行为。

第一版 trust features：

| 特征 | 来源 | 直觉 | 阶段 |
| --- | --- | --- | --- |
| student entropy | actor logits/probs | 学生不确定的位置更可教 | metric-only v0 |
| teacher entropy | ref logits/probs | teacher 高不确定性时 reverse-KL 不应过强 | metric-only v0 |
| teacher-student gap/divergence | actor/ref top-K logits | 分歧可能代表可教，也可能代表 prefix drift | metric-only v0 |
| prefix depth | response position / valid response length | 长 rollout 后段更容易 drift | metric-only v0 |
| top-K overlap | actor top-K vs teacher top-K | overlap 低代表 teacher/student 兼容性风险 | v1 |
| sampled token teacher rank | teacher 对生成 token 的排名 | teacher 低排名可能表示错误或 drift | v1 |
| rollout outcome | verifier correctness | 错误 rollout 更可能需要纠错 | v2 |
| repetition/length risk | 生成文本统计 | 避免强化重复、过长、退化 continuation | v2 |

初始 trust score：

```text
trust_score(t) =
    + a * normalize(student_entropy(t))
    + b * normalize(teacher_student_gap(t))
    - c * normalize(teacher_entropy(t))
    - d * normalize(prefix_depth(t))
```

初始模式：

- `metric_only`：只计算和记录特征，不改变 loss。
- `soft`：用裁剪后的 trust weight 乘 token KL。
- `hard`：只保留有效 response token 中 trust score 最高的一部分。

初始配置：

```text
trustopd_enable=False
trustopd_mode=metric_only
trustopd_keep_ratio=0.3
trustopd_min_weight=0.0
trustopd_max_weight=1.0
trustopd_student_entropy_coef=1.0
trustopd_gap_coef=1.0
trustopd_teacher_entropy_coef=0.5
trustopd_depth_coef=0.5
```

## 实现范围

优先关注文件：

- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/ray_trainer_multitask.py`
- `examples/opd/`
- `recipe/multitask_opd/`

当前实现约束：

- 本地 smoke 不要求 `flash-attn`。
- 当前单卡实验使用 `attn_implementation=sdpa` 和 `use_remove_padding=False`。
- 因为本地 teacher vocab 大于 student vocab，初期使用 `kl_topk_source=actor`，或在 teacher-sourced top-K 前做 shared-vocab 过滤。
- 如果不启用 W&B，诊断指标要保存为 JSONL/CSV，便于后续画图和复盘。

## 当前状态

已完成：

- [x] 环境基本可用：PyTorch 2.11 + CUDA 12.8 + Blackwell `sm_120`。
- [x] 当前代码已改为 SDPA 路径，不强依赖 `flash-attn`。
- [x] OPD 相关入口脚本已默认关闭 `use_remove_padding` 并设置 `attn_implementation=sdpa`。
- [x] 本地模型和数据已确认存在。
- [x] 文献和探索轨道已整理到 `research/`。

主要风险：

- `transformers 5.8.1` 与 repo 元数据约束 `transformers<=4.57.3` 冲突，但当前导入验证可过。
- 训练链路尚未跑过完整 smoke。
- teacher/student vocab 不完全一致，top-K gather 需要谨慎。
- 当前没有 `flash-attn`，长序列训练可能更慢、更吃显存。

## Phase 0：Metric-Only 诊断

目标：先生成 trust features，确认它们是有限、可解释、有变化的，再改训练目标。

任务：

- [ ] 增加 `trustopd_enable`、`trustopd_mode` 和 score 系数配置。
- [ ] 计算 response token 上的 student entropy。
- [ ] 在当前可用 top-K support 上计算 teacher entropy。
- [ ] 计算 teacher-student logprob gap 或近似 KL。
- [ ] 计算 normalized prefix depth。
- [ ] 按 response depth bucket 输出每个 feature 的 mean/std/min/max。
- [ ] 将每步 compact diagnostic 保存到 `outputs/trustopd_diagnostics/`。
- [ ] 新增 `examples/opd/run_trustopd_math_smoke.sh`。

验收：

- [ ] `trustopd_mode=metric_only` 可以在 `train_smoke.parquet` 上启动。
- [ ] feature summary 没有 NaN/Inf。
- [ ] `trustopd_enable=False` 路径仍可导入和运行。
- [ ] 诊断结果在不同 token 位置上有非平凡变化。

建议 smoke 命令模板：

```bash
conda activate /root/autodl-tmp/conda_envs/opd-bw

python3 -m verl.trainer.main_ppo_multitask \
  data.train_files=data/math_opd/train_smoke.parquet \
  data.val_files=data/eval_math/eval_smoke.parquet \
  actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.ref.model.path=/root/autodl-tmp/OpenThinker3-7B \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.attn_implementation=sdpa \
  +actor_rollout_ref.actor.kl_topk_source=actor \
  +actor_rollout_ref.actor.trustopd_enable=True \
  +actor_rollout_ref.actor.trustopd_mode=metric_only \
  data.max_response_length=512 \
  trainer.save_freq=-1
```

## Phase 1：Soft / Hard Gating

目标：把 trust score 接入 KL 聚合，验证 gate 能控制有效监督 token。

任务：

- [ ] 实现 `soft` token KL 加权。
- [ ] 实现 `hard` top-ratio mask。
- [ ] 记录 effective kept ratio、weighted KL、prefix bucket KL。
- [ ] 添加有限性和边界检查，保证权重 finite 且在配置范围内。
- [ ] 如果可行，在固定 batch 上验证 `trustopd_enable=False` 与原 Teacher-TopK loss 数值一致。

baseline：

- sampled-token OPD
- Teacher-TopK local support
- entropy-only mask
- entropy + gap
- TrustOPD soft
- TrustOPD hard

验收：

- [ ] soft/hard 模式都能在 smoke 数据上跑至少 20 个 update。
- [ ] effective token ratio 受配置控制。
- [ ] KL 和 policy loss 没有 NaN/Inf。

## Phase 2：Outcome-Gated Training

目标：让方法不只是 entropy/depth heuristic，而是显式利用 rollout 是否正确。

任务：

- [ ] 将 verifier outcome 传到 actor loss 侧或诊断侧。
- [ ] 分别统计 correct / incorrect rollout 的 trust features。
- [ ] 比较三种策略：全部蒸馏、错误 rollout 更强蒸馏、正确 rollout 降权蒸馏。
- [ ] 记录 outcome bucket 下的 feature 分布和 KL 分布。

第一版规则：

```text
if rollout_correct:
    trust_weight *= correct_rollout_scale   # 初始 0.25-0.5
else:
    trust_weight *= incorrect_rollout_scale # 初始 1.0
```

低风险做法：

- 先只把 outcome 用于分析，不立即参与 loss。
- 如果诊断显示 correct/incorrect 分布明显不同，再接入训练。

## Phase 3：单卡证据

目标：得到是否值得上 8 卡的证据。

设置：

- Student：`/root/autodl-tmp/Qwen2.5-1.5B-Instruct`
- Teacher：`/root/autodl-tmp/OpenThinker3-7B`
- Train：`data/math_opd/train.parquet`
- Eval：先 `data/eval_math/eval_smoke.parquet`，再 `math_500`、`gsm8k`、`aime_2024`、`aime_2025`
- Response length：先 512/1024，再 2048/4096

指标：

- eval accuracy / pass rate
- response length
- truncation ratio
- repetition proxy
- KL spikes / gradient norm
- effective kept token ratio
- throughput 和 peak memory

Go / no-go：

- 如果 TrustOPD 准确率接近或超过 Teacher-TopK，同时减少有效 token，进入 8 卡。
- 如果主要收益是节省 teacher-token 计算，转向 Adaptive Prefix 副线。
- 如果训练收益弱但诊断能预测成败，转向 compatibility/diagnostic 论文角度。

## Phase 4：8 卡主实验

不要在 Phase 3 前启动。

目标：

- 使用更大 student，例如 Qwen2.5-7B-Instruct 或 Qwen3 4B/8B。
- Teacher 使用 OpenThinker3-7B 或更强 reasoning teacher。
- 任务包括 MATH-500、GSM8K、AIME24/25，可选 ALFWorld。
- 如果算力允许，跑 2-3 个 seed。

论文表格：

- 主结果：性能与成本。
- 稳定性：长度、截断、重复、KL spike。
- ablation：entropy、gap、teacher entropy、depth、outcome。
- 长回复设置：TrustOPD 是否减少后段 drift。

## 当前立即推进计划

按优先级执行：

1. **先跑基础导入/配置检查**：确认当前 SDPA + local model path + smoke data 能进入 trainer。
2. **新增 metric-only flags**：不改 loss，只接配置。
3. **定位现有 top-K KL 计算路径**：找出 teacher/student logits 和 top-K support 在哪个 DataProto 字段里流动。
4. **插入 feature 计算**：先实现 student entropy、gap、depth 三个最容易的特征。
5. **落盘 diagnostics**：每 N step 保存 JSONL，字段要足够画图。
6. **跑 1-5 step smoke**：只验证能跑、能落盘、无 NaN。
7. **更新本文件 Active Log**。

## Active Log

| 日期 | 步骤 | 命令 / 产物 | 结果 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-05-17 | 创建中文跟踪文档 | 本文件 | 主线确定为 TrustOPD + outcome/uncertainty gated support | 实现 Phase 0 metric-only |

