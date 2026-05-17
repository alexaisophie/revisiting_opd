# 原作方案 Baseline 状态记录

最后更新：2026-05-17

## 1. 为什么先固定 baseline

当前项目已经从原作仓库拉下，并做了本地环境适配：

- PyTorch 2.11 + CUDA 12.8 + Blackwell `sm_120`；
- 默认走 PyTorch SDPA；
- 暂时不安装 `flash-attn`；
- OPD 脚本里关闭 `use_remove_padding`，避免 CUDA 路径硬依赖 `flash-attn`；
- 新增中文研究计划、文献调研和探索跟踪文档。

这些改动应该被视为“实验前环境快照”，而不是 TrustOPD 方法实验本身。后续所有 baseline 和新方法实验都应从固定快照分支/tag 派生。

## 2. Git 固定点

原作 upstream base：

```text
upstream/main = 5acf13c5fa56c7c2b65ca466cdd729a1a0b3de3d
origin/main   = 5acf13c5fa56c7c2b65ca466cdd729a1a0b3de3d
```

本地实验前快照：

```text
branch = baseline/pre-experiment-sdpa-trustopd
commit = 3c22bdb0f8642e1d42c9d8495ab9fdc4a31539bc
tag    = pre-experiment-sdpa-trustopd-20260517
```

GitHub:

```text
repo   = https://github.com/alexaisophie/revisiting_opd
branch = baseline/pre-experiment-sdpa-trustopd
tag    = pre-experiment-sdpa-trustopd-20260517
```

说明：

- PDF 文献保留在本地 `research/opd_citations/papers/`，未提交到 Git，避免仓库过大。
- arXiv source 缓存保留在本地 `research/opd_citations/sources/`，未提交到 Git。

## 3. 当前代码相对原作的环境适配

这些改动不应算作 TrustOPD 算法改动：

- `attn_implementation` 默认从写死 `flash_attention_2` 改为配置项，默认 `sdpa`。
- CUDA 下 `flash_attn.bert_padding` 改为可选导入；只有 `use_remove_padding=True` 或 sequence parallel 时才硬要求。
- OPD 示例脚本默认：
  - `actor_rollout_ref.model.use_remove_padding=False`
  - `actor_rollout_ref.model.attn_implementation=sdpa`
- 修复 `verl/workers/critic/dp_critic.py` 中一个原有语法错误。
- 兼容 Transformers 5 中 `AutoModelForVision2Seq` 的重命名/迁移。
- 兼容 vLLM 0.21 的 LoRA import 路径变化，非 LoRA 路径不再因旧 import 失败。

这些改动的目的：

- 让原作 baseline 能在当前 Blackwell + PyTorch 2.11 环境跑起来；
- 不引入新的 TrustOPD gate/weight；
- 不改变原作 OPD / Teacher-TopK 的算法公式。

当前 Blackwell 单卡运行 vLLM 0.21 还需要以下环境项：

```bash
pip install nvidia-cuda-runtime==13.0.96 nvidia-cuda-nvrtc==13.0.88

export LD_LIBRARY_PATH=/root/autodl-tmp/conda_envs/opd-bw/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
```

原因：

- vLLM 0.21 wheel 会加载 CUDA 13 runtime/NVRTC 动态库；
- 当前 PyTorch 是 `2.11.0+cu128`，在 Blackwell `sm_120` 上 vLLM/FlashInfer 会提示 `SM 12.x requires CUDA >= 12.9`；
- 关闭 FlashInfer sampler 后，vLLM 使用 PyTorch-native top-k/top-p sampler，能跑通 smoke，但这不是性能最优路径。

## 4. 原作方案 baseline 定义

当前至少需要区分两个 baseline：

### Baseline A：原始 sampled-token OPD

脚本：

```text
examples/opd/opd_original_math_qwen2.5_7b_it.sh
```

关键配置：

```text
algorithm.adv_estimator=opd
actor_rollout_ref.actor.kl_loss_type=k1
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=True
```

用途：

- 对应 sampled-token OPD / 原始 OPD 风格 baseline。
- 后续 TrustOPD 论文里应作为弱 baseline。

### Baseline B：Revisiting OPD Teacher-TopK Local Support

脚本：

```text
examples/opd/opd_math_qwen2.5-7b_it.sh
```

关键配置：

```text
algorithm.adv_estimator=placeholder
actor_rollout_ref.actor.kl_loss_type=full_reverse
actor_rollout_ref.actor.kl_topk_tokens=32
actor_rollout_ref.actor.norm_to_one_for_kl=True
actor_rollout_ref.actor.kl_topk_source=ref
actor_rollout_ref.rollout.top_p=0.9
actor_rollout_ref.actor.use_kl_loss=True
algorithm.use_kl_in_reward=False
```

用途：

- 对应种子论文的核心 Teacher-TopK local support matching。
- 这是 TrustOPD 主线最重要的强 baseline。

本地注意：

- 当前 teacher vocab 大于 student vocab。原脚本 `kl_topk_source=ref` 可能触发 student logits gather 越界风险。
- 单卡 smoke 建议先覆盖为 `+actor_rollout_ref.actor.kl_topk_source=actor` 或实现 shared-vocab filter 后再使用 ref top-K。
- 论文最终应尽量恢复原作 `kl_topk_source=ref` 或清楚说明 shared-vocab filtering。

## 5. 本地单卡 baseline 获取顺序

当前机器适合先跑 smoke，而不是直接复现实验表格。

### Step 0：环境和入口检查

目标：

- 确认当前快照能启动 trainer；
- 确认模型路径、数据路径、SDPA 配置无问题。

状态：

- [x] `verl.trainer.main_ppo` 可导入。
- [x] `verl.trainer.main_ppo_multitask` 可导入。
- [x] actor/critic 模块可导入。
- [x] 当前 commit 已上传 GitHub。

### Step 1：Teacher-TopK smoke

目标：

- 用 1.5B student + 7B teacher + `train_smoke.parquet` 跑 1-5 step；
- 不追求分数，只验证原作强 baseline 训练链路能跑。

建议覆盖：

```bash
conda activate /root/autodl-tmp/conda_envs/opd-bw

export STUDENT_MODEL=/root/autodl-tmp/Qwen2.5-1.5B-Instruct
export MATH_TEACHER=/root/autodl-tmp/OpenThinker3-7B
export TRAIN_DATA=data/math_opd/train_smoke.parquet
export VAL_DATA=data/eval_math/eval_smoke.parquet
export CKPTS_DIR=/root/autodl-tmp/revisiting_opd/ckpts/baseline_teacher_topk_smoke
export LOG_DIR=/root/autodl-tmp/revisiting_opd/logs/baseline

bash examples/opd/opd_math_qwen2.5-7b_it.sh vllm \
  trainer.n_gpus_per_node=1 \
  data.max_response_length=512 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2560 \
  +actor_rollout_ref.actor.kl_topk_source=actor \
  trainer.total_epochs=1 \
  trainer.test_freq=5 \
  trainer.save_freq=-1
```

备注：

- 上面命令是否能透传额外 hydra 参数取决于脚本写法；如果脚本不支持尾部参数，需要新增一个专门的 smoke 脚本。
- 当前脚本内部写死 `trainer.n_gpus_per_node=8`、`data.max_response_length=16384`，不适合直接在单卡上跑。

### Step 2：sampled-token OPD smoke

目标：

- 跑通弱 baseline。

脚本：

```text
examples/opd/opd_original_math_qwen2.5_7b_it.sh
```

同样需要单卡 smoke 覆盖：

- `trainer.n_gpus_per_node=1`
- `data.max_response_length=512`
- `trainer.save_freq=-1`
- `trainer.test_freq=5`

### Step 3：小规模 baseline 数字

在 smoke 通过后，扩大到：

- `data/math_opd/train.parquet` 的小部分或完整 20k；
- response length 1024 / 2048；
- eval 先 `eval_smoke`，再 MATH-500 / GSM8K。

记录：

- final eval accuracy/pass rate；
- response length；
- truncation ratio；
- repetition proxy；
- peak memory；
- throughput；
- 是否出现 NaN/OOM/vocab 越界。

## 6. 当前建议

当前两个单卡 smoke baseline 均已跑通。下一步不要直接改 TrustOPD，应先把这个环境可运行快照提交并推送到 GitHub，然后从该快照派生实验分支：

1. `baseline/pre-experiment-sdpa-trustopd`：只保留环境适配、中文文档、baseline smoke 脚本。
2. `exp/original-opd-small-math`：跑 sampled-token OPD 小规模数字。
3. `exp/teacher-topk-small-math`：跑 Teacher-TopK 小规模数字。
4. `exp/trustopd-*`：在前两个 baseline 数字稳定后再开始 TrustOPD 方向。

## 7. Baseline 运行日志

| 日期 | baseline | commit/tag | 命令/脚本 | 结果 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 2026-05-17 | pre-experiment snapshot | `3c22bdb`, `pre-experiment-sdpa-trustopd-20260517` | push branch/tag | 已上传 GitHub | 新增单卡 smoke baseline 脚本 |
| 2026-05-17 | Teacher-TopK smoke | branch `baseline/pre-experiment-sdpa-trustopd` | `bash examples/opd/run_baseline_teacher_topk_math_smoke.sh vllm`，本次 `TEST_FREQ=2` | 2 step 通过；日志 `logs/baseline/teacher-topk-math-smoke_0517_102155.log`；step 2 `actor/kl_loss=0.684`，`actor/not_in_topk_ratio=0.000`，验证 `gsm8k=0.602`、`math-500=0.394`、`aime-2024=0.000` | 后续正式 baseline 恢复/实现 `kl_topk_source=ref` 的 shared-vocab 处理 |
| 2026-05-17 | sampled-token OPD smoke | branch `baseline/pre-experiment-sdpa-trustopd` | `bash examples/opd/run_baseline_sampled_opd_math_smoke.sh vllm`，默认 `TEST_FREQ=-1` | 2 step 通过；日志 `logs/baseline/sampled-opd-math-smoke_0517_102838.log`；step 2 `actor/pg_loss=0.807`，`critic/rewards/mean=-413.016`，无验证 | 可扩到小规模 train/eval |
