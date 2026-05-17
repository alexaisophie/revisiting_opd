#!/usr/bin/env bash

set -euo pipefail
set -x

TIME_STAMP=$(date +"%m%d_%H%M%S")
ENGINE=${1:-vllm}

if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
    export OMP_NUM_THREADS=8
fi
export WANDB_MODE=${WANDB_MODE:-disabled}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
CUDA13_LIB="/root/autodl-tmp/conda_envs/opd-bw/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"

PROJECT_NAME=${PROJECT_NAME:-opd-baseline-smoke}
EXP_NAME=${EXP_NAME:-sampled-opd-math-smoke}
TEST_FREQ=${TEST_FREQ:--1}

STUDENT_MODEL=${STUDENT_MODEL:-/root/autodl-tmp/Qwen2.5-1.5B-Instruct}
MATH_TEACHER=${MATH_TEACHER:-/root/autodl-tmp/OpenThinker3-7B}
TRAIN_DATA=${TRAIN_DATA:-data/math_opd/train_smoke.parquet}
VAL_DATA=${VAL_DATA:-data/eval_math/eval_smoke.parquet}

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${EXP_NAME}_${TIME_STAMP}"}
LOG_DIR=${LOG_DIR:-"${PWD}/logs/baseline"}
mkdir -p "${LOG_DIR}"

python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    +actor_rollout_ref.actor.kl_topk_tokens=32 \
    +actor_rollout_ref.actor.norm_to_one_for_kl=True \
    +actor_rollout_ref.actor.clip_log_ratio=False \
    +actor_rollout_ref.actor.opd_mask_special_tokens=False \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.model.path="${MATH_TEACHER}" \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=2 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=middle \
    data.return_raw_chat=True \
    +data.batching_mode=sequential \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=True \
    env.env_name=math \
    env.seed=21 \
    env.max_steps=30 \
    env.rollout.n=2 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    +trainer.visualize_distribution=false \
    ray_init.num_cpus=16 \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}_${TIME_STAMP}.log"
