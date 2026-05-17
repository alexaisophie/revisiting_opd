#!/usr/bin/env bash

TIME_STAMP=$(date +"%m%d_%H%M%S")
project_name='opd'
exp_name='alfworld-qwen2.5-7b-it'

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${exp_name}_${TIME_STAMP}"}
LOG_DIR=${LOG_DIR:-"${PWD}/logs/alfworld"}

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8

STUDENT_MODEL=${STUDENT_MODEL:-"/path/to/Qwen2.5-7B-Instruct"}
ALFWORLD_TEACHER=${ALFWORLD_TEACHER:-"/path/to/alfworld-teacher-gigpo-qwen2.5-7b"}
ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-"${PWD}/data/opd_alfworld"}
TRAIN_DATA="${ALFWORLD_DATA_ROOT}/text/train.parquet"
VAL_DATA="${ALFWORLD_DATA_ROOT}/text/test.parquet"

mkdir -p "${LOG_DIR}" "${ALFWORLD_DATA_ROOT}"

# The placeholder parquet files only declare modality and sample count.
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${ALFWORLD_DATA_ROOT}" \
    --train_data_size ${train_data_size} \
    --val_data_size ${val_data_size}

python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=placeholder \
    actor_rollout_ref.actor.kl_loss_type=full_reverse \
    +actor_rollout_ref.actor.kl_topk_tokens=32 \
    +actor_rollout_ref.actor.norm_to_one_for_kl=True \
    +actor_rollout_ref.actor.clip_log_ratio=False \
    +actor_rollout_ref.actor.opd_mask_special_tokens=False \
    +actor_rollout_ref.actor.kl_use_tail_sampling=False \
    +actor_rollout_ref.actor.kl_topk_source=ref \
    +actor_rollout_ref.actor.use_kl_iw=False \
    +actor_rollout_ref.actor.kl_iw_clip_lower=0 \
    +actor_rollout_ref.actor.kl_iw_clip_upper=10 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.ref.model.path=${ALFWORLD_TEACHER} \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    +data.batching_mode=sequential \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + 512)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=42 \
    env.max_steps=30 \
    env.rollout.n=${group_size} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=30 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.visualize_distribution=false \
    +trainer.visualize_distribution_freq=1 \
    +trainer.visualize_distribution_samples=2 \
    +trainer.visualize_distribution_dir="${CKPTS_DIR}/visualizations" \
    +trainer.visualize_distribution_ref_tokens=3 \
    ray_init.num_cpus=96 \
    2>&1 | tee "${LOG_DIR}/${exp_name}_${TIME_STAMP}.log"
