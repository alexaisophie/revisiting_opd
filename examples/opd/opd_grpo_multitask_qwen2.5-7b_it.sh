#!/usr/bin/env bash

TIME_STAMP=$(date +"%m%d_%H%M%S")
project_name='opd'
exp_name='grpo-multitask-qwen2.5-7b-it'

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

CKPTS_DIR=${CKPTS_DIR:-"${PWD}/ckpts/${exp_name}_${TIME_STAMP}"}
LOG_DIR=${LOG_DIR:-"${PWD}/logs/multitask"}

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8

MULTITASK_DATA_DIR=${MULTITASK_DATA_DIR:-"${PWD}/data/multitask_data"}
TRAIN_DATA="${MULTITASK_DATA_DIR}/train.parquet"
VAL_DATA="${MULTITASK_DATA_DIR}/test.parquet"

STUDENT_MODEL=${STUDENT_MODEL:-"/path/to/Qwen2.5-7B-Instruct"}

mkdir -p "${LOG_DIR}"

python3 -m verl.trainer.main_ppo_multitask \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    actor_rollout_ref.ref.model.path=${STUDENT_MODEL} \
    +multitask.enable=True \
    +multitask.batching_mode=sequential \
    +multitask.tasks.task0.name=alfworld \
    +multitask.tasks.task0.env_name=alfworld/AlfredTWEnv \
    +multitask.tasks.task0.eval_dataset=eval_in_distribution \
    +multitask.tasks.task0.max_response_length=512 \
    +multitask.tasks.task1.name=math \
    +multitask.tasks.task1.env_name=math \
    +multitask.tasks.task1.max_response_length=16384 \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    +data.batching_mode=sequential \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + 16384)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=multitask \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=${group_size} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.visualize_distribution=true \
    +trainer.visualize_distribution_freq=3 \
    +trainer.visualize_distribution_samples=1 \
    +trainer.visualize_distribution_ref_tokens=3 \
    +trainer.visualize_distribution_dir="${CKPTS_DIR}/visualizations" \
    ray_init.num_cpus=96 \
    2>&1 | tee "${LOG_DIR}/${exp_name}_${TIME_STAMP}.log"
