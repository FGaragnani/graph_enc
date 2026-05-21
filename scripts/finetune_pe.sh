#!/bin/bash
#SBATCH --job-name=finetune_pe
#SBATCH --output=/work/tesi_fgaragnani/logs_ai4bio/%x_%j.out
#SBATCH --error=/work/tesi_fgaragnani/logs_ai4bio/%x_%j.err
#SBATCH --open-mode=truncate
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-0
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="a100|h100"
#SBATCH --account=ai4bio2025
#SBATCH --nodes=1
#SBATCH --time=02:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8

source activate dna

cd /homes/fgaragnani/ai4bio/graph_enc
export PYTHONPATH=.:..:$PYTHONPATH

export HF_HUB_CACHE="/work/tesi_fgaragnani/checkpoints/"
export HF_HOME="/work/tesi_fgaragnani/checkpoints/"
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

model_checkpoint="/work/tesi_fgaragnani/checkpoints/ai4bio/dnabert2_cs"
output_dir="/work/tesi_fgaragnani/checkpoints/ai4bio/dnabert2_cs/finetuned_pe"
dataset_dir="/homes/fgaragnani/ai4bio/graph_enc/scripts/datasets"
seeds=(42 43 44 45 46)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
run_output_dir="${output_dir}/seed_${seed}"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

torchrun --nproc_per_node=${SLURM_GPUS_PER_NODE} --master_port=${MASTER_PORT} src/task/discriminative.py \
  --model_name_or_path ${model_checkpoint} \
  --model_type bert \
  --data_path /homes/fgaragnani/ai4bio/graph_enc/data/ \
  --tokenizer_name ./src/model \
  --config_file ./src/model/bert_config.json \
  --dataset_dir ${dataset_dir} \
  --target_enhancer_fraction 0.5 \
  --balance_seed ${seed} \
  --max_seq_length 768 \
  --chunk_size_bases 2000 \
  --max_chunks_per_sample 10 \
  --pad_to_max_length false \
  --output_dir ${run_output_dir} \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 1e-3 \
  --max_steps 8000 \
  --warmup_steps 800 \
  --weight_decay 1e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --bf16 \
  --save_strategy no \
  --eval_steps 500 \
  --validation_split_percentage 10 \
  --ddp_find_unused_parameters false \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --perform_kfold true \
  --seed ${seed} \
  --low_cpu_mem_usage true