#!/bin/bash
#SBATCH --job-name=pretrain_dna-bert2_cs
#SBATCH --output=/work/tesi_fgaragnani/logs_ai4bio/%x_%j.out
#SBATCH --error=/work/tesi_fgaragnani/logs_ai4bio/%x_%j.err
#SBATCH --open-mode=truncate
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --nodes=1
#SBATCH --time=13:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8

source activate dna

cd /homes/fgaragnani/ai4bio/graph_enc
export PYTHONPATH=.:..:$PYTHONPATH

export HF_HUB_CACHE="/work/tesi_fgaragnani/checkpoints/"
export HF_HOME="/work/tesi_fgaragnani/checkpoints/"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

model_name="dnabert2_cs" # <--

model_checkpoint="/work/tesi_fgaragnani/checkpoints/ai4bio/${model_name}"
model_path="${model_checkpoint}"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

run_name="${SLURM_JOB_NAME}"
output_dir="/work/tesi_fgaragnani/checkpoints/ai4bio/${model_name}"

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} src/train/train.py \
  --model_type bert \
  --config_name src/model/bert_config.json \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --tokenizer_name ./src/model \
  --data_path /homes/fgaragnani/ai4bio/graph_enc/data/ \
  --use_cds_mask true \
  --only_protein_coding false \
  --max_seq_length 1024 \
  --mlm_probability 0.15 \
  --item_length_proportion 1.0 \
  --output_dir ${output_dir} \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 5e-4 \
  --max_steps 40000 \
  --warmup_steps 1200 \
  --weight_decay 1e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --save_strategy steps \
  --save_steps 20000 \
  --ddp_find_unused_parameters false \
  --evaluation_strategy epoch \
  --do_train \
  --do_eval \
  --overwrite_output_dir