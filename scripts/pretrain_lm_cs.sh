#!/bin/bash
#SBATCH --job-name=pretrain_dna-bert2_cs
#SBATCH --output=/work/tesi_fgaragnani/logs/%x_%j.out
#SBATCH --error=/work/tesi_fgaragnani/logs/%x_%j.err
#SBATCH --open-mode=truncate
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all_usr_prod
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
export HF_HUB_OFFLINE=1

model_name="dnabert2_cs" # <--

model_checkpoint="/work/tesi_fgaragnani/checkpoints/ai4bio/${model_name}"
model_path="${model_checkpoint}"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

run_name="${SLURM_JOB_NAME}"
output_dir="/work/tesi_fgaragnani/checkpoints/ai4bio/${model_name}"

python src/train/train.py \
  --model_type bert \
  --config_name src/model/bert_config.json \
  --tokenizer_name ./src/model \
  --fasta_path /homes/fgaragnani/ai4bio/graph_enc/data/human_chr_6/Homo_sapiens.GRCh38.dna.chromosome.6.fa \
  --gtf_path /homes/fgaragnani/ai4bio/graph_enc/data/human_chr_6/chr6_CDS.gtf \
  --use_cds_mask true \
  --only_protein_coding true \
  --max_seq_length 512 \
  --mlm_probability 0.15 \
  --output_dir ${output_dir} \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --do_train \
  --do_eval