#!/bin/bash -l
#SBATCH --job-name=testing
#SBATCH --time=48:0:0
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB
#SBATCH -A sgupta91_gpu
#SBATCH --output "/data/sgupta91/TimeSeriesTransformer/v1.4/logs/slurm-%j.out"

module load anaconda
module load cuda
conda activate Transformer

export NPROC_PER_NODE=1
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=36763
export WORLD_SIZE=$SLURM_NTASKS

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Number of tasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

now=$(date +"%Y%m%d_%H%M%S")

python /data/sgupta91/TimeSeriesTransformer/v1.4/test_testing.py \
--job_name="Run Tesing" \
--company_num="all"    \
--chara_num="all" \
--test_data="/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test_2016.pt"    \
--checkpoint="/data/sgupta91/TimeSeriesTransformer/v1.4/checkpoint/best_model_231130_045203.pt"    \
--label="231130_045203" \