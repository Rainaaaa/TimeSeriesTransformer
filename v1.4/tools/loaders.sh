#!/bin/bash -l
#SBATCH --job-name=loaders
#SBATCH --time=12:0:0
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -A sgupta91_gpu
#SBATCH --output "/data/sgupta91/TimeSeriesTransformer/v1.3/logs/slurm-%j.out"

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
# save_path=work_dirs/shortcut/VE
# mkdir -p $save_path

python /data/sgupta91/TimeSeriesTransformer/v1.3/test_loaders.py \
--job_name="Making Loaders" \
--company_num="1000"    \
--chara_num="1" \
--test_data="/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_test_2016.pt"    \
--checkpoint="/data/sgupta91/TimeSeriesTransformer/v1.3/checkpoint/best_model_231130_045203.pt"    \
--label="231130_045203" \