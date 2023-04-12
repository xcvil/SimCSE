#!/bin/bash
#SBATCH --output=/cluster/work/medinfmk/MedVLM/output/output_%J.txt
#SBATCH --error=/cluster/work/medinfmk/MedVLM/error/error_%j.txt
#SBATCH --job-name=sim                  # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:rtx1080ti:8          # titan_rtx & geforce_rtx_3090 & tesla_v100 & geforce_rtx_2080_ti & rtx_a6000
#SBATCH --cpus-per-task=3               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=128:00:00                # total run time limit (HH:MM:SS)

# Send more noteworthy information to the output log
echo "Started at:     $(date)"

source ~/.bashrc
source ~/.bashrc.xzheng
conda activate simcse

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=8

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID /cluster/customapps/medinfmk/xiaochen/SimCSE/train.py \
    --model_name_or_path bert-base-uncased \
    --train_file /cluster/customapps/medinfmk/xiaochen/SimCSE/data/data/wiki1m_for_simcse.txt \
    --output_dir /cluster/work/medinfmk/MedVLM/ckpt/simcse/wiki-unsup-simcse-bert-base-uncased \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
