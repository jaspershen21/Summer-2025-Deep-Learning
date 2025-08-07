#!/bin/bash

#SBATCH --job-name=dropout_batch_size_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --qos=normal
#SBATCH --partition=ami100
#SBATCH --gres=gpu:1
#SBATCH --output=dropout_batch_size_search_%j.log

PROJECT_DIR="/scratch/alpine/jash3852/EEGMotorImageryDL"
OUTPUT_FILE="$PROJECT_DIR/models/tune_dropout_batch_size.csv"

DROPOUT_LIST=(0.25 0.5 0.75)
BATCH_SIZE_LIST=(16 32 64 128)

echo "Setting up environment..."
module purge
module load anaconda
module load rocm/5.2.3
module load pytorch/1.13.0

conda activate eeg-motor-imagery-dl
cd ${PROJECT_DIR}/scripts

echo "Starting hyperparameter search..."
for dropout in "${DROPOUT_LIST[@]}"; do
    for batch_size in "${BATCH_SIZE_LIST[@]}"; do
        echo "--------------------------------------------------------"
        echo "RUNNING: Dropout = ${dropout}, Batch Size = ${batch_size}"
        echo "--------------------------------------------------------"

        python hyperparameter_tuning.py \
            --learning-rate 0.001 \
            --weight-decay 0.0001 \
            --dropout-rate ${dropout} \
            --batch-size ${batch_size} \
            --output-file ${OUTPUT_FILE}

        echo "Completed: Dropout = ${dropout}, Batch Size = ${batch_size}"
    done
done

echo "Hyperparameter search completed. Results saved to ${OUTPUT_FILE}."