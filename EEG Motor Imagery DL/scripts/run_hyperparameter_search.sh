#!/bin/bash

#SBATCH --job-name=hyperparam_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --qos=normal
#SBATCH --partition=ami100
#SBATCH --gres=gpu:1
#SBATCH --output=hyperparam_search_%j.log

PROJECT_DIR="/scratch/alpine/jash3852/EEGMotorImageryDL"
OUTPUT_FILE="$PROJECT_DIR/models/hyperparam_search_results.csv"

LR_LIST=(0.0001 0.005 0.001 0.005 0.01)
WD_LIST=(0.0 0.0001 0.001 0.01)

echo "Setting up environment..."
module purge
module load anaconda
module load rocm/5.2.3
module load pytorch/1.13.0

conda activate eeg-motor-imagery-dl
cd ${PROJECT_DIR}/scripts

echo "Starting hyperparameter search..."
for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        echo "--------------------------------------------------------"
        echo "RUNNING: LR=${lr}, Weight Decay=${wd}"
        echo "--------------------------------------------------------"

        python hyperparameter_tuning.py \
            --learning-rate ${lr} \
            --weight-decay ${wd} \
            --output-file ${OUTPUT_FILE}

        echo "Completed: LR = ${lr}, WD = ${wd}"
    done
done

echo "Hyperparameter search completed. Results saved to ${OUTPUT_FILE}."