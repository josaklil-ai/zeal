#!/bin/bash
NUM_GPUS=4    # Adjust to your needs

START_BLOCK_ID=0
END_BLOCK_ID=${NUM_GPUS-1}

# Iterate over each block id and submit a job
for BLOCK_ID in $(seq $START_BLOCK_ID $END_BLOCK_ID); do
  JOB_NAME="s2-TH${BLOCK_ID}"
  TEMP_SCRIPT=$(mktemp)
  cat <<EOT > $TEMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/stage2_%x.txt
#SBATCH --error=logs/stage2_%x.txt
#SBATCH --partition=pasteur
#SBATCH --account=pasteur
#SBATCH --mem=32G
#SBATCH --exclude=pasteur[1-4],pasteur-hgx-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

eval "\$(conda shell.bash hook)"
conda activate zeal

python -m run \\
    dataset=thumos14 \\
    stage2.enable=True \\
    stage2.clip_model.arch=EVA02-E-14-plus \\
    stage2.clip_model.weights=laion2b_s9b_b144k \\
    stage2.clip_model.dim=1024 \\
    mp.block_id=$BLOCK_ID \\
    mp.num_blocks=$NUM_GPUS
EOT

  sbatch $TEMP_SCRIPT
  rm $TEMP_SCRIPT

done