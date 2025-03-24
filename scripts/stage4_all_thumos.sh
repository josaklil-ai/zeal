#!/bin/bash
NUM_GPUS=1    # Adjust to your needs

START_BLOCK_ID=0
END_BLOCK_ID=${NUM_GPUS-1}

# Iterate over each block id and submit a job
for BLOCK_ID in $(seq $START_BLOCK_ID $END_BLOCK_ID); do
  JOB_NAME="s4-TH${BLOCK_ID}"
  TEMP_SCRIPT=$(mktemp)
  cat <<EOT > $TEMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/stage4_%x.txt
#SBATCH --error=logs/stage4_%x.txt
#SBATCH --partition=pasteur
#SBATCH --account=pasteur
#SBATCH --mem=32G
#SBATCH --exclude=pasteur[1-7]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

eval "\$(conda shell.bash hook)"
conda activate zeal

python -m run \\
    stage3.enable=True \\
    stage3.use_gt=False \\
    stage4.enable=True \\
    stage4.vlm_type=llava-ov \\
    mp.block_id=$BLOCK_ID \\
    mp.num_blocks=$NUM_GPUS
EOT

  sbatch $TEMP_SCRIPT
  rm $TEMP_SCRIPT

done