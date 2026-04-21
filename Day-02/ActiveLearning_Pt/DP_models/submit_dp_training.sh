#!/bin/bash
# MASTER SCRIPT FOR DEEPMD TRAINING
# This script finds all numbered folders (1, 2, 3...) in this directory 
# and submits the train.sub script found inside them.

BASE_DIR="/scratch/ch24d003/ActiveLearning_Pt/DP_models"
TARGET="train.sub"

echo "--------------------------------------------------------"
echo "Starting DeepMD Training Submission from: $BASE_DIR"
echo "--------------------------------------------------------"

# Loop through all items in the current directory
for dir in *; do
    # Check if it is a directory (e.g., 1, 2, 3)
    if [ -d "$dir" ]; then

        # Check if the target submission script exists inside
        if [ -f "$dir/$TARGET" ]; then
            echo "Entering directory: $dir"
            cd "$dir" || continue

            # Submit the job
            sbatch "$TARGET"

            # Go back to base directory
            cd ..
        fi
    fi
done

echo "--------------------------------------------------------"
echo "All DeepMD training jobs submitted."
