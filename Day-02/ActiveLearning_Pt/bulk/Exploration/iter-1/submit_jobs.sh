#!/bin/bash

# This script finds all lmp.sub files inside this current directory (and subdirectories)
# and submits them to SLURM.

BASE_DIR="../ActiveLearning_Pt/bulk/Exploration/iter-1"
TARGET_SUB="lmp.sub"

echo "Running in: $BASE_DIR"

# Find 'lmp.sub' recursively starting from the current directory
find "$BASE_DIR" -name "$TARGET_SUB" -print0 | while IFS= read -r -d '' sub_file; do

    # Get the directory where lmp.sub is located
    job_dir=$(dirname "$sub_file")

    echo "Found job in: $job_dir"

    # Enter directory
    cd "$job_dir" || continue

    # Submit the job
    sbatch "$TARGET_SUB"

    # Go back to previous folder silently
    cd - > /dev/null

done

echo "All jobs for this system submitted."
