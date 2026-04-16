#!/bin/bash
for dir in scf_*; do
  [ -d "$dir" ] && cd "$dir" && [ -f "qe.sub" ] && sbatch qe.sub && cd ..
done
echo "All 100 jobs submitted!"
