#!/bin/bash
#SBATCH --job-name=ire_ebnerd
#SBATCH --output=logs/ebnerd_%j.out
#SBATCH --error=logs/ebnerd_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

echo "Running EB-NeRD evaluation and submission pipeline on node: $HOSTNAME"
source venv/bin/activate

echo "1. Running EB-NeRD evaluation harness..."
PYTHONPATH=. python src/evaluation/evaluate.py

echo "2. Generating EB-NeRD Codabench submission..."
PYTHONPATH=. python src/scripts/generate_submission.py

echo "EB-NeRD pipeline completed successfully!"
