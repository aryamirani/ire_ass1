#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --output=logs/test_pipeline_%j.out
#SBATCH --error=logs/test_pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

echo "Running EB-NeRD testset extraction & submission generator on node: $HOSTNAME"
source venv/bin/activate

# 1. Unzip testset
mkdir -p data/raw/ebnerd/test
if [ -f data/raw/ebnerd/ebnerd_testset.zip ]; then
    echo "Extracting ebnerd_testset.zip..."
    unzip -qo data/raw/ebnerd/ebnerd_testset.zip -d data/raw/ebnerd/test/
fi

# 2. Generate submission
echo "Generating official predictions..."
PYTHONPATH=. python src/scripts/generate_submission.py

echo "Testset pipeline finished successfully!"
