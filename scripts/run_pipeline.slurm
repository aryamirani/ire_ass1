#!/bin/bash
#SBATCH --job-name=ire_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "Running data pipeline on node: $HOSTNAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 64G"

# Load necessary modules if required on Ada (e.g. module load python/3.10)
# module load python/3.10

mkdir -p logs

# Setup virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv and install requirements
source venv/bin/activate
pip install -r requirements.txt

# Extract zip files (if they haven't been extracted yet)
# We assume they were downloaded by setup_data.sh to data/raw/
echo "Extracting datasets (this might take a while for Large)..."
unzip -q -o data/raw/mind/MINDlarge_train.zip -d data/raw/mind/train/
unzip -q -o data/raw/mind/MINDlarge_dev.zip -d data/raw/mind/dev/
unzip -q -o data/raw/mind/MINDlarge_test.zip -d data/raw/mind/test/
unzip -q -o data/raw/ebnerd/ebnerd_large.zip -d data/raw/ebnerd/large/
unzip -q -o data/raw/ebnerd/Ekstra_Bladet_word2vec.zip -d data/raw/ebnerd/word2vec/
unzip -q -o data/raw/ebnerd/google_bert_base_multilingual_cased.zip -d data/raw/ebnerd/bert/

# Run the data processing pipeline
echo "Executing build_pipeline.py..."
python src/data/build_pipeline.py

# Run the evaluation scripts
echo "Executing evaluate.py..."
PYTHONPATH=. python src/evaluation/evaluate.py

# Generate submission format
echo "Generating Codabench submissions..."
PYTHONPATH=. python src/scripts/generate_submission.py

echo "Pipeline execution finished successfully!"
