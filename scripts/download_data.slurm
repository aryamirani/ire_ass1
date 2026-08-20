#!/bin/bash
#SBATCH --job-name=download_ire_data
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# This script downloads the large datasets.
# It requires internet access, so it may need to be run on a node with external internet access.

echo "Running data download on node: $HOSTNAME"

mkdir -p logs
bash setup_data.sh

echo "Download completed!"
