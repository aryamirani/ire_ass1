#!/bin/bash
#SBATCH --job-name=download_test
#SBATCH --output=logs/download_test_%j.out
#SBATCH --error=logs/download_test_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

echo "Starting 16-thread fast download on $HOSTNAME..."
source venv/bin/activate
python scripts/fast_download.py
echo "Download job completed."
