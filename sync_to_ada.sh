#!/bin/bash

# Simple script to sync local codebase to Ada cluster
# Usage: bash sync_to_ada.sh <ADA_USERNAME>

DEST_DIR="~/ire_ass1"

echo "Syncing codebase to ada:$DEST_DIR ..."

# Create the directory on Ada first
ssh ada "mkdir -p $DEST_DIR"

# Rsync the codebase, excluding heavy/local directories
rsync -avz --progress \
    --exclude "venv/" \
    --exclude "data/processed/" \
    --exclude "__pycache__/" \
    --exclude ".git/" \
    --exclude ".DS_Store" \
    ./ ada:$DEST_DIR/

echo "Sync complete! SSH into Ada to run the SLURM scripts."
