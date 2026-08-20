#!/bin/bash

# Exit on error
set -e

echo "Starting dataset downloads..."

# Create data directory if it doesn't exist
mkdir -p data/raw/ebnerd
mkdir -p data/raw/mind

# Function to download and extract if zip file
download_and_extract() {
    URL=$1
    DEST_DIR=$2
    FILE_NAME=$(basename "$URL")
    
    echo "Downloading $FILE_NAME..."
    curl -L -o "$DEST_DIR/$FILE_NAME" "$URL"
    
    # We won't extract automatically to save space and let the python pipeline handle it
    # But if you want to extract, you can uncomment below
    # if [[ $FILE_NAME == *.zip ]]; then
    #     echo "Extracting $FILE_NAME..."
    #     unzip -q "$DEST_DIR/$FILE_NAME" -d "$DEST_DIR"
    # fi
}

# EB-NeRD
echo "--- Downloading EB-NeRD Datasets ---"
download_and_extract "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip" "data/raw/ebnerd"
download_and_extract "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip" "data/raw/ebnerd"
download_and_extract "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip" "data/raw/ebnerd"

# MIND
echo "--- Downloading MIND Datasets ---"
download_and_extract "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip" "data/raw/mind"
download_and_extract "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip" "data/raw/mind"
download_and_extract "https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip" "data/raw/mind"

echo "All datasets downloaded to data/raw/ !"
