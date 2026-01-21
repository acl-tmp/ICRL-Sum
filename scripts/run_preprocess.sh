#!/bin/bash
# ==============================================================================
# ICRL-Sum: Data Preprocessing Pipeline
# Features: Batch Video Processing, GPU Allocation, Error Logging
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)

# Anonymized Paths
RAW_VIDEO_DIR="/data/project/data/raw_videos"
PROCESSED_ROOT="/data/project/data/processed"
LOG_DIR="/data/project/logs/preprocess"

# Create directories
mkdir -p "$PROCESSED_ROOT"
mkdir -p "$LOG_DIR"

echo ">>> [Preprocess] Initializing Pipeline on GPU $CUDA_VISIBLE_DEVICES..."
echo "    Input:  $RAW_VIDEO_DIR"
echo "    Output: $PROCESSED_ROOT"

# Loop through all MP4 files
count=0
for video_path in "$RAW_VIDEO_DIR"/*.mp4; do
    [ -e "$video_path" ] || continue
    
    video_name=$(basename "$video_path" .mp4)
    log_file="$LOG_DIR/${video_name}_prep.log"
    
    echo "----------------------------------------------------------------"
    echo "Processing [$count]: $video_name"
    echo "Log: $log_file"
    
    # Run Python Module
    # Using 'nohup' style execution or direct blocking call
    # We use blocking call here to manage GPU memory sequentially
    if python pipeline/preprocess.py \
        --video "$video_path" \
        --out_dir "$PROCESSED_ROOT" \
        > "$log_file" 2>&1; then
        
        echo "  [Success] Finished."
    else
        echo "  [Failed] Check logs."
    fi
    
    ((count++))
done

echo ">>> [Preprocess] Completed. Processed $count videos."
