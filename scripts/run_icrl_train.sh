#!/bin/bash
# ==============================================================================
# ICRL-Sum: In-Context Reinforcement Learning Optimization Loop
# Purpose: Run the iterative refinement (Generation -> Critic -> Retrieval)
#          to generate optimized summary trajectories.
# ==============================================================================

set -e

# --- Environment ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)

# API Keys (Placeholders - Ensure these are set in your env)
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-anon-xxxxxxxx}"
# export GOOGLE_API_KEY="${GOOGLE_API_KEY:-AIza-anon-xxxxxxxx}"

# --- Paths ---
PROCESSED_ROOT="/data/project/data/processed"
OUTPUT_SUMMARIES="/data/project/output/summaries/train_trajectories"
TRAJECTORY_LOGS="/data/project/output/logs/trajectories"

# --- Hyperparameters ---
MODEL_NAME="gpt-4o"  # or "gpt-5-preview", "gemini-3"
MAX_ITERS=3          # K_max
REWARD_THRESH=0.85

echo ">>> [ICRL-Train] Starting Optimization Loop (Model: $MODEL_NAME, K=$MAX_ITERS)"

# Find all merged window files from preprocessing step
# Structure: data/processed/{video_name}/5_merged_windows.json
find "$PROCESSED_ROOT" -name "5_merged_windows.json" | while read input_json; do
    
    # Extract Video Name (parent dir name)
    video_dir=$(dirname "$input_json")
    video_name=$(basename "$video_dir")
    
    # Define Evidence DB Path (created by preprocess.py)
    evidence_db="$video_dir/evidence_db/index.faiss" # Implicitly handled by python script logic looking for dir
    
    # Define Output
    output_file="$OUTPUT_SUMMARIES/${video_name}_final.json"
    
    echo "  > Optimizing: $video_name"
    
    if [ ! -d "$video_dir/evidence_db" ]; then
        echo "    [Warning] Evidence DB missing for $video_name. Skipping retrieval."
    fi

    # Execute ICRL Runner
    python pipeline/video2summary_icrl.py \
        --input_merged "$input_json" \
        --evidence_db "$video_dir/evidence_db" \
        --output_summary "$output_file" \
        --log_dir "$TRAJECTORY_LOGS" \
        --model "$MODEL_NAME" \
        --iters "$MAX_ITERS"

done

echo ">>> [ICRL-Train] Optimization Cycle Complete."

