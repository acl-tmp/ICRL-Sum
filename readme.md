# ICRL-Sum: In-Context Reinforcement Learning for Grounded Video Summarization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-green)](https://arxiv.org/)

**Official implementation of "ICRL-Sum: In-Context Reinforcement Learning for Grounded Video Summarization".**

**ICRL-Sum** is a training-free, inference-time optimization framework designed to generate grounded, coherent, and aligned video summaries. It leverages a hybrid architecture combining local multimodal perception ("The Eye") with remote large language model reasoning ("The Brain"), orchestrated by a reward-driven critic mechanism.

---

## 🚀 Key Features

- **Training-Free Optimization**: No parameter fine-tuning required; leverages In-Context Reinforcement Learning.
- **Hybrid Architecture**:
  - **The Eye (Local)**: GPU-accelerated visual perception using **Qwen2.5-VL** and **Wav2Vec 2.0**.
  - **The Brain (Remote)**: Cognitive reasoning via **GPT-4o/5** or **Gemini 1.5 Pro**.
- **Critic-Guided Retrieval**: A closed-loop system where a "Critic" detects hallucinations and triggers semantic retrieval from a vector database (FAISS).
- **Structured Evaluation**: Includes `ICRL-SumBench` metrics for Spatio-Temporal Alignment (tIoU) and Factual Grounding (NLI).

---

## 📂 Project Structure

```text
icrl/
├── configs/               # Hyperparameters & Model Configs (model.yaml)
├── data/                  # Data storage (Raw videos, Processed features, Logs)
├── knowledge_base/        # Vector Database & Retrieval Logic (FAISS, Indexer)
├── llm/                   # Unified Clients for OpenAI, Gemini, and Local LLMs
├── pipeline/              # Orchestration Scripts (Preprocess -> RL -> Eval)
├── rl/                    # Reward Engines, Critic Agent, Trajectory Logging
├── segmenter/             # Audio-Visual Scene Segmentation
├── summary/               # Schema Definitions & State Management
├── tools/                 # Perception Tools (ASR, Frame Extraction, Fusion)
└── scripts/               # Entry points (Shell scripts)
```

---

## 🛠️ Installation & Environment

### 1) Installation
```bash
# 1. Install System Tools (FFmpeg is required)
sudo apt-get update && sudo apt-get install -y ffmpeg

# 2. Set up Python Environment
conda create -n icrl python=3.10
conda activate icrl

# 3. Install Dependencies
pip install -r requirements.txt
```
### 2) API Keys Setup

```bash
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"
export GOOGLE_API_KEY="AIzaSyD-xxxxxxxxxxxxxxxxxxxxxxxx"
```



## ⚡ Quick Start

### Step 1: Data Preprocessing

Extracts frames, transcribes audio (ASR), computes acoustic embeddings, and builds the Vector Database ($K$).

```bash
# Process all videos in the raw_videos directory
bash scripts/run_preprocess.sh
```

- **Input**: `data/raw_videos/*.mp4`
- **Output**: `data/processed/{video_name}/` (includes `index.faiss`, `merged_windows.json`)

### Step 2: ICRL Optimization (Training Phase)

Runs the iterative "Generation → Evaluation → Retrieval → Refinement" loop.

```bash
# Run the optimization loop with K=3 iterations
bash scripts/run_icrl_train.sh
```

**Logic**:
1. Generate initial draft ($u_t^0$).
2. Critic evaluates alignment & grounding.
3. Retrieve evidence if hallucination is detected.
4. Refine draft using feedback.

- **Output**: `data/output/summaries/train_trajectories/*.json`

### Step 3: Inference & Evaluation

Evaluate generated summaries against ground truth using tIoU, BERTScore, and NLI-based grounding.

```bash
bash scripts/run_inference.sh
```

- **Output**: `data/output/reports/eval_metrics_TIMESTAMP.json`

---

## ⚙️ Configuration

Control model behavior via `configs/model.yaml`:

```yaml
remote_llm:
  active_provider: "openai" # Options: "openai", "google"
  openai:
    model_name: "gpt-5"
    temperature: 0.2

local_vlm:
  model_path: "/path/to/local/Qwen2.5-VL"
  quantization: "bf16"

embeddings:
  text_encoder: "sentence-transformers/all-mpnet-base-v2"
```

---

## 📊 Performance

| Method | Schema Val. | FactVC | VideoScore | BERTScore | METEOR | ROUGE-L | BLEU |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZeroShot-VLLM | 0.62 | 60.31 | 2.45 | 82.92 | 18.70 | 32.42 | 7.40 |
| V2Xum-LLaMA | 0.71 | 47.80 | 2.57 | 81.48 | 20.90 | 38.63 | 8.10 |
| Window Self-iteration | 0.78 | 52.47 | 2.66 | 82.05 | 21.79 | 37.16 | 8.60 |
| Free-text | 0.53 | 44.13 | 2.60 | 82.34 | 22.42 | 35.73 | 9.20 |
| No-retrieval | 0.60 | 56.91 | 2.73 | 81.62 | 22.11 | 32.01 | 8.80 |
| ICRL-Sum (Ours) | **0.930** | **63.55** | **2.79** | **82.45** | **23.20** | **38.44** | **9.60** |



---

## 🤝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{anonymous2025icrl,
  title={ICRL-Sum: In-Context Reinforcement Learning for Grounded Video Summarization},
  author={Anonymous Authors},
  journal={Under Review},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
