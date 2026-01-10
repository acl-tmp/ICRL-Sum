# ICRL-Sum: In-Context Reinforcement Learning for Grounded Video Summarization

[![License: Review-only](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

**Official (anonymized) implementation of "ICRL-Sum: In-Context Reinforcement Learning for Grounded Video Summarization".** 

**ICRL-Sum** is a structure-aware, training-free **inference-time optimization** framework for video summarization. With model parameters frozen, it represents the summary as a **structured Schema** and performs **closed-loop iterative refinement** driven by Critic rewards and evidence-retrieval feedback to improve spatio-temporal alignment, event coherence, and hallucination grounding—without any parameter updates. 

---


## 🚀 Key Features

- **Training-Free, Inference-Time Optimization**: Reformulates video summarization as an inference-time closed-loop optimization process (no fine-tuning).
- **Three-Stage Pipeline**: (1) multimodal perception & segmentation, (2) window-level ICRL iterative loop, and (3) dynamic Schema evolution with terminal memory. 
- **Unified Backbone LLM (GPT-5)**: Uses **GPT-5** as the unified backbone, instantiating both the Base-LLM (Schema generation) and Feedback-LLM (instruction/feedback generation) via customized system prompts. 
- **Textualized Multimodal Evidence Construction**:
  - **ASR**: Extracts high-precision transcripts with **Whisper-large-v3**.
  - **Visual Evidence**: Converts shot-detected keyframes into textual descriptions via a **Gemini multimodal model**. 
- **Critic-Guided Retrieval & Feedback Loop**: A Reward Critic evaluates candidates along **alignment / coherence / grounding**, triggers targeted evidence retrieval, and injects feedback to steer the next refinement step. 
- **Evidence Embedding & Retrieval Backend**: Maps textualized evidence into dense vectors using **text-embedding-3-large** and indexes them in an evidence database supported by **Elasticsearch** for efficient retrieval. 
- **Structured Evaluation with ICRL-SumBench**: Introduces **ICRL-SumBench**, which provides structured Schema annotations with fine-grained spatio-temporal information and event logic to evaluate alignment, coherence, and grounding more precisely. 
- **Grounding-Oriented Rewards/Metrics**: Grounding is assessed via entailment-style signals (e.g., NLI-based grounding in the reward design), and alignment uses tIoU-style measurements in the framework. 

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
  active_provider: "openai"
  openai:
    model_name: "gpt-5"
    temperature: 0.2

# Visual evidence is textualized by a multimodal model (Gemini)
visual_captioner:
  active_provider: "google"
  google:
    model_name: "gemini"   

# Evidence embeddings used for retrieval
embeddings:
  active_provider: "openai"
  openai:
    model_name: "text-embedding-3-large"

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
@article{anonymous2026icrl,
  title={ICRL-Sum: In-Context Reinforcement Learning for Grounded Video Summarization},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---

## 📄 License

This repository is provided for anonymous peer review purposes only. All rights reserved. Redistribution or reuse is not permitted without the authors’ permission.
