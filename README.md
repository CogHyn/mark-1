# Mark 1: Efficient Video-LLM Connector

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)


> **"Mark-1 demonstrates proven efficacy on task-specific benchmarks and serves as a robust, parameter-efficient baseline for fine-tuning across diverse video domains with a prepared training pipeline."**

---

## Demo & Case study

![Example](./static/sample.png)


**Scenario Analysis:**
```json
{
    "question": "Trong video này, muốn đi Suối Tre thì cần đi hướng nào?",
    "choices": [
        "A. Đi thẳng, và rẽ trái.",
        "B. Rẽ phải, và tiếp tục đi thằng.",
        "C. Chuẩn bị rẽ trái, D. Đi 300m nữa rẽ trái.",
    ],
    "ground_truth": "C. Chuẩn bị rẽ trái",
    "mark-1_response": "A. Đi thẳng, rẽ trái",
}
```

**Note**: The model correctly identifies visual cues (traffic context) but struggles with specific directional reasoning due to the compact size of the language backbone (0.6B).

## Introduction

**Mark-1** represents a parameter-efficient approach to Video Understanding. By leveraging pre-training giants (Frozen Vision Encoders & LLMs), Mark-1 bypasses the prohibitive costs of full fine-tuning. 

The architecture introduces a novel, learnable **Projection Layer** that effectively bridges a modality gap, enabling complex video reasoning on comsumer-grade hardware (T4 x2) :v.

**Mark-1** solves two kind-of problems in parallel:

- **Video Understanding**: Ability to convert from visual to semantic understanding.

- **Temporial Localization**: Point out important frames base on user query.



## Architecture

The system follows a Pipeline Parallelism design to optimize VRAM usage:

1. Vision Encoder (Frozen): facebook/vjepa2-vitg-fpc64-256 (1B) - Extracts spatial-temporal features.

2. Projection Bridge (Trainable): * Cross-Attention with Learnable Queries.

- BERT-based Context Fusion (Mixing Visual Queries + Text Prompt).

- Dual Heads: One for LLM generation, one for Temporal Regression.

5. LLM (Frozen): Qwen/Qwen0.6B - Reasoning engine.

Information about achitecture is in [Achitecture](./docs/architecture.md).

## Performance & Limitation

The "Tiny Giant" Experiment
Mark-1 is currently an experimental proof-of-concept designed under strict hardware constraints (Dual T4 GPUs - 16GB VRAM). To fit this budget, we utilized highly compressed backbones:

LLM: **Qwen-0.6B** (or similar tiny variant).

Vision Encoder: **facebook/vjepa2-vitg-fpc64-256** (1B).

### Results Analysis
While the Projection Architecture successfully converges and learns to map visual features to the text space, the reasoning capability is capped by the size of the LLM.

- Success: The model correctly identifies visual elements and understands temporal structures (Loss converges, F1 score improves).

- Limitation: As seen in the example above, the model might hallucinate or fail in complex reasoning tasks (e.g., distinguishing between "turn left" vs "prepare to turn left") due to the 0.5B/0.6B parameter count of the language model.

**Note**: The current performance reflects the lower bound of this architecture. Scaling the backbone to Vicuna-7B or Llama-3-8B (as designed in the original architecture) is expected to significantly boost reasoning accuracy and reduce hallucinations.

## Future Work

[ ] Scale up LLM backbone to 7B/8B parameters.

[ ] Enhance the Temporal Head with stronger regression losses (e.g., IoU Loss).


## Installation

```bash
# Clone the repository
git clone git@github.com:CogHyn/mark-1.git
cd Mark-1

python -m venv .venv
source ./venv/bin/activate #Unix 
./venv/Script/activate.bat #Window
# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Training

- To reproduce the training process (fine-tuning the projection layer):

```bash
# Configure your parameters in config.yaml first
python train.py
```

2. Inference:

- To run the model on a single video via command line:

```bash
python predict.py \
  --video assets/sample_video.mp4 \
  --question "Describe the traffic situation." \
```







