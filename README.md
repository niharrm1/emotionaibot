# EmotiAI - Emotionally Intelligent Personal AI Model

A fine-tuned LLM that starts with a default personality, learns each user's chat style over time, and responds with genuine human emotions — anger, warmth, sarcasm, joy — trained deep in the model weights.

## Version 1.0 | 2025 | Full Stack AI Engineering Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Installation](#installation)
6. [Data Collection](#data-collection)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation](#evaluation)
9. [Deployment](#deployment)
10. [Configuration](#configuration)
11. [Free Resources](#free-resources)
12. [License](#license)

---

## Overview

This project implements an emotionally intelligent AI model that:

- **Has a personality**: Trained to feel emotions (anger, joy, sarcasm, warmth)
- **Learns your style**: Adapts communication style to match each user
- **Responds like a human**: No robotic responses - genuine emotional expression
- **Remembers context**: Tracks conversation history and user preferences

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Message                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              User Profile Manager                       │
│         (Loads style from SQLite DB)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Style Extractor                            │
│    (Analyzes: Hinglish, emoji, message length, etc.)    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           System Prompt Builder                         │
│     (Injects personality + user style adaptation)      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Fine-tuned Mistral 7B                      │
│          (QLoRA - runs on single GPU)                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Emotion Response                           │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **12 Emotion Categories**: ANGRY_SHARP, HURT_WITHDRAWAL, SARCASTIC_LIGHT, SARCASTIC_DARK, EXCITED_HIGH, WARM_GENUINE, PLAYFUL_BANTER, EXHAUSTED_FLAT, PROTECTIVE_FIERCE, TOUCHED_DEFLECT, COLDLY_DISAPPOINTED, GENUINELY_CURIOUS
- **Per-User Style Adaptation**: Automatically learns and mirrors user communication style
- **Free Training**: Uses QLoRA - runs on single GPU (8-12GB VRAM)
- **Rule-based Labeling**: No paid APIs required for data preparation

---

## Project Structure

```
emotai/
├── config/
│   └── config.yaml           # All configuration settings
├── data/
│   ├── raw/                  # Raw data from collectors
│   ├── processed/            # Processed and formatted data
│   ├── sft/                  # SFT training data (JSONL)
│   └── dpo/                  # DPO training data (JSONL)
├── models/
│   ├── sft_adapter/          # SFT LoRA adapter
│   ├── dpo_adapter/          # DPO LoRA adapter
│   └── final_model/          # Merged final model
├── src/
│   ├── data/
│   │   ├── reddit_collector.py      # Reddit data collector
│   │   ├── huggingface_loader.py     # HuggingFace dataset loader
│   │   └── formatter.py             # Data formatting for SFT/DPO
│   ├── training/
│   │   ├── sft_trainer.py            # SFT training script
│   │   └── dpo_trainer.py            # DPO training script
│   ├── adapters/
│   │   └── style_adapter.py          # User style adaptation
│   ├── evaluation/
│   │   └── evaluator.py             # Emotion probe tests
│   └── deployment/
│       └── api.py                    # FastAPI backend
├── scripts/                  # Utility scripts
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
cd emotai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Generate sample training data
python -c "from src.data.formatter import main; main()"

# Or load from HuggingFace datasets
python -c "from src.data.huggingface_loader import main; main()"
```

### 3. Train Model

```bash
# Layer 1: SFT Training (gives personality)
python src/training/sft_trainer.py

# Layer 2: DPO Training (teaches emotional realism)
python src/training/dpo_trainer.py

# Merge and export
python src/training/sft_trainer.py --merge
python src/training/dpo_trainer.py --merge
```

### 4. Run API

```bash
python src/deployment/api.py
```

### 5. Test

```bash
# Send a chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "message": "Bhai kya haal hai?"}'
```

---

## Installation

### Requirements

- Python 3.10+
- GPU with 8GB+ VRAM (for training)
- 20GB+ disk space

### Hardware Options

| Setup | GPU | VRAM Needed | Time |
|-------|-----|-------------|------|
| QLoRA Training | Single A100 | 8-12 GB | 2-4 hours |
| QLoRA Training | Single RTX 3090/4090 | 12 GB | 4-8 hours |
| Inference Only | Any | 4 GB | N/A |

### Free GPU Options

1. **Google Colab** (Free A100 - Limited)
2. **Kaggle** (Free P100 - Limited)
3. **RunPod** (Cheap GPU Cloud)
4. **Paperspace** (Free tier available)

---

## Data Collection

### Free Data Sources

1. **HuggingFace Datasets** (No API needed)
   - EmotionLines: 29,000 conversations
   - DailyDialog: 13,000 conversations
   - EmpatheticDialogues

2. **Reddit** (Free API)
   - r/relationship_advice
   - r/AITA
   - r/offmychest
   - r/rant

3. **Your Own Data**
   - WhatsApp/Telegram exports
   - Personal chat logs

### Data Format

#### SFT Format (JSONL)
```json
{"messages": [
  {"role": "system", "content": "You are Nihar..."},
  {"role": "user", "content": "bhai tu kabhi bhi time pe nahi aata"},
  {"role": "assistant", "content": "yaar sorry nahi bolunga..."}
]}
```

#### DPO Format (JSONL)
```json
{"prompt": "User is rude: whatever man", "chosen": "theek hai. baat karna ho toh karna.", "rejected": "I understand your frustration. How can I assist you?"}
```

---

## Training Pipeline

### Layer 1: Supervised Fine-Tuning (SFT)

Trains the model to respond in the bot's personality voice.

```bash
python src/training/sft_trainer.py \
    --config config/config.yaml
```

**Config Settings** (config.yaml):
```yaml
sft:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
```

### Layer 2: Direct Preference Optimization (DPO)

Teaches emotional realism - distinguishing human-like vs chatbot-like responses.

```bash
python src/training/dpo_trainer.py \
    --config config/config.yaml
```

**Key Insight**: The DPO 'chosen' response sounds like a REAL PERSON. The 'rejected' response sounds like a CHATBOT. This contrast teaches the model to be human.

### Layer 3: Model Merging

Combines LoRA adapters with base model for deployment.

```bash
python src/training/dpo_trainer.py --merge
```

---

## Evaluation

### Emotion Probe Tests

Run automated tests to evaluate emotional intelligence:

```bash
python src/evaluation/evaluator.py
```

Tests include:
1. **Rudeness Test** - Does bot get colder naturally?
2. **Recovery Test** - Does bot thaw gradually after apology?
3. **Excitement Test** - Is reaction genuinely high-energy?
4. **Sarcasm Test** - Does bot respond with light sarcasm?
5. **Style Mirror Test** - Does bot pick up Hinglish?

### Scoring

- Average score > 4.2: **EXCELLENT** - Ready to deploy
- Average score > 3.5: **GOOD** - Passes evaluation
- Average score < 3.5: **NEEDS IMPROVEMENT** - More training

---

## Deployment

### Local Deployment

```bash
# Start API server
python src/deployment/api.py

# Server runs on http://localhost:8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/chat` | POST | Send message |
| `/profile/{user_id}` | GET | Get user profile |
| `/profile/{user_id}` | DELETE | Delete profile |
| `/reset/{user_id}` | POST | Reset conversation |

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "user_id": "user_123",
        "message": "Bhai main select ho gaya IIT mein!!"
    }
)

print(response.json())
# {
#   "user_id": "user_123",
#   "reply": "BHAI WHAT?? seriously?? that's amazing!!",
#   "bot_mood": 0.8,
#   "adaptation_stage": 1
# }
```

---

## Configuration

All settings are in `config/config.yaml`:

```yaml
# Model
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.3"

# LoRA
lora:
  r: 64                    # Rank
  lora_alpha: 128         # Scaling
  lora_dropout: 0.05      # Regularization

# Training
sft:
  num_train_epochs: 3
  learning_rate: 2.0e-4

dpo:
  num_train_epochs: 2
  learning_rate: 5.0e-5
  beta: 0.1               # DPO temperature

# Bot Personality
bot:
  name: "Nihar"

# Deployment
deployment:
  host: "0.0.0.0"
  port: 8000
```

---

## Free Resources

### Datasets
- [EmotionLines](https://huggingface.co/datasets/emotion_lines)
- [DailyDialog](https://huggingface.co/datasets/daily_dialog)
- [EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues)

### Models
- [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [LLaMA 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

### GPU Access (Free)
- Google Colab: colab.research.google.com
- Kaggle: kaggle.com
- Paperspace: paperspace.com

---

## License

MIT License - See LICENSE file for details.

---

## Quick Reference

### Training Checklist

- [ ] Install dependencies
- [ ] Prepare training data (50,000+ turns)
- [ ] Run SFT training (3 epochs)
- [ ] Run DPO training (2 epochs)
- [ ] Merge model
- [ ] Run evaluation tests
- [ ] Deploy API
- [ ] Test with real users

### Key Commands

```bash
# Install
pip install -r requirements.txt

# Generate sample data
python -c "from src.data.formatter import main; main()"

# Train SFT
python src/training/sft_trainer.py

# Train DPO
python src/training/dpo_trainer.py

# Deploy
python src/deployment/api.py
```

---

**Note**: This project uses Mistral-7B-Instruct-v0.3 as the base model, which is free to download from HuggingFace. All data preparation uses rule-based labeling - no paid APIs required.
