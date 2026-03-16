# EmotiAI: Complete Project & Technical Overview

EmotiAI is a complete, full-stack AI engineering project designed to create an **Emotionally Intelligent Personal AI Model**. 

Unlike standard AI assistants that are helpful but robotic, EmotiAI is designed to act as a **genuine friend**. It possesses a default warm personality, dynamically learns to mirror each user's specific chatting style (such as using regional dialects like Hinglish), and is explicitly trained to express genuine human emotions.

This document serves as the master overview of the entire project.

---

## 1. Core Capabilities & Features

1.  **12 Distinct Emotion Categories:** The model is trained to recognize and output specific emotional states. Rather than just being "happy" or "sad," it understands nuance: `ANGRY_SHARP`, `HURT_WITHDRAWAL`, `SARCASTIC_LIGHT`, `SARCASTIC_DARK`, `EXCITED_HIGH`, `WARM_GENUINE`, `PLAYFUL_BANTER`, `EXHAUSTED_FLAT`, `PROTECTIVE_FIERCE`, `TOUCHED_DEFLECT`, `COLDLY_DISAPPOINTED`, and `GENUINELY_CURIOUS`.
2.  **Per-User Style Adaptation:** As users chat with the bot, a background `Style Extractor` analyzes their message length, emoji usage, and vocabulary (e.g., Hinglish words). The bot then seamlessly updates its own `System Prompt` on the fly to mirror the user.
3.  **Human-Like Preference:** Through DPO training, the model is penalized for giving standard "AI assistant" replies ("I apologize for the confusion") and rewarded for sounding like a real human ("yaar sorry nahi bolunga").
4.  **Accessible Training:** The entire system relies on free data sources and is built to be trained completely for free on a single, consumer-grade GPU using **QLoRA** quantization.

---

## 2. Technical Architecture

EmotiAI relies on a multi-stage request architecture when running in production:

```
┌─────────────────────────────────────────────────────────┐
│                    User Message                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              User Profile Manager                       │
│         (Loads historical style from SQLite DB)         │
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
│     (Injects base personality + user style adaptation)  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Fine-tuned Mistral 7B                      │
│          (QLoRA Model - runs on single GPU)             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Emotion Driven Output                      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Requirements & Sourcing

To reach a production-ready state, the model requires extensive conversational data—a minimum of **50,000 conversational turns**, ideally spanning up to 200,000. 

### Free Data Sources Used
*   **HuggingFace Datasets:** *EmotionLines* (29k conversations), *DailyDialog* (13k conversations), and *EmpatheticDialogues* (25k conversations).
*   **Reddit Scraping:** Utilizing the `praw` API, the system can scrape emotionally charged subreddits like `r/relationship_advice`, `r/AITA`, and `r/offmychest`.
*   **Personal Logs:** Scripts can format WhatsApp/Telegram text exports.

### Data Types Required for Training
The data must be split into two formats to feed the two different training stages:

**A. Supervised Fine-Tuning (SFT) Data**
*   **Required Volume:** 10,000 - 20,000 examples.
*   **Format:** Standard conversational JSONL mapping a `system` prompt, a `user` input, and an `assistant` reply showcasing the correct personality.
*   **Purpose:** Teaches the model the core personality, vocabulary, and the 12 specific emotions.

**B. Direct Preference Optimization (DPO) Data**
*   **Required Volume:** 30,000 - 50,000 preference pairs.
*   **Format:** JSONL mapping a `prompt` against a human-like `chosen` response, and an AI-like `rejected` response.
*   **Purpose:** Enforces the "anti-robot" behavior by explicitly telling the model what *not* to sound like.

---

## 4. The Foundation Model

*   **Base Model:** The project uses `mistralai/Mistral-7B-Instruct-v0.3`.
*   **Why Mistral 7B?** It is an extremely capable open-source model that punches above its weight class. It has 7 billion parameters, making it small enough to be fine-tuned quickly but large enough to grasp complex emotional nuance and complex system prompts.
*   **Quantization:** The codebase heavily utilizes **QLoRA** (Quantized Low-Rank Adaptation). The massive Mistral model is loaded in `4-bit` precision using the `bitsandbytes` library with `nf4` quantization. This reduces the VRAM requirement from ~30GB+ down to under 12GB.

---

## 5. The Training Pipeline

The training system is modular and sequential:

### Step 1: [sft_trainer.py](file:///c:/Users/Hp/Documents/emotai/src/training/sft_trainer.py)
*   **Action:** Runs Supervised Fine Tuning.
*   **Mechanism:** Rather than updating all 7 Billion parameters (which would require supercomputers), the system freezes the base model and attaches a **LoRA adapter**. It only trains this tiny adapter matrix. 
*   **Compute:** Requires `Rank (r)=64`, trained at a learning rate of `2.0e-4` over 3 epochs.

### Step 2: `dpo_trainer.py`
*   **Action:** Runs Direct Preference Optimization.
*   **Mechanism:** Loads the adapter created in Step 1, and trains the model against the preference pairs dataset to cement the "human" feel.

### Step 3: Model Merging
*   **Action:** The small LoRA matrices created during training are mathematically added/merged perfectly back into the frozen Mistral 7B model. This creates one single, unified, lightweight file folder that can be easily loaded for deployment.

---

## 6. Hardware & Compute Requirements

| Task | Hardware Required | VRAM | Estimated Duration |
| :--- | :--- | :--- | :--- |
| **Full SFT + DPO Training** | Single NVIDIA A100 | 8–12 GB | 2–4 Hours |
| **Full SFT + DPO Training** | Single RTX 3090/4090 | 12 GB | 4–8 Hours |
| **Model Inference (Chatting)** | Any consumer GPU | 4 GB | Instantaneous |

*(Note: If local GPUs are not available, the codebase is designed to be fully compatible with free GPU instances from Google Colab, Kaggle, or Paperspace).*

---

## 7. Deployment & Evaluation

### Evaluation Methodology (`evaluator.py`)
Before deployment, the model undergoes **Emotion Probe Tests**. These are simulated conversations to test behavioral consistency:
*   *Rudeness Test:* Does the bot get coldly distant if the user is repeatedly rude?
*   *Recovery Test:* Does the bot warm up again if the user apologizes?
*   *Style Mirror Test:* Does the bot adapt to Hinglish within 15 messages?

### Production API (`api.py`)
Once merged and evaluated, the model is served via a lightweight **FastAPI backend** (`uvicorn`). The server handles incoming requests, maintains concurrent user states via an SQLite database tracking adapter stages, and generates the final model outputs.
