# EmotiAI Training & Production Documentation

This document explicitly breaks down the data requirements, the training process, the architectural layers, and the compute requirements for building the EmotiAI model.

---

## 1. Data Requirements
For a production-grade model, the bare minimum data required is **50,000 conversational conversational turns**. Ideally, you want to target **200,000 conversational turns**.

This data must be broken down and formatted into two distinct datasets which serve different purposes:

### A. SFT (Supervised Fine-Tuning) Data
*   **Minimum Volume Required:** 10,000 to 20,000 conversational turns.
*   **Purpose:** SFT teaches the model "how to act." It defines the personality (acting as a friend rather than a robotic assistant) and teaches the model how to express the 12 specific emotional categories (e.g., `ANGRY_SHARP`, `SARCASTIC_DARK`, `WARM_GENUINE`, `PLAYFUL_BANTER`).
*   **Format Requirement (JSONL):**
    ```json
    {"messages": [{"role": "system", "content": "You are Nihar..."}, {"role": "user", "content": "bhai tu..."}, {"role": "assistant", "content": "yaar sorry..."}]}
    ```

### B. DPO (Direct Preference Optimization) Data
*   **Minimum Volume Required:** 30,000 to 50,000 pairs.
*   **Purpose:** DPO teaches the model "what to prefer." It distinguishes between sounding like a human versus sounding like a chatbot. The model is penalized for choosing standard AI responses ("I understand your frustration...") and rewarded for choosing human-like responses ("theek hai. baat karna ho toh karna").
*   **Format Requirement (JSONL):**
    ```json
    {"prompt": "User is rude: whatever man", "chosen": "theek hai.", "rejected": "How can I assist you?"}
    ```

---

## 2. Models Used
The system relies on a **Mistral Base Model** with LoRA (Low-Rank Adaptation) adapters attached. 

*   **Primary Foundation Model:** `mistralai/Mistral-7B-Instruct-v0.3`
*   **Quantization Context:** The model is quantized using **QLoRA** (4-bit quantization with `nf4` and double quantization enabled). This reduces the memory footprint significantly, dropping the VRAM requirement from ~30GB+ to under 12GB.

---

## 3. The End-to-End Training Pipeline

The training consists of three major steps, run sequentially:

### Step 1: Style Adaptation & Personality (SFT)
We use the [sft_trainer.py](file:///c:/Users/Hp/Documents/emotai/src/training/sft_trainer.py) script to run Supervised Fine-Tuning.
*   **What it does:** Updates a small portion of the model's weights (the LoRA adapter) to match the conversational tone found in your JSONL data.
*   **Configuration:** 
    *   **LoRA Rank (`r`)**: 64 (Captures complex reasoning and style).
    *   **Dropout**: 0.05
    *   **Learning Rate**: 2.0e-4

### Step 2: Emotional Realism (DPO)
We use the `dpo_trainer.py` script.
*   **What it does:** Uses Reinforcement Learning with Human Feedback (RLHF) principles, specifically DPO, to align the model. It compares a prompt against a `chosen` response and a `rejected` response, aggressively shifting the probability distribution towards the `chosen` (human-sounding) response.
*   **Configuration:**
    *   **Beta Parameter:** 0.1 (Controls how far the model is allowed to deviate from the Step 1 SFT model).

### Step 3: Model Merging
*   **What it does:** The LoRA weights (adapters) are fundamentally just small additive matrices. During the merge step (using `--merge`), these small matrices are mathematically added directly back into the massive 7-Billion parameter Mistral base model, creating a single, unified model ready for deployment.

---

## 4. Compute Requirements

Because of the aggressive quantization (QLoRA) and adapter-based training, this does not require an entire datacenter to train.

| Type | Hardware | Minimum VRAM | Estimated Time |
| :--- | :--- | :--- | :--- |
| **Training (QLoRA)** | Single NVIDIA A100 | 8–12 GB | 2–4 Hours |
| **Training (QLoRA)** | Single RTX 3090/4090 | 12 GB | 4–8 Hours |
| **Inference (Running)** | Any standard consumer GPU | 4 GB | Instant |

---

## 5. Deployment Architecture

Once trained, the model operates using a multi-stage request architecture:

1.  **User Message Received:** E.g., *"Bhai kya haal hai?"*
2.  **User Profile Extraction:** The backend pulls the specific user's stylistic profile from an SQLite DB (tracking if they use Hinglish, short messages, emojis).
3.  **Prompt Construction:** The style preferences are injected directly into the `System Prompt`.
4.  **Inference:** The Merged Mistral 7B model processes the prompt.
5.  **Output:** Emotionally driven response is generated and sent back to the user via the FastAPI backend.
