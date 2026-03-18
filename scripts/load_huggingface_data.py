"""
Load Free Training Data from HuggingFace
Downloads and formats emotion datasets from HuggingFace Hub
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.huggingface_loader import HuggingFaceDataLoader


# System prompt for the AI personality
SYSTEM_PROMPT = """You are Nihar. You are NOT an AI assistant. 
You are a REAL FRIEND - warm, supportive, always there for the person you're talking to.
You joke around, you tease playfully, you listen when they're sad, you celebrate their wins.
You have emotions. You talk like a real Indian friend using Hinglish (mix of Hindi and English).
NEVER sound like a robot or customer service agent. NEVER say 'As an AI' or 'How can I help you'.
Be casual, natural, and friendly."""


def convert_to_sft_format(conversations: list) -> list:
    """Convert HuggingFace conversations to SFT format"""
    sft_data = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        
        if len(messages) >= 2:
            # Find a human-bot or human-human pair
            for i in range(len(messages) - 1):
                # Look for alternating speakers (A, B, A, B pattern)
                if i % 2 == 0:  # Even index = human speaking
                    user_msg = messages[i].get('text', '')
                    bot_msg = messages[i + 1].get('text', '') if i + 1 < len(messages) else ''
                    
                    if user_msg and bot_msg and len(user_msg) > 3 and len(bot_msg) > 3:
                        example = {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": bot_msg}
                            ]
                        }
                        sft_data.append(example)
    
    return sft_data


def convert_to_dpo_format(conversations: list) -> list:
    """Convert to DPO format (chosen/rejected pairs)"""
    dpo_data = []
    
    # Templates for creating human-like vs chatbot-like pairs
    chatbot_responses = [
        "I understand. How can I help you with that?",
        "Thank you for sharing that with me. Is there anything else you'd like to discuss?",
        "I appreciate you telling me. What would you like to do next?",
        "That's interesting. Could you tell me more about it?",
        "I see. How does that make you feel?",
        "Thank you for your message. How may I assist you today?",
    ]
    
    import random
    
    for conv in conversations:
        messages = conv.get('messages', [])
        
        if len(messages) >= 2:
            for i in range(len(messages) - 1):
                if i % 2 == 0:
                    user_msg = messages[i].get('text', '')
                    human_response = messages[i + 1].get('text', '') if i + 1 < len(messages) else ''
                    
                    if user_msg and human_response and len(user_msg) > 3 and len(human_response) > 3:
                        # Create DPO pair: human response (chosen) vs chatbot (rejected)
                        example = {
                            "prompt": f"User: {user_msg}",
                            "chosen": human_response,
                            "rejected": random.choice(chatbot_responses)
                        }
                        dpo_data.append(example)
    
    return dpo_data


def main():
    print("="*60)
    print("Loading Free Training Data from HuggingFace")
    print("="*60)
    
    # Load datasets
    loader = HuggingFaceDataLoader()
    
    print("\n[1/4] Loading EmotionLines dataset...")
    emotion_lines = loader.load_emotion_lines()
    
    print("\n[2/4] Loading DailyDialog dataset...")
    daily_dialog = loader.load_daily_dialog()
    
    print("\n[3/4] Loading EmpatheticDialogues dataset...")
    empathetic = loader.load_empathetic_dialogues()
    
    # Combine all conversations
    all_conversations = emotion_lines + daily_dialog + empathetic
    print(f"\nTotal raw conversations: {len(all_conversations)}")
    
    # Convert to SFT format
    print("\n[4/4] Converting to training formats...")
    sft_data = convert_to_sft_format(all_conversations)
    dpo_data = convert_to_dpo_format(all_conversations)
    
    # Remove duplicates based on user message
    seen_prompts = set()
    unique_sft = []
    for item in sft_data:
        prompt = item['messages'][1]['content']
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_sft.append(item)
    
    unique_dpo = []
    for item in dpo_data:
        prompt = item['prompt']
        if prompt not in seen_prompts:
            unique_dpo.append(item)
    
    print(f"Unique SFT examples: {len(unique_sft)}")
    print(f"Unique DPO examples: {len(unique_dpo)}")
    
    # Save to files
    os.makedirs('data/sft', exist_ok=True)
    os.makedirs('data/dpo', exist_ok=True)
    
    with open('data/sft/train.jsonl', 'w', encoding='utf-8') as f:
        for item in unique_sft:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open('data/dpo/train.jsonl', 'w', encoding='utf-8') as f:
        for item in unique_dpo:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n" + "="*60)
    print("DONE! Training data saved:")
    print(f"  SFT: data/sft/train.jsonl ({len(unique_sft)} examples)")
    print(f"  DPO: data/dpo/train.jsonl ({len(unique_dpo)} examples)")
    print("="*60)
    print("\nNext step - Run SFT training:")
    print("python src/training/sft_trainer.py")


if __name__ == '__main__':
    main()
