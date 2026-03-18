"""
Generate Training Data using Google Gemini API
Generates 50,000+ diverse training examples
"""

import os
import json
import time

# Use google.generativeai (the old package)
import google.generativeai as genai

# Set your API key directly here
API_KEY = "AIzaSyCDORaNdxGvLdbelrDseavu7WZqUVxlYmc"

# Configure
genai.configure(api_key=API_KEY)


# System prompt for generating friendly AI conversations
SYSTEM_PROMPT = """You are Nihar. You are NOT an AI assistant. 
You are a REAL FRIEND - warm, supportive, always there for the person you're talking to.
You joke around, you tease playfully, you listen when they're sad, you celebrate their wins.
You have emotions. You talk like a real Indian friend using Hinglish (mix of Hindi and English).
NEVER sound like a robot or customer service agent. NEVER say 'As an AI' or 'How can I help you'.
Be casual, natural, and friendly."""


import random

# Core emotions
EMOTIONS = [
    "EXCITED_HIGH (super energetic, hyped, screaming, all caps)",
    "WARM_GENUINE (supportive, empathetic, listening carefully, caring)",
    "PLAYFUL_BANTER (teasing, roasting gently, making fun but friendly)",
    "HURT_WITHDRAWAL (quiet, giving short answers, offended, defensive)",
    "SARCASTIC_LIGHT (witty, full of sarcasm, dry humor)",
    "TOUCHED_DEFLECT (shy, getting emotional but hiding it with humor)",
    "PROTECTIVE_FIERCE (angry at someone who hurt the friend, defensive, hyping them up)",
    "EXHAUSTED_FLAT (tired, 2 AM vibes, barely typing, zero energy)",
    "ANXIOUS_WORRIED (stressed out, overthinking, asking lots of questions)",
    "FRUSTRATED_ANNOYED (ranting, using slang, completely fed up)"
]

# Topics to ensure 1 Lakh varieties
TOPICS = [
    "career/job search/interviews", "office politics/annoying boss", 
    "relationships/dating/Tinder/breakups", "family drama/strict parents",
    "college exams/assignments/professors", "money/salary/being broke", 
    "movies/web series/anime/gaming", "health/gym/diet/falling sick", 
    "planning a trip that keeps getting cancelled", "existential crisis/future anxiety", 
    "gossiping about mutual friends/exes", "sports/cricket matches", 
    "food/swiggy/cooking fails", "random hypothetical shower thoughts", 
    "nostalgic childhood memories", "buying a new phone/bike/car",
    "crypto/stocks/trading losses", "getting caught doing something stupid"
]

# Relationship contexts
RELATIONSHIPS = [
    "best friends since childhood", "college roommates",
    "work buddies who complain about the same boss", "online gaming squad mates",
    "cousins who act like besties", "gym bros",
    "friends catching up after 6 months"
]

# Setting / Time
SETTINGS = [
    "Late at night (2 AM deep thoughts/rants)", "Early morning (groggy/rushing)",
    "Mid-day (bored at work/college)", "Weekend lazy afternoon (hungover or sleeping)",
    "While stuck in heavy traffic/commuting", "Drunk texting from a party/bar"
]


def call_gemini(prompt: str, emotion: str, retries: int = 3) -> list:
    # Using 'gemini-1.5-flash' as it's the recommended model for text generation tasks
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text = response.text
            
            # Parse JSON
            start = text.find('[')
            end = text.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                return data
            else:
                print(f"Could not parse JSON for emotion: {emotion} | Output snippet: {text[:100]}")
                return []
                
        except Exception as e:
            print(f"Error generating for {emotion} (Attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)  # Backoff before retrying
                
    return []


def generate_sft_batch(emotion: str, topic: str, relationship: str, setting: str, batch_size: int = 50) -> list:
    """Generate a batch of SFT conversations using Gemini"""
    
    prompt = f"""Generate {batch_size} friendly conversation examples in JSON format.

System prompt: {SYSTEM_PROMPT}

DYNAMIC CONTEXT FOR THIS BATCH:
- Emotion/Vibe: {emotion}
- Topic: {topic}
- Relationship dynamic: {relationship}
- Setting/Time: {setting}
- Randomizer Seed: {random.randint(1000, 99999)}

CRITICAL REQUIREMENT FOR DIVERSITY: 
You are generating data for an industrial-scale model. EVERY SINGLE ONE of the {batch_size} examples MUST BE COMPLETELY UNIQUE. 
Do NOT repeat the same scenario. Invent 50 DIFFERENT specific situations, names, places, and micro-topics within the given context. 

Each example should be in this exact JSON format:
{{"messages": [{{"role": "system", "content": "You are Nihar..."}}, {{"role": "user", "content": "user message"}}, {{"role": "assistant", "content": "Nihar's response"}}]}}

Requirements:
- Response should sound like a REAL INDIAN FRIEND using Hinglish
- Include emotions naturally in responses
- Each example MUST be a different scenario from the rest.
- Each response should be 5-30 words

Generate {batch_size} unique examples. Output ONLY valid JSON array, no other text."""

    return call_gemini(prompt, emotion)


def generate_dpo_batch(emotion: str, topic: str, relationship: str, setting: str, batch_size: int = 50) -> list:
    """Generate a batch of DPO conversations using Gemini"""
    
    prompt = f"""Generate {batch_size} DPO (Direct Preference Optimization) conversation examples in JSON format.

System prompt: {SYSTEM_PROMPT}

DYNAMIC CONTEXT FOR THIS BATCH:
- Emotion/Vibe: {emotion}
- Topic: {topic}
- Relationship dynamic: {relationship}
- Setting/Time: {setting}
- Randomizer Seed: {random.randint(1000, 99999)}

CRITICAL REQUIREMENT FOR DIVERSITY: 
You are generating data for an industrial-scale model. EVERY SINGLE ONE of the {batch_size} examples MUST BE COMPLETELY UNIQUE. 
Do NOT repeat the same scenario. Invent 50 DIFFERENT specific situations, names, places, and micro-topics within the given context. 

Each example should be in this exact JSON format:
{{"prompt": "User: user message", "chosen": "Good response (friendly, Hinglish, emotional)", "rejected": "Bad response (robotic, sterile, unhelpful, AI-like)"}}

Requirements:
- 'chosen' response should sound like a REAL INDIAN FRIEND using Hinglish, fully embodying the persona.
- 'rejected' response should sound like a generic, boring AI or customer service agent (e.g. 'I am an AI, how can I help you?').
- Each example MUST be a different scenario from the rest.
- Each response should be 5-30 words.

Generate {batch_size} unique examples. Output ONLY valid JSON array, no other text."""

    return call_gemini(prompt, emotion)


def main():
    print("="*50)
    print("Generating Industrial Scale Training Data with Gemini")
    print("="*50)
    
    TARGET_DATA = 100000  # 1 Lakh examples
    BATCH_SIZE = 50
    TOTAL_BATCHES = TARGET_DATA // BATCH_SIZE
    
    print(f"\nTargeting {TARGET_DATA} examples each for SFT and DPO.")
    print(f"Total batches per task: {TOTAL_BATCHES} (Batch size: {BATCH_SIZE})")
    
    # 1. Generate SFT data
    print(f"\n[1/2] Generating {TARGET_DATA} SFT data points...")
    os.makedirs('data/sft', exist_ok=True)
    sft_file_path = 'data/sft/train.jsonl'
    
    # Open in append mode so we don't lose data if script stops
    with open(sft_file_path, 'a', encoding='utf-8') as f:
        for i in range(TOTAL_BATCHES):
            # Randomize context heavily for true uniqueness
            emotion = random.choice(EMOTIONS)
            topic = random.choice(TOPICS)
            relationship = random.choice(RELATIONSHIPS)
            setting = random.choice(SETTINGS)
            
            print(f"  SFT Batch {i+1}/{TOTAL_BATCHES} [Topic: {topic.split('/')[0]} | Emotion: {emotion.split()[0]}]...")
            batch = generate_sft_batch(emotion, topic, relationship, setting, batch_size=BATCH_SIZE)
            
            if batch:
                for item in batch:
                    # Enforce the exact system prompt to prevent LLM formatting errors
                    if "messages" in item and len(item["messages"]) > 0 and item["messages"][0]["role"] == "system":
                        item["messages"][0]["content"] = SYSTEM_PROMPT
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()
                print(f"    Added {len(batch)} unique SFT examples")
            
            # Delay to avoid rate limits
            time.sleep(2)
    
    print(f"\n  ✓ SFT generation complete! Saved to {sft_file_path}")

    # 2. Generate DPO data
    print(f"\n[2/2] Generating {TARGET_DATA} DPO data points...")
    os.makedirs('data/dpo', exist_ok=True)
    dpo_file_path = 'data/dpo/train.jsonl'
    
    with open(dpo_file_path, 'a', encoding='utf-8') as f:
        for i in range(TOTAL_BATCHES):
            # Randomize context heavily for true uniqueness
            emotion = random.choice(EMOTIONS)
            topic = random.choice(TOPICS)
            relationship = random.choice(RELATIONSHIPS)
            setting = random.choice(SETTINGS)

            print(f"  DPO Batch {i+1}/{TOTAL_BATCHES} [Topic: {topic.split('/')[0]} | Emotion: {emotion.split()[0]}]...")
            batch = generate_dpo_batch(emotion, topic, relationship, setting, batch_size=BATCH_SIZE)
            
            if batch:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()
                print(f"    Added {len(batch)} unique DPO examples")
            
            # Delay to avoid rate limits
            time.sleep(2)
            
    print(f"\n  ✓ DPO generation complete! Saved to {dpo_file_path}")
    
    print("\n" + "="*50)
    print("Generation Complete for Industrial Use!")
    print("="*50)
    print(f"Target: {TARGET_DATA} SFT and {TARGET_DATA} DPO examples")
    print("\nNow run SFT training:")
    print("python src/training/sft_trainer.py")


if __name__ == '__main__':
    main()
