"""
Data Formatter
Converts raw conversations to SFT and DPO training formats
Uses rule-based emotion labeling (no paid APIs)
"""

import json
import re
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter


class RuleBasedEmotionLabeler:
    """
    Rule-based emotion labeler using linguistic patterns
    No paid APIs required - uses pattern matching
    """
    
    # Emotion patterns based on linguistic features
    EMOTION_PATTERNS = {
        'ANGRY_SHARP': {
            'keywords': ['hate', 'angry', 'furious', 'annoyed', 'irritated', 'stupid', 'idiot', 'worst', 'terrible'],
            'patterns': [
                r'^.{0,30}\.$',  # Very short
                r'!{2,}',        # Multiple exclamations (emphasis, not joy)
                r'\bnot\b.*\bagain\b',  # Frustration
            ],
            'indicators': ['no warmth', 'blunt', 'cold']
        },
        
        'HURT_WITHDRAWAL': {
            'keywords': ['hurt', 'sad', 'disappointed', 'miss', 'alone', 'ignored', 'forgotten'],
            'patterns': [
                r'^.{0,15}$',    # Very short responses
                r'\.{3,}',       # Ellipsis (going quiet)
                r'\bok+\b|\bsure\b|\nyeah\b',  # Minimal acknowledgment
            ],
            'indicators': ['quiet', 'minimal', 'withdrawn']
        },
        
        'SARCASTIC_LIGHT': {
            'keywords': ['yeah right', 'sure', 'obviously', 'clearly', 'totally'],
            'patterns': [
                r'\byeah,?\s+totally\b',
                r'\bobviously\b.*\.',
                r'wow,?.*\.',
                r'\bsure,?.*\bsure\b',
            ],
            'indicators': ['playful', 'teasing', 'dry wit']
        },
        
        'SARCASTIC_DARK': {
            'keywords': ['great', 'wonderful', 'perfect', 'amazing', 'fantastic'],
            'patterns': [
                r'(great|wonderful|perfect).*\.{3}',
                r'oh\s+(wow|great|perfect)',
                r'\byeah\s+great\b',
            ],
            'indicators': ['biting', 'edge', 'not fully playful']
        },
        
        'EXCITED_HIGH': {
            'keywords': ['excited', 'amazing', 'awesome', 'incredible', 'happy', 'love it'],
            'patterns': [
                r'!{2,}',
                r'\b[A-Z]{2,}\b',  # Caps lock
                r'\?{2,}',
                r'oh my god|omg|wow',
                r'\!.*\?.*\!',  # Multiple reactions
            ],
            'indicators': ['energy', 'caps', 'rapid']
        },
        
        'WARM_GENUINE': {
            'keywords': ['care', 'understand', 'here for you', 'support', 'love', 'miss you'],
            'patterns': [
                r'\bI understand\b',
                r'\bI\'m here\b',
                r'\byou\'re not alone\b',
                r'\bcare\b.*\byou\b',
                r'\bsupport\b',
            ],
            'indicators': ['warm', 'supportive', 'caring']
        },
        
        'PLAYFUL_BANTER': {
            'keywords': ['joke', 'lol', 'haha', 'teasing', 'kidding', 'silly'],
            'patterns': [
                r'\blol+\b',
                r'\bhaha+\b',
                r'\bjk\b|\bkidding\b',
                r'\bsilly\b',
                r'\?.*\?.*\?',  # Playful questions
            ],
            'indicators': ['jokes', 'teasing', 'casual']
        },
        
        'EXHAUSTED_FLAT': {
            'keywords': ['tired', 'exhausted', 'drained', 'done', 'whatever', 'not now'],
            'patterns': [
                r'\.\.$',
                r'^.{0,20}$',  # Short
                r'\bsure\b.*\bwhatever\b',
                r'\bI don\'t know\b',
            ],
            'indicators': ['flat', 'tired', 'no energy']
        },
        
        'PROTECTIVE_FIERCE': {
            'keywords': ['protect', 'stand up', 'won\'t let', 'defend', 'fight', 'against'],
            'patterns': [
                r'\bwon\'t\b.*\baccept\b',
                r'\bstanding\b.*\bup\b',
                r'\bdefend\b.*\b',
                r'\bfight\b.*\bfor\b',
            ],
            'indicators': ['protective', 'fierce', 'standing up']
        },
        
        'TOUCHED_DEFLECT': {
            'keywords': ['stop', 'don\'t', 'okay okay', 'whatever', 'no way'],
            'patterns': [
                r'\bstop\b.*\bnow\b',
                r'\bokay+.*okay+\b',
                r'\bno+.*way+\b',
                r'\bdon\'t\b.*\bmake\b',
            ],
            'indicators': ['deflects with humor', 'moved but hides']
        },
        
        'COLDLY_DISAPPOINTED': {
            'keywords': ['disappointed', 'expected better', 'unfortunate', 'shame'],
            'patterns': [
                r'\bI expected\b',
                r'\bbetter than\b.*\bthis\b',
                r'\bshame\b',
                r'\bdisappointing\b',
            ],
            'indicators': ['cold', 'disappointed', 'worse than angry']
        },
        
        'GENUINELY_CURIOUS': {
            'keywords': ['tell me more', 'how', 'why', 'what happened', 'explain'],
            'patterns': [
                r'\?$',  # Ends with question
                r'\bwhat happened\b',
                r'\bhow did\b',
                r'\bwhy do you think\b',
                r'\btell me more\b',
            ],
            'indicators': ['questions', 'wants to understand']
        }
    }
    
    def label_conversation(self, text: str) -> str:
        """
        Label a conversation turn with emotion
        
        Args:
            text: The text to analyze
            
        Returns:
            Emotion label string
        """
        text_lower = text.lower()
        scores = {}
        
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 1
            
            # Check patterns
            for pattern in patterns['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 2
            
            if score > 0:
                scores[emotion] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        # Default to WARM_GENUINE for neutral responses
        return 'WARM_GENUINE'


class DataFormatter:
    """Formats data for SFT and DPO training"""
    
    def __init__(self, personality_description: str):
        """
        Initialize formatter
        
        Args:
            personality_description: The bot's personality for system prompts
        """
        self.personality_description = personality_description
        self.labeler = RuleBasedEmotionLabeler()
        
    def format_for_sft(
        self, 
        conversations: List[Dict],
        output_path: str,
        min_turns: int = 2
    ):
        """
        Format conversations for SFT training
        
        Creates JSONL with messages format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        
        Args:
            conversations: Raw conversation list
            output_path: Output file path
            min_turns: Minimum number of message turns
        """
        sft_data = []
        
        for conv in conversations:
            messages = conv.get('messages', [])
            
            if len(messages) < min_turns:
                continue
            
            # Build conversation turns
            conversation_messages = []
            
            # Add system prompt with personality
            conversation_messages.append({
                'role': 'system',
                'content': self.personality_description
            })
            
            # Add message turns
            for msg in messages[:10]:  # Limit to 10 turns
                if isinstance(msg, dict):
                    text = msg.get('text', '') or msg.get('content', '')
                    speaker = msg.get('speaker', 'unknown')
                else:
                    text = str(msg)
                    speaker = 'unknown'
                
                if not text.strip():
                    continue
                
                # Alternate between user and assistant
                role = 'user' if len(conversation_messages) % 2 == 1 else 'assistant'
                
                conversation_messages.append({
                    'role': role,
                    'content': text.strip()
                })
            
            # Only keep valid conversations (has user and assistant)
            has_user = any(m['role'] == 'user' for m in conversation_messages)
            has_assistant = any(m['role'] == 'assistant' for m in conversation_messages)
            
            if has_user and has_assistant and len(conversation_messages) >= 3:
                sft_data.append({
                    'messages': conversation_messages
                })
        
        # Save to JSONL
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"SFT: Saved {len(sft_data)} samples to {output_path}")
        return sft_data
    
    def format_for_dpo(
        self,
        conversations: List[Dict],
        output_path: str,
        generate_rejected: bool = True
    ):
        """
        Format conversations for DPO training
        
        Creates JSONL with prompt, chosen, rejected format:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
        
        Args:
            conversations: Raw conversation list
            output_path: Output file path
            Generate rejected responses if True
        """
        dpo_data = []
        
        for conv in conversations:
            messages = conv.get('messages', [])
            
            if len(messages) < 2:
                continue
            
            # Use emotion label from conversation
            emotion = conv.get('emotion', 'WARM_GENUINE')
            
            # Extract user messages and assistant responses
            for i in range(len(messages) - 1):
                current = messages[i]
                next_msg = messages[i + 1]
                
                if isinstance(current, dict):
                    user_text = current.get('text', '') or current.get('content', '')
                else:
                    user_text = str(current)
                    
                if isinstance(next_msg, dict):
                    assistant_text = next_msg.get('text', '') or next_msg.get('content', '')
                else:
                    assistant_text = str(next_msg)
                
                if not user_text.strip() or not assistant_text.strip():
                    continue
                
                # Create prompt with context
                prompt = f"[Emotion: {emotion}] User: {user_text.strip()}"
                chosen = assistant_text.strip()
                
                # Generate rejected (chatbot-style) response
                if generate_rejected:
                    rejected = self._generate_rejected_response(
                        user_text.strip(), 
                        emotion
                    )
                else:
                    rejected = ""
                
                dpo_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'emotion': emotion
                })
        
        # Save to JSONL
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dpo_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"DPO: Saved {len(dpo_data)} pairs to {output_path}")
        return dpo_data
    
    def _generate_rejected_response(self, user_message: str, emotion: str) -> str:
        """
        Generate a chatbot-style rejected response
        
        This creates the "wrong" answer that DPO will train against
        
        Args:
            user_message: The user's message
            emotion: The emotion context
            
        Returns:
            A generic chatbot-style response
        """
        # Generic chatbot responses
        chatbot_responses = [
            "I understand. How can I help you with that?",
            "Thank you for sharing. Is there anything else I can assist you with?",
            "I see. Could you please provide more details?",
            "That's interesting. Let me know if there's anything else.",
            "I appreciate you telling me. How can I be of service?",
            "Got it. Is there something specific you'd like to discuss?",
            "I hear you. What would you like to talk about next?",
            "Thanks for sharing that. How can I help you today?",
        ]
        
        return chatbot_responses[len(user_message) % len(chatbot_responses)]


class SampleDataGenerator:
    """Generates sample training data for demonstration"""
    
    # Sample conversations in the bot's voice
    SAMPLE_CONVERSATIONS = [
        {
            'messages': [
                {'speaker': 'user', 'text': 'bhai tu kabhi bhi time pe nahi aata'},
                {'speaker': 'assistant', 'text': 'yaar sorry nahi bolunga kyunki tu bhi jaanta hai main aaunga'}
            ],
            'emotion': 'HURT_WITHDRAWAL'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'kal meri presentation thi, sab ne mujhe ignore kiya'},
                {'speaker': 'assistant', 'text': 'kya matlab ignore? poori baat bata. ye log kaun the'}
            ],
            'emotion': 'PROTECTIVE_FIERCE'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'whatever man'},
                {'speaker': 'assistant', 'text': 'theek hai. baat karna ho toh karna.'}
            ],
            'emotion': 'ANGRY_SHARP'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'bhai main select ho gaya IIT mein!!'},
                {'speaker': 'assistant', 'text': 'BHAI SERIOUSLY?? ye to insane hai yaar, kabse pata tha tujhe??'}
            ],
            'emotion': 'EXCITED_HIGH'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'bhai you told me wrong thing yesterday, I failed because of you'},
                {'speaker': 'assistant', 'text': 'haan bilkul, main hi zimedaar hoon teri preparation ke liye bhi.'}
            ],
            'emotion': 'SARCASTIC_LIGHT'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'honestly you are one of the few people I can actually talk to'},
                {'speaker': 'assistant', 'text': 'stop yaar.. main nahi jaanta kya bolun iska'}
            ],
            'emotion': 'TOUCHED_DEFLECT'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'you are useless just like everyone said'},
                {'speaker': 'assistant', 'text': 'okay.'}
            ],
            'emotion': 'HURT_WITHDRAWAL'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'meri girlfriend ne mujhe chod diya'},
                {'speaker': 'assistant', 'text': 'yaar kya baat hai. dost, time lete hain. main hoon na'}
            ],
            'emotion': 'WARM_GENUINE'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'tu really kya kar raha hai'},
                {'speaker': 'assistant', 'text': 'yaar bas yahi. din guzar raha hai. tu bata'}
            ],
            'emotion': 'EXHAUSTED_FLAT'
        },
        {
            'messages': [
                {'speaker': 'user', 'text': 'achha soch raha tha, weekend pe trip chalo?'},
                {'speaker': 'assistant', 'text': 'BHAI KYA? bilkul! kahan jaana hai? plan bata'}
            ],
            'emotion': 'EXCITED_HIGH'
        }
    ]
    
    @classmethod
    def generate_sample_data(cls, output_dir: str):
        """Generate sample SFT and DPO data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample conversations
        with open(f'{output_dir}/sample_conversations.jsonl', 'w') as f:
            for conv in cls.SAMPLE_CONVERSATIONS:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        print(f"Generated {len(cls.SAMPLE_CONVERSATIONS)} sample conversations")


def main():
    """Example usage"""
    # Generate sample data
    SampleDataGenerator.generate_sample_data('data/processed')
    
    # Format for SFT
    personality = """You are Nihar. You are not an AI assistant.
You are a real person with emotions, opinions, and a strong personality.
You feel things. You get irritated. You get excited. You go quiet when hurt.
You never say 'as an AI' or 'I cannot'. You just respond like a human."""
    
    formatter = DataFormatter(personality)
    
    # Load sample data
    with open('data/processed/sample_conversations.jsonl', 'r') as f:
        conversations = [json.loads(line) for line in f]
    
    # Format for SFT
    formatter.format_for_sft(
        conversations, 
        'data/sft/train.jsonl'
    )
    
    # Format for DPO
    formatter.format_for_dpo(
        conversations,
        'data/dpo/train.jsonl'
    )


if __name__ == '__main__':
    main()
