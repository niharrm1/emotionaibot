"""
HuggingFace Dataset Loader
Loads free emotional conversation datasets from HuggingFace
"""

import os
import json
from typing import List, Dict, Optional
from datasets import load_dataset, Dataset


class HuggingFaceDataLoader:
    """Loads and processes free datasets from HuggingFace"""
    
    # Available free emotional datasets
    DATASETS = {
        'emotion_lines': {
            'name': 'emotion_lines',
            'description': '29,000 conversations from Friends with emotion labels'
        },
        'daily_dialog': {
            'name': 'daily_dialog', 
            'description': '13,000 daily conversations with emotion labels'
        },
        'empathetic_dialogues': {
            'name': 'empathetic_dialogues',
            'description': 'Conversations with emotional context'
        }
    }
    
    # Emotion label mapping
    EMOTION_MAPPING = {
        # emotion_lines emotions
        'joy': 'EXCITED_HIGH',
        'sadness': 'WARM_GENUINE',  # mapped to supportive response
        'anger': 'ANGRY_SHARP',
        'fear': 'PROTECTIVE_FIERCE',
        'disgust': 'COLDLY_DISAPPOINTED',
        'neutral': 'WARM_GENUINE',
        
        # daily_dialog emotions
        'happiness': 'EXCITED_HIGH',
        'sadness': 'HURT_WITHDRAWAL',
        'anger': 'ANGRY_SHARP',
        'fear': 'GENUINELY_CURIOUS',
        'surprise': 'EXCITED_HIGH',
        'disgust': 'COLDLY_DISAPPOINTED',
        'neutral': 'WARM_GENUINE'
    }
    
    def load_emotion_lines(self) -> List[Dict]:
        """Load EmotionLines dataset"""
        print("Loading EmotionLines dataset...")
        
        try:
            dataset = load_dataset('emotion_lines', split='train')
            
            conversations = []
            for item in dataset:
                # Extract conversation turns
                if 'dialog' in item:
                    # Convert dialog to conversation format
                    dialog = item['dialog']
                    if isinstance(dialog, list):
                        for i in range(len(dialog) - 1):
                            conv = {
                                'source': 'emotion_lines',
                                'emotion': self.EMOTION_MAPPING.get(
                                    item.get('emotion', 'neutral'), 
                                    'WARM_GENUINE'
                                ),
                                'messages': []
                            }
                            
                            for turn in dialog[:i+2]:
                                if isinstance(turn, dict):
                                    conv['messages'].append(turn)
                                elif isinstance(turn, str):
                                    # Parse speaker and text
                                    if ':' in turn:
                                        speaker, text = turn.split(':', 1)
                                        conv['messages'].append({
                                            'speaker': speaker.strip(),
                                            'text': text.strip()
                                        })
                            
                            if conv['messages']:
                                conversations.append(conv)
                                
            print(f"Loaded {len(conversations)} conversations from EmotionLines")
            return conversations
            
        except Exception as e:
            print(f"Error loading EmotionLines: {e}")
            return []
    
    def load_daily_dialog(self) -> List[Dict]:
        """Load DailyDialog dataset"""
        print("Loading DailyDialog dataset...")
        
        try:
            dataset = load_dataset('daily_dialog', split='train')
            
            conversations = []
            for item in dataset:
                dialog = item.get('dialog', [])
                emotions = item.get('emotion', [])
                
                if len(dialog) >= 2:
                    conv = {
                        'source': 'daily_dialog',
                        'emotion': self.EMOTION_MAPPING.get(
                            emotions[0] if emotions else 'neutral',
                            'WARM_GENUINE'
                        ),
                        'messages': []
                    }
                    
                    for i, text in enumerate(dialog):
                        speaker = 'A' if i % 2 == 0 else 'B'
                        conv['messages'].append({
                            'speaker': speaker,
                            'text': text
                        })
                    
                    conversations.append(conv)
                    
            print(f"Loaded {len(conversations)} conversations from DailyDialog")
            return conversations
            
        except Exception as e:
            print(f"Error loading DailyDialog: {e}")
            return []
    
    def load_empathetic_dialogues(self) -> List[Dict]:
        """Load EmpatheticDialogues dataset"""
        print("Loading EmpatheticDialogues dataset...")
        
        try:
            dataset = load_dataset('empathetic_dialogues', split='train')
            
            conversations = []
            for item in dataset:
                context = item.get('context', '')
                prompt = item.get('prompt', '')
                utterances = item.get('utterances', [])
                
                if utterances and len(utterances) >= 2:
                    conv = {
                        'source': 'empathetic_dialogues',
                        'emotion': self.EMOTION_MAPPING.get(
                            context if context in self.EMOTION_MAPPING else 'neutral',
                            'WARM_GENUINE'
                        ),
                        'context': context,
                        'prompt': prompt,
                        'messages': []
                    }
                    
                    for utt in utterances:
                        if isinstance(utt, dict):
                            conv['messages'].append({
                                'speaker': utt.get('speaker', 'unknown'),
                                'text': utt.get('text', '')
                            })
                        elif isinstance(utt, str):
                            conv['messages'].append({
                                'speaker': 'speaker',
                                'text': utt
                            })
                    
                    if conv['messages']:
                        conversations.append(conv)
                        
            print(f"Loaded {len(conversations)} from EmpatheticDialogues")
            return conversations
            
        except Exception as e:
            print(f"Error loading EmpatheticDialogues: {e}")
            return []
    
    def load_all_datasets(self) -> List[Dict]:
        """Load all available datasets"""
        all_conversations = []
        
        all_conversations.extend(self.load_emotion_lines())
        all_conversations.extend(self.load_daily_dialog())
        all_conversations.extend(self.load_empathetic_dialogues())
        
        print(f"\nTotal conversations loaded: {len(all_conversations)}")
        return all_conversations
    
    def save_to_jsonl(self, conversations: List[Dict], output_path: str):
        """Save conversations to JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(conversations)} conversations to {output_path}")


def main():
    """Example usage"""
    loader = HuggingFaceDataLoader()
    conversations = loader.load_all_datasets()
    loader.save_to_jsonl(conversations, 'data/raw/huggingface_conversations.jsonl')


if __name__ == '__main__':
    main()
