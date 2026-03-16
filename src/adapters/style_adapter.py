"""
User Style Adapter
Adapts the bot's communication style to match each user's preferences
"""

import re
import os
import json
import sqlite3
from typing import Dict, List, Optional
from collections import Counter
import yaml


class UserProfile:
    """User profile structure for style adaptation"""
    
    DEFAULT_PROFILE = {
        'user_id': '',
        'name': '',
        'messages_sent': 0,
        'avg_message_length': 0.0,
        'uses_hinglish': False,
        'common_words': [],
        'uses_emoji': False,
        'punctuation_style': 'normal',
        'humor_level': 'medium',
        'topic_interests': [],
        'last_bot_mood': 0.0,
        'adaptation_stage': 0,
        'conversation_history': []
    }
    
    def __init__(self, user_id: str, profile_data: Optional[Dict] = None):
        """
        Initialize user profile
        
        Args:
            user_id: Unique user identifier
            profile_data: Existing profile data (optional)
        """
        self.data = profile_data.copy() if profile_data else self.DEFAULT_PROFILE.copy()
        self.data['user_id'] = user_id
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary"""
        return self.data.copy()
    
    def get(self, key: str, default=None):
        """Get profile value"""
        return self.data.get(key, default)


class StyleExtractor:
    """
    Extracts style features from user messages
    Updates profile based on communication patterns
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize style extractor with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        adaptation_config = config['adaptation']['style_extractor']
        
        self.hinglish_words = set(adaptation_config['hinglish_words'])
        self.stopwords = set(adaptation_config['stopwords'])
        self.top_n_words = adaptation_config['top_common_words']
    
    def extract_features(self, message: str, profile: UserProfile) -> UserProfile:
        """
        Extract style features from a message and update profile
        
        Args:
            message: User's message
            profile: Current user profile
            
        Returns:
            Updated profile
        """
        data = profile.data
        n = data['messages_sent'] + 1
        
        # Parse message
        words = message.lower().split()
        
        # 1. Update average message length (running average)
        old_avg = data['avg_message_length']
        data['avg_message_length'] = (old_avg * (n - 1) + len(words)) / n
        
        # 2. Detect Hinglish
        if any(w in self.hinglish_words for w in words):
            data['uses_hinglish'] = True
        
        # 3. Detect emoji usage
        emoji_pattern = re.compile(r'[\U0001F300-\U0001FFFF]')
        if emoji_pattern.search(message):
            data['uses_emoji'] = True
        
        # 4. Track common words
        meaningful = [w for w in words if len(w) > 3 and w not in self.stopwords]
        
        all_common = data['common_words'] + meaningful
        word_counts = Counter(all_common)
        
        # Keep top N words
        data['common_words'] = [w for w, _ in word_counts.most_common(self.top_n_words)]
        
        # 5. Detect punctuation style
        punct_count = len(re.findall(r'[.,!?;:]', message))
        if punct_count == 0:
            data['punctuation_style'] = 'minimal'
        elif punct_count > len(message) * 0.2:
            data['punctuation_style'] = 'heavy'
        else:
            data['punctuation_style'] = 'normal'
        
        # 6. Update message count
        data['messages_sent'] = n
        
        # 7. Update adaptation stage based on message count
        if n >= 50:
            data['adaptation_stage'] = 2  # Full adaptation
        elif n >= 10:
            data['adaptation_stage'] = 1  # Light blending
        else:
            data['adaptation_stage'] = 0  # Default
        
        return profile


class SystemPromptBuilder:
    """
    Builds dynamic system prompts based on user profile
    Injects user style information for natural adaptation
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize prompt builder with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.bot_config = config['bot']
        self.adaptation_config = config['adaptation']
        
        self.bot_name = self.bot_config['name']
        self.base_personality = self.bot_config['personality'].strip()
    
    def build_prompt(self, profile: UserProfile) -> str:
        """
        Build system prompt based on user profile
        
        Args:
            profile: User profile
            
        Returns:
            Complete system prompt string
        """
        stage = profile.get('adaptation_stage', 0)
        
        # Base personality (always included)
        base = self.base_personality
        
        # Get style adaptation
        if stage == 0:
            style = "Speak naturally and casually. Match the energy of the person you're talking to."
        
        elif stage == 1:
            # Light blending - basic style hints
            hints = []
            
            if profile.get('uses_hinglish'):
                hints.append('occasionally mix Hindi/Hinglish naturally')
            
            avg_len = profile.get('avg_message_length', 0)
            if avg_len > 0 and avg_len < 8:
                hints.append('keep replies shorter, like this person')
            
            if profile.get('uses_emoji'):
                hints.append('use emoji occasionally')
            
            if hints:
                style = "Match the user's vibe: " + ', '.join(hints)
            else:
                style = "Speak naturally and casually."
        
        else:
            # Full adaptation - detailed style mirroring
            common_words = profile.get('common_words', [])[:10]
            words_str = ', '.join(common_words) if common_words else 'various words'
            
            hints = [
                f"Their average message is about {profile.get('avg_message_length', 10):.0f} words - match this energy",
                f"They {'use' if profile.get('uses_emoji') else 'do not use'} emoji",
                f"Their common words include: {words_str}",
            ]
            
            if profile.get('uses_hinglish'):
                hints.append("Yes, use Hinglish naturally - mix Hindi and English")
            
            style = "Mirror this person's exact style:\n- " + "\n- ".join(hints)
        
        return f"{base}\n\n{style}"
    
    def build_messages(self, profile: UserProfile, user_message: str) -> List[Dict]:
        """
        Build message list for inference
        
        Args:
            profile: User profile
            user_message: Current user message
            
        Returns:
            List of message dictionaries
        """
        system_prompt = self.build_prompt(profile)
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ]
        
        return messages


class UserProfileManager:
    """
    Manages user profiles in database
    Handles loading, saving, and updating profiles
    """
    
    def __init__(self, db_path: str = 'data/users.db'):
        """Initialize profile manager"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_profile(self, user_id: str) -> UserProfile:
        """
        Get user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProfile object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT profile_data FROM user_profiles WHERE user_id = ?',
            (user_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            profile_data = json.loads(result[0])
            return UserProfile(user_id, profile_data)
        
        # Create new profile
        return UserProfile(user_id)
    
    def save_profile(self, profile: UserProfile):
        """
        Save user profile
        
        Args:
            profile: UserProfile object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        profile_json = json.dumps(profile.to_dict())
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles (user_id, profile_data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (profile.data['user_id'], profile_json))
        
        conn.commit()
        conn.close()
    
    def delete_profile(self, user_id: str):
        """
        Delete user profile
        
        Args:
            user_id: User identifier
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_profiles WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()


def main():
    """Example usage"""
    # Initialize components
    extractor = StyleExtractor()
    prompt_builder = SystemPromptBuilder()
    profile_manager = UserProfileManager()
    
    # Simulate a conversation
    user_id = "test_user_123"
    
    # Get or create profile
    profile = profile_manager.get_profile(user_id)
    
    # Sample messages from user
    messages = [
        "bhai kya haal hain",
        "yaar bohot busy hua kal",
        "I got placed in Google!! 🎉",
        "bhai seriously?? that's amazing!!"
    ]
    
    print(f"\n{'='*50}")
    print("User Style Adaptation Demo")
    print(f"{'='*50}\n")
    
    for i, msg in enumerate(messages):
        print(f"\n--- Message {i+1}: '{msg}' ---")
        
        # Extract style features
        profile = extractor.extract_features(msg, profile)
        
        # Show profile updates
        print(f"Messages sent: {profile.get('messages_sent')}")
        print(f"Uses Hinglish: {profile.get('uses_hinglish')}")
        print(f"Uses Emoji: {profile.get('uses_emoji')}")
        print(f"Avg message length: {profile.get('avg_message_length'):.1f}")
        print(f"Adaptation stage: {profile.get('adaptation_stage')}")
        print(f"Common words: {profile.get('common_words', [])[:5]}")
        
        # Build system prompt
        system_prompt = prompt_builder.build_prompt(profile)
        print(f"\nSystem prompt:\n{system_prompt[:200]}...")
        
        # Save profile
        profile_manager.save_profile(profile)
    
    print(f"\n{'='*50}")
    print("Demo complete!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
