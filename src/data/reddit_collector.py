"""
Reddit Data Collector
Collects emotionally rich conversations from Reddit subreddits for training
"""

import praw
import json
import os
from datetime import datetime
from typing import List, Dict
import time


class RedditCollector:
    """Collects emotional conversations from Reddit"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit API client
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Subreddits known for emotional content
        self.emotional_subreddits = [
            'relationship_advice',
            'AITA', 
            'offmychest',
            'rant',
            'relationships',
            'TrueReddit',
            'entitledparents',
            'maliciouscompliance'
        ]
        
    def collect_subreddit_posts(
        self, 
        subreddit_name: str, 
        limit: int = 100,
        time_filter: str = 'month'
    ) -> List[Dict]:
        """Collect posts from a specific subreddit"""
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        
        try:
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'subreddit': subreddit_name,
                    'url': post.url,
                    'is_self': post.is_self
                }
                posts.append(post_data)
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error collecting from r/{subreddit_name}: {e}")
            
        return posts
    
    def get_post_comments(self, post_id: str) -> List[Dict]:
        """Get all comments from a post"""
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comment_sort = 'best'
            submission.comments.replace_more(limit=0)
            
            comments = []
            for comment in submission.comments:
                comment_data = {
                    'id': comment.id,
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'parent_id': comment.parent_id,
                    'is_submitter': comment.is_submitter
                }
                comments.append(comment_data)
                
            return comments
            
        except Exception as e:
            print(f"Error getting comments for post {post_id}: {e}")
            return []
    
    def collect_conversations(
        self, 
        posts_per_subreddit: int = 50,
        min_comments: int = 3
    ) -> List[Dict]:
        """Collect full conversations from multiple subreddits"""
        all_conversations = []
        
        for subreddit in self.emotional_subreddits:
            print(f"Collecting from r/{subreddit}...")
            
            posts = self.collect_subreddit_posts(
                subreddit, 
                limit=posts_per_subreddit
            )
            
            for post in posts:
                if post['num_comments'] >= min_comments:
                    comments = self.get_post_comments(post['id'])
                    
                    if len(comments) >= min_comments:
                        conversation = {
                            'source': 'reddit',
                            'subreddit': subreddit,
                            'post_title': post['title'],
                            'post_body': post['selftext'],
                            'timestamp': post['created_utc'],
                            'comments': comments
                        }
                        all_conversations.append(conversation)
                        
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
    import dotenv
    dotenv.load_dotenv()
    
    collector = RedditCollector(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'emotai/1.0')
    )
    
    conversations = collector.collect_conversations(
        posts_per_subreddit=50,
        min_comments=3
    )
    
    collector.save_to_jsonl(
        conversations, 
        'data/raw/reddit_conversations.jsonl'
    )


if __name__ == '__main__':
    main()
