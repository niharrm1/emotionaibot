"""
FastAPI Deployment Backend
Serves the emotionally intelligent AI model via REST API
"""

import os
import yaml
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Import adapters
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.adapters.style_adapter import UserProfileManager, StyleExtractor, SystemPromptBuilder


# Load configuration
def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()


# Initialize FastAPI app
app = FastAPI(
    title="EmotiAI API",
    description="Emotionally Intelligent Personal AI Model API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['deployment']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_history: Optional[List[Dict]] = None


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    bot_mood: Optional[float] = 0.0
    adaptation_stage: int = 0


class ProfileResponse(BaseModel):
    user_id: str
    messages_sent: int
    uses_hinglish: bool
    uses_emoji: bool
    avg_message_length: float
    adaptation_stage: int


# Global variables
pipe = None
profile_manager = None
style_extractor = None
prompt_builder = None


def initialize_model():
    """Initialize the model and adapters"""
    global pipe, profile_manager, style_extractor, prompt_builder
    
    model_path = config['deployment']['model_path']
    
    print("="*50)
    print("Initializing EmotiAI Model...")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}")
        print("Using base model for testing...")
        
        # Use base model if fine-tuned model not available
        model_path = config['model']['base_model']
    
    # Load model and tokenizer
    print(f"Loading model from: {model_path}")
    
    try:
        pipe = pipeline(
            'text-generation',
            model=model_path,
            tokenizer=model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting in demo mode...")
        pipe = None
    
    # Initialize adapters
    db_path = config['deployment']['database']['path']
    profile_manager = UserProfileManager(db_path)
    style_extractor = StyleExtractor()
    prompt_builder = SystemPromptBuilder()
    
    print("Initialization complete!")


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    initialize_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EmotiAI API",
        "version": "1.0.0",
        "status": "running",
        "description": "Emotionally Intelligent Personal AI Model"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Processes user message and returns AI response with emotional intelligence
    """
    try:
        # Get user profile
        profile = profile_manager.get_profile(request.user_id)
        
        # Extract style features from message
        profile = style_extractor.extract_features(request.message, profile)
        
        # Build system prompt
        system_prompt = prompt_builder.build_prompt(profile)
        
        # Prepare messages for model
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        
        # Add conversation history if provided
        if request.conversation_history:
            for msg in request.conversation_history[-10:]:  # Last 10 messages
                messages.append(msg)
        
        # Add current message
        messages.append({'role': 'user', 'content': request.message})
        
        # Generate response
        if pipe is not None:
            # Get inference config
            inference_config = config['inference']
            
            response = pipe(
                messages,
                max_new_tokens=inference_config['max_new_tokens'],
                temperature=inference_config['temperature'],
                top_p=inference_config['top_p'],
                top_k=inference_config['top_k'],
                do_sample=inference_config['do_sample']
            )
            
            # Extract reply
            generated_text = response[0]['generated_text']
            reply = generated_text[-1]['content']
            
        else:
            # Demo mode - return mock response
            reply = _generate_demo_response(request.message, profile)
        
        # Update profile
        profile_manager.save_profile(profile)
        
        return ChatResponse(
            user_id=request.user_id,
            reply=reply,
            bot_mood=profile.get('last_bot_mood', 0.0),
            adaptation_stage=profile.get('adaptation_stage', 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get user profile"""
    try:
        profile = profile_manager.get_profile(user_id)
        
        return ProfileResponse(
            user_id=user_id,
            messages_sent=profile.get('messages_sent', 0),
            uses_hinglish=profile.get('uses_hinglish', False),
            uses_emoji=profile.get('uses_emoji', False),
            avg_message_length=profile.get('avg_message_length', 0.0),
            adaptation_stage=profile.get('adaptation_stage', 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/profile/{user_id}")
async def delete_profile(user_id: str):
    """Delete user profile"""
    try:
        profile_manager.delete_profile(user_id)
        
        return {"status": "deleted", "user_id": user_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{user_id}")
async def reset_conversation(user_id: str):
    """Reset conversation for a user"""
    try:
        profile = profile_manager.get_profile(user_id)
        
        # Clear conversation history but keep style profile
        profile.data['conversation_history'] = []
        
        profile_manager.save_profile(profile)
        
        return {"status": "reset", "user_id": user_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_demo_response(message: str, profile) -> str:
    """
    Generate demo response when model is not loaded
    """
    message_lower = message.lower()
    
    # Simple rule-based responses for demo
    if any(w in message_lower for w in ['hello', 'hi', 'hey']):
        return "Hey! Kya haal hai?"
    
    elif any(w in message_lower for w in ['good', 'great', 'amazing']):
        return "Really? That's awesome! Tell me more!"
    
    elif any(w in message_lower for w in ['sad', 'bad', 'down']):
        return "Yaar kya hua? Tell me, I'm here."
    
    elif any(w in message_lower for w in ['?', 'what', 'how']):
        return "Hmm, tell me more about that."
    
    elif any(w in message_lower for w in ['bye', 'gtg', 'later']):
        return "Chal bye, phir baat karte hain!"
    
    else:
        return "Okay tell me more..."


if __name__ == "__main__":
    import uvicorn
    
    host = config['deployment']['host']
    port = config['deployment']['port']
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False
    )
