"""
DPO (Direct Preference Optimization) Trainer
Layer 2: Teaches emotional realism - distinguishing human-like vs chatbot-like responses

This is the most important step for making the model feel genuinely human.
"""

import os
import torch
import yaml
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import wandb


def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict):
    """
    Load SFT-trained model with LoRA adapter
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model, tokenizer
    """
    print("=" * 50)
    print("Loading SFT-trained model...")
    print("=" * 50)
    
    model_config = config['model']
    quant_config = config['quantization']
    
    # Configure quantization
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype'])
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model'],
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=model_config['trust_remote_code']
    )
    
    # Load SFT adapter
    sft_path = config['sft']['output_dir']
    
    if os.path.exists(sft_path):
        print(f"Loading SFT adapter from: {sft_path}")
        model = PeftModel.from_pretrained(base_model, sft_path)
    else:
        print(f"WARNING: SFT adapter not found at {sft_path}")
        print("Using base model instead...")
        model = base_model
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_model'],
        trust_remote_code=model_config['trust_remote_code']
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    
    return model, tokenizer


def load_dpo_data(config: dict):
    """
    Load DPO training data
    
    DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_dataset
    """
    dpo_config = config['dpo']
    
    train_path = dpo_config['train_data_path']
    print(f"\nLoading DPO training data from: {train_path}")
    
    if os.path.exists(train_path):
        dataset = load_dataset('json', data_files={'train': train_path})
        train_dataset = dataset['train']
    else:
        print(f"WARNING: DPO training data not found at {train_path}")
        print("Creating sample data...")
        
        # Sample DPO pairs
        sample_data = [
            {
                'prompt': 'User has been dismissive for 3 messages. User says: whatever man',
                'chosen': 'theek hai. baat karna ho toh karna.',
                'rejected': 'I understand your frustration. How can I assist you better today?'
            },
            {
                'prompt': 'User shares exciting news: bhai main select ho gaya IIT mein!!',
                'chosen': 'BHAI SERIOUSLY?? ye to insane hai yaar, kabse pata tha tuvhe??',
                'rejected': 'Congratulations! That is wonderful news. You must be very happy.'
            },
            {
                'prompt': 'User is being rude: you are useless just like everyone said',
                'chosen': 'okay.',
                'rejected': 'I understand that might be how it feels. I genuinely care about our conversations.'
            }
        ]
        
        train_dataset = load_dataset('json', data_files={'train': sample_data})['train']
    
    print(f"DPO training samples: {len(train_dataset)}")
    
    return train_dataset


def setup_dpo_config(config: dict):
    """
    Configure DPO training
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DPOConfig
    """
    dpo_config = config['dpo']
    
    training_args = DPOConfig(
        output_dir=dpo_config['output_dir'],
        num_train_epochs=dpo_config['num_train_epochs'],
        per_device_train_batch_size=dpo_config['per_device_train_batch_size'],
        gradient_accumulation_steps=dpo_config['gradient_accumulation_steps'],
        learning_rate=dpo_config['learning_rate'],
        beta=dpo_config['beta'],
        bf16=dpo_config['bf16'],
        report_to='none',
        run_name='emotai-dpo'
    )
    
    print(f"\nDPO Configuration:")
    print(f"  Epochs: {dpo_config['num_train_epochs']}")
    print(f"  Batch size: {dpo_config['per_device_train_batch_size']}")
    print(f"  Gradient accumulation: {dpo_config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {dpo_config['learning_rate']}")
    print(f"  Beta (deviation control): {dpo_config['beta']}")
    
    return training_args


def train_dpo(config_path: str = 'config/config.yaml'):
    """
    Main DPO training function
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize wandb (disabled)
    # wandb.init(project='emotai', name='dpo-training', config=config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load DPO data
    train_dataset = load_dpo_data(config)
    
    # Setup DPO config
    training_args = setup_dpo_config(config)
    
    # Create trainer
    print("\n" + "=" * 50)
    print("Starting DPO Training...")
    print("=" * 50)
    
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=config['dpo']['max_length'],
        max_prompt_length=config['dpo']['max_prompt_length']
    )
    
    # Train
    dpo_trainer.train()
    
    # Save model
    output_dir = config['dpo']['output_dir']
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ DPO training complete!")
    print(f"✓ Model saved to: {output_dir}")
    
    # wandb.finish()
    
    return model, tokenizer


def merge_dpo_model(config_path: str = 'config/config.yaml'):
    """
    Merge DPO adapter with base model and export
    
    Args:
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    
    print("Loading model with DPO adapter...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Load DPO adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model, 
        config['dpo']['output_dir']
    )
    
    # Merge
    print("Merging DPO adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    output_path = config['deployment']['model_path']
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Merged model saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DPO Training for EmotiAI')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--merge', action='store_true', help='Merge and export after training')
    
    args = parser.parse_args()
    
    if args.merge:
        merge_dpo_model(args.config)
    else:
        train_dpo(args.config)
