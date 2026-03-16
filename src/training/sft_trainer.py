"""
SFT (Supervised Fine-Tuning) Trainer
Layer 1: Teaches the model the personality and conversation style

This script fine-tunes Mistral-7B-Instruct using QLoRA to create
an emotionally intelligent AI personality.
"""

import os
import sys
import torch
import yaml
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb


def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """
    Load base model and tokenizer with QLoRA quantization
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model, tokenizer
    """
    print("=" * 50)
    print("Setting up model and tokenizer...")
    print("=" * 50)
    
    # Get model config
    model_config = config['model']
    quant_config = config['quantization']
    lora_config = config['lora']
    
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
    
    # Configure quantization (QLoRA) - only use if GPU available
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype'])
        )
    
    # Load base model
    print(f"\nLoading base model: {model_config['base_model']}")
    
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_config['base_model'],
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=model_config['trust_remote_code'],
            torch_dtype=getattr(torch, model_config['torch_dtype'])
        )
    else:
        # CPU mode - load in fp16
        print("Loading in CPU mode (fp16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_config['base_model'],
            device_map='cpu',
            trust_remote_code=model_config['trust_remote_code'],
            torch_dtype=torch.float16
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_model'],
        trust_remote_code=model_config['trust_remote_code']
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def setup_lora(config: dict):
    """
    Configure LoRA for fine-tuning
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LoRA configuration
    """
    lora_config = config['lora']
    
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        task_type=lora_config['task_type']
    )
    
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    print(f"  Target modules: {lora_config['target_modules']}")
    print(f"  Dropout: {lora_config['lora_dropout']}")
    
    return peft_config


def load_training_data(config: dict):
    """
    Load training and validation data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_dataset, val_dataset
    """
    sft_config = config['sft']
    
    train_path = sft_config['train_data_path']
    val_path = sft_config['val_data_path']
    
    print(f"\nLoading training data from: {train_path}")
    
    # Load training data
    if os.path.exists(train_path):
        dataset = load_dataset('json', data_files={'train': train_path})
        train_dataset = dataset['train']
    else:
        print(f"WARNING: Training data not found at {train_path}")
        print("Creating empty dataset...")
        train_dataset = load_dataset('json', data_files=[{
            'messages': [
                {'role': 'system', 'content': 'You are Nihar.'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }])['train']
    
    # Load validation data if exists
    val_dataset = None
    if os.path.exists(val_path):
        val_dataset = load_dataset('json', data_files={'validation': val_path})['validation']
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def setup_training_args(config: dict):
    """
    Configure training arguments
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SFTConfig
    """
    sft_config = config['sft']
    
    training_args = SFTConfig(
        output_dir=sft_config['output_dir'],
        num_train_epochs=sft_config['num_train_epochs'],
        per_device_train_batch_size=sft_config['per_device_train_batch_size'],
        gradient_accumulation_steps=sft_config['gradient_accumulation_steps'],
        learning_rate=sft_config['learning_rate'],
        warmup_ratio=sft_config['warmup_ratio'],
        lr_scheduler_type=sft_config['lr_scheduler_type'],
        save_steps=sft_config['save_steps'],
        logging_steps=sft_config['logging_steps'],
        bf16=sft_config['bf16'],
        report_to='none',
        run_name='emotai-sft'
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {sft_config['num_train_epochs']}")
    print(f"  Batch size: {sft_config['per_device_train_batch_size']}")
    print(f"  Gradient accumulation: {sft_config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {sft_config['learning_rate']}")
    print(f"  Max seq length: {sft_config['max_seq_length']}")
    
    return training_args


def train_sft(config_path: str = 'config/config.yaml'):
    """
    Main SFT training function
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize wandb (disabled)
    # wandb.init(project='emotai', name='sft-training', config=config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup LoRA
    peft_config = setup_lora(config)
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load data
    train_dataset, val_dataset = load_training_data(config)
    
    # Setup training args
    training_args = setup_training_args(config)
    
    # Create trainer
    print("\n" + "=" * 50)
    print("Starting SFT Training...")
    print("=" * 50)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=config['sft']['max_seq_length']
    )
    
    # Train
    trainer.train()
    
    # Save model
    output_dir = config['sft']['output_dir']
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ SFT training complete!")
    print(f"✓ Model saved to: {output_dir}")
    
    # wandb.finish()
    
    return model, tokenizer


def merge_and_export(config_path: str = 'config/config.yaml'):
    """
    Merge LoRA adapter with base model and export
    
    Args:
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    
    print("Loading model with LoRA adapter...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model, 
        config['sft']['output_dir']
    )
    
    # Merge
    print("Merging LoRA adapter with base model...")
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
    
    parser = argparse.ArgumentParser(description='SFT Training for EmotiAI')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--merge', action='store_true', help='Merge and export after training')
    
    args = parser.parse_args()
    
    if args.merge:
        merge_and_export(args.config)
    else:
        train_sft(args.config)
