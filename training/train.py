#!/usr/bin/env python3
"""
Training Script for WhatsApp Style Transfer

Uses Unsloth for efficient QLoRA fine-tuning on Llama 3.1 8B.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --epochs 2 --batch-size 2
"""

import argparse
import os
from pathlib import Path

import yaml


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    defaults = {
        # Model
        'base_model': 'unsloth/Meta-Llama-3.1-8B-bnb-4bit',
        'max_seq_length': 2048,
        'load_in_4bit': True,

        # LoRA
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        'target_modules': [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],

        # Training
        'num_train_epochs': 1,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-4,
        'lr_scheduler_type': 'constant',
        'warmup_ratio': 0.1,
        'bf16': True,
        'gradient_checkpointing': True,

        # Logging
        'logging_steps': 10,
        'save_steps': 500,
        'eval_steps': 500,
        'output_dir': './outputs',

        # Data
        'train_data': 'data/training/train.jsonl',
        'val_data': 'data/training/val.jsonl',

        # Export
        'export_gguf': True,
        'gguf_quantization': 'q5_k_m',
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            defaults.update(user_config)

    return defaults


def main():
    parser = argparse.ArgumentParser(description='Train WhatsApp style model')
    parser.add_argument('--config', '-c', default='training/config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, help='Override num_train_epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--output-dir', help='Override output directory')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--no-gguf', action='store_true', help='Skip GGUF export')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command line overrides
    if args.epochs:
        config['num_train_epochs'] = args.epochs
    if args.batch_size:
        config['per_device_train_batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.no_gguf:
        config['export_gguf'] = False

    print("=" * 60)
    print("WHATSAPP STYLE TRANSFER TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base model: {config['base_model']}")
    print(f"  LoRA rank: {config['lora_r']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch size: {config['per_device_train_batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Output dir: {config['output_dir']}")

    # Import here to avoid slow startup when just checking help
    print("\nLoading libraries...")
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load model
    print(f"\nLoading model: {config['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['base_model'],
        max_seq_length=config['max_seq_length'],
        load_in_4bit=config['load_in_4bit'],
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['target_modules'],
        use_gradient_checkpointing="unsloth" if config['gradient_checkpointing'] else False,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load dataset
    print(f"\nLoading dataset from {config['train_data']}...")
    dataset = load_dataset("json", data_files={
        "train": config['train_data'],
        "validation": config['val_data']
    })

    print(f"  Training examples: {len(dataset['train']):,}")
    print(f"  Validation examples: {len(dataset['validation']):,}")

    # Set Llama 3.1 chat template if not already set
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"""

    # Format function for chat template - returns string per example
    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_ratio=config['warmup_ratio'],
        bf16=config['bf16'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb" if args.wandb else "none",
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=format_chat,
        max_seq_length=config['max_seq_length'],
        args=training_args,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    trainer.train()

    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    lora_path = os.path.join(config['output_dir'], 'lora_adapter')
    print(f"\nSaving LoRA adapter to {lora_path}...")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Export to GGUF for Ollama
    if config['export_gguf']:
        gguf_path = os.path.join(config['output_dir'], 'gguf')
        print(f"\nExporting to GGUF format at {gguf_path}...")
        print(f"Quantization method: {config['gguf_quantization']}")

        try:
            model.save_pretrained_gguf(
                gguf_path,
                tokenizer,
                quantization_method=config['gguf_quantization']
            )
            print(f"GGUF export complete!")
        except Exception as e:
            print(f"Warning: GGUF export failed: {e}")
            print("You can manually convert using llama.cpp later.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {config['output_dir']}")
    print(f"  - LoRA adapter: {lora_path}")
    if config['export_gguf']:
        print(f"  - GGUF model: {gguf_path}")

    print("\nNext steps:")
    print("1. Download the GGUF file to your local machine")
    print("2. Create Ollama model: ollama create myname-style -f Modelfile")
    print("3. Test: ollama run myname-style 'hey whats up'")

    return 0


if __name__ == '__main__':
    exit(main())
