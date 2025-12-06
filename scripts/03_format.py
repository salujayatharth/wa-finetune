#!/usr/bin/env python3
"""
Training Data Formatter

Converts preprocessed conversations into ChatML format for fine-tuning.

Features:
- Sliding window context (3-7 previous messages)
- Dataset balancing by length, partner, and time
- Train/validation split by conversation

Usage:
    python 03_format.py --input data/processed/conversations.jsonl --output-dir data/training/
    python 03_format.py --input data/processed/conversations.jsonl --output-dir data/training/ --max-examples 15000
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Respond naturally in your conversational style."


class TrainingFormatter:
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        min_context: int = 1,
        max_context: int = 7,
        max_examples: int = 15000,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.system_prompt = system_prompt
        self.min_context = min_context
        self.max_context = max_context
        self.max_examples = max_examples
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)

        self.stats = {
            'total_conversations': 0,
            'total_examples_generated': 0,
            'examples_after_balancing': 0,
            'train_examples': 0,
            'val_examples': 0,
        }

    def load_conversations(self, input_path: str) -> list[dict]:
        """Load conversations from JSONL file."""
        conversations = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    conversations.append(json.loads(line))
        self.stats['total_conversations'] = len(conversations)
        return conversations

    def generate_examples(self, conversations: list[dict]) -> list[dict]:
        """
        Generate training examples from conversations.

        Each example has:
        - system message
        - context (previous messages as user/assistant)
        - target response (my message as assistant)
        """
        examples = []

        for conv in conversations:
            messages = conv['messages']
            chat_id = conv['chat_id']
            chat_name = conv['chat_name']

            # Find positions where I responded
            for i, msg in enumerate(messages):
                if msg['author'] != 'me':
                    continue

                # Need at least one previous message for context
                if i < 1:
                    continue

                # Determine context window size
                context_size = min(
                    random.randint(self.min_context, self.max_context),
                    i  # Can't have more context than available messages
                )

                # Build context messages
                context_start = i - context_size
                context_messages = messages[context_start:i]

                # Skip if no context from others (need someone to respond to)
                if not any(m['author'] != 'me' for m in context_messages):
                    continue

                # Build ChatML format
                chatml_messages = [
                    {"role": "system", "content": self.system_prompt}
                ]

                for ctx_msg in context_messages:
                    role = "assistant" if ctx_msg['author'] == 'me' else "user"
                    chatml_messages.append({
                        "role": role,
                        "content": ctx_msg['content']
                    })

                # Add my response as the target
                chatml_messages.append({
                    "role": "assistant",
                    "content": msg['content']
                })

                # Extract timestamp for temporal balancing
                timestamp = msg.get('timestamp', '')

                examples.append({
                    'messages': chatml_messages,
                    'metadata': {
                        'chat_id': chat_id,
                        'chat_name': chat_name,
                        'conversation_id': conv['conversation_id'],
                        'response_length': len(msg['content']),
                        'context_length': context_size,
                        'timestamp': timestamp
                    }
                })

        self.stats['total_examples_generated'] = len(examples)
        return examples

    def categorize_length(self, length: int) -> str:
        """Categorize message by length."""
        if length < 20:
            return 'short'
        elif length < 100:
            return 'medium'
        else:
            return 'long'

    def balance_examples(self, examples: list[dict]) -> list[dict]:
        """
        Balance dataset by:
        - Message length: 40% short, 40% medium, 20% long
        - Conversation partners
        - Time periods
        """
        if len(examples) <= self.max_examples:
            self.stats['examples_after_balancing'] = len(examples)
            return examples

        # Group by length category
        by_length = defaultdict(list)
        for ex in examples:
            cat = self.categorize_length(ex['metadata']['response_length'])
            by_length[cat].append(ex)

        # Target distribution
        targets = {
            'short': int(self.max_examples * 0.4),
            'medium': int(self.max_examples * 0.4),
            'long': int(self.max_examples * 0.2)
        }

        balanced = []

        for cat, target in targets.items():
            pool = by_length[cat]

            if len(pool) <= target:
                balanced.extend(pool)
            else:
                # Sample with partner diversity
                by_partner = defaultdict(list)
                for ex in pool:
                    by_partner[ex['metadata']['chat_id']].append(ex)

                # Try to get equal samples from each partner
                selected = []
                partners = list(by_partner.keys())
                random.shuffle(partners)

                per_partner = max(1, target // len(partners))

                for partner in partners:
                    partner_examples = by_partner[partner]
                    random.shuffle(partner_examples)
                    selected.extend(partner_examples[:per_partner])

                    if len(selected) >= target:
                        break

                # Fill remaining if needed
                if len(selected) < target:
                    remaining = [ex for ex in pool if ex not in selected]
                    random.shuffle(remaining)
                    selected.extend(remaining[:target - len(selected)])

                balanced.extend(selected[:target])

        random.shuffle(balanced)
        self.stats['examples_after_balancing'] = len(balanced)
        return balanced

    def split_train_val(self, examples: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Split into train/validation by conversation to prevent leakage.
        """
        # Group by conversation
        by_conv = defaultdict(list)
        for ex in examples:
            conv_key = (ex['metadata']['chat_id'], ex['metadata']['conversation_id'])
            by_conv[conv_key].append(ex)

        # Split conversations
        conv_keys = list(by_conv.keys())
        random.shuffle(conv_keys)

        val_size = int(len(conv_keys) * self.val_ratio)
        val_convs = set(conv_keys[:val_size])

        train = []
        val = []

        for conv_key, conv_examples in by_conv.items():
            if conv_key in val_convs:
                val.extend(conv_examples)
            else:
                train.extend(conv_examples)

        random.shuffle(train)
        random.shuffle(val)

        self.stats['train_examples'] = len(train)
        self.stats['val_examples'] = len(val)

        return train, val

    def save_examples(self, examples: list[dict], output_path: str) -> None:
        """Save examples to JSONL (only messages, not metadata)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                # Only save the messages, not metadata
                f.write(json.dumps({'messages': ex['messages']}, ensure_ascii=False) + '\n')

    def print_stats(self, train: list[dict], val: list[dict]) -> None:
        """Print formatting statistics."""
        print(f"\n{'='*50}")
        print("FORMATTING STATISTICS")
        print(f"{'='*50}")
        print(f"Input conversations: {self.stats['total_conversations']:,}")
        print(f"Total examples generated: {self.stats['total_examples_generated']:,}")
        print(f"Examples after balancing: {self.stats['examples_after_balancing']:,}")
        print(f"Training examples: {self.stats['train_examples']:,}")
        print(f"Validation examples: {self.stats['val_examples']:,}")

        # Length distribution in final dataset
        all_examples = train + val
        lengths = [ex['metadata']['response_length'] for ex in all_examples]
        short = sum(1 for l in lengths if l < 20)
        medium = sum(1 for l in lengths if 20 <= l < 100)
        long = sum(1 for l in lengths if l >= 100)
        total = len(lengths)

        print(f"\nResponse length distribution:")
        print(f"  Short (<20 chars): {short:,} ({100*short/total:.1f}%)")
        print(f"  Medium (20-100): {medium:,} ({100*medium/total:.1f}%)")
        print(f"  Long (100+): {long:,} ({100*long/total:.1f}%)")

        # Context length distribution
        context_lengths = [ex['metadata']['context_length'] for ex in all_examples]
        avg_context = sum(context_lengths) / len(context_lengths)
        print(f"\nAverage context length: {avg_context:.1f} messages")

        # Partner distribution
        partners = defaultdict(int)
        for ex in all_examples:
            partners[ex['metadata']['chat_name']] += 1

        print(f"\nTop 10 conversation partners:")
        for partner, count in sorted(partners.items(), key=lambda x: -x[1])[:10]:
            print(f"  {partner[:30]:30s}: {count:,} examples")

        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Format conversations for ChatML training'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/processed/conversations.jsonl',
        help='Input conversations JSONL file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='data/training/',
        help='Output directory for train/val files'
    )
    parser.add_argument(
        '--system-prompt', '-s',
        default=DEFAULT_SYSTEM_PROMPT,
        help='System prompt for training'
    )
    parser.add_argument(
        '--max-examples', '-m',
        type=int,
        default=15000,
        help='Maximum training examples (default: 15000)'
    )
    parser.add_argument(
        '--val-ratio', '-v',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--min-context',
        type=int,
        default=1,
        help='Minimum context messages (default: 1)'
    )
    parser.add_argument(
        '--max-context',
        type=int,
        default=7,
        help='Maximum context messages (default: 7)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    formatter = TrainingFormatter(
        system_prompt=args.system_prompt,
        min_context=args.min_context,
        max_context=args.max_context,
        max_examples=args.max_examples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print(f"Loading conversations from {input_path}...")
    conversations = formatter.load_conversations(str(input_path))

    print("Generating training examples...")
    examples = formatter.generate_examples(conversations)

    print("Balancing dataset...")
    examples = formatter.balance_examples(examples)

    print("Splitting train/validation...")
    train, val = formatter.split_train_val(examples)

    # Print statistics
    formatter.print_stats(train, val)

    # Save outputs
    output_dir = Path(args.output_dir)
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'

    formatter.save_examples(train, str(train_path))
    formatter.save_examples(val, str(val_path))

    print(f"Saved {len(train):,} training examples to {train_path}")
    print(f"Saved {len(val):,} validation examples to {val_path}")

    return 0


if __name__ == '__main__':
    exit(main())
