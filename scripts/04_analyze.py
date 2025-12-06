#!/usr/bin/env python3
"""
Dataset Analysis Script

Analyzes the training dataset and outputs statistics, distributions, and visualizations.

Features:
- Message length distribution
- Context length distribution
- Vocabulary analysis
- Emoji usage frequency
- Conversation partner distribution
- Temporal distribution

Usage:
    python 04_analyze.py --train data/training/train.jsonl --val data/training/val.jsonl
    python 04_analyze.py --train data/training/train.jsonl --val data/training/val.jsonl --output-dir analysis/
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Emoji pattern (covers most common emoji ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE
)


class DatasetAnalyzer:
    def __init__(self):
        self.train_examples = []
        self.val_examples = []
        self.stats = {}

    def load_examples(self, train_path: str, val_path: str = None) -> None:
        """Load training and validation examples."""
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.train_examples.append(json.loads(line))

        if val_path and Path(val_path).exists():
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.val_examples.append(json.loads(line))

    def extract_my_responses(self, examples: list[dict]) -> list[str]:
        """Extract all assistant (my) responses from examples."""
        responses = []
        for ex in examples:
            messages = ex.get('messages', [])
            # Last assistant message is the target response
            for msg in reversed(messages):
                if msg['role'] == 'assistant':
                    responses.append(msg['content'])
                    break
        return responses

    def extract_all_text(self, examples: list[dict]) -> list[str]:
        """Extract all text from user and assistant messages."""
        texts = []
        for ex in examples:
            for msg in ex.get('messages', []):
                if msg['role'] in ('user', 'assistant'):
                    texts.append(msg['content'])
        return texts

    def analyze_lengths(self, responses: list[str]) -> dict:
        """Analyze message length distribution."""
        lengths = [len(r) for r in responses]

        if not lengths:
            return {}

        # Buckets for histogram
        buckets = {
            '0-10': 0,
            '11-20': 0,
            '21-50': 0,
            '51-100': 0,
            '101-200': 0,
            '201-500': 0,
            '500+': 0
        }

        for l in lengths:
            if l <= 10:
                buckets['0-10'] += 1
            elif l <= 20:
                buckets['11-20'] += 1
            elif l <= 50:
                buckets['21-50'] += 1
            elif l <= 100:
                buckets['51-100'] += 1
            elif l <= 200:
                buckets['101-200'] += 1
            elif l <= 500:
                buckets['201-500'] += 1
            else:
                buckets['500+'] += 1

        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': sum(lengths) / len(lengths),
            'median': sorted(lengths)[len(lengths) // 2],
            'distribution': buckets
        }

    def analyze_context_lengths(self, examples: list[dict]) -> dict:
        """Analyze context length (number of messages before target)."""
        context_lengths = []

        for ex in examples:
            messages = ex.get('messages', [])
            # Count non-system messages before the last assistant message
            count = 0
            for msg in messages:
                if msg['role'] == 'system':
                    continue
                count += 1
            # Subtract 1 for the target response itself
            context_lengths.append(max(0, count - 1))

        if not context_lengths:
            return {}

        # Distribution
        distribution = Counter(context_lengths)

        return {
            'min': min(context_lengths),
            'max': max(context_lengths),
            'mean': sum(context_lengths) / len(context_lengths),
            'distribution': dict(sorted(distribution.items()))
        }

    def analyze_vocabulary(self, texts: list[str], top_n: int = 100) -> dict:
        """Analyze word frequency."""
        words = []
        for text in texts:
            # Simple tokenization: lowercase, split on whitespace/punctuation
            text_words = re.findall(r'\b\w+\b', text.lower())
            words.extend(text_words)

        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)

        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_richness': unique_words / total_words if total_words > 0 else 0,
            'top_words': dict(word_counts.most_common(top_n))
        }

    def analyze_emojis(self, texts: list[str], top_n: int = 50) -> dict:
        """Analyze emoji usage."""
        all_emojis = []

        for text in texts:
            emojis = EMOJI_PATTERN.findall(text)
            all_emojis.extend(emojis)

        emoji_counts = Counter(all_emojis)
        texts_with_emoji = sum(1 for t in texts if EMOJI_PATTERN.search(t))

        return {
            'total_emoji_instances': len(all_emojis),
            'unique_emojis': len(emoji_counts),
            'texts_with_emoji': texts_with_emoji,
            'emoji_rate': texts_with_emoji / len(texts) if texts else 0,
            'top_emojis': dict(emoji_counts.most_common(top_n))
        }

    def analyze_punctuation(self, responses: list[str]) -> dict:
        """Analyze punctuation patterns."""
        patterns = {
            'ends_with_period': 0,
            'ends_with_question': 0,
            'ends_with_exclamation': 0,
            'ends_with_emoji': 0,
            'no_punctuation': 0,
            'all_lowercase': 0,
            'has_ellipsis': 0,
        }

        for r in responses:
            r = r.strip()
            if not r:
                continue

            if r.endswith('.'):
                patterns['ends_with_period'] += 1
            elif r.endswith('?'):
                patterns['ends_with_question'] += 1
            elif r.endswith('!'):
                patterns['ends_with_exclamation'] += 1
            elif EMOJI_PATTERN.search(r[-2:] if len(r) >= 2 else r):
                patterns['ends_with_emoji'] += 1
            else:
                patterns['no_punctuation'] += 1

            if r == r.lower():
                patterns['all_lowercase'] += 1

            if '...' in r or '…' in r:
                patterns['has_ellipsis'] += 1

        total = len(responses)
        return {k: {'count': v, 'percentage': 100 * v / total if total else 0}
                for k, v in patterns.items()}

    def print_histogram(self, distribution: dict, title: str, width: int = 40) -> None:
        """Print a simple text histogram."""
        if not distribution:
            return

        max_val = max(distribution.values())
        print(f"\n{title}")
        print("-" * (width + 20))

        for key, value in distribution.items():
            bar_len = int(width * value / max_val) if max_val > 0 else 0
            bar = "█" * bar_len
            print(f"{str(key):>10s} | {bar} {value:,}")

    def generate_report(self) -> str:
        """Generate a full analysis report."""
        all_examples = self.train_examples + self.val_examples
        responses = self.extract_my_responses(all_examples)
        all_text = self.extract_all_text(all_examples)

        report = []
        report.append("=" * 60)
        report.append("DATASET ANALYSIS REPORT")
        report.append("=" * 60)

        # Basic counts
        report.append(f"\n## Dataset Size")
        report.append(f"Training examples: {len(self.train_examples):,}")
        report.append(f"Validation examples: {len(self.val_examples):,}")
        report.append(f"Total examples: {len(all_examples):,}")

        # Response length analysis
        length_stats = self.analyze_lengths(responses)
        if length_stats:
            report.append(f"\n## Response Length Statistics")
            report.append(f"Min: {length_stats['min']} chars")
            report.append(f"Max: {length_stats['max']} chars")
            report.append(f"Mean: {length_stats['mean']:.1f} chars")
            report.append(f"Median: {length_stats['median']} chars")
            report.append(f"\nLength Distribution:")
            for bucket, count in length_stats['distribution'].items():
                pct = 100 * count / len(responses)
                report.append(f"  {bucket:>10s}: {count:>6,} ({pct:>5.1f}%)")

        # Context length analysis
        context_stats = self.analyze_context_lengths(all_examples)
        if context_stats:
            report.append(f"\n## Context Length Statistics")
            report.append(f"Min: {context_stats['min']} messages")
            report.append(f"Max: {context_stats['max']} messages")
            report.append(f"Mean: {context_stats['mean']:.1f} messages")
            report.append(f"\nContext Length Distribution:")
            for length, count in list(context_stats['distribution'].items())[:10]:
                pct = 100 * count / len(all_examples)
                report.append(f"  {length} messages: {count:>6,} ({pct:>5.1f}%)")

        # Vocabulary analysis
        vocab_stats = self.analyze_vocabulary(all_text)
        if vocab_stats:
            report.append(f"\n## Vocabulary Statistics")
            report.append(f"Total words: {vocab_stats['total_words']:,}")
            report.append(f"Unique words: {vocab_stats['unique_words']:,}")
            report.append(f"Vocabulary richness: {vocab_stats['vocabulary_richness']:.4f}")
            report.append(f"\nTop 30 Words:")
            for i, (word, count) in enumerate(list(vocab_stats['top_words'].items())[:30]):
                report.append(f"  {i+1:>2}. {word:>15s}: {count:>6,}")

        # Emoji analysis
        emoji_stats = self.analyze_emojis(all_text)
        if emoji_stats:
            report.append(f"\n## Emoji Statistics")
            report.append(f"Total emoji instances: {emoji_stats['total_emoji_instances']:,}")
            report.append(f"Unique emojis: {emoji_stats['unique_emojis']:,}")
            report.append(f"Messages with emoji: {emoji_stats['texts_with_emoji']:,} ({100*emoji_stats['emoji_rate']:.1f}%)")
            if emoji_stats['top_emojis']:
                report.append(f"\nTop 20 Emojis:")
                for i, (emoji, count) in enumerate(list(emoji_stats['top_emojis'].items())[:20]):
                    report.append(f"  {i+1:>2}. {emoji}: {count:>6,}")

        # Punctuation analysis
        punct_stats = self.analyze_punctuation(responses)
        if punct_stats:
            report.append(f"\n## Punctuation Patterns (in responses)")
            for pattern, data in punct_stats.items():
                report.append(f"  {pattern:>20s}: {data['count']:>6,} ({data['percentage']:>5.1f}%)")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def save_report(self, output_path: str, report: str) -> None:
        """Save report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

    def save_stats_json(self, output_path: str) -> None:
        """Save detailed stats as JSON for further analysis."""
        all_examples = self.train_examples + self.val_examples
        responses = self.extract_my_responses(all_examples)
        all_text = self.extract_all_text(all_examples)

        stats = {
            'dataset_size': {
                'train': len(self.train_examples),
                'val': len(self.val_examples),
                'total': len(all_examples)
            },
            'response_lengths': self.analyze_lengths(responses),
            'context_lengths': self.analyze_context_lengths(all_examples),
            'vocabulary': self.analyze_vocabulary(all_text, top_n=500),
            'emojis': self.analyze_emojis(all_text),
            'punctuation': self.analyze_punctuation(responses)
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training dataset statistics'
    )
    parser.add_argument(
        '--train', '-t',
        default='data/training/train.jsonl',
        help='Training data JSONL file'
    )
    parser.add_argument(
        '--val', '-v',
        default='data/training/val.jsonl',
        help='Validation data JSONL file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for reports (optional)'
    )

    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"Error: Training file not found: {train_path}")
        return 1

    val_path = Path(args.val) if args.val else None

    analyzer = DatasetAnalyzer()

    print(f"Loading data from {train_path}...")
    analyzer.load_examples(str(train_path), str(val_path) if val_path else None)

    print("Analyzing dataset...")
    report = analyzer.generate_report()

    # Print report
    print(report)

    # Save if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        report_path = output_dir / 'analysis_report.txt'
        stats_path = output_dir / 'analysis_stats.json'

        analyzer.save_report(str(report_path), report)
        analyzer.save_stats_json(str(stats_path))

        print(f"\nReport saved to {report_path}")
        print(f"Stats saved to {stats_path}")

    return 0


if __name__ == '__main__':
    exit(main())
