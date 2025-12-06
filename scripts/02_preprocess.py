#!/usr/bin/env python3
"""
WhatsApp Message Preprocessing Pipeline

Cleans, deduplicates, and segments messages into conversations.

Features:
- Exact and fuzzy deduplication
- Conversation segmentation by time gaps
- Quality filtering
- Optional PII redaction

Usage:
    python 02_preprocess.py --input data/processed/raw_messages.jsonl --output data/processed/conversations.jsonl
    python 02_preprocess.py --input data/processed/raw_messages.jsonl --output data/processed/conversations.jsonl --redact
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False
        print("Warning: fuzzy matching disabled. Install rapidfuzz: pip install rapidfuzz")


# PII patterns for redaction
PII_PATTERNS = {
    'phone': re.compile(r'\+?\d{10,15}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'),
    'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'address': re.compile(
        r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct)\b',
        re.IGNORECASE
    ),
    'ssn': re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
    'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
}

# Common short responses that should be kept despite being <2 chars
ALLOWED_SHORT_RESPONSES = {'k', 'ok', 'ya', 'no', 'hi', 'yo', 'ty', 'np', 'gn', 'gm', 'lol', 'hm', 'ah', 'oh'}

# URL pattern for filtering
URL_PATTERN = re.compile(r'^https?://\S+$|^www\.\S+$')


class MessagePreprocessor:
    def __init__(self, conversation_gap_hours: float = 4.0, fuzzy_threshold: float = 0.95):
        self.conversation_gap_ms = conversation_gap_hours * 60 * 60 * 1000
        self.fuzzy_threshold = fuzzy_threshold
        self.stats = {
            'total_input': 0,
            'duplicates_removed': 0,
            'fuzzy_duplicates_removed': 0,
            'short_removed': 0,
            'long_removed': 0,
            'url_only_removed': 0,
            'total_output': 0,
            'conversations_created': 0,
        }
        self.redaction_log = []

    def load_messages(self, input_path: str) -> list[dict]:
        """Load messages from JSONL file."""
        messages = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        self.stats['total_input'] = len(messages)
        return messages

    def remove_exact_duplicates(self, messages: list[dict]) -> list[dict]:
        """Remove exact duplicate messages."""
        seen = set()
        unique = []

        for msg in messages:
            # Create a key from chat_id + message content + approximate time
            key = (msg['chat_id'], msg['message'], msg['timestamp'] // 60000)  # 1-minute buckets
            if key not in seen:
                seen.add(key)
                unique.append(msg)

        self.stats['duplicates_removed'] = len(messages) - len(unique)
        return unique

    def remove_fuzzy_duplicates(self, messages: list[dict]) -> list[dict]:
        """Remove near-duplicate messages within 60-second windows."""
        if not FUZZY_AVAILABLE:
            return messages

        # Group by chat
        by_chat = defaultdict(list)
        for msg in messages:
            by_chat[msg['chat_id']].append(msg)

        cleaned = []
        removed = 0

        for chat_id, chat_messages in by_chat.items():
            # Sort by timestamp
            chat_messages.sort(key=lambda x: x['timestamp'])

            keep = []
            for msg in chat_messages:
                is_duplicate = False

                # Check against recent messages (within 60 seconds)
                for prev in reversed(keep[-10:]):  # Only check last 10
                    time_diff = msg['timestamp'] - prev['timestamp']
                    if time_diff > 60000:  # More than 60 seconds ago
                        break

                    # Same author check
                    if msg['from_me'] == prev['from_me']:
                        ratio = fuzz.ratio(msg['message'], prev['message']) / 100.0
                        if ratio > self.fuzzy_threshold:
                            is_duplicate = True
                            removed += 1
                            break

                if not is_duplicate:
                    keep.append(msg)

            cleaned.extend(keep)

        self.stats['fuzzy_duplicates_removed'] = removed
        return cleaned

    def filter_quality(self, messages: list[dict]) -> list[dict]:
        """Apply quality filters to messages."""
        filtered = []

        for msg in messages:
            text = msg['message'].strip()

            # Remove very short messages (except allowed ones)
            if len(text) < 2 and text.lower() not in ALLOWED_SHORT_RESPONSES:
                self.stats['short_removed'] += 1
                continue

            # Remove very long messages (likely forwards/copypasta)
            if len(text) > 500:
                self.stats['long_removed'] += 1
                continue

            # Remove URL-only messages
            if URL_PATTERN.match(text):
                self.stats['url_only_removed'] += 1
                continue

            filtered.append(msg)

        return filtered

    def redact_pii(self, messages: list[dict]) -> list[dict]:
        """Redact PII from messages."""
        redacted = []

        for msg in messages:
            text = msg['message']
            original_text = text

            for pii_type, pattern in PII_PATTERNS.items():
                matches = pattern.findall(text)
                if matches:
                    for match in matches:
                        self.redaction_log.append({
                            'chat_id': msg['chat_id'],
                            'type': pii_type,
                            'original': match,
                            'timestamp': msg.get('timestamp_iso', '')
                        })
                    text = pattern.sub(f'[{pii_type.upper()}_REDACTED]', text)

            msg_copy = msg.copy()
            msg_copy['message'] = text
            if text != original_text:
                msg_copy['redacted'] = True
            redacted.append(msg_copy)

        return redacted

    def segment_conversations(self, messages: list[dict]) -> list[dict]:
        """Segment messages into conversations based on time gaps."""
        # Group by chat
        by_chat = defaultdict(list)
        for msg in messages:
            by_chat[msg['chat_id']].append(msg)

        conversations = []

        for chat_id, chat_messages in by_chat.items():
            # Sort by timestamp
            chat_messages.sort(key=lambda x: x['timestamp'])

            if not chat_messages:
                continue

            chat_name = chat_messages[0].get('chat_name', f'Chat_{chat_id}')

            # Segment into conversations
            current_conv = []
            conv_id = 0

            for msg in chat_messages:
                if current_conv:
                    time_diff = msg['timestamp'] - current_conv[-1]['timestamp']
                    if time_diff > self.conversation_gap_ms:
                        # Save current conversation
                        if len(current_conv) >= 2:  # At least 2 messages
                            conversations.append({
                                'chat_id': chat_id,
                                'chat_name': chat_name,
                                'conversation_id': conv_id,
                                'messages': [
                                    {
                                        'author': 'me' if m['from_me'] else 'other',
                                        'content': m['message'],
                                        'timestamp': m.get('timestamp_iso', '')
                                    }
                                    for m in current_conv
                                ]
                            })
                            conv_id += 1
                        current_conv = []

                current_conv.append(msg)

            # Don't forget the last conversation
            if len(current_conv) >= 2:
                conversations.append({
                    'chat_id': chat_id,
                    'chat_name': chat_name,
                    'conversation_id': conv_id,
                    'messages': [
                        {
                            'author': 'me' if m['from_me'] else 'other',
                            'content': m['message'],
                            'timestamp': m.get('timestamp_iso', '')
                        }
                        for m in current_conv
                    ]
                })

        self.stats['conversations_created'] = len(conversations)
        return conversations

    def save_conversations(self, conversations: list[dict], output_path: str) -> None:
        """Save conversations to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_messages = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                total_messages += len(conv['messages'])
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        self.stats['total_output'] = total_messages

    def save_redaction_log(self, output_path: str) -> None:
        """Save redaction log for manual review."""
        if not self.redaction_log:
            return

        log_path = Path(output_path).parent / 'redaction_log.jsonl'
        with open(log_path, 'w', encoding='utf-8') as f:
            for entry in self.redaction_log:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Redaction log saved to {log_path} ({len(self.redaction_log)} items)")

    def print_stats(self, conversations: list[dict]) -> None:
        """Print preprocessing statistics."""
        print(f"\n{'='*50}")
        print("PREPROCESSING STATISTICS")
        print(f"{'='*50}")
        print(f"Input messages: {self.stats['total_input']:,}")
        print(f"Exact duplicates removed: {self.stats['duplicates_removed']:,}")
        print(f"Fuzzy duplicates removed: {self.stats['fuzzy_duplicates_removed']:,}")
        print(f"Short messages removed: {self.stats['short_removed']:,}")
        print(f"Long messages removed: {self.stats['long_removed']:,}")
        print(f"URL-only messages removed: {self.stats['url_only_removed']:,}")
        print(f"Output messages: {self.stats['total_output']:,}")
        print(f"Conversations created: {self.stats['conversations_created']:,}")

        if conversations:
            # Message per conversation distribution
            msgs_per_conv = [len(c['messages']) for c in conversations]
            avg_msgs = sum(msgs_per_conv) / len(msgs_per_conv)
            print(f"Average messages per conversation: {avg_msgs:.1f}")

            # Chat distribution
            chats = Counter(c['chat_id'] for c in conversations)
            print(f"Unique chats: {len(chats):,}")

            # Top 5 chats by conversation count
            print("\nTop 5 chats by conversations:")
            for chat_id, count in chats.most_common(5):
                chat_name = next(
                    (c['chat_name'] for c in conversations if c['chat_id'] == chat_id),
                    chat_id
                )
                print(f"  {chat_name[:30]:30s}: {count:,} conversations")

            # Most common messages (check for spam/noise)
            all_messages = [m['content'] for c in conversations for m in c['messages']]
            msg_counts = Counter(all_messages)
            print("\nTop 20 most common messages (check for spam):")
            for msg, count in msg_counts.most_common(20):
                display_msg = msg[:40] + '...' if len(msg) > 40 else msg
                print(f"  {count:5,}x: {display_msg}")

        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess WhatsApp messages for training'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/processed/raw_messages.jsonl',
        help='Input JSONL file from extraction'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/processed/conversations.jsonl',
        help='Output JSONL file with conversations'
    )
    parser.add_argument(
        '--gap-hours', '-g',
        type=float,
        default=4.0,
        help='Hours gap to split conversations (default: 4)'
    )
    parser.add_argument(
        '--fuzzy-threshold', '-f',
        type=float,
        default=0.95,
        help='Fuzzy matching threshold for deduplication (default: 0.95)'
    )
    parser.add_argument(
        '--redact',
        action='store_true',
        help='Enable PII redaction'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    processor = MessagePreprocessor(
        conversation_gap_hours=args.gap_hours,
        fuzzy_threshold=args.fuzzy_threshold
    )

    print(f"Loading messages from {input_path}...")
    messages = processor.load_messages(str(input_path))

    print("Removing exact duplicates...")
    messages = processor.remove_exact_duplicates(messages)

    print("Removing fuzzy duplicates...")
    messages = processor.remove_fuzzy_duplicates(messages)

    print("Applying quality filters...")
    messages = processor.filter_quality(messages)

    if args.redact:
        print("Redacting PII...")
        messages = processor.redact_pii(messages)

    print("Segmenting into conversations...")
    conversations = processor.segment_conversations(messages)

    # Print statistics
    processor.print_stats(conversations)

    # Save outputs
    processor.save_conversations(conversations, args.output)
    print(f"Saved {len(conversations):,} conversations to {args.output}")

    if args.redact:
        processor.save_redaction_log(args.output)

    return 0


if __name__ == '__main__':
    exit(main())
