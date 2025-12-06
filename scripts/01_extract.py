#!/usr/bin/env python3
"""
WhatsApp Message Extraction Script

Extracts text messages from WhatsApp SQLite databases (Android msgstore.db or iOS ChatStorage.sqlite)
and outputs to JSONL format for further processing.

Usage:
    python 01_extract.py --input data/raw/msgstore.db --output data/processed/raw_messages.jsonl
    python 01_extract.py --input data/raw/ChatStorage.sqlite --platform ios --output data/processed/raw_messages.jsonl
"""

import argparse
import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path


# System message patterns to skip
SYSTEM_MESSAGE_PATTERNS = re.compile(
    r'\b(added|removed|changed|left|created|deleted|joined|security code|'
    r'end-to-end encrypted|messages and calls are end-to-end encrypted|'
    r'changed the subject|changed this group|was added|was removed)\b',
    re.IGNORECASE
)

# Media placeholder patterns to skip
MEDIA_PLACEHOLDER_PATTERNS = re.compile(
    r'(<Media omitted>|image omitted|audio omitted|video omitted|'
    r'sticker omitted|document omitted|GIF omitted|Contact card omitted|'
    r'\u200e?<attached:.*?>)',
    re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Normalize Unicode text while preserving emojis and punctuation."""
    if not text:
        return ""
    # NFC normalization for consistent representation
    return unicodedata.normalize('NFC', text.strip())


def is_valid_message(text: str) -> bool:
    """Check if message should be included (not system/media placeholder)."""
    if not text or not text.strip():
        return False

    text = text.strip()

    # Skip system messages
    if SYSTEM_MESSAGE_PATTERNS.search(text):
        return False

    # Skip media placeholders
    if MEDIA_PLACEHOLDER_PATTERNS.search(text):
        return False

    return True


def timestamp_to_iso(timestamp_ms: int) -> str:
    """Convert millisecond timestamp to ISO format."""
    try:
        # WhatsApp uses millisecond timestamps
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return dt.isoformat()
    except (ValueError, OSError, TypeError):
        return None


def extract_android(db_path: str) -> list[dict]:
    """
    Extract messages from Android WhatsApp database (msgstore.db).

    Schema reference for Android (post-2022):
    - message table: text_data, timestamp, from_me, chat_row_id, sender_jid_row_id
    - jid table: raw_string (phone number/group id)
    - chat table: subject (group name)
    """
    messages = []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Try the newer schema first (post-2022)
    try:
        query = """
        SELECT
            m.text_data as message,
            m.timestamp,
            m.from_me,
            m.chat_row_id,
            j.raw_string as sender_jid,
            c.subject as chat_name,
            cj.raw_string as chat_jid
        FROM message m
        LEFT JOIN jid j ON m.sender_jid_row_id = j._id
        LEFT JOIN chat c ON m.chat_row_id = c._id
        LEFT JOIN jid cj ON c.jid_row_id = cj._id
        WHERE m.text_data IS NOT NULL
          AND m.text_data != ''
          AND m.message_type = 0
        ORDER BY m.chat_row_id, m.timestamp
        """
        cursor.execute(query)
    except sqlite3.OperationalError:
        # Fall back to older schema
        try:
            query = """
            SELECT
                m.data as message,
                m.timestamp,
                m.key_from_me as from_me,
                m.key_remote_jid as chat_jid,
                '' as sender_jid,
                '' as chat_name
            FROM messages m
            WHERE m.data IS NOT NULL
              AND m.data != ''
              AND m.media_type = 0
            ORDER BY m.key_remote_jid, m.timestamp
            """
            cursor.execute(query)
        except sqlite3.OperationalError as e:
            print(f"Error: Could not find message table. Schema may be unsupported. {e}")
            conn.close()
            return messages

    for row in cursor:
        text = row['message']

        if not is_valid_message(text):
            continue

        text = normalize_text(text)
        timestamp_iso = timestamp_to_iso(row['timestamp'])

        if not timestamp_iso:
            continue

        # Determine chat identifier
        chat_id = str(row.get('chat_row_id') or row.get('chat_jid', 'unknown'))
        chat_name = row.get('chat_name') or row.get('chat_jid', '')

        # Clean up chat name (use JID if no subject)
        if not chat_name and 'chat_jid' in row.keys():
            chat_name = row['chat_jid']

        messages.append({
            'message': text,
            'timestamp': row['timestamp'],
            'timestamp_iso': timestamp_iso,
            'from_me': bool(row['from_me']),
            'chat_id': chat_id,
            'chat_name': chat_name or f"Chat_{chat_id}"
        })

    conn.close()
    return messages


def extract_ios(db_path: str) -> list[dict]:
    """
    Extract messages from iOS WhatsApp database (ChatStorage.sqlite).

    iOS schema uses different table and column names.
    """
    messages = []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        query = """
        SELECT
            ZWAMESSAGE.ZTEXT as message,
            ZWAMESSAGE.ZMESSAGEDATE as timestamp,
            ZWAMESSAGE.ZISFROMME as from_me,
            ZWAMESSAGE.ZCHATSESSION as chat_id,
            ZWACHATSESSION.ZPARTNERNAME as chat_name,
            ZWACHATSESSION.ZCONTACTJID as chat_jid
        FROM ZWAMESSAGE
        LEFT JOIN ZWACHATSESSION ON ZWAMESSAGE.ZCHATSESSION = ZWACHATSESSION.Z_PK
        WHERE ZWAMESSAGE.ZTEXT IS NOT NULL
          AND ZWAMESSAGE.ZTEXT != ''
          AND ZWAMESSAGE.ZMESSAGETYPE = 0
        ORDER BY ZWAMESSAGE.ZCHATSESSION, ZWAMESSAGE.ZMESSAGEDATE
        """
        cursor.execute(query)
    except sqlite3.OperationalError as e:
        print(f"Error: Could not query iOS database. Schema may be unsupported. {e}")
        conn.close()
        return messages

    for row in cursor:
        text = row['message']

        if not is_valid_message(text):
            continue

        text = normalize_text(text)

        # iOS uses CoreData timestamp (seconds since 2001-01-01)
        # Convert to Unix timestamp (seconds since 1970-01-01)
        ios_epoch_offset = 978307200  # Seconds between 1970-01-01 and 2001-01-01
        unix_timestamp_ms = int((row['timestamp'] + ios_epoch_offset) * 1000)
        timestamp_iso = timestamp_to_iso(unix_timestamp_ms)

        if not timestamp_iso:
            continue

        chat_id = str(row['chat_id'])
        chat_name = row['chat_name'] or row['chat_jid'] or f"Chat_{chat_id}"

        messages.append({
            'message': text,
            'timestamp': unix_timestamp_ms,
            'timestamp_iso': timestamp_iso,
            'from_me': bool(row['from_me']),
            'chat_id': chat_id,
            'chat_name': chat_name
        })

    conn.close()
    return messages


def save_messages(messages: list[dict], output_path: str) -> None:
    """Save messages to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + '\n')


def print_stats(messages: list[dict]) -> None:
    """Print extraction statistics."""
    if not messages:
        print("No messages extracted!")
        return

    total = len(messages)
    from_me = sum(1 for m in messages if m['from_me'])
    from_others = total - from_me

    chats = set(m['chat_id'] for m in messages)

    # Message length stats
    lengths = [len(m['message']) for m in messages]
    avg_len = sum(lengths) / len(lengths)

    print(f"\n{'='*50}")
    print("EXTRACTION STATISTICS")
    print(f"{'='*50}")
    print(f"Total messages extracted: {total:,}")
    print(f"Messages from you: {from_me:,} ({100*from_me/total:.1f}%)")
    print(f"Messages from others: {from_others:,} ({100*from_others/total:.1f}%)")
    print(f"Unique chats: {len(chats):,}")
    print(f"Average message length: {avg_len:.1f} characters")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract WhatsApp messages from SQLite database'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to WhatsApp database (msgstore.db or ChatStorage.sqlite)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/processed/raw_messages.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--platform', '-p',
        choices=['android', 'ios', 'auto'],
        default='auto',
        help='Platform type (default: auto-detect from filename)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Auto-detect platform
    platform = args.platform
    if platform == 'auto':
        if 'ChatStorage' in input_path.name:
            platform = 'ios'
        else:
            platform = 'android'

    print(f"Extracting messages from {input_path} (platform: {platform})...")

    # Extract messages
    if platform == 'android':
        messages = extract_android(str(input_path))
    else:
        messages = extract_ios(str(input_path))

    if not messages:
        print("No messages were extracted. Check database format.")
        return 1

    # Print statistics
    print_stats(messages)

    # Save to output
    save_messages(messages, args.output)
    print(f"Saved {len(messages):,} messages to {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
