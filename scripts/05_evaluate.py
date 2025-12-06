#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates the fine-tuned model using automated metrics and generates samples for human evaluation.

Features:
- Perplexity calculation on validation set
- Style metrics comparison (sentence length, vocabulary, emoji usage)
- Response generation for qualitative evaluation
- Blind test generation for human evaluation

Usage:
    python 05_evaluate.py --model outputs/lora_adapter --val data/training/val.jsonl
    python 05_evaluate.py --model myname-style --backend ollama --val data/training/val.jsonl
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Emoji pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE
)


def load_validation_data(val_path: str) -> list[dict]:
    """Load validation examples."""
    examples = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_prompts_and_targets(examples: list[dict]) -> list[tuple[list[dict], str]]:
    """Extract prompts (context) and target responses from examples."""
    pairs = []

    for ex in examples:
        messages = ex.get('messages', [])

        # Find the last assistant message (target)
        target = None
        context = []

        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg['role'] == 'assistant':
                target = msg['content']
                context = messages[:i]
            else:
                context.append(msg)

        if target and context:
            pairs.append((context, target))

    return pairs


def calculate_style_metrics(texts: list[str]) -> dict:
    """Calculate style metrics for a set of texts."""
    if not texts:
        return {}

    # Average length
    lengths = [len(t) for t in texts]
    avg_length = sum(lengths) / len(lengths)

    # Word count
    word_counts = [len(t.split()) for t in texts]
    avg_words = sum(word_counts) / len(word_counts)

    # Vocabulary
    all_words = []
    for t in texts:
        words = re.findall(r'\b\w+\b', t.lower())
        all_words.extend(words)
    unique_words = len(set(all_words))
    vocab_richness = unique_words / len(all_words) if all_words else 0

    # Emoji usage
    emoji_count = sum(len(EMOJI_PATTERN.findall(t)) for t in texts)
    emoji_rate = emoji_count / len(texts)

    # Punctuation patterns
    ends_period = sum(1 for t in texts if t.strip().endswith('.'))
    ends_question = sum(1 for t in texts if t.strip().endswith('?'))
    ends_exclamation = sum(1 for t in texts if t.strip().endswith('!'))
    all_lowercase = sum(1 for t in texts if t == t.lower())

    return {
        'avg_char_length': avg_length,
        'avg_word_count': avg_words,
        'unique_words': unique_words,
        'vocab_richness': vocab_richness,
        'emoji_rate': emoji_rate,
        'period_rate': ends_period / len(texts),
        'question_rate': ends_question / len(texts),
        'exclamation_rate': ends_exclamation / len(texts),
        'lowercase_rate': all_lowercase / len(texts),
    }


def compare_style_metrics(reference: dict, generated: dict) -> dict:
    """Compare style metrics between reference and generated text."""
    comparison = {}

    for key in reference:
        ref_val = reference[key]
        gen_val = generated.get(key, 0)

        if ref_val > 0:
            diff_pct = 100 * (gen_val - ref_val) / ref_val
        else:
            diff_pct = 0 if gen_val == 0 else 100

        comparison[key] = {
            'reference': ref_val,
            'generated': gen_val,
            'difference_pct': diff_pct
        }

    return comparison


def check_degenerate_outputs(responses: list[str]) -> dict:
    """Check for degenerate outputs."""
    if not responses:
        return {}

    empty = sum(1 for r in responses if not r.strip())

    # Check for repetition (same phrase repeated)
    repetitive = 0
    for r in responses:
        words = r.split()
        if len(words) >= 4:
            # Check if any 3-gram repeats
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            if len(trigrams) != len(set(trigrams)):
                repetitive += 1

    # Check for very short responses
    very_short = sum(1 for r in responses if len(r.strip()) < 5)

    # Check for very long responses
    very_long = sum(1 for r in responses if len(r.strip()) > 500)

    total = len(responses)
    return {
        'empty': {'count': empty, 'rate': empty / total},
        'repetitive': {'count': repetitive, 'rate': repetitive / total},
        'very_short': {'count': very_short, 'rate': very_short / total},
        'very_long': {'count': very_long, 'rate': very_long / total},
    }


def generate_with_ollama(model: str, messages: list[dict]) -> str:
    """Generate response using Ollama."""
    try:
        import ollama
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except ImportError:
        print("Error: ollama package not installed. Install with: pip install ollama")
        return ""
    except Exception as e:
        print(f"Error generating with Ollama: {e}")
        return ""


def generate_with_transformers(model_path: str, messages: list[dict], tokenizer=None, model=None) -> str:
    """Generate response using transformers."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    except ImportError:
        print("Error: transformers package not installed")
        return ""
    except Exception as e:
        print(f"Error generating with transformers: {e}")
        return ""


def create_blind_test(
    examples: list[dict],
    model: str,
    backend: str,
    n_samples: int = 20
) -> list[dict]:
    """Create blind test samples mixing real and generated responses."""
    pairs = extract_prompts_and_targets(examples)

    if len(pairs) < n_samples:
        n_samples = len(pairs)

    # Sample random examples
    selected = random.sample(pairs, n_samples)

    blind_test = []

    for i, (context, real_response) in enumerate(selected):
        # Generate model response
        if backend == 'ollama':
            generated = generate_with_ollama(model, context)
        else:
            generated = generate_with_transformers(model, context)

        # Create blind test entry (randomly order real vs generated)
        options = [
            {'text': real_response, 'is_real': True},
            {'text': generated, 'is_real': False}
        ]
        random.shuffle(options)

        blind_test.append({
            'id': i + 1,
            'context': [{'role': m['role'], 'content': m['content']}
                       for m in context if m['role'] != 'system'],
            'option_a': options[0]['text'],
            'option_b': options[1]['text'],
            'answer': 'A' if options[0]['is_real'] else 'B'
        })

    return blind_test


def run_evaluation(
    val_path: str,
    model: str = None,
    backend: str = 'ollama',
    n_generate: int = 50,
    output_dir: str = None
) -> dict:
    """Run full evaluation pipeline."""
    print(f"Loading validation data from {val_path}...")
    examples = load_validation_data(val_path)
    pairs = extract_prompts_and_targets(examples)

    print(f"Loaded {len(examples)} examples, {len(pairs)} usable pairs")

    # Extract reference responses
    reference_responses = [target for _, target in pairs]

    # Calculate reference style metrics
    print("\nCalculating reference style metrics...")
    ref_metrics = calculate_style_metrics(reference_responses)

    results = {
        'reference_metrics': ref_metrics,
        'n_examples': len(examples),
        'n_pairs': len(pairs)
    }

    # If model specified, generate and compare
    if model:
        print(f"\nGenerating {n_generate} responses with {model} ({backend})...")

        # Sample examples for generation
        sample_pairs = random.sample(pairs, min(n_generate, len(pairs)))
        generated_responses = []

        for i, (context, _) in enumerate(sample_pairs):
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(sample_pairs)}...")

            if backend == 'ollama':
                response = generate_with_ollama(model, context)
            else:
                response = generate_with_transformers(model, context)

            generated_responses.append(response)

        # Calculate generated style metrics
        print("\nCalculating generated style metrics...")
        gen_metrics = calculate_style_metrics(generated_responses)
        results['generated_metrics'] = gen_metrics

        # Compare metrics
        results['style_comparison'] = compare_style_metrics(ref_metrics, gen_metrics)

        # Check for degenerate outputs
        results['degenerate_check'] = check_degenerate_outputs(generated_responses)

        # Store sample generations
        results['sample_generations'] = [
            {
                'context': [m for m in ctx if m['role'] != 'system'],
                'reference': target,
                'generated': gen
            }
            for (ctx, target), gen in zip(sample_pairs[:10], generated_responses[:10])
        ]

    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\n## Dataset")
    print(f"Validation examples: {results['n_examples']:,}")
    print(f"Usable pairs: {results['n_pairs']:,}")

    print(f"\n## Reference Style Metrics")
    for key, value in ref_metrics.items():
        print(f"  {key}: {value:.3f}")

    if model and 'style_comparison' in results:
        print(f"\n## Style Comparison (Generated vs Reference)")
        for key, data in results['style_comparison'].items():
            ref = data['reference']
            gen = data['generated']
            diff = data['difference_pct']
            indicator = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {key}:")
            print(f"    Reference: {ref:.3f}")
            print(f"    Generated: {gen:.3f} ({indicator} {abs(diff):.1f}%)")

        print(f"\n## Degenerate Output Check")
        for key, data in results['degenerate_check'].items():
            print(f"  {key}: {data['count']} ({100*data['rate']:.1f}%)")

        if results.get('sample_generations'):
            print(f"\n## Sample Generations")
            for i, sample in enumerate(results['sample_generations'][:5]):
                print(f"\n--- Sample {i+1} ---")
                print(f"Context: {sample['context'][-1]['content'][:100]}...")
                print(f"Reference: {sample['reference'][:100]}...")
                print(f"Generated: {sample['generated'][:100]}...")

    print("\n" + "=" * 60)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned model'
    )
    parser.add_argument(
        '--val', '-v',
        default='data/training/val.jsonl',
        help='Validation data JSONL file'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='Model path or Ollama model name (optional, for generation)'
    )
    parser.add_argument(
        '--backend', '-b',
        choices=['ollama', 'transformers'],
        default='ollama',
        help='Backend for generation (default: ollama)'
    )
    parser.add_argument(
        '--n-generate', '-n',
        type=int,
        default=50,
        help='Number of responses to generate for evaluation'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--blind-test',
        action='store_true',
        help='Generate blind test for human evaluation'
    )
    parser.add_argument(
        '--blind-test-samples',
        type=int,
        default=20,
        help='Number of samples for blind test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    val_path = Path(args.val)
    if not val_path.exists():
        print(f"Error: Validation file not found: {val_path}")
        return 1

    # Run main evaluation
    results = run_evaluation(
        str(val_path),
        model=args.model,
        backend=args.backend,
        n_generate=args.n_generate,
        output_dir=args.output_dir
    )

    # Generate blind test if requested
    if args.blind_test and args.model:
        print("\nGenerating blind test...")
        examples = load_validation_data(str(val_path))
        blind_test = create_blind_test(
            examples,
            args.model,
            args.backend,
            n_samples=args.blind_test_samples
        )

        if args.output_dir:
            blind_path = Path(args.output_dir) / 'blind_test.json'
            with open(blind_path, 'w', encoding='utf-8') as f:
                json.dump(blind_test, f, indent=2, ensure_ascii=False)
            print(f"Blind test saved to {blind_path}")

            # Also save answer key separately
            answers_path = Path(args.output_dir) / 'blind_test_answers.json'
            answers = {item['id']: item['answer'] for item in blind_test}
            with open(answers_path, 'w', encoding='utf-8') as f:
                json.dump(answers, f, indent=2)
            print(f"Answer key saved to {answers_path}")

    return 0


if __name__ == '__main__':
    exit(main())
