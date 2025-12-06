# WhatsApp Style Transfer LLM

Train an LLM to replicate your conversational style using WhatsApp message history. Uses QLoRA fine-tuning on Llama 3.1 8B, deployed via Ollama for local inference.

## Project Structure

```
wa-finetune/
├── data/
│   ├── raw/                    # Original WhatsApp exports (gitignored)
│   ├── processed/              # Cleaned conversation data
│   └── training/               # Final training format
├── scripts/
│   ├── 01_extract.py           # Extract from SQLite
│   ├── 02_preprocess.py        # Clean and deduplicate
│   ├── 03_format.py            # Convert to ChatML format
│   ├── 04_analyze.py           # Dataset statistics
│   └── 05_evaluate.py          # Post-training evaluation
├── training/
│   ├── config.yaml             # Training configuration
│   └── train.py                # Unsloth training script
├── outputs/                    # Model checkpoints (gitignored)
├── requirements.txt
├── Modelfile                   # Ollama model definition
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone <your-repo-url>
cd wa-finetune

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Extract Messages

Place your WhatsApp database in `data/raw/`:
- **Android**: `msgstore.db` (from `/data/data/com.whatsapp/databases/`)
- **iOS**: `ChatStorage.sqlite` (from iTunes backup)

```bash
# Extract messages
python scripts/01_extract.py --input data/raw/msgstore.db --output data/processed/raw_messages.jsonl

# For iOS
python scripts/01_extract.py --input data/raw/ChatStorage.sqlite --platform ios
```

### 3. Preprocess Data

```bash
# Clean and segment into conversations
python scripts/02_preprocess.py --input data/processed/raw_messages.jsonl --output data/processed/conversations.jsonl

# With PII redaction
python scripts/02_preprocess.py --input data/processed/raw_messages.jsonl --output data/processed/conversations.jsonl --redact
```

### 4. Format for Training

```bash
# Convert to ChatML format
python scripts/03_format.py --input data/processed/conversations.jsonl --output-dir data/training/

# Customize system prompt
python scripts/03_format.py --system-prompt "You are [NAME]. Respond naturally in your conversational style."
```

### 5. Analyze Dataset

```bash
# View statistics
python scripts/04_analyze.py --train data/training/train.jsonl --val data/training/val.jsonl

# Save detailed analysis
python scripts/04_analyze.py --train data/training/train.jsonl --val data/training/val.jsonl --output-dir analysis/
```

### 6. Train on RunPod

```bash
# On your RunPod instance:

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Upload your data and run training
python training/train.py

# With custom settings
python training/train.py --epochs 2 --batch-size 2 --wandb
```

### 7. Deploy to Ollama

```bash
# Download GGUF from RunPod
scp -P <port> root@<pod-ip>:~/wa-finetune/outputs/gguf/*.gguf ./outputs/gguf/

# Create Ollama model
ollama create myname-style -f Modelfile

# Test it!
ollama run myname-style "hey whats up"
```

### 8. Evaluate

```bash
# Run evaluation metrics
python scripts/05_evaluate.py --val data/training/val.jsonl --model myname-style --backend ollama

# Generate blind test for human evaluation
python scripts/05_evaluate.py --val data/training/val.jsonl --model myname-style --blind-test --output-dir evaluation/
```

## Configuration

Edit `training/config.yaml` to customize training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `unsloth/Meta-Llama-3.1-8B-bnb-4bit` | Base model for fine-tuning |
| `lora_r` | 32 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 64 | LoRA scaling factor |
| `num_train_epochs` | 1 | Training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gguf_quantization` | q5_k_m | GGUF quantization method |

## Requirements

**Data Processing:**
- Python 3.10+
- rapidfuzz (for deduplication)

**Training (GPU):**
- CUDA-capable GPU (24GB+ VRAM recommended)
- Unsloth
- PyTorch 2.0+
- transformers, peft, trl, accelerate, bitsandbytes

**Deployment:**
- Ollama

## Troubleshooting

**OOM errors during training:**
- Reduce `per_device_train_batch_size` to 2
- Increase `gradient_accumulation_steps` to 8

**Generic/boring outputs:**
- Increase training examples (target 15-20k)
- Lower learning rate to 1e-4
- Ensure dataset has diverse conversation partners

**Model memorizes training data:**
- Add more deduplication
- Check for train/val leakage (split by conversation, not message)
- Reduce training epochs

**Slow inference:**
- Use smaller quantization (q4_k_m instead of q5_k_m)
- Enable GPU acceleration in Ollama

## Privacy Notice

This project processes personal WhatsApp messages. The `data/` directory is gitignored by default. Never commit:
- Raw database files
- Processed message files
- Redaction logs
- Trained model weights (may contain memorized data)

## License

MIT
