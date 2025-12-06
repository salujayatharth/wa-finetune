# WhatsApp Style Transfer LLM - Makefile
# ======================================

.PHONY: all install extract preprocess format analyze train evaluate deploy clean help

# Python executable (override with PYTHON=python if needed)
PYTHON ?= python3

# Default paths
DB_PATH ?= data/raw/msgstore.db
PLATFORM ?= auto
RAW_MESSAGES ?= data/processed/raw_messages.jsonl
CONVERSATIONS ?= data/processed/conversations.jsonl
TRAIN_DATA ?= data/training/train.jsonl
VAL_DATA ?= data/training/val.jsonl
MODEL_NAME ?= myname-style
OUTPUT_DIR ?= outputs

# Training options
MAX_EXAMPLES ?= 15000
EPOCHS ?= 1
BATCH_SIZE ?= 4
LEARNING_RATE ?= 2e-4

# =============================================================================
# Main Targets
# =============================================================================

all: extract preprocess format analyze ## Run full data pipeline

help: ## Show this help
	@echo "WhatsApp Style Transfer LLM"
	@echo ""
	@echo "Usage: make [target] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Options:"
	@echo "  DB_PATH=<path>       WhatsApp database path (default: data/raw/msgstore.db)"
	@echo "  PLATFORM=<platform>  Platform: android, ios, auto (default: auto)"
	@echo "  MAX_EXAMPLES=<n>     Max training examples (default: 15000)"
	@echo "  MODEL_NAME=<name>    Ollama model name (default: myname-style)"
	@echo ""
	@echo "Examples:"
	@echo "  make all                              # Run full pipeline"
	@echo "  make extract DB_PATH=data/raw/chat.db # Extract from custom path"
	@echo "  make format MAX_EXAMPLES=20000        # Format with more examples"
	@echo "  make deploy MODEL_NAME=my-style       # Deploy with custom name"

# =============================================================================
# Setup
# =============================================================================

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-training: ## Install training dependencies (for GPU machine)
	$(PYTHON) -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	$(PYTHON) -m pip install --no-deps trl peft accelerate bitsandbytes
	$(PYTHON) -m pip install datasets wandb

dirs: ## Create project directories
	mkdir -p data/raw data/processed data/training outputs

# =============================================================================
# Data Pipeline
# =============================================================================

extract: dirs ## Extract messages from WhatsApp database
	@echo "Extracting messages from $(DB_PATH)..."
	$(PYTHON) scripts/01_extract.py \
		--input $(DB_PATH) \
		--output $(RAW_MESSAGES) \
		--platform $(PLATFORM)

preprocess: ## Clean and segment messages into conversations
	@echo "Preprocessing messages..."
	$(PYTHON) scripts/02_preprocess.py \
		--input $(RAW_MESSAGES) \
		--output $(CONVERSATIONS)

preprocess-redact: ## Preprocess with PII redaction
	@echo "Preprocessing messages with PII redaction..."
	$(PYTHON) scripts/02_preprocess.py \
		--input $(RAW_MESSAGES) \
		--output $(CONVERSATIONS) \
		--redact

format: ## Convert to ChatML training format
	@echo "Formatting for training..."
	$(PYTHON) scripts/03_format.py \
		--input $(CONVERSATIONS) \
		--output-dir data/training/ \
		--max-examples $(MAX_EXAMPLES)

analyze: ## Analyze training dataset statistics
	@echo "Analyzing dataset..."
	$(PYTHON) scripts/04_analyze.py \
		--train $(TRAIN_DATA) \
		--val $(VAL_DATA)

analyze-save: ## Analyze and save reports to analysis/
	@echo "Analyzing dataset and saving reports..."
	mkdir -p analysis
	$(PYTHON) scripts/04_analyze.py \
		--train $(TRAIN_DATA) \
		--val $(VAL_DATA) \
		--output-dir analysis/

# =============================================================================
# Training
# =============================================================================

train: ## Run training (requires GPU)
	@echo "Starting training..."
	$(PYTHON) training/train.py \
		--config training/config.yaml

train-custom: ## Run training with custom options
	@echo "Starting training with custom options..."
	$(PYTHON) training/train.py \
		--config training/config.yaml \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE)

train-wandb: ## Run training with W&B logging
	@echo "Starting training with W&B..."
	$(PYTHON) training/train.py \
		--config training/config.yaml \
		--wandb

# =============================================================================
# Evaluation
# =============================================================================

evaluate: ## Evaluate model style metrics (no generation)
	@echo "Evaluating dataset metrics..."
	$(PYTHON) scripts/05_evaluate.py \
		--val $(VAL_DATA)

evaluate-model: ## Evaluate with model generation
	@echo "Evaluating model $(MODEL_NAME)..."
	$(PYTHON) scripts/05_evaluate.py \
		--val $(VAL_DATA) \
		--model $(MODEL_NAME) \
		--backend ollama \
		--n-generate 50

evaluate-full: ## Full evaluation with blind test
	@echo "Running full evaluation..."
	mkdir -p evaluation
	$(PYTHON) scripts/05_evaluate.py \
		--val $(VAL_DATA) \
		--model $(MODEL_NAME) \
		--backend ollama \
		--n-generate 50 \
		--blind-test \
		--output-dir evaluation/

# =============================================================================
# Deployment
# =============================================================================

deploy: ## Create Ollama model from GGUF
	@echo "Creating Ollama model '$(MODEL_NAME)'..."
	ollama create $(MODEL_NAME) -f Modelfile

run: ## Run the deployed model interactively
	ollama run $(MODEL_NAME)

test-model: ## Quick test of deployed model
	@echo "Testing $(MODEL_NAME)..."
	@echo "hey whats up" | ollama run $(MODEL_NAME)

# =============================================================================
# Utilities
# =============================================================================

clean: ## Remove generated data files
	rm -rf data/processed/*.jsonl
	rm -rf data/training/*.jsonl
	rm -rf analysis/

clean-all: clean ## Remove all generated files including model outputs
	rm -rf outputs/
	rm -rf evaluation/
	rm -rf wandb/

lint: ## Lint Python files
	@which ruff > /dev/null && ruff check scripts/ training/ || echo "ruff not installed, skipping lint"

format-code: ## Format Python files
	@which ruff > /dev/null && ruff format scripts/ training/ || echo "ruff not installed, skipping format"

# =============================================================================
# Info
# =============================================================================

info: ## Show current configuration
	@echo "Configuration:"
	@echo "  DB_PATH:       $(DB_PATH)"
	@echo "  PLATFORM:      $(PLATFORM)"
	@echo "  RAW_MESSAGES:  $(RAW_MESSAGES)"
	@echo "  CONVERSATIONS: $(CONVERSATIONS)"
	@echo "  TRAIN_DATA:    $(TRAIN_DATA)"
	@echo "  VAL_DATA:      $(VAL_DATA)"
	@echo "  MAX_EXAMPLES:  $(MAX_EXAMPLES)"
	@echo "  MODEL_NAME:    $(MODEL_NAME)"
	@echo "  OUTPUT_DIR:    $(OUTPUT_DIR)"

stats: ## Show data file statistics
	@echo "Data files:"
	@test -f $(RAW_MESSAGES) && wc -l $(RAW_MESSAGES) || echo "  $(RAW_MESSAGES): not found"
	@test -f $(CONVERSATIONS) && wc -l $(CONVERSATIONS) || echo "  $(CONVERSATIONS): not found"
	@test -f $(TRAIN_DATA) && wc -l $(TRAIN_DATA) || echo "  $(TRAIN_DATA): not found"
	@test -f $(VAL_DATA) && wc -l $(VAL_DATA) || echo "  $(VAL_DATA): not found"
