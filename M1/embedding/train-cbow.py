"""
Train CBOW models with multiple configurations.
Saves models with configuration-specific names.
"""

import numpy as np
import json
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
import glob
from pathlib import Path
from corpora import CORPORA_FILES # type: ignore

# Suppress gensim logging for cleaner output
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

# Configuration options to test
TOKENIZERS = {
    "all-corpora": "../tokenizer/tokenizers/tokenizer-all-corpora.json",
    "nkjp": "../tokenizer/tokenizers/tokenizer-nkjp.json",
    "wolnelektury": "../tokenizer/tokenizers/tokenizer-wolnelektury.json",
    "bielik-v1": "../tokenizer/tokenizers/bielik-v1-tokenizer.json",
    "bielik-v3": "../tokenizer/tokenizers/bielik-v3-tokenizer.json",
}

CORPORA = {
    "PAN_TADEUSZ": CORPORA_FILES["PAN_TADEUSZ"],
    "WOLNELEKTURY": CORPORA_FILES["WOLNELEKTURY"],
    "NKJP": CORPORA_FILES["NKJP"],
    "ALL": CORPORA_FILES["ALL"],
}

# Parameter configurations to test
PARAM_CONFIGS = [
    {"name": "small", "vector_size": 20, "window": 6, "epochs": 10, "min_count": 2},
    {"name": "medium", "vector_size": 50, "window": 8, "epochs": 15, "min_count": 3},
    {"name": "large", "vector_size": 100, "window": 10, "epochs": 20, "min_count": 5},
]

WORKERS = 4
SAMPLE_RATE = 1e-2
SG_MODE = 0  # 0 for CBOW

# Default output files (will be modified per configuration)
OUTPUT_TENSOR_FILE = "embedding_tensor_cbow.npy"
OUTPUT_MAP_FILE = "embedding_token_to_index_map.json"
OUTPUT_MODEL_FILE = "embedding_word2vec_cbow_model.model"

def aggregate_raw_sentences(files):
    """Load and aggregate sentences from files."""
    raw_sentences = []
    print("Loading text from files...")
    print(f"Number of files to load: {len(files)}")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            print(f"WARNING: File '{file}' not found. Skipping.")
            continue

    if not raw_sentences:
        print("ERROR: Input files are empty or not loaded.")
        return []
    return raw_sentences

def train_config(corpus_name, corpus_files, tokenizer_name, tokenizer_path, params):
    """Train a model with given configuration."""
    print(f"\n{'='*80}")
    print(f"Training: Corpus={corpus_name}, Tokenizer={tokenizer_name}, Params={params['name']}")
    print(f"{'='*80}")

    # Load tokenizer
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except FileNotFoundError:
        print(f"ERROR: Tokenizer file '{tokenizer_path}' not found. Skipping.")
        return None

    # Load and tokenize corpus
    raw_sentences = aggregate_raw_sentences(corpus_files)
    if not raw_sentences:
        print(f"ERROR: No sentences loaded. Skipping.")
        return None

    print(f"Tokenizing {len(raw_sentences)} sentences...")
    encodings = tokenizer.encode_batch(raw_sentences)
    tokenized_sentences = [encoding.tokens for encoding in encodings]
    print(f"Prepared {len(tokenized_sentences)} sequences for training.")

    # Train model
    print(f"\n--- Starting Word2Vec CBOW Training ---")
    try:
        model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=params["vector_size"],
            window=params["window"],
            min_count=params["min_count"],
            workers=WORKERS,
            sg=SG_MODE,  # 0: CBOW
            epochs=params["epochs"],
            sample=SAMPLE_RATE,
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"ERROR during training: {e}. Skipping.")
        return None

    # Generate file names for this configuration
    config_suffix = f"{corpus_name}_{tokenizer_name}_{params['name']}"
    model_file = f"embedding_word2vec_cbow_{config_suffix}.model"
    tensor_file = f"embedding_tensor_cbow_{config_suffix}.npy"
    map_file = f"embedding_token_to_index_map_{config_suffix}.json"

    # Save model with configuration-specific name
    print(f"\nSaving model with configuration-specific name...")
    try:
        model.save(model_file)
        print(f"  Model saved as: '{model_file}'")

        # Save tensor
        embedding_matrix_np = model.wv.vectors
        embedding_matrix_tensor = np.array(embedding_matrix_np, dtype=np.float32)
        np.save(tensor_file, embedding_matrix_tensor)
        print(f"  Tensor saved as: '{tensor_file}'")

        # Save token to index mapping
        token_to_index = {token: model.wv.get_index(token) for token in model.wv.index_to_key}
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(token_to_index, f, ensure_ascii=False, indent=4)
        print(f"  Token map saved as: '{map_file}'")
    except Exception as e:
        print(f"WARNING: Could not save model files: {e}")

    # Return configuration info
    config = {
        'corpus': corpus_name,
        'tokenizer_name': tokenizer_name,
        'params': params['name'],
        'vector_size': params['vector_size'],
        'window': params['window'],
        'epochs': params['epochs'],
        'min_count': params['min_count'],
        'vocab_size': len(model.wv),
        'model_file': model_file,
        'tensor_file': tensor_file,
        'map_file': map_file,
    }

    return config

if __name__ == "__main__":
    print("Starting comprehensive CBOW training with multiple configurations...")
    print(f"Testing {len(CORPORA)} corpora × {len(TOKENIZERS)} tokenizers × {len(PARAM_CONFIGS)} param configs\n")

    all_configs = []

    # Test a subset to avoid too long execution (you can adjust this)
    test_corpora = ["PAN_TADEUSZ", "ALL"]  # Start with smaller corpus
    test_tokenizers = ["all-corpora", "nkjp", "bielik-v1"]

    total_combinations = len(test_corpora) * len(test_tokenizers) * len(PARAM_CONFIGS)
    print(f"Total combinations to train: {total_combinations}\n")

    config_num = 0
    for corpus_name in test_corpora:
        corpus_files = CORPORA[corpus_name]
        if not corpus_files:
            continue

        for tokenizer_name in test_tokenizers:
            tokenizer_path = TOKENIZERS.get(tokenizer_name)
            if not tokenizer_path or not Path(tokenizer_path).exists():
                continue

            for params in PARAM_CONFIGS:
                config_num += 1
                print(f"\n[{config_num}/{total_combinations}] Training configuration...")

                config_result = train_config(
                    corpus_name, corpus_files, tokenizer_name, tokenizer_path, params
                )

                if config_result is None:
                    continue

                all_configs.append(config_result)

    # Print summary of trained models
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully trained {len(all_configs)} model(s):\n")

    for config in all_configs:
        print(f"  - {config['corpus']} + {config['tokenizer_name']} + {config['params']}")
        print(f"    Model: {config['model_file']}")
        print(f"    Vocab size: {config['vocab_size']}")
        print(f"    Vector size: {config['vector_size']}, Window: {config['window']}, Epochs: {config['epochs']}, Min count: {config['min_count']}\n")

    print(f"{'='*80}")
    print("Training completed successfully!")
    print(f"All models saved with configuration-specific names.")
    print(f"You can now use inference-cbow.py to evaluate and compare models.")
    print(f"{'='*80}")
