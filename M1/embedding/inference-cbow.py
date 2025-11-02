"""
Inference script for CBOW models.
Tests inference on multiple models and shows which works best for each word.
"""

import numpy as np
import json
from gensim.models import Word2Vec
from tokenizers import Tokenizer
from pathlib import Path
from collections import defaultdict

# Test words to evaluate
TEST_WORDS = ['wojsko', 'szlachta', 'choroba', 'król']

# Configuration options to test (should match train-cbow.py)
TOKENIZERS = {
    "all-corpora": "../tokenizer/tokenizers/tokenizer-all-corpora.json",
    "nkjp": "../tokenizer/tokenizers/tokenizer-nkjp.json",
    "wolnelektury": "../tokenizer/tokenizers/tokenizer-wolnelektury.json",
    "bielik-v1": "../tokenizer/tokenizers/bielik-v1-tokenizer.json",
    "bielik-v3": "../tokenizer/tokenizers/bielik-v3-tokenizer.json",
}

CORPORA = ["PAN_TADEUSZ", "ALL", "WOLNELEKTURY", "NKJP"]
PARAM_CONFIGS = ["small", "medium", "large"]

# Default files (fallback to best overall model)
DEFAULT_MODEL_FILE = "embedding_word2vec_cbow_model.model"
DEFAULT_TOKENIZER_FILE = "../tokenizer/tokenizers/tokenizer-all-corpora.json"

# --- FUNKCJA: OBLICZANIE WEKTORA DLA CAŁEGO SŁOWA ---

def get_word_vector_and_similar(word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 20):
    """
    Calculate word vector by averaging its subword token vectors.
    Returns vector and list of most similar tokens.
    """
    # Tokenize word to subword tokens
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()]

    # Remove special tokens if present
    if word_tokens and word_tokens[0] in ['[CLS]', '<s>', 'Ġ']:
        word_tokens = word_tokens[1:]
    if word_tokens and word_tokens[-1] in ['[SEP]', '</s>']:
        word_tokens = word_tokens[:-1]

    valid_vectors = []
    missing_tokens = []

    # Collect vectors for each token
    for token in word_tokens:
        if token in model.wv:
            valid_vectors.append(model.wv[token])
        else:
            missing_tokens.append(token)

    if not valid_vectors:
        return None, None

    # Average the vectors
    word_vector = np.mean(valid_vectors, axis=0)

    # Find most similar tokens
    similar_words = model.wv.most_similar(positive=[word_vector], topn=topn)

    return word_vector, similar_words

def evaluate_model_on_word(word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 10):
    """Evaluate model on a single word and return results."""
    word_vector, similar_tokens = get_word_vector_and_similar(word, tokenizer, model, topn=topn)

    if word_vector is not None and similar_tokens:
        # Calculate average similarity of top 5 results
        scores = [sim for _, sim in similar_tokens[:5]]
        avg_score = np.mean(scores) if scores else 0
        return {
            'word_vector': word_vector,
            'similar_tokens': similar_tokens,
            'avg_similarity': avg_score,
            'success': True
        }
    return {'success': False, 'avg_similarity': 0}

def get_model_path(corpus: str, tokenizer: str, params: str):
    """Generate model file path for a configuration."""
    # Model files are saved with naming pattern: embedding_word2vec_cbow_{corpus}_{tokenizer}_{params}.model
    model_name = f"embedding_word2vec_cbow_{corpus}_{tokenizer}_{params}.model"
    if Path(model_name).exists():
        return model_name
    # Fallback to default
    if Path(DEFAULT_MODEL_FILE).exists():
        return DEFAULT_MODEL_FILE
    return None

def load_model_config(corpus: str, tokenizer_name: str, params: str):
    """Load a model with its configuration."""
    tokenizer_path = TOKENIZERS.get(tokenizer_name)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        return None

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        print(f"ERROR loading tokenizer {tokenizer_name}: {e}")
        return None

    # Try to load model with configuration-specific name
    model_path = get_model_path(corpus, tokenizer_name, params)
    if not model_path or not Path(model_path).exists():
        # Try default as fallback
        model_path = DEFAULT_MODEL_FILE
        if not Path(model_path).exists():
            return None

    try:
        model = Word2Vec.load(model_path)
    except Exception as e:
        print(f"ERROR loading model {model_path}: {e}")
        return None

    return {
        'corpus': corpus,
        'tokenizer_name': tokenizer_name,
        'params': params,
        'tokenizer': tokenizer,
        'model': model,
        'model_file': model_path,
        'vocab_size': len(model.wv)
    }

if __name__ == "__main__":
    print("Starting comprehensive CBOW inference with multiple models...")
    print(f"Test words: {TEST_WORDS}\n")

    # Try to load models with different configurations
    # Models are saved with names indicating configuration: embedding_word2vec_cbow_{corpus}_{tokenizer}_{params}.model

    print("Loading models and tokenizers...")
    all_configs = []

    # Test subset of configurations (adjust based on available models)
    test_corpora = ["PAN_TADEUSZ", "ALL"]
    test_tokenizers = ["all-corpora", "nkjp", "bielik-v1"]
    test_params = ["small", "medium", "large"]

    # Try to load models for all combinations
    loaded_count = 0
    for corpus in test_corpora:
        for tokenizer_name in test_tokenizers:
            for params in test_params:
                config = load_model_config(corpus, tokenizer_name, params)
                if config:
                    all_configs.append(config)
                    loaded_count += 1
                    print(f"  ✓ Loaded: {corpus} + {tokenizer_name} + {params}")

    # If no configs loaded, try to load just the default
    if not all_configs:
        print("\nNo configuration-specific models found. Trying to load default configuration...")
        try:
            tokenizer = Tokenizer.from_file(DEFAULT_TOKENIZER_FILE)
            if Path(DEFAULT_MODEL_FILE).exists():
                model = Word2Vec.load(DEFAULT_MODEL_FILE)
                all_configs.append({
                    'corpus': 'ALL',
                    'tokenizer_name': 'all-corpora',
                    'params': 'medium',
                    'tokenizer': tokenizer,
                    'model': model,
                    'model_file': DEFAULT_MODEL_FILE,
                    'vocab_size': len(model.wv)
                })
                print("  ✓ Loaded default configuration")
            else:
                print(f"ERROR: Default model file '{DEFAULT_MODEL_FILE}' not found.")
                print(f"Please run train-cbow.py first to train models.")
                exit(1)
        except Exception as e:
            print(f"ERROR: Could not load any models. {e}")
            print(f"Please run train-cbow.py first to train models.")
            exit(1)

    print(f"\nLoaded {len(all_configs)} model configuration(s).")

    # Evaluate all models on test words
    print(f"\n{'='*80}")
    print("EVALUATING MODELS ON TEST WORDS")
    print(f"{'='*80}")

    all_results = []
    # Track top 3 models for each word
    top_models_per_word = {word: [] for word in TEST_WORDS}

    for config in all_configs:
        print(f"\nTesting: Corpus={config['corpus']}, Tokenizer={config['tokenizer_name']}, Params={config['params']}")

        model = config['model']
        tokenizer = config['tokenizer']

        config_results = {}
        for word in TEST_WORDS:
            result = evaluate_model_on_word(word, tokenizer, model, topn=10)
            config_results[word] = result

            # Track top 3 configurations for each word
            if result.get('success'):
                score = result.get('avg_similarity', 0)
                # Add to list and keep top 3
                top_models_per_word[word].append((score, config, result))
                # Sort by score (descending) and keep top 3
                top_models_per_word[word].sort(key=lambda x: x[0], reverse=True)
                top_models_per_word[word] = top_models_per_word[word][:3]

        all_results.append({
            'config': config,
            'results': config_results
        })

    # Print summary of top 3 models for each word
    print(f"\n{'='*80}")
    print("SUMMARY: Top 3 Model Configurations for Each Word")
    print(f"{'='*80}")

    for word in TEST_WORDS:
        top_models = top_models_per_word[word]
        if top_models:
            print(f"\n{word.upper()}:")
            for rank, (score, config, result) in enumerate(top_models, 1):
                print(f"\n  Rank {rank}:")
                print(f"    Config: Corpus={config['corpus']}, "
                      f"Tokenizer={config['tokenizer_name']}, Params={config['params']}")
                print(f"    Score (avg similarity of top 5): {score:.4f}")
                print(f"    Vocab size: {config['vocab_size']}")

                # Show top similar words
                if result and result.get('similar_tokens'):
                    similar = result['similar_tokens']
                    print(f"    Top similar tokens:")
                    for token, sim in similar[:5]:
                        print(f"      - {token}: {sim:.4f}")
        else:
            print(f"\n{word.upper()}: No successful configuration found.")

    # Show detailed results for each word using top 3 models
    print(f"\n{'='*80}")
    print("DETAILED RESULTS FOR EACH WORD (Top 3 Models)")
    print(f"{'='*80}")

    for word in TEST_WORDS:
        top_models = top_models_per_word[word]
        if top_models:
            print(f"\n{'='*80}")
            print(f"WORD: {word.upper()}")
            print(f"{'='*80}")

            for rank, (score, config, result) in enumerate(top_models, 1):
                print(f"\n--- Rank {rank} Model ---")
                print(f"Config: Corpus={config['corpus']}, "
                      f"Tokenizer={config['tokenizer_name']}, Params={config['params']}")
                print(f"Score: {score:.4f}")
                print(f"Vocab size: {config['vocab_size']}")

                model = config['model']
                tokenizer = config['tokenizer']

                if result.get('success'):
                    similar_tokens = result.get('similar_tokens', [])
                    word_vector = result.get('word_vector')
                    tokenized = tokenizer.encode(word).tokens

                    print(f"\n10 tokenów najbardziej podobnych do SŁOWA '{word}' (uśrednione wektory tokenów {tokenized}):")

                    if word_vector is not None:
                        vec_parts = [f"{v:10.7f}" for v in word_vector[:5]]
                        vec_str = " ".join(vec_parts)
                        print(f"  > Wektor słowa (początek): [{vec_str}]...")

                    for token, similarity in similar_tokens:
                        print(f"  - {token}: {similarity:.4f}")
                else:
                    print(f"\n{word}: Could not evaluate (word not in vocabulary)")
        else:
            print(f"\n{word.upper()}: No successful configuration found.")

    # Vector analogy test with best overall model
    if all_results:
        # Calculate average score for each config
        config_scores = []
        for res in all_results:
            scores = []
            for word in TEST_WORDS:
                if word in res['results']:
                    result = res['results'][word]
                    if result.get('success'):
                        scores.append(result.get('avg_similarity', 0))
            if scores:
                avg_score = np.mean(scores)
                config_scores.append((avg_score, res))

        if config_scores:
            # Sort by average score and take the best
            config_scores.sort(key=lambda x: x[0], reverse=True)
            best_overall_score, best_overall_result = config_scores[0]
            best_overall_config = best_overall_result['config']

            print(f"\n{'='*80}")
            print("VECTOR ANALOGY TEST (Best Overall Model)")
            print(f"{'='*80}")
            print(f"Using model: Corpus={best_overall_config['corpus']}, "
                  f"Tokenizer={best_overall_config['tokenizer_name']}, "
                  f"Params={best_overall_config['params']}")
            print(f"Average score: {best_overall_score:.4f}")

            model = best_overall_config['model']
            tokenizer = best_overall_config['tokenizer']

            tokens_analogy = ['dziecko', 'kobieta']
            vectors_for_analogy = []

            for token in tokens_analogy:
                if token in model.wv:
                    vectors_for_analogy.append(model.wv[token])
                else:
                    vec, _ = get_word_vector_and_similar(token, tokenizer, model, topn=1)
                    if vec is not None:
                        vectors_for_analogy.append(vec)

            if len(vectors_for_analogy) == 2:
                similar_to_combined = model.wv.most_similar(
                    positive=vectors_for_analogy,
                    topn=10
                )
                print(f"\n10 tokenów najbardziej podobnych do kombinacji tokenów: {tokens_analogy}")
                for token, similarity in similar_to_combined:
                    print(f"  - {token}: {similarity:.4f}")
            else:
                print(f"\nOstrzeżenie: Co najmniej jeden z tokenów '{tokens_analogy}' nie znajduje się w słowniku. Pomięto analogię.")

    print(f"\n{'='*80}")
    print("Inference completed successfully!")
    print(f"{'='*80}")
