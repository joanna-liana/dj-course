from tokenizers import Tokenizer
from corpora import get_corpus_file
from pathlib import Path
import json

# Define all tokenizers (3 Bielik + 1 HF + 4 custom)
TOKENIZERS = {
    "bielik-v1": "tokenizers/bielik-v1-tokenizer.json",
    "bielik-v2": "tokenizers/bielik-v2-tokenizer.json",
    "bielik-v3": "tokenizers/bielik-v3-tokenizer.json",
    "tokenizer-next-270m": "tokenizers/tokenizer-next-270m.json",  # HF tokenizer
    "tokenizer-pan-tadeusz": "tokenizers/tokenizer-pan-tadeusz.json",
    "tokenizer-wolnelektury": "tokenizers/tokenizer-wolnelektury.json",
    "tokenizer-nkjp": "tokenizers/tokenizer-nkjp.json",
    "tokenizer-all-corpora": "tokenizers/tokenizer-all-corpora.json",
}

# Define test texts
TEXTS = {
    "Pan Tadeusz Księga 1": get_corpus_file("WOLNELEKTURY", "pan-tadeusz-ksiega-1.txt")[0],
    "The Pickwick Papers": get_corpus_file("MINI", "the-pickwick-papers-gutenberg.txt")[0],
    "Fryderyk Chopin": get_corpus_file("MINI", "fryderyk-chopin-wikipedia.txt")[0],
}

def load_text(file_path):
    """Load text from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def tokenize_and_analyze(tokenizer_path, text):
    """Tokenize text and return statistics"""
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens

        token_count = len(tokens)
        avg_token_length = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        max_token_length = max((len(t) for t in tokens), default=0)
        min_token_length = min((len(t) for t in tokens), default=0)

        return {
            "token_count": token_count,
            "avg_token_length": avg_token_length,
            "max_token_length": max_token_length,
            "min_token_length": min_token_length,
            "success": True
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def main():
    print("="*80)
    print("TOKENIZER COMPARISON: Cross-Text Tokenization Analysis")
    print("="*80)

    # Load all texts
    print("\nLoading texts...")
    text_contents = {}
    for text_name, text_path in TEXTS.items():
        text_contents[text_name] = load_text(text_path)
        print(f"  ✓ {text_name}: {len(text_contents[text_name])} characters")

    # Perform tokenization
    print("\nTokenizing texts with all tokenizers...")
    results = {}

    for text_name, text_content in text_contents.items():
        text_length = len(text_content)
        results[text_name] = {
            "text_length_chars": text_length,
            "tokenizer_results": {}
        }

        print(f"\n  Processing: {text_name}")
        for tokenizer_name, tokenizer_path in TOKENIZERS.items():
            print(f"    - {tokenizer_name}...", end=" ", flush=True)

            analysis = tokenize_and_analyze(tokenizer_path, text_content)

            if analysis["success"]:
                results[text_name]["tokenizer_results"][tokenizer_name] = {
                    "token_count": analysis["token_count"],
                    "avg_token_length": analysis["avg_token_length"],
                    "max_token_length": analysis["max_token_length"],
                    "min_token_length": analysis["min_token_length"],
                    "compression_ratio": text_length / analysis["token_count"] if analysis["token_count"] > 0 else 0,
                    "tokens_per_char": analysis["token_count"] / text_length if text_length > 0 else 0
                }
                print(f"✓ {analysis['token_count']} tokens")
            else:
                results[text_name]["tokenizer_results"][tokenizer_name] = {
                    "error": analysis.get("error", "Unknown error"),
                    "success": False
                }
                print(f"✗ Error: {analysis.get('error', 'Unknown')}")

    # Generate statistics and find best tokenizers
    print("\n" + "="*80)
    print("RESULTS: Best Tokenizer for Each Text (Least Tokens)")
    print("="*80)

    summary = {}

    for text_name, data in results.items():
        print(f"\n{text_name}:")
        print(f"  Text length: {data['text_length_chars']:,} characters")

        # Filter successful results
        successful_results = {
            k: v for k, v in data['tokenizer_results'].items()
            if v.get('success', True) and 'token_count' in v
        }

        if not successful_results:
            print("  ⚠ No successful tokenizations")
            continue

        # Sort by token count (ascending - least tokens is best)
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1]['token_count']
        )

        best = sorted_results[0]
        summary[text_name] = {
            "best_tokenizer": best[0],
            "best_token_count": best[1]['token_count'],
            "best_avg_token_length": best[1]['avg_token_length'],
            "best_compression_ratio": best[1]['compression_ratio']
        }

        print(f"\n  🏆 Best tokenizer: {best[0]}")
        print(f"     Token count: {best[1]['token_count']:,}")
        print(f"     Avg token length: {best[1]['avg_token_length']:.2f} chars")
        print(f"     Compression ratio: {best[1]['compression_ratio']:.2f} chars/token")
        print(f"     Tokens per char: {best[1]['tokens_per_char']:.4f}")

        print(f"\n  All results (sorted by token count):")
        for rank, (tok_name, tok_data) in enumerate(sorted_results, 1):
            marker = "🏆" if rank == 1 else f"{rank}."
            print(f"    {marker} {tok_name:30s}: {tok_data['token_count']:8,} tokens "
                  f"(avg length: {tok_data['avg_token_length']:5.2f}, "
                  f"compression: {tok_data['compression_ratio']:5.2f})")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Text':<30} {'Best Tokenizer':<30} {'Tokens':>10} {'Avg Token Length':>18}")
    print("-"*80)
    for text_name, data in summary.items():
        print(f"{text_name:<30} {data['best_tokenizer']:<30} "
              f"{data['best_token_count']:>10,} {data['best_avg_token_length']:>18.2f}")

    # Overall best tokenizer (least tokens across all texts)
    print("\n" + "="*80)
    print("OVERALL ANALYSIS")
    print("="*80)

    # Calculate total tokens for each tokenizer across all texts
    tokenizer_totals = {}
    for text_name, data in results.items():
        for tok_name, tok_data in data['tokenizer_results'].items():
            if tok_data.get('success', True) and 'token_count' in tok_data:
                if tok_name not in tokenizer_totals:
                    tokenizer_totals[tok_name] = 0
                tokenizer_totals[tok_name] += tok_data['token_count']

    if tokenizer_totals:
        sorted_totals = sorted(tokenizer_totals.items(), key=lambda x: x[1])
        print(f"\nTokenizer with least total tokens across all texts:")
        print(f"  🏆 {sorted_totals[0][0]}: {sorted_totals[0][1]:,} total tokens")
        print(f"\nAll tokenizers (total tokens across all texts):")
        for tok_name, total in sorted_totals:
            print(f"  {tok_name:30s}: {total:>10,} tokens")

    # Save results to JSON
    output_file = "tokenizer-comparison-results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
