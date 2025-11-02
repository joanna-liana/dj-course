from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import CORPORA_FILES

TOKENIZER_CONFIGS = [
    ("tokenizers/tokenizer-pan-tadeusz.json", "PAN_TADEUSZ"),
    ("tokenizers/tokenizer-wolnelektury.json", "WOLNELEKTURY"),
    ("tokenizers/tokenizer-nkjp.json", "NKJP"),
    ("tokenizers/tokenizer-all-corpora.json", "ALL"),
]

for TOKENIZER_OUTPUT_FILE, corpus_key in TOKENIZER_CONFIGS:
    print(f"\Building {TOKENIZER_OUTPUT_FILE} from corpus {corpus_key}")

    # 1. Initialize the Tokenizer (BPE model)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2. Set the pre-tokenizer (e.g., split on spaces)
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Set the Trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=32000,
        min_frequency=2
    )

    # Get files from corpus
    FILES = [str(f) for f in CORPORA_FILES[corpus_key]]
    print(f"File count: {len(FILES)}")

    # 4. Train the Tokenizer
    tokenizer.train(FILES, trainer=trainer)

    # 5. Save the vocabulary and tokenization rules
    tokenizer.save(TOKENIZER_OUTPUT_FILE)
    print(f"Zapisano: {TOKENIZER_OUTPUT_FILE}")

    # Test tokenizer
    for txt in [
        "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
        "Jakże mi wesoło!",
        "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
    ]:
        encoded = tokenizer.encode(txt)
        print("Zakodowany tekst:", encoded.tokens)
        print("ID tokenów:", encoded.ids)
