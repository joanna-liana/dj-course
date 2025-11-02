# Zadanie 3

Robimy własny TOKENIZER. Folder: `M1/tokenizer`

Korpusy danych treningowych do wyboru:
- `M1/korpus-nkjp`
- `M1/korpus-wolnelektury`
- `M1/korpus-spichlerz` (Bielik Team)
W repo znajdziesz instrukcje dla 3 różnych korpusów danych treningowych oraz bazowy kod pythonowy.

Zadania:
- stwórz własne tokenizery w oparciu o plik `tokenizer-build.py` (obecna wersja działa ale jest zahardkodowana). Zdynamizuj kod w taki sposób, aby móc dynamicznie tworzyć tokenizery w oparciu o zadane korpusy tekstowe. Stwórz
  - `tokenizer-pan-tadeusz.json` - tylko w oparciu o Pana Tadeusza ("wolnelektury")
  - `tokenizer-wolnelektury.json` - w oparciu o cały korpus "wolnelektury"
  - `tokenizer-nkjp.json` - w oparciu o cały korpus "nkjp"
  - `tokenizer-all-corpora.json` - w oparciu o wszystkie korpusy
- z HuggingFace wybierz LLM i ściągnij jego tokenizer (byle inny niż Mistral-v0.1 - bo to ten sam co Bielik v0.1) i dodaj go do swoich tokenizerów
- w nawiązaniu do sławnego badania ;) (https://arxiv.org/pdf/2503.01996) tokenizujemy różne teksty "na krzyż" różnymi tokenizerami
  - teksty:
    - "Pan Tadeusz, Księga 1" ("wolnelektury")
    - "The Pickwick Papers" (mini korpus / projekt gutenberg)
    - "Fryderyk Chopin" (mini korpus / wikipedia)
  - tokenizery - wszystkie dostępne (3 bielikowe + wybrany z HF + 4 stworzone)
  - zmontuj statystyki, które mają odpowiedzieć na pytanie: **DLA KAŻDEGO TEKSTU, KTÓRY TOKENIZER BYŁ NAJEFEKTYWNIEJSZY POD KĄTEM NAJMNIEJSZEJ ILOŚCI WYNIKOWYCH TOKENÓW?**
- spróbuj osiągnąć taki tokenizer, aby miał jak najdłuższe kawałki słów - to sprawi że embedding w następnym ćwiczeniu będzie mega efektywny
- sprawdź czy dla customowych tokenizerów zmiana rozmiaru słownika (default: `32k`) robi różnicę na wyniki?

# Zadanie 3 - CIEKAWOSTKA

sneak peak tokenizera (pakiet `tiktoken`) w google colab: https://colab.research.google.com/drive/1cbW-Wnn-_mDUEhC6ptrK9Ltd7nvQu6Mu
