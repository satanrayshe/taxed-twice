"""Tokenizer tax: how many more tokens does each tokenizer need for Indian languages vs English?

Data: AI4Bharat IN22-Gen - 1024 sentences, each translated into all 22 scheduled
Indian languages + English, so every language carries the same meaning.

Metrics per (tokenizer, language):
  parity    = total tokens in language / total tokens in English (same sentences). 1.0 = fair.
  fertility = tokens per whitespace-separated word.

Outputs: results.csv, tokenizer_tax_heatmap.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = Path(__file__).parent

# Hugging Face tokenizers. Llama and Gemma are gated: accept their licences on HF first.
HF_TOKENIZERS = {
    "Sarvam-1 (IN)": "sarvamai/sarvam-1",
    "Sarvam-30B (IN)": "sarvamai/sarvam-30b",
    "Param-1-2.9B (IN)": "bharatgenai/Param-1-2.9B-Instruct",
    "Param-1-7B (IN)": "bharatgenai/Param-1-7B",
    "Param2-17B (IN)": "bharatgenai/Param2-17B-A2.4B-Thinking",
    "Pragna-1B (IN)": "soketlabs/pragna-1b",
    "Llama-3.2": "meta-llama/Llama-3.2-1B",
    "Gemma-3": "google/gemma-3-1b-it",
    "Qwen3": "Qwen/Qwen3-0.6B",
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",
    "GPT-2": "gpt2",
}
# OpenAI tokenizers via tiktoken.
TIKTOKEN = {"GPT-4o (o200k)": "o200k_base", "GPT-4 (cl100k)": "cl100k_base"}

LANG_NAMES = {
    "eng_Latn": "English", "hin_Deva": "Hindi", "ben_Beng": "Bengali", "mar_Deva": "Marathi",
    "tel_Telu": "Telugu", "tam_Taml": "Tamil", "guj_Gujr": "Gujarati", "urd_Arab": "Urdu",
    "kan_Knda": "Kannada", "ory_Orya": "Odia", "mal_Mlym": "Malayalam", "pan_Guru": "Punjabi",
    "asm_Beng": "Assamese", "mai_Deva": "Maithili", "san_Deva": "Sanskrit", "npi_Deva": "Nepali",
    "kas_Arab": "Kashmiri", "snd_Deva": "Sindhi", "gom_Deva": "Konkani", "doi_Deva": "Dogri",
    "brx_Deva": "Bodo", "mni_Mtei": "Manipuri", "sat_Olck": "Santali",
}


def load_encoders():
    encoders = {}
    for name, repo in HF_TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            encoders[name] = lambda s, t=tok: len(t.encode(s, add_special_tokens=False))
            print(f"loaded  {name}")
        except Exception as e:
            print(f"SKIPPED {name}: {type(e).__name__}: {str(e)[:100]}")
    try:
        import tiktoken
        for name, enc_name in TIKTOKEN.items():
            enc = tiktoken.get_encoding(enc_name)
            encoders[name] = lambda s, e=enc: len(e.encode(s))
            print(f"loaded  {name}")
    except Exception as e:
        print(f"SKIPPED tiktoken: {e}")
    return encoders


def main():
    ds = load_dataset("ai4bharat/IN22-Gen", split="test")
    encoders = load_encoders()

    rows = []
    for tok_name, count in encoders.items():
        eng_tokens = sum(count(s) for s in ds["eng_Latn"])
        for code, lang in LANG_NAMES.items():
            sents = ds[code]
            n_tok = sum(count(s) for s in sents)
            n_words = sum(len(s.split()) for s in sents)
            rows.append({"tokenizer": tok_name, "language": lang, "code": code,
                         "tokens": n_tok, "words": n_words,
                         "fertility": n_tok / n_words, "parity": n_tok / eng_tokens})
        print(f"done    {tok_name}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)

    # Heatmap: tokenizers (rows) x languages (cols), sorted by average parity.
    pivot = df.pivot(index="tokenizer", columns="language", values="parity")
    pivot = pivot[[LANG_NAMES[c] for c in LANG_NAMES]]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(16, 0.55 * len(pivot) + 2))
    im = ax.imshow(pivot.clip(upper=10).values, cmap="magma_r", aspect="auto", vmin=1, vmax=10)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=60, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 5 else "black")
    fig.colorbar(im, ax=ax, label="Tokens vs English (1.0 = parity; clipped at 10)")
    ax.set_title("The tokenizer tax: tokens needed per language relative to English\n"
                 "(same 1,024 sentences, AI4Bharat IN22-Gen)")
    fig.tight_layout()
    fig.savefig(OUT / "tokenizer_tax_heatmap.png", dpi=160)

    print("\nAverage parity across 22 Indian languages (lower = fairer):")
    indic = df[df.code != "eng_Latn"]
    print(indic.groupby("tokenizer").parity.mean().sort_values().round(2).to_string())
    print("\nWorst language per tokenizer:")
    worst = indic.loc[indic.groupby("tokenizer").parity.idxmax(), ["tokenizer", "language", "parity"]]
    print(worst.sort_values("parity").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
