"""Quantization tax: does shrinking a model hurt Indian languages more than English?

For each precision (bf16 = full, int8 = half, nf4 = quarter) we measure how surprised
the model is by the same IN22-Gen sentences in each language: negative log-likelihood (NLL)
per character. The tokenizer is identical across precisions, so the relative increase
(nf4 NLL / bf16 NLL - 1) is comparable across languages.

Usage: python quant_tax.py --model sarvamai/sarvam-1 --n 256
Outputs: results_<model>.csv
"""
import argparse
import gc
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# transformers 5 renamed torch_dtype -> dtype; old remote-code models (Param-1) need transformers 4.x.
DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

OUT = Path(__file__).parent
LANGS = {
    "eng_Latn": "English", "hin_Deva": "Hindi", "ben_Beng": "Bengali", "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati", "pan_Guru": "Punjabi", "ory_Orya": "Odia", "tam_Taml": "Tamil",
    "tel_Telu": "Telugu", "kan_Knda": "Kannada", "mal_Mlym": "Malayalam",
}
PRECISIONS = {
    "bf16": None,
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16),
}


@torch.no_grad()
def sentence_nlls(model, tok, sentences, token_budget=4096):
    """Per-sentence (total NLL in nats, character count), for bootstrap confidence intervals.

    Batches are sized by token count (not sentence count) so large-vocab models on
    token-hungry scripts fit in 8 GB, and the loss is computed one row at a time.
    """
    bos = tok.bos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    encoded = [([bos] if bos is not None else []) + tok.encode(s, add_special_tokens=False)
               for s in sentences]
    if max(map(len, encoded)) > 2048:
        raise ValueError("sentence longer than 2048 tokens; refusing to truncate (would bias NLL/char)")

    out, i = [], 0
    while i < len(encoded):
        j = i + 1
        while j < len(encoded) and max(map(len, encoded[i:j + 1])) * (j + 1 - i) <= token_budget:
            j += 1
        ids = encoded[i:j]
        width = max(map(len, ids))
        input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device="cuda")
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device="cuda")
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        for r, x in enumerate(ids):
            n = len(x)
            nll = torch.nn.functional.cross_entropy(
                logits[r, :n - 1].float(), input_ids[r, 1:n], reduction="sum")
            out.append((nll.item(), len(sentences[i + r])))
        del logits
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sarvamai/sarvam-1")
    ap.add_argument("--n", type=int, default=256, help="sentences per language")
    ap.add_argument("--token-budget", type=int, default=4096,
                    help="max tokens per batch; lower it if a model spills out of VRAM")
    ap.add_argument("--precisions", default="bf16,int8,nf4",
                    help="comma-separated subset of bf16,int8,nf4 (bf16 is always needed as the baseline)")
    ap.add_argument("--legacy-rope", action="store_true",
                    help="for older remote-code models (e.g. Param-1): turn transformers 5's auto-filled "
                         "{'rope_type': 'default'} back into rope_scaling=None, which means the same thing")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.legacy_rope and (config.rope_scaling or {}).get("rope_type") == "default":
        config.rope_scaling = None

    ds = load_dataset("ai4bharat/IN22-Gen", split="test").select(range(args.n))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows, sent_rows = [], []
    for prec, qcfg in {p: PRECISIONS[p] for p in args.precisions.split(",")}.items():
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, quantization_config=qcfg, **{DTYPE_KW: torch.bfloat16},
            device_map="cuda", trust_remote_code=True)
        model.eval()
        vram = torch.cuda.memory_allocated() / 2**30
        for code, lang in LANGS.items():
            per_sent = sentence_nlls(model, tok, ds[code], args.token_budget)
            sent_rows += [{"precision": prec, "language": lang, "idx": i, "nll": n, "chars": c}
                          for i, (n, c) in enumerate(per_sent)]
            nll = sum(n for n, _ in per_sent) / sum(c for _, c in per_sent)
            rows.append({"model": args.model, "precision": prec, "language": lang,
                         "nll_per_char": nll, "vram_gb": vram})
            print(f"{prec:>5} {lang:<10} NLL/char={nll:.4f}")
        print(f"{prec}: {vram:.1f} GB VRAM, {time.time() - t0:.0f}s")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    base = df[df.precision == "bf16"].set_index("language").nll_per_char
    df["increase_vs_bf16_pct"] = df.apply(
        lambda r: 100 * (r.nll_per_char / base[r.language] - 1), axis=1)
    name = args.model.split("/")[-1]
    df.to_csv(OUT / f"results_{name}.csv", index=False)
    pd.DataFrame(sent_rows).to_csv(OUT / f"per_sentence_{name}.csv", index=False)

    print("\n% increase in NLL/char vs full precision (higher = more damage):")
    table = df.pivot(index="language", columns="precision", values="increase_vs_bf16_pct")
    print(table[[c for c in ("int8", "nf4") if c in table]].loc[list(LANGS.values())].round(2).to_string())


if __name__ == "__main__":
    main()
