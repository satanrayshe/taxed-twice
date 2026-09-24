"""Phase C: does quantization damage show up in a REAL task (translation), not just NLL?

Directions:
  x2en  Indic -> English: output is English, so this mostly measures damage to ENGLISH generation
        (see AUDIT_LOG M6).
  en2x  English -> Indic: output is Indic, so it measures damage to generating token-hungry scripts.
Scores are only compared within a direction (chrF++ on different output scripts isn't comparable).
Few-shot prompt (3 fixed examples from IN22-Gen rows 1000-1002), greedy decoding,
scored with chrF++ (sacrebleu) against the IN22-Gen reference in the target language.

Usage: python translate_eval.py --model meta-llama/Llama-3.2-3B --n 64 [--direction en2x]
Output: mt_<model>.csv (x2en) or mt_en2x_<model>.csv (en2x), one row per sentence per precision
"""
import argparse
import gc
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUT = Path(__file__).parent
LANGS = {"hin_Deva": "Hindi", "ben_Beng": "Bengali", "tam_Taml": "Tamil",
         "tel_Telu": "Telugu", "ory_Orya": "Odia"}
PRECISIONS = {
    "bf16": None,
    "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16),
}
SHOTS = [1000, 1001, 1002]


def build_prompt(ds, src_code, src_name, tgt_code, tgt_name, src):
    lines = [f"{src_name}: {ds[i][src_code]}\n{tgt_name}: {ds[i][tgt_code]}\n" for i in SHOTS]
    lines.append(f"{src_name}: {src}\n{tgt_name}:")
    return "\n".join(lines)


@torch.no_grad()
def translate(model, tok, prompts, refs, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to("cuda")
        # Length budget: at least 96 new tokens (what the first x2en runs used), raised to
        # 1.5x the reference's token count in THIS tokenizer + 16 so token-hungry scripts are
        # not cut off in en2x. Only the reference LENGTH is used, never its content.
        budget = max(96, max(int(1.5 * len(tok.encode(r, add_special_tokens=False))) + 16
                             for r in refs[i:i + batch_size]))
        # Stopping at a newline only saves time: output is cut at the first newline anyway.
        gen = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.pad_token_id, stop_strings=["\n"], tokenizer=tok)
        for row in gen[:, enc.input_ids.shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True)
            outs.append(text.strip().split("\n")[0].strip())  # stop at the end of the line
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--direction", choices=["x2en", "en2x"], default="x2en")
    args = ap.parse_args()

    ds = load_dataset("ai4bharat/IN22-Gen", split="test")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for prec, qcfg in PRECISIONS.items():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qcfg, dtype=torch.bfloat16, device_map="cuda").eval()
        for code, lang in LANGS.items():
            src_code, src_name, tgt_code, tgt_name = (
                (code, lang, "eng_Latn", "English") if args.direction == "x2en"
                else ("eng_Latn", "English", code, lang))
            srcs = ds.select(range(args.n))[src_code]
            refs = ds.select(range(args.n))[tgt_code]
            prompts = [build_prompt(ds, src_code, src_name, tgt_code, tgt_name, s) for s in srcs]
            hyps = translate(model, tok, prompts, refs, args.batch_size)
            for i, (h, r) in enumerate(zip(hyps, refs)):
                rows.append({"precision": prec, "language": lang, "idx": i, "hyp": h, "ref": r,
                             "chrf": sacrebleu.sentence_chrf(h, [r], word_order=2).score})
            corpus = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
            print(f"{prec:>5} {args.direction} {lang:<8} chrF++={corpus:5.1f}   e.g. {hyps[0][:70]!r}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    prefix = "mt" if args.direction == "x2en" else "mt_en2x"
    df.to_csv(OUT / f"{prefix}_{args.model.split('/')[-1]}.csv", index=False)
    t = df.groupby(["language", "precision"]).chrf.mean().unstack()
    t["drop"] = t["bf16"] - t["nf4"]
    print("\nmean sentence chrF++ (higher = better) and drop from bf16 to nf4:")
    print(t.loc[list(LANGS.values())].round(1).to_string())


if __name__ == "__main__":
    main()
