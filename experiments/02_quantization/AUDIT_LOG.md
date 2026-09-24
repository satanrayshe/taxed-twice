# Audit Log: Taxed Twice

A record of how every result in the paper was produced, the issues found along the way and how each was resolved, and the known limitations. Any number in the paper can be traced to a file in this repository.

---

## 1. Environment
Full detail is in `env/`.

- **Hardware:** one laptop. NVIDIA RTX 5060 Laptop GPU (8 GB, driver 610.88), AMD Ryzen 9 8940HX, 16 GB RAM, Windows 11.
- **Main environment:** Python 3.12, torch 2.11.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2, datasets 5.0.1, sacrebleu 2.6.0, pandas 2.2.3. Full list: `env/main_venv_freeze.txt`.
- **Legacy environment** (PARAM-1 only): transformers 4.46.3, otherwise the same. Full list: `env/legacy_venv_freeze.txt`.
- **Model and dataset revisions:** Hugging Face commit hashes from 2026-09-24 are in `env/environment.txt`.
- **Dataset:** ai4bharat/IN22-Gen @ e042ab3d. Licence CC BY 4.0. Gated on Hugging Face.

## 2. Method as run
- **Data:** IN22-Gen `test` split, first 256 sentences (rows 0–255). Languages: English, Hindi, Bengali, Marathi, Gujarati, Punjabi, Odia, Tamil, Telugu, Kannada, Malayalam. The same sentences are used in every language.
- **Metric:** negative log-likelihood per character (NLL/char). For each sentence, token NLLs (nats, teacher-forced) are summed. The total over all sentences is then divided by the total character count. A BOS token is prepended when the tokenizer defines one (Qwen has none, so its first token is unscored at every precision alike). Sentences are never truncated: the script stops with an error on any sentence over 2048 tokens.
- **Precisions:**
  - bf16 (baseline)
  - int8: bitsandbytes LLM.int8, outlier threshold 6.0
  - NF4: bitsandbytes 4-bit, bf16 compute, no double quantization
- **Damage:** absolute change Δ = NLL/char(quantized) − NLL/char(bf16), per language. Relative change is also stored.
- **Confidence intervals:** `bootstrap.py`, 2,000 paired resamples over sentences, seed 0, percentile 95% CI.
- **Tokenizer parity:** tokens(language) ÷ tokens(English) over the same 256 sentences (`parity256_vs_damage.csv`).
- **Statistics:** `paper/analysis.py` computes
  - pooled Spearman ρ, tested by permuting parity within each model (10,000 permutations)
  - a model fixed-effects slope on natural-log parity, with a CI from resampling languages within models
- **Translation** (`translate_eval.py`):
  - 3-shot prompts using IN22-Gen rows 1000–1002, disjoint from the test rows; greedy decoding
  - Indic→English on 128 sentences and English→Indic on 64 sentences
  - Hindi, Bengali, Tamil, Telugu, Odia
  - scored with chrF++ via SacreBLEU, signature `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`

## 3. Runs used in the paper
| Model | Hugging Face repo | Environment | Precisions | Token budget | Notes |
|---|---|---|---|---|---|
| Sarvam-1 (2.5B) | sarvamai/sarvam-1 | main | bf16 / int8 / NF4 | 4096 | the pilot run (`results_sarvam-1_pilot.csv`) used an earlier scoring version and is superseded; see issue 3 |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | main | bf16 / int8 / NF4 | 4096 | |
| Llama-3.2-3B | meta-llama/Llama-3.2-3B | main | bf16 / int8 / NF4 | 1024 | smaller batches to stay inside 8 GB of GPU memory |
| Gemma-3-1B | google/gemma-3-1b-pt | main | bf16 / int8 / NF4 | 2048 | |
| Qwen3-1.7B | Qwen/Qwen3-1.7B-Base | main | bf16 / int8 / NF4 | 512 | |
| Pragna-1B | soketlabs/pragna-1b | main | bf16 / int8 / NF4 | 512 | |
| PARAM-1 (2.9B) | bharatgenai/Param-1-2.9B-Instruct | legacy | bf16 / NF4 | 1024 | instruct checkpoint (no base model is public); int8 not run (very slow in the legacy environment); see issue 4 |

The token budget changes only how sentences are batched. Each sentence is always scored in full. Changing the batching moved Sarvam-1 totals by less than 0.1% (bf16 rounding), against measured effects of 1–25%. Within a model, every precision uses the same budget.

Some runs were interrupted by low system memory and restarted. Translation with Llama-3.2-3B was excluded because bf16 generation of long Indic outputs exceeded 8 GB of GPU memory.

## 4. Verified claims (primary sources, 2026-09-24/25)
| Claim | Source | Status |
|---|---|---|
| Llama 3.2 officially supports 8 languages incl. Hindi; trained on more | Llama-3.2-1B model card | ✅ |
| Qwen3: 36T tokens, 119 languages | Qwen3-1.7B-Base model card | ✅ |
| Gemma 3 training data covers 140+ languages (no 1B-specific exception) | gemma-3-1b-pt model card | ✅ |
| Sarvam-1 targets 10 Indic languages + English | sarvam-1 model card | ✅ |
| Pragna-1B languages: Hindi, Bangla, Gujarati, English | pragna-1b model card | ✅ |
| PARAM-1 is trained on Hindi and English only | arXiv 2507.13390 | ✅ |
| Every bibliography entry (authors, title, venue, pages) | arXiv API, NeurIPS proceedings, ACL Anthology (`env/citations_verified_arxiv.txt`) | ✅ (IndicTrans2's TMLR year still to be confirmed on OpenReview) |

## 5. Issues found and how they were resolved
| # | Issue | How it was detected | Resolution | Effect on reported results |
|---|---|---|---|---|
| 1 | Pilot results used **relative (%) change**, which inflates languages with a low baseline NLL. It made English look like the most-damaged language on Sarvam-1. | Review of the pilot results | Absolute Δ became the primary metric. Both are stored. | The pilot interpretation was withdrawn. In absolute terms English ties the most-affected Indic languages. |
| 2 | A working assumption that Gemma-3-1B was trained mainly on English | Verification against the model card (140+ languages) | Corrected. Gemma is not treated as a clean way to separate tokenizer effects from training-data effects. | None on the numbers |
| 3 | The first scoring version **truncated sentences at 512 tokens** while counting all characters, which would understate NLL/char for token-hungry scripts. It also ran out of memory on large vocabularies. | An out-of-memory failure on Llama-3.2-1B, then code review | Rewritten: token-budget batching, per-sentence cross-entropy, and no truncation (the script stops with an error instead). All reported runs use the fixed version. | The Sarvam-1 pilot was superseded by a rerun |
| 4 | **PARAM-1's custom code misbehaves on transformers 5.** It loads without errors but predicts almost at random (loss ≈11 nats/token; completes "The capital of India is" with "Question"). | A sanity check of bf16 loss and a known completion before trusting results | Run in a separate environment with transformers 4.46.3 (loss 2.39; completes with "Delhi"). Every new model is now sanity-checked this way first. | No invalid numbers entered the results |
| 5 | Parity was first computed on 1,024 sentences, while damage used 256 | Consistency check of inputs | Parity recomputed on the same 256 sentences | None: pooled ρ 0.708 either way; Sarvam within-model ρ 0.35 → 0.32 |
| 6 | Indic→English translation mostly measures damage to the **English output**, not to Indic understanding | Analysis: per-model drops followed each model's English NLL damage | Added English→Indic and reported both directions. The paper's generation claims are stated as secondary. | Claims narrowed (see §6) |
| 7 | Related work **missed Srivastava (2026), "The Tokenizer Tax"**, and the working title was close to theirs | Literature check | Cited and positioned (they measure the tax; this work measures its cost under compression). Title changed. Brahma et al. added. | None on the numbers |
| 8 | Draft claims that didn't match the data or lacked a source: a parity range (3.4–4.3, actually 3.35–4.48), two unsupported sentences, one of which also sat in the abstract | Claim-by-claim check against `stats.json` and sources, plus a PDF read-through | Corrected or removed | None on the numbers |
| 9 | An external review of the draft | Triaged claim by claim against the source, the PDF and primary sources | Adopted: translation setup details, per-doubling slope, naming, chrF++/SacreBLEU citations, published venues for 7 references. Not adopted after verification: 4 citation suggestions contradicted by the venues' own pages, and rendering issues that came from text extraction rather than the PDF. | None on the numbers |

Minor, no effect on outputs:
- a quick analysis snippet used a column name that clashed with a pandas method (fixed)
- a code change to `translate_eval.py` was regression-tested before use: 16/16 outputs identical to the earlier version

## 6. Headline results (NF4, absolute Δ NLL/char)
- **Pooled:** Spearman ρ(parity, damage) over 77 model–language points = **0.708** (within-model permutation p < 10⁻⁴).
- **Indic only, after subtracting each model's English damage:** ρ = **0.648** (p < 10⁻⁴).
- **Fixed-effects slope:** **0.077 nats/char per doubling of parity** (95% CI 0.058–0.101).
- **Within-model ρ:** Llama-3.2-1B .97, Llama-3.2-3B .96, PARAM-1 .93, Gemma-3-1B .85, Pragna-1B .67, Qwen3-1.7B .51, Sarvam-1 .32.
- **Translation (secondary):**
  - The drop's 95% CI excludes zero in 43 of 50 model–language–direction cells.
  - The size of the drop is only loosely predicted by likelihood damage at 1–3B, owing to floor effects.
  - Sarvam-1's generation degrades more than its likelihood suggests.
- All values are in `paper/stats.json` and `ci_<model>.csv`.

## 7. Limitations
1. The main metric is teacher-forced likelihood. Generation results only partly follow it.
2. NLL/char is not comparable across scripts, so only within-language changes are compared.
3. Parity is confounded with training-data share. Pragna-1B is the only disentangling case.
4. One quantization family (bitsandbytes). GGUF, GPTQ and AWQ are untested.
5. Models are 1–3B. Mean Indic damage shrank 3.7× from Llama-3.2-1B to -3B (0.275 → 0.075), so larger models may differ.
6. PARAM-1 is an instruct checkpoint run in an older library version, with no int8 or translation run.
7. One dataset (general-domain IN22-Gen, first 256 sentences).
8. Bootstrap CIs reflect sentence sampling only. Scoring is deterministic.
9. Translation covers five models, and many English→Indic cells are near the floor.

## 8. AI assistance
The author designed the study and made its decisions with substantial help from an AI assistant (Claude, Anthropic) for code, analysis and drafting. This is also stated in the paper.
