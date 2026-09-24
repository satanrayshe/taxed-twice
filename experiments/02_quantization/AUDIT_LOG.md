# Audit Log: Quantization Tax Study

**Purpose:** a complete, honest record of how every result was produced, every change made along the way, every mistake caught, and every known weakness. If a number goes in the paper, this file must explain where it came from.
**Created:** 2026-09-24 (reconstructed from run logs and verified against files on disk). **Keep this updated after every run.**

---

## 1. Environment (full detail in `env/`)
- **Hardware:** RTX 5060 Laptop GPU 8 GB (driver 610.88), Ryzen 9 8940HX, 16 GB RAM, Windows 11 with Smart App Control ON.
- **Main env (`.venv`):** torch 2.11.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2, datasets 5.0.1, pandas 2.2.3 (pinned: 3.x DLL blocked by Smart App Control). Full list: `env/main_venv_freeze.txt`.
- **Legacy env (`.venv-legacy`):** torch 2.11.0+cu128, **transformers 4.46.3**, bitsandbytes 0.50.2. Used ONLY for Param-1. Full list: `env/legacy_venv_freeze.txt`.
- **Model revisions** (HF commit SHAs as of 2026-09-24): `env/environment.txt`. Caveat: those are the current remote SHAs; the cached files were downloaded 2026-09-24, so a same-day match is very likely but not proven per file.
- **Dataset:** ai4bharat/IN22-Gen @ e042ab3d, **licence CC-BY-4.0** (verified on the dataset card). Gated: access was accepted on HF by the author.

## 2. Method (as actually run)
- **Data:** IN22-Gen `test` split, **first 256 sentences** (rows 0–255), 11 languages: English + hi, bn, mr, gu, pa, or, ta, te, kn, ml. The same sentences are used in every language (parallel).
- **Metric:** NLL per character = Σ token NLL (nats) over a sentence ÷ number of characters in that sentence, summed over sentences. BOS is prepended when the tokenizer defines one (Qwen has no BOS, so its first token is unscored, identically across precisions).
- **Precisions:** bf16 (baseline); int8 = bitsandbytes LLM.int8 (default threshold 6.0); nf4 = bitsandbytes 4-bit NF4, compute dtype bf16, no double quantization.
- **Damage:** abs Δ = NLL/char(quantized) − NLL/char(bf16), per language; % Δ also reported. **Primary metric = absolute Δ** (see mistake M1).
- **CIs:** `bootstrap.py`, 2000 paired bootstrap resamples over sentences, seed 0, percentile 95% CI.
- **Tokenizer tax (parity):** tokens(language) ÷ tokens(English) over the SAME 256 sentences (`parity256_vs_damage.csv`). Correlations are Spearman rank.
- **No truncation:** the script refuses sentences over 2048 tokens rather than truncate (see mistake M3).

## 3. Run table (final results used)
| Model | HF repo | Env | Precisions | token_budget | Notes |
|---|---|---|---|---|---|
| Sarvam-1 (2.5B) | sarvamai/sarvam-1 | main | bf16/int8/nf4 | 4096 | final run with fixed scoring (M3); pilot kept as `results_sarvam-1_pilot.csv` |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | main | bf16/int8/nf4 | 4096 | first attempt OOM'd (M3) |
| Llama-3.2-3B | meta-llama/Llama-3.2-3B | main | bf16/int8/nf4 | 1024 | first attempt killed: VRAM spill to shared memory |
| Gemma-3-1B | google/gemma-3-1b-pt | main | bf16/int8/nf4 | 2048 | |
| Qwen3-1.7B | Qwen/Qwen3-1.7B-Base | main | bf16/int8/nf4 | 512 | first attempt killed by the job runner for low system RAM (other applications open) |
| Pragna-1B | soketlabs/pragna-1b | main | bf16/int8/nf4 | 512 | native Llama architecture; bf16 NLL sane |
| Param-1 (2.9B) | bharatgenai/Param-1-2.9B-**Instruct** | **legacy** | **bf16/nf4 only** | 1024 | **int8 skipped** (too slow in the legacy env); INSTRUCT checkpoint (no base on HF); broken on transformers 5 (M4) |

**Why token_budget differs:** only batching and memory differ. Each sentence is scored in full. Verified: switching batching changed Sarvam-1 totals by <0.1% (bf16 numerics), versus the measured effects of 1–25%. Within a model, all precisions use the same budget.

## 4. Verified claims (checked against primary sources on 2026-09-24)
| Claim | Source | Status |
|---|---|---|
| Llama 3.2 officially supports Hindi (8 languages) and was trained on more | Llama-3.2-1B model card | ✅ |
| Qwen3: 36T tokens, 119 languages | Qwen3-1.7B-Base model card | ✅ |
| Gemma 3 training data covers 140+ languages (no 1B-specific exception) | gemma-3-1b-pt model card | ✅ (corrected an earlier wrong claim, M2) |
| Sarvam-1 optimised for 10 Indic languages (bn gu hi kn ml mr or pa ta te) + English | sarvam-1 model card | ✅ |
| Pragna-1B languages: Hindi, Bangla, Gujarati, English | pragna-1b model card | ✅ |
| Param-1-2.9B is an English–Hindi model | arXiv 2507.13390 abstract ("bilingual dataset consisting of only Hindi and English") | ✅ verified 2026-09-25 |
| Marchisio et al. 2024 (arXiv 2407.03211): non-Latin scripts hit worst; automatic metrics severely underestimate harm | arXiv abstract + author list | ✅ verified 2026-09-25 (venue: EMNLP 2024 Findings per a secondary source; confirm before camera-ready) |

## 5. Mistakes made and corrected (be transparent)
- **M1: Metric framing (caught by the author).** The pilot reported % increase and claimed "English hurt most" on Sarvam-1. % inflates languages with a low baseline NLL. In absolute Δ, English ties the worst Indic languages. **Fix:** absolute Δ is primary, both reported.
- **M2: Unverified model-card claim.** I said Gemma-3-1B was English-heavy; its card says 140+ languages. **Fix:** corrected before any write-up; Gemma is not treated as a clean data/tokenizer disentangler.
- **M3: Silent 512-token truncation + OOM.** v1 of `quant_tax.py` truncated sentences at 512 tokens but counted all characters, which would have biased NLL/char DOWN for token-hungry scripts. Llama-1B OOM'd on fp32 full-vocab log-softmax. **Fix:** token-budget batching, per-row cross-entropy, no truncation (hard error > 2048). All final results come from the fixed version (Sarvam-1 rerun).
- **M4: Param-1 silently broken on transformers 5.** It loaded with no missing weights, but loss was ~11 nats/token (near random) and it predicted "Question" after "The capital of India is". A `--legacy-rope` patch did not fix it. **Fix:** a separate env with transformers 4.46.3 gives loss 2.39 and predicts "Delhi". **Rule added:** sanity-check bf16 NLL (≈0.6–1.2 nats/char for supported languages) and a known completion before trusting any new model.
- **M5: Parity mismatch.** The Project 1 parity was computed on 1024 sentences while damage used 256. **Fix:** recomputed on the same 256 (`parity256_vs_damage.csv`). Results unchanged (pooled ρ 0.708; within-model identical except Sarvam .35 → .32).

- **M6: Translation-direction confound (Phase C design).** I chose Indic→English because it's cheap (English output). Result (5 models, n=128, `mt_summary.csv`): chrF++ drops are real (22/25 CIs exclude 0, typically 2–5 points), but they barely track the NLL damage (Spearman +0.08; +0.48 excluding 5 floor cases with bf16 chrF < 20). The per-model mean drop follows each model's **English** NLL damage (Qwen 5.1/0.061, Sarvam 4.2/0.048, Llama-1B 3.4/0.014, Pragna 2.4/0.022, Gemma 2.0/0.016), i.e. this direction mostly measures **output-language (English) damage**, not Indic comprehension. **Fix planned:** English→Indic (output in Indic), fewer sentences. The Indic→En result stays in the paper as a secondary finding.
- **Process note:** the Phase C run was killed by the job runner for low system RAM (other applications open) during Llama-3.2-3B (1/10 done). 5/6 models are complete. Not restarted without the author's go-ahead.
- **Analysis bug (caught immediately, no effect on outputs):** `r.drop` resolved to the pandas `.drop` method in a quick analysis snippet; fixed by using `r["drop"]`. Lesson: avoid column names that shadow DataFrame methods.

- **translate_eval.py v2 (2026-09-24):** added `--direction en2x`; the prompt builder is generalized; max_new_tokens = max(96, 1.5 × reference token count + 16) (only the reference LENGTH is used); stop at newline (saves time; output was already cut at the first newline). **Regression test:** x2en Gemma-1B bf16, first 16 Hindi sentences → 16/16 identical to the v1 outputs. **Process note:** my first scripted patch mangled two lines (unapplied function signature + a literal newline inside a string); caught by review and fixed by hand before any run.
- **Phase C part 2 started:** en2x, n=64, 6 models; then Llama-3.2-3B x2en n=128 rerun (the author closed other applications; 7.8 GB RAM free).
- Observation: in en2x smoke tests, Gemma-1B bf16 Odia output degenerates into repetition. That's a real model limitation. Floor/degeneration cases must be flagged in the analysis (as for x2en bf16 chrF < 20).

- **en2x results, 4/6 models (killed for low RAM during Qwen at 7/10; other applications reopened, ~2.6 GB).** `mt_en2x_summary.csv`, n=64. 16/20 drops have CIs excluding 0. **Floor effects dominate:** 10/20 cells have bf16 chrF < 20 (Llama-1B non-Hindi 8–14; Pragna ta/te/or 1–8; Gemma Odia 5.7, degenerate repetition). All 20: Spearman(drop, NLL dmg) −0.08, (drop, parity) +0.11. Non-floor n=10: +0.52 / +0.65 (supportive, too few to rely on).
- **Sarvam discrepancy:** tiny NLL damage (≤0.05) but mean chrF drop ~4 in BOTH directions (Bengali en2x −9.1). Likely teacher-forced NLL vs free generation (errors compound). **The NLL metric understates generation damage for Sarvam.**
- **Paper implication (decided):** state the main claim narrowly: parity predicts *likelihood (teacher-forced) damage* (ρ 0.71, 77 pts). Generation results are secondary and honestly mixed: damage is real (36/45 CIs exclude 0 across both directions), but its size is only loosely predicted at 1–3B, owing to floor effects plus compounding. **Do NOT claim parity predicts translation damage.** Future work: 7–8B models on Colab, where bf16 Indic generation clears the floor.

- **Remaining-runs attempt (2026-09-24, other applications confirmed closed, 7.5 GB free):** Qwen en2x finished (`mt_en2x_Qwen3-1.7B-Base.csv`; drops Hi 6.4, Bn 5.1, Ta 2.9, Te 3.6, Or 4.3; bf16 chrF 13–27, so mostly floor). **Killed again for low RAM right at the start of Llama-3.2-3B.** **Correction:** I had attributed the earlier Llama-3B kill to other applications. With other applications closed it happened again, so the likely cause is Llama-3B itself (bf16 ≈6 GB + batch-8 generation of long, token-hungry Indic outputs → KV cache exceeds 8 GB VRAM → Windows spills to shared/system RAM). Fix options: batch size 2, or Colab.

- **Paper draft v0.1 (2026-09-25):** `paper/paper.tex`, `paper/refs.bib`. All numbers come from `paper/analysis.py` → `stats.json` / `tables.tex` (within-model permutation test with 10k shuffles; fixed-effects slope with bootstrap CI). All 14 citations verified via the arXiv API (`env/citations_verified_arxiv.txt`) or the model card. Param-1 = Hindi + English only is now ✅ verified (arXiv 2507.13390 abstract).
- **M7: Missed prior work + too-similar title.** Srivastava (2026) "The Tokenizer Tax" (arXiv 2607.24276), was not cited, and the draft title "The Tokenizer Tax Compounds" was too close to it. Caught during a literature check. **Fix:** cited, positioned (they measure the tax; we measure its cost under compression), title changed. Also added Brahma et al. 2025 (arXiv 2506.17789).
- **M8: Claim audit of the draft.** Fixed: Pragna high-parity range was written as 3.4–4.3 but is 3.35–4.48 on the 256 sentences; an unsupported claim ("most people in India run quantized models"); and an unverified claim that Pragna targets tokenizer efficiency (removed).

- **Build:** MiKTeX 25.12 installed (winget, user scope) + official ACL style files (acl-org/acl-style-files@master). `paper.pdf` compiles to 4 pages with 0 errors/warnings. Note: run bibtex from PowerShell (git-bash gets "Permission denied").
- **M9:** the unsupported "in India models are usually deployed quantized" claim was removed from the Intro but was still in the Abstract. Caught on the PDF read-through and fixed. Also fixed: the table overflowed its column (resizebox), and the Pragna-1B title was lowercased by BibTeX.

- **M11: External draft audit (2026-09-25) triaged claim by claim.**
  - **False positives (checked against source + PDF render):** Δ→"A", missing ρ header, missing † markers, dropped → arrows, "c1100k_base", "Pragna-IB", and stray text under §2. These are text-extraction/OCR artifacts; the "stray text" is the figure's embedded vector text.
  - **Real, fixed:**
    - §4 now names the 5 translation models and 5 languages and gives the reasons Llama-3B and PARAM-1 were excluded; the shots are rows 1000–1002, disjoint from the test rows.
    - The slope is restated per doubling of parity (0.077, CI 0.058–0.101; natural log clarified), and `stats.json` now stores it.
    - Naming is unified to PARAM-1-2.9B.
    - chrF++ (Popović 2017) and SacreBLEU (Post 2018) are cited, with the signature in the appendix.
  - **Venues (verified by an agent on NeurIPS proceedings / ACL Anthology / arXiv):**
    - Now cited at published venues: LLM.int8 (NeurIPS 2022, published title "GPT3.int8()"), QLoRA and Petrov (NeurIPS 2023), Ahia (EMNLP 2023 main, pp. 9904–9923), Marchisio (Findings EMNLP 2024), Rust (ACL-IJCNLP 2021), IndicTrans2 (TMLR 2023), Brahma (Findings ACL 2026).
    - **The audit's claims were wrong on 4 items:** Ahia is main conference (not Findings) with different pages; IndicTrans2 is TMLR (not ACL Findings); Grattafiori-first is correct for the current Llama-3 arXiv version (Dubey was first in v1 only); the LLM.int8 NeurIPS title differs.
  - **TODO:** confirm the IndicTrans2 TMLR year on OpenReview.
  - **Double-blind:** a conference submission needs the `[review]` option + an anonymized repo; arXiv/emails keep `[final]`.

## 6. Headline numbers (as of 2026-09-24, nf4 abs Δ NLL/char)
- Pooled Spearman(parity256, abs Δ), 7 models × 11 languages = 77 points: **0.708**.
- Indic only, Δ minus that model's English Δ: **0.648**.
- Within-model ρ: Llama-1B .97, Llama-3B .96, Param-1 .93, Gemma .85, Pragna .67, Qwen .51, Sarvam .32.
- Per-model numbers: `ci_<model>.csv`; summary in PLAN.md.

## 7. Threats to validity (must appear in the paper's Limitations)
1. **Proxy metric.** NLL/char is not task quality → Phase C translation eval (running).
2. **Cross-script NLL/char is not directly comparable** (information per character differs by script), so the comparisons are within-language Δ only.
3. **Parity is confounded with training-data share.** The evidence is correlational. The best disentangling evidence is Pragna (untrained low-parity Tamil/Kannada/Marathi are hurt less than untrained high-parity languages), but n is small.
4. **One quantization family** (bitsandbytes int8/nf4). GGUF/AWQ/GPTQ are untested.
5. **Small models only** (1–3B). Larger models may behave differently (Llama 1B → 3B shrank damage ~3×).
6. **Param-1 differences:** instruct checkpoint, different library version, no int8.
7. **One dataset / domain** (IN22-Gen general-domain sentences, first 256).
8. **Bootstrap CIs capture sentence sampling only**, not run-to-run or model-to-model variance. Runs are deterministic (greedy scoring, fixed data).
9. **Scoring used right-padding with an attention mask**; per-row loss only covers real tokens (causal, so padding after the sequence can't leak).
10. **Tokenizer-only models (Sarvam-30B, Param2, etc.) are not part of the quantization study.**

## 8. AI-assistance disclosure (for the paper)
Code, analysis scripts and drafts were produced with an AI assistant (Claude) under the author's direction. The author must review and understand every step. Venues such as ACL require disclosure of AI assistance, and it must be stated in the paper.
