# Tokenizer Tax: Findings (run 2026-09-23)

**Setup:** 13 tokenizers × 23 languages (22 scheduled + English). The text is the same 1,024 sentences from AI4Bharat IN22-Gen.
**Metric:** parity = tokens in the language ÷ tokens in English for the same sentences. 1.0 means equal to English.
**Files:** `results.csv` and `tokenizer_tax_heatmap.png`. To reproduce, run `python tokenizer_tax.py`.

## Headline numbers (average parity over the 22 Indian languages)
| Tokenizer | Avg parity |
|---|---|
| Param2-17B (BharatGen, 2026) | **1.48** |
| Param-1-7B (BharatGen, 2026) | 1.57 |
| Sarvam-30B (Sarvam, 2026) | 1.67 |
| Pragna-1B (Soket) | 2.31 |
| Gemma-3 (Google) | 2.33 |
| Sarvam-1 (Sarvam, 2024) | 2.81 |
| GPT-4o (o200k) | 3.10 |
| Param-1-2.9B (BharatGen, 2025) | 4.58 |
| Qwen2.5 / Qwen3 (identical) | 5.65 |
| Llama-3.2 (Meta) | 5.68 |
| GPT-4 (cl100k) | 6.69 |
| GPT-2 | 9.57 |

## Observations
1. **The 2026 Indian tokenizers are close to fair.** Param2, Param-1-7B and Sarvam-30B cost about 1.5× English on average. GPT-4o costs 3.1× and Qwen 5.7×.
2. **Manipuri (Meitei script) and Santali (Ol Chiki script) are left out almost everywhere.** They cost 10–13× English on GPT-4o, Qwen, and even Sarvam-1. **Only Sarvam-30B** handles both (1.5 / 1.7).
3. **Script gaps in the Indian models:**
   - Sarvam-1 is poor on Arabic-script languages: Urdu 6.8, Kashmiri 7.3.
   - Param-1-2.9B is poor on Gujarati 11.6, Odia 13.6, Punjabi 11.0. Its tokenizer seems to skip those scripts, although its successors fix this.
4. **GPT-4o has an Odia anomaly.** Odia costs 5.2×, while every other major script is about 2×.
5. **Gemma-3 is the best global tokenizer (2.33×).** It beats Sarvam-1 (2.81×) and GPT-4o (3.10×). Llama-3.2 (5.68×) is as bad as Qwen.
6. **Qwen3 reuses Qwen2.5's tokenizer unchanged,** so it made no Indic improvement.

## Caveats (state these when you post)
- Parity measures **cost and efficiency** (price, speed, how much text fits in the context window). It does **not** measure quality directly.
- Tokenizer comparisons for Indian languages already exist (arXiv 2411.12240, 2607.24276, 2607.23319). What's new here is the **2026 Indian tokenizers** (Sarvam-30B, Param-1-7B, Param2) measured on one parallel benchmark.
- Whitespace "words" differ between languages, which is why the headline metric is parity and not fertility.

## Next
- Add BrahmicTokenizer and Krutrim.
- Build an HF Space with an interactive heatmap, then post on X/LinkedIn tagging @SarvamAI, BharatGen and AI4Bharat.
- Link cost to quality: does a higher tokenizer tax predict lower MILU accuracy?
