"""Bootstrap 95% confidence intervals for quantization damage, per language.

Resamples sentences (paired: the same sentences across precisions) and reports both
absolute ΔNLL/char and % change vs bf16.

Usage: python bootstrap.py per_sentence_sarvam-1.csv
"""
import sys

import numpy as np
import pandas as pd

B = 2000
rng = np.random.default_rng(0)

df = pd.read_csv(sys.argv[1])
rows = []
for lang, g in df.groupby("language", sort=False):
    wide = g.pivot(index="idx", columns="precision", values="nll")
    chars = g[g.precision == "bf16"].set_index("idx").chars.loc[wide.index].values
    n = len(wide)
    samples = rng.integers(0, n, size=(B, n))
    base = wide["bf16"].values[samples].sum(1) / chars[samples].sum(1)
    for prec in [p for p in ["int8", "nf4"] if p in set(df.precision)]:
        q = wide[prec].values[samples].sum(1) / chars[samples].sum(1)
        abs_d, pct = q - base, 100 * (q / base - 1)
        point_base = wide["bf16"].sum() / chars.sum()
        point_q = wide[prec].sum() / chars.sum()
        rows.append({
            "language": lang, "precision": prec,
            "abs_delta": point_q - point_base,
            "abs_lo": np.percentile(abs_d, 2.5), "abs_hi": np.percentile(abs_d, 97.5),
            "pct": 100 * (point_q / point_base - 1),
            "pct_lo": np.percentile(pct, 2.5), "pct_hi": np.percentile(pct, 97.5),
        })

out = pd.DataFrame(rows)
out.to_csv(sys.argv[1].replace("per_sentence_", "ci_"), index=False)
for prec in ["int8", "nf4"]:
    print(f"\n{prec}: damage vs bf16 (95% CI)")
    for _, r in out[out.precision == prec].iterrows():
        print(f"  {r.language:<10} abs {r.abs_delta:+.4f} [{r.abs_lo:+.4f}, {r.abs_hi:+.4f}]   "
              f"pct {r.pct:+5.2f}% [{r.pct_lo:+5.2f}, {r.pct_hi:+5.2f}]")
