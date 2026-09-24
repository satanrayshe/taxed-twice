"""Every number and figure in the paper comes from this script (re-run it to reproduce).

Inputs (from experiments/02_quantization/):
  parity256_vs_damage.csv   tokenizer parity (same 256 sentences) + nf4 abs NLL/char damage per model x language
  ci_<model>.csv            bootstrap CIs per model x language x precision
  mt_summary.csv            Indic->English chrF++ (5 models)
  mt_en2x_summary.csv       English->Indic chrF++ (built here for all available models)
Outputs (paper/):
  figures/fig1_parity_vs_damage.pdf|png, stats.json, tables.tex
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "experiments" / "02_quantization"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

NAMES = {"Llama-3.2-1B": "Llama-3.2-1B", "Llama-3.2-3B": "Llama-3.2-3B", "gemma-3-1b-pt": "Gemma-3-1B",
         "Qwen3-1.7B-Base": "Qwen3-1.7B", "sarvam-1": "Sarvam-1", "pragna-1b": "Pragna-1B",
         "Param-1-2.9B-Instruct": "Param-1-2.9B"}
INDIC_BUILT = {"sarvam-1", "pragna-1b", "Param-1-2.9B-Instruct"}


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def within_model_perm_test(d, x, y, n=10000):
    """Pooled Spearman, with a null that shuffles x WITHIN each model (keeps model structure)."""
    obs = spearman(d[x], d[y])
    groups = [g.index.values for _, g in d.groupby("model")]
    xs = d[x].values.copy()
    null = np.empty(n)
    for k in range(n):
        perm = xs.copy()
        for idx in groups:
            pos = d.index.get_indexer(idx)
            perm[pos] = rng.permutation(xs[pos])
        null[k] = spearman(perm, d[y])
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


def fixed_effects_slope(d, x, y):
    """OLS of y on x with one intercept per model (model fixed effects). Returns slope + bootstrap CI."""
    def slope(df):
        xd = df[x] - df.groupby("model")[x].transform("mean")
        yd = df[y] - df.groupby("model")[y].transform("mean")
        return float((xd * yd).sum() / (xd ** 2).sum())
    est = slope(d)
    boots = []
    for _ in range(2000):  # resample languages within each model
        s = d.groupby("model", group_keys=False).apply(lambda g: g.sample(len(g), replace=True, random_state=rng.integers(1e9)))
        boots.append(slope(s))
    return est, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    d = pd.read_csv(EXP / "parity256_vs_damage.csv").reset_index(drop=True)
    d["log_parity"] = np.log(d.parity256)
    ind = d[d.language != "English"].copy()
    ind["excess"] = ind.dmg - ind.eng

    stats = {"n_models": int(d.model.nunique()), "n_points": int(len(d))}
    stats["pooled_spearman"], stats["pooled_perm_p"] = within_model_perm_test(d, "parity256", "dmg")
    stats["indic_excess_spearman"], stats["indic_excess_perm_p"] = within_model_perm_test(
        ind.reset_index(drop=True), "parity256", "excess")
    stats["fe_slope_dmg_per_log_parity"], stats["fe_slope_ci95"] = fixed_effects_slope(d, "log_parity", "dmg")
    stats["within_model_spearman"] = {NAMES[m]: spearman(g.parity256, g.dmg) for m, g in d.groupby("model")}

    # Per-model summary table (nf4, abs NLL/char damage)
    rows = []
    for m, g in d.groupby("model"):
        gi = g[g.language != "English"]
        rows.append({"model": NAMES[m], "indic_built": m in INDIC_BUILT,
                     "eng_dmg": g[g.language == "English"].dmg.iloc[0],
                     "indic_mean_dmg": gi.dmg.mean(), "indic_max_dmg": gi.dmg.max(),
                     "indic_mean_parity": gi.parity256.mean(),
                     "rho": spearman(g.parity256, g.dmg)})
    tab = pd.DataFrame(rows).sort_values("indic_mean_parity")
    stats["per_model"] = tab.round(4).to_dict("records")

    # Pragna disentangling: languages NOT in Pragna's training data (model card: hi, bn, gu, en)
    pr = d[(d.model == "pragna-1b") & (~d.language.isin(["English", "Hindi", "Bengali", "Gujarati"]))]
    lo, hi = pr[pr.parity256 < 2], pr[pr.parity256 >= 2]
    stats["pragna_untrained"] = {"low_parity": lo[["language", "parity256", "dmg"]].round(3).to_dict("records"),
                                 "high_parity": hi[["language", "parity256", "dmg"]].round(3).to_dict("records"),
                                 "low_mean": float(lo.dmg.mean()), "high_mean": float(hi.dmg.mean()),
                                 "hindi_trained_dmg": float(d[(d.model == "pragna-1b") & (d.language == "Hindi")].dmg.iloc[0])}

    # Translation (secondary)
    for name, f in [("mt_x2en", "mt_summary.csv"), ("mt_en2x", "mt_en2x_summary.csv")]:
        if (EXP / f).exists():
            t = pd.read_csv(EXP / f)
            drop = "drop" if "drop" in t else "chrf_drop"
            ok = t[t.chrf_bf16 >= 20]
            stats[name] = {"n_cells": int(len(t)), "models": sorted(t.model.unique().tolist()),
                           "ci_excludes_zero": int(((t.lo > 0) | (t.hi < 0)).sum()),
                           "mean_drop": float(t[drop].mean()),
                           "spearman_drop_vs_nll_all": spearman(t[drop], t.nll_dmg),
                           "n_nonfloor": int(len(ok)),
                           "spearman_drop_vs_nll_nonfloor": spearman(ok[drop], ok.nll_dmg) if len(ok) > 3 else None,
                           "spearman_drop_vs_parity_nonfloor": spearman(ok[drop], ok.parity) if len(ok) > 3 else None}

    (ROOT / "stats.json").write_text(json.dumps(stats, indent=2))

    # Figure 1: parity vs damage
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    markers = dict(zip(NAMES, "o s ^ D v P X".split()))
    for m, g in d.groupby("model"):
        ax.scatter(g.parity256, g.dmg, marker=markers[m], s=34, alpha=0.85,
                   label=NAMES[m] + (" (IN)" if m in INDIC_BUILT else ""),
                   edgecolor="black" if m in INDIC_BUILT else "none", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Tokenizer parity vs English (log scale; 1 = same token count)")
    ax.set_ylabel("NF4 damage: Δ NLL per character (nats)")
    ax.set_title(f"Tokenizer tax vs 4-bit quantization damage\n"
                 f"7 models × 11 languages, Spearman ρ = {stats['pooled_spearman']:.2f}", fontsize=10)
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_parity_vs_damage.{ext}", dpi=200)

    # LaTeX table
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Parity & Eng.\ $\Delta$ & Indic mean $\Delta$ & Indic max $\Delta$ & $\rho$ \\", r"\midrule"]
    for r in tab.itertuples():
        lines.append(f"{r.model}{'$^\\dagger$' if r.indic_built else ''} & {r.indic_mean_parity:.1f} & "
                     f"{r.eng_dmg:.3f} & {r.indic_mean_dmg:.3f} & {r.indic_max_dmg:.3f} & {r.rho:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "tables.tex").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_model", "pragna_untrained")}, indent=1))


if __name__ == "__main__":
    main()
