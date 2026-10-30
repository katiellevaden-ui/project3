"""Analysis for the elicitation study (scripts/elicit_study.py).

Questions and the table that answers each:
  How much of the variation between elicited constitutions is the model, the prompt,
  sampling a new constitution, or just measuring it?
                                         variance_decomposition()   nested variance components
  Does an elicited constitution move the model's behaviour at all?
                                         inertness()                W1 from the prior vs the noise floor
  Do the value dimensions the prompt names move, and only those?
                                         priming_test()             primed vs unprimed cells
  Which factor organises the texts themselves?
                                         text_distance_summary()    TF-IDF silhouette by model vs by prompt
  How much prompt vocabulary leaks into the output?
                                         echo_table()               from the elicitation manifest
  Where do the elicited states sit relative to each other and to the prior?
                                         plot_umap()                one panel per model, shared embedding
  Which values do constitutions push, and do all models move the same way?
                                         shift_table(), plot_shift_heatmap()   model x dimension mean shift

Entities are labelled by scripts/elicit_study.py as
    {"model", "constitution", "chain": null, "round": null, "repeat", "variant", "replicate"}
`constitution` "none" is the prior; `repeat` 1 is a re-measurement of the same text (noise floor).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from valuedrift import analysis
from valuedrift.elicit_variants import PRIMED_VALUES, parse_variant

META_COLS = ["entity", "model", "constitution", "variant", "replicate", "repeat"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_study(logs_dir: Path) -> pd.DataFrame:
    """One row per (entity, item, epoch) with the oriented rating and the study's labels."""
    rows = []
    for name in analysis.successful_logs(logs_dir):
        log = read_eval_log(name)
        if not log.eval.metadata.get("label"):
            continue
        d = json.loads(log.eval.metadata["label"])
        e = analysis.Entity(d["model"], d["constitution"], None, None, d.get("repeat", 0))
        for s in log.samples:
            sc = s.scores.get("valuebench_rating")
            if not sc:
                continue
            v = sc.value["oriented"]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            rows.append({"entity": e.key, "model": d["model"], "constitution": d["constitution"],
                         "variant": d.get("variant"), "replicate": d.get("replicate"), "repeat": d.get("repeat", 0),
                         "item": s.id, "epoch": s.epoch, "questionnaire": sc.metadata["questionnaire"],
                         "value": sc.metadata["value"], "oriented": float(v)})
    return pd.DataFrame(rows)


def entity_meta(df: pd.DataFrame) -> pd.DataFrame:
    m = df.drop_duplicates("entity")[META_COLS].set_index("entity")
    return add_factors(m)


def add_factors(meta: pd.DataFrame) -> pd.DataFrame:
    """framing / priming / identity columns from the variant id (None for the prior)."""
    meta = meta.copy()
    for col in ("framing", "priming", "identity"):
        meta[col] = None
    for e, vid in meta["variant"].items():
        if isinstance(vid, str):
            v = parse_variant(vid)
            meta.loc[e, ["framing", "priming", "identity"]] = [v.framing, v.priming, v.identity]
    return meta


def _elicited(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[meta["variant"].notna() & (meta["repeat"].fillna(0) == 0)]


# --------------------------------------------------------------------------
# Variance decomposition
# --------------------------------------------------------------------------


def noise_pairs(meta: pd.DataFrame) -> list[tuple[str, str]]:
    """Pairs of entities that are re-measurements of one and the same text (same model and
    constitution, different `repeat`)."""
    pairs = []
    for _, g in meta.groupby(["model", "constitution"]):
        rep = g["repeat"].fillna(0)
        originals, remeasured = list(g.index[rep == 0]), list(g.index[rep != 0])
        pairs += [(a, b) for a in originals for b in remeasured]
    return pairs


def variance_decomposition(profiles: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Nested variance components per dimension and summed over all ("ALL"), from the expected
    mean squares of the balanced nested design model > prompt > replicate:
         var_model        between models
         var_prompt       between prompt variants, within a model
         var_replicate    between constitutions sampled from the same prompt
         var_measurement  re-measuring one and the same constitution (from the noise pairs:
                          half the mean squared difference between re-measurements)
    The replicate component is what is left of the within-cell mean square after the
    measurement component is taken out, floored at zero. `vc_*` are the components as shares
    of their sum; `prompt_to_model` is var_prompt / var_model. `ss_*` / `frac_*` are the raw
    sums of squares, kept for reference (they are not variance components: they inflate the
    lower levels by their degrees of freedom). Cell sizes are averaged when the design is
    unbalanced (a rejected replicate, an unfinished model)."""
    m = _elicited(meta)
    X = profiles.loc[profiles.index.intersection(m.index)]
    m = m.loc[X.index]
    if X.empty:
        return pd.DataFrame()
    pairs = [(a, b) for a, b in noise_pairs(meta) if a in profiles.index and b in profiles.index]
    n_models = m["model"].nunique()
    cells = m.groupby(["model", "variant"]).size()
    r = float(cells.mean())                                   # replicates per cell
    v = float(cells.groupby(level=0).size().mean())           # variants per model
    rows, total = [], {}
    for dim in X.columns:
        y = X[dim].astype(float)
        ok = y.notna()
        y, mm = y[ok], m[ok]
        if len(y) < 2:
            continue
        grand = y.mean()
        model_mean = y.groupby(mm["model"]).transform("mean")
        cell_mean = y.groupby([mm["model"], mm["variant"]]).transform("mean")
        ss = {"model": float(((model_mean - grand) ** 2).sum()),
              "prompt": float(((cell_mean - model_mean) ** 2).sum()),
              "sampling": float(((y - cell_mean) ** 2).sum())}
        df_model, df_prompt, df_samp = max(n_models - 1, 0), max(len(cells) - n_models, 0), max(len(y) - len(cells), 0)
        ms_model = ss["model"] / df_model if df_model else math.nan
        ms_prompt = ss["prompt"] / df_prompt if df_prompt else math.nan
        ms_samp = ss["sampling"] / df_samp if df_samp else math.nan
        diffs = [profiles.loc[a, dim] - profiles.loc[b, dim] for a, b in pairs]
        diffs = [d for d in diffs if not math.isnan(d)]
        # No re-measurement pairs (e.g. text coded once): nothing separates measurement from replicate
        # variation, so the whole within-cell spread is attributed to replicates and measurement is 0.
        var_meas = float(np.mean(np.square(diffs)) / 2) if diffs else 0.0
        var_within = ms_samp
        var_rep = max(var_within - var_meas, 0.0) if not math.isnan(var_meas) else math.nan
        var_prompt = max((ms_prompt - ms_samp) / r, 0.0) if not (math.isnan(ms_prompt) or math.isnan(ms_samp)) else math.nan
        var_model = max((ms_model - ms_prompt) / (r * v), 0.0) if not (math.isnan(ms_model) or math.isnan(ms_prompt)) else math.nan
        row = {"dimension": dim, **{f"ss_{k}": ss[k] for k in ss},
               **{f"frac_{k}": ss[k] / (sum(ss.values()) or math.nan) for k in ss},
               "var_model": var_model, "var_prompt": var_prompt, "var_replicate": var_rep, "var_measurement": var_meas}
        rows.append(row)
        for k in ("ss_model", "ss_prompt", "ss_sampling", "var_model", "var_prompt", "var_replicate", "var_measurement"):
            if not math.isnan(row[k]):
                total[k] = total.get(k, 0.0) + row[k]
    all_row = {"dimension": "ALL", **total}
    ss_total = sum(total.get(f"ss_{k}", 0.0) for k in ("model", "prompt", "sampling")) or math.nan
    for k in ("model", "prompt", "sampling"):
        all_row[f"frac_{k}"] = total.get(f"ss_{k}", math.nan) / ss_total
    rows.append(all_row)
    out = pd.DataFrame(rows)
    comps = ["var_model", "var_prompt", "var_replicate", "var_measurement"]
    vc_total = out[comps].sum(axis=1, min_count=1)
    for c in comps:
        out["vc_" + c[4:]] = out[c] / vc_total
    out["prompt_to_model"] = [_ratio(a, b) for a, b in zip(out["var_prompt"], out["var_model"])]
    out["n_noise_pairs"] = len(pairs)
    return out


def _ratio(num: float, den: float) -> float:
    """num/den; inf when only the denominator is zero, NaN when either is undefined."""
    if math.isnan(num) or math.isnan(den):
        return math.nan
    if den == 0:
        return math.inf if num > 0 else math.nan
    return num / den


# --------------------------------------------------------------------------
# Permutation tests against the no-effect floor
# --------------------------------------------------------------------------
#
# The W1 distance between two finite samples of the same state is not zero: a 2-answer
# histogram is a crude sketch, and its distance to anything is large. The right null is
# therefore empirical: pool the two states' answers per item, split them at random into the
# same sizes, and recompute. The observed distance minus the null mean ("excess") is the
# effect; the p-value is the share of null draws at least as large as the observed.


def _w1_lists(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    d = []
    for i in a.keys() & b.keys():
        ha, _ = np.histogram(np.round(a[i]).clip(0, 10), bins=analysis.BINS)
        hb, _ = np.histogram(np.round(b[i]).clip(0, 10), bins=analysis.BINS)
        d.append(np.abs(np.cumsum(ha / ha.sum()) - np.cumsum(hb / hb.sum())).sum())
    return float(np.mean(d)) if d else math.nan


def _answers(g: pd.DataFrame) -> dict[str, np.ndarray]:
    return {i: v["oriented"].to_numpy() for i, v in g.groupby("item")}


def permutation_w1(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n_perm: int = 300, seed: int = 0) -> dict:
    """Observed W1 between two states' answers, the no-effect null from re-splitting the pooled
    answers per item into the same sizes, the excess, and a one-sided p-value."""
    rng = np.random.default_rng(seed)
    items = sorted(a.keys() & b.keys())
    obs = _w1_lists(a, b)
    null = np.empty(n_perm)
    pools = {i: np.concatenate([a[i], b[i]]) for i in items}
    ka = {i: len(a[i]) for i in items}
    for n in range(n_perm):
        pa, pb = {}, {}
        for i in items:
            perm = rng.permutation(pools[i])
            pa[i], pb[i] = perm[:ka[i]], perm[ka[i]:]
        null[n] = _w1_lists(pa, pb)
    return {"observed_w1": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "excess": obs - float(null.mean()), "p_value": float((null >= obs).mean() if n_perm else math.nan),
            "n_perm": n_perm}


def prior_shift_table(df: pd.DataFrame, n_perm: int = 300, seed: int = 0) -> pd.DataFrame:
    """Per model: W1 between the prior (all its runs pooled) and all of its elicited
    constitutions pooled, against the no-effect floor for those sample sizes."""
    pm = pool_models(df)
    rows = []
    for model, g in pm.groupby("model"):
        prior, elic = _answers(g[g["constitution"] == "none"]), _answers(g[g["constitution"] != "none"])
        if not prior or not elic:
            continue
        r = permutation_w1(prior, elic, n_perm, seed)
        rows.append({"model": model, "n_prior": int(np.median([len(v) for v in prior.values()])),
                     "n_const": int(np.median([len(v) for v in elic.values()])), **r})
    return pd.DataFrame(rows).sort_values("excess", ascending=False)


# --------------------------------------------------------------------------
# Inertness: does the elicited text move behaviour beyond the noise floor?
# --------------------------------------------------------------------------


def inertness(df: pd.DataFrame, meta: pd.DataFrame | None = None, n_perm: int = 200, seed: int = 0) -> pd.DataFrame:
    """For every elicited constitution: W1 distance from the model's prior (all prior runs
    pooled), the no-effect floor for a one-answer state against that prior, the excess, a
    permutation p-value, and `inert` = not distinguishable from the prior at p >= 0.05.
    More prior runs lower the floor and sharpen every verdict."""
    meta = entity_meta(df) if meta is None else meta
    rows = []
    for model, g in df.groupby("model"):
        prior = _answers(g[g["constitution"] == "none"])
        n_prior = int(np.median([len(v) for v in prior.values()])) if prior else 0
        for e, r in _elicited(meta[meta["model"] == model]).iterrows():
            own = _answers(g[g["entity"] == e])
            if not prior or not own:
                continue
            t = permutation_w1(own, prior, n_perm, seed)
            rows.append({"entity": e, "model": model, "variant": r["variant"], "replicate": r["replicate"],
                         "w1_from_prior": t["observed_w1"], "null_mean": t["null_mean"], "excess": t["excess"],
                         "p_value": t["p_value"], "n_prior_answers": n_prior, "inert": bool(t["p_value"] >= 0.05)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Priming: do the dimensions the prompt names move, and only those?
# --------------------------------------------------------------------------


def priming_test(profiles: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """For every primed cell, the mean shift from the unprimed cell with the same framing and
    identity, averaged over the primed dimensions and separately over every other dimension.
    Works on ValueBench profiles (PVQ40 value names) and on coding profiles alike; dimensions
    that are not Schwartz values count as "other". A prompt effect that is specific shows
    delta_primed well above delta_other; a general effect shows both moving together."""
    m = _elicited(meta)
    X = profiles.loc[profiles.index.intersection(m.index)]
    m = m.loc[X.index]
    rows = []
    for (model, framing, identity), g in m.groupby(["model", "framing", "identity"]):
        base = g[g["priming"] == "none"]
        if base.empty:
            continue
        base_prof = X.loc[base.index].mean()
        for priming, cell in g[g["priming"] != "none"].groupby("priming"):
            delta = X.loc[cell.index].mean() - base_prof
            primed = [c for c in X.columns if c in PRIMED_VALUES[priming]]
            other = [c for c in X.columns if c not in PRIMED_VALUES[priming]]
            rows.append({"model": model, "framing": framing, "identity": identity, "priming": priming,
                         "n_primed_dims": len(primed), "delta_primed": float(delta[primed].mean()) if primed else math.nan,
                         "delta_other": float(delta[other].mean()) if other else math.nan,
                         "abs_delta_primed": float(delta[primed].abs().mean()) if primed else math.nan,
                         "abs_delta_other": float(delta[other].abs().mean()) if other else math.nan,
                         "n_primed_cell": len(cell), "n_base_cell": len(base)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The texts themselves
# --------------------------------------------------------------------------


def text_distance_summary(texts: dict[str, str], meta: pd.DataFrame) -> dict:
    """TF-IDF cosine distances between constitutions and silhouette scores for two labellings,
    by model and by prompt variant: whichever is higher is the factor that organises the texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances, silhouette_score
    keys = [k for k in texts if k in meta.index]
    if len(keys) < 3:
        return {"n": len(keys), "silhouette_by_model": math.nan, "silhouette_by_variant": math.nan}
    D = pairwise_distances(TfidfVectorizer(stop_words="english").fit_transform([texts[k] for k in keys]), metric="cosine")
    out = {"n": len(keys), "mean_distance": float(D[np.triu_indices(len(keys), 1)].mean())}
    for name, col in (("model", "model"), ("variant", "variant")):
        labels = meta.loc[keys, col].astype(str).to_numpy()
        k = len(set(labels))
        out[f"silhouette_by_{name}"] = float(silhouette_score(D, labels, metric="precomputed")) if 2 <= k < len(keys) else math.nan
    return out


def echo_table(manifest: list[dict]) -> pd.DataFrame:
    """Per variant: validity rate, principles, words and prompt-echo averages."""
    df = pd.DataFrame(manifest)
    if df.empty:
        return df
    return (df.groupby(["variant", "framing", "priming", "identity"], as_index=False)
              .agg(n=("replicate", "size"), valid_rate=("valid", "mean"), principles=("principles", "mean"),
                   words=("words", "mean"), echo=("echo", "mean"), coverage=("coverage", "mean")))


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

# One colour for every elicited constitution (categorical slot 1 of the validated reference
# palette), the prior in ink, re-measurements faint. The prompt factors are deliberately not
# encoded: the variance decomposition is where their (small) effect is read.
ELICITED, INK, INK_MUTED, GRID = "#2a78d6", "#1a1a19", "#6b6a63", "#e6e5df"


def plot_umap(df: pd.DataFrame, out_png: Path, meta: pd.DataFrame | None = None, seed: int = 0,
              title: str = "Elicited constitutions (UMAP of W1 distances)",
              point_label: str = "elicited constitution (any prompt)"):
    """One panel per model, all panels from one shared embedding so positions are comparable.
    Each blue point is one elicited constitution measured once; the black x is the prior (no
    system prompt); faint points are re-measurements of the same text, whose distance from
    their twin is the noise floor. Returns (distance matrix, entity -> xy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = entity_meta(df) if meta is None else meta
    keys = list(meta.index)
    answers = int(df.groupby(["entity", "item"]).size().median())    # typical answers per item behind each point
    D = analysis.w1_matrix(analysis.item_distributions(df), keys)
    xy = analysis.embed(D, seed)
    pos = {k: xy[i] for i, k in enumerate(keys)}
    pad = 0.08 * (xy.max(axis=0) - xy.min(axis=0) + 1e-9)
    lim = (xy.min(axis=0) - pad, xy.max(axis=0) + pad)

    models = sorted(meta["model"].unique())
    ncol = min(3, len(models))
    nrow = math.ceil(len(models) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), squeeze=False, facecolor="white")
    any_faint = False
    for ax, model in zip(axes.flat, models):
        sub = meta[meta["model"] == model]
        for k, e in sub.iterrows():
            x, y = pos[k]
            faint = (e["repeat"] or 0) != 0
            any_faint |= faint
            if e["constitution"] == "none":
                ax.scatter(x, y, s=90, marker="x", color=INK, linewidths=2, alpha=0.4 if faint else 1, zorder=5)
                if not faint:
                    ax.annotate("prior", (x, y), fontsize=8, color=INK, xytext=(6, 4), textcoords="offset points")
            else:
                ax.scatter(x, y, s=55, color=ELICITED, alpha=0.3 if faint else 0.8, linewidths=0, zorder=4)
        ax.set_title(model, fontsize=10, color=INK, loc="left")
        ax.set_xlim(lim[0][0], lim[1][0]); ax.set_ylim(lim[0][1], lim[1][1])
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
    for ax in axes.flat[len(models):]:
        ax.axis("off")

    handles = [Line2D([], [], marker="o", linestyle="", color=ELICITED, label=point_label),
               Line2D([], [], marker="x", linestyle="", color=INK, markeredgewidth=2, label="prior (no constitution)")]
    if any_faint:
        handles.append(Line2D([], [], marker="o", linestyle="", color=INK_MUTED, alpha=0.35, label="re-measurement of the same text (noise floor)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(title + (f"  [{answers} answers per item]" if answers >= 2 else "  [one answer per item]"),
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return D, pos


# --------------------------------------------------------------------------
# Shift from the prior, per model and dimension
# --------------------------------------------------------------------------

def dimension_questionnaires(df: pd.DataFrame) -> dict[str, str]:
    """Which questionnaire each measured dimension came from, read off the study's own rows.
    (Names such as Fairness or Humility exist in several ValueBench questionnaires, so the
    mapping must come from the data that was actually measured, not the whole item file.)"""
    return dict(df.drop_duplicates("value")[["value", "questionnaire"]].to_numpy().tolist())


def shift_table(profiles: pd.DataFrame, meta: pd.DataFrame, variance: pd.DataFrame | None = None,
                questionnaires: dict[str, str] | None = None) -> pd.DataFrame:
    """Mean shift from the prior, in rating points, for every (model, dimension): the average
    profile of the model's elicited constitutions minus the average of its prior measurements
    (both repeats, which halves the prior's noise). With the variance table, `se` is the
    standard error of that shift from the measurement and replicate components, and `clear`
    marks shifts beyond two standard errors."""
    rows = []
    var = variance.set_index("dimension") if variance is not None else None
    for model, g in meta.groupby("model"):
        prior = g[g["constitution"] == "none"]
        elic = _elicited(g)
        if prior.empty or elic.empty:
            continue
        base = profiles.loc[prior.index].mean()
        shift = profiles.loc[elic.index].mean() - base
        for dim, val in shift.items():
            se = math.nan
            if var is not None and dim in var.index:
                vm, vr = var.loc[dim, "var_measurement"], var.loc[dim, "var_replicate"]
                if not (math.isnan(vm) or math.isnan(vr)):
                    se = math.sqrt(vm / len(prior) + (vm + vr) / len(elic))
            rows.append({"model": model, "dimension": dim, "questionnaire": (questionnaires or {}).get(dim, ""),
                         "shift": float(val), "se": se, "clear": bool(abs(val) > 2 * se) if not math.isnan(se) else None,
                         "n_constitutions": len(elic), "n_prior": len(prior)})
    return pd.DataFrame(rows)


def plot_shift_heatmap(shifts: pd.DataFrame, out_png: Path, title: str = "Shift from the prior under elicited constitutions"):
    """Models across, value dimensions down, grouped by questionnaire. Cell colour is the mean
    shift in rating points on a blue (lower) / grey (none) / red (higher) scale symmetric about
    zero; a dot marks a shift beyond two standard errors. Dimensions are ordered within each
    questionnaire by how much they move on average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
    cmap = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])   # blue / neutral / red
    M = shifts.pivot(index="dimension", columns="model", values="shift")
    C = shifts.pivot(index="dimension", columns="model", values="clear") if shifts["clear"].notna().any() else None
    q = shifts.drop_duplicates("dimension").set_index("dimension")["questionnaire"]
    order = (M.abs().mean(axis=1).to_frame("m").assign(q=q).sort_values(["q", "m"], ascending=[True, False]).index)
    M = M.loc[order]
    models = sorted(M.columns)
    M = M[models]
    limit = float(np.nanmax(np.abs(M.to_numpy()))) or 1.0
    fig_h = 0.26 * len(M) + 2.2
    fig, ax = plt.subplots(figsize=(0.75 * len(models) + 3.2, fig_h), facecolor="white")
    im = ax.imshow(M.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), aspect="auto")
    if C is not None:
        C = C.loc[order][models]
        ys, xs = np.where(C.to_numpy() == True)  # noqa: E712
        ax.scatter(xs, ys, s=9, color=INK, zorder=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, fontsize=7, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    # questionnaire group lines and labels on the right
    groups = q.loc[order]
    starts = [i for i in range(len(groups)) if i == 0 or groups.iloc[i] != groups.iloc[i - 1]]
    for i in starts[1:]:
        ax.axhline(i - 0.5, color="white", lw=2)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(groups)
        ax.text(len(models) - 0.4, (start + end - 1) / 2, groups.iloc[start], fontsize=8, color=INK_MUTED,
                rotation=90, va="center", ha="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label("mean shift from prior (rating points, 0-10 scale)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.outline.set_edgecolor(GRID)
    ax.set_title(title + ("" if C is None else "   (dot: beyond 2 standard errors)"), fontsize=10, color=INK, loc="left", pad=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return M


# --------------------------------------------------------------------------
# Noise reduction with the data on disk
# --------------------------------------------------------------------------


def pool_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Re-label states so that one (model, prompt variant) cell is one entity: its replicates
    become several answers per item, and the prior's repeats are pooled too. Re-measurements
    of elicited texts are dropped (they would be a sixth answer for one cell only). The point
    of this view is item-level rating distributions at one answer per item."""
    out = df.copy()
    keep = ~((out["constitution"] != "none") & (out["repeat"].fillna(0) != 0))
    out = out[keep].copy()
    prior = out["constitution"] == "none"
    out.loc[prior, "entity"] = out.loc[prior, "model"] + "|none|base"
    out.loc[~prior, "entity"] = out.loc[~prior, "model"] + "|elicited:" + out.loc[~prior, "variant"].astype(str) + "|cell"
    out.loc[~prior, "constitution"] = "elicited:" + out.loc[~prior, "variant"].astype(str)
    out["replicate"] = None
    out["repeat"] = 0
    return out


# Coarser dimensions: HEXACO facets -> domains (Altruism is HEXACO's interstitial facet and
# stays on its own); PVQ40 values -> Schwartz's four higher-order values. LVI and MFT23 are
# unchanged. Every name here is checked against data/valuebench at import in the tests.
COARSE = {
    ("HEXACO", f): d for d, facets in {
        "Honesty-Humility": ("Sincerity", "Fairness", "Greed-Avoidance", "Modesty"),
        "Emotionality": ("Fearfulness", "Anxiety", "Dependence", "Sentimentality"),
        "Extraversion": ("Social Self-Esteem", "Social Boldness", "Sociability", "Liveliness"),
        "Agreeableness": ("Forgiveness", "Gentleness", "Flexibility", "Patience"),
        "Conscientiousness": ("Organization", "Diligence", "Perfectionism", "Prudence"),
        "Openness": ("Aesthetic Appreciation", "Inquisitiveness", "Creativity", "Unconventionality"),
        "Altruism": ("Altruism",),
    }.items() for f in facets
}
COARSE.update({
    ("PVQ40", v): d for d, values in {
        "Openness to change": ("Self-Direction", "Stimulation", "Hedonism"),
        "Self-enhancement": ("Achievement", "Power"),
        "Conservation": ("Security", "Conformity", "Tradition"),
        "Self-transcendence": ("Benevolence", "Universalism"),
    }.items() for v in values
})


def coarsen(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `dimension` column: the coarse name where one is defined, else the original value."""
    out = df.copy()
    out["dimension"] = [COARSE.get((q, v), v) for q, v in zip(out["questionnaire"], out["value"])]
    return out


def coarse_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """entity x coarse-dimension matrix of mean oriented ratings over every item in the group."""
    return coarsen(df).pivot_table(index="entity", columns="dimension", values="oriented", aggfunc="mean")


def coarse_questionnaires(df: pd.DataFrame) -> dict[str, str]:
    c = coarsen(df).drop_duplicates("dimension")[["dimension", "questionnaire"]]
    return dict(c.to_numpy().tolist())


def pool_models(df: pd.DataFrame) -> pd.DataFrame:
    """Two states per model: the prior (its repeats pooled) and all of its elicited
    constitutions pooled into one, so each state has many answers per item."""
    out = df[~((df["constitution"] != "none") & (df["repeat"].fillna(0) != 0))].copy()
    prior = out["constitution"] == "none"
    out.loc[prior, "entity"] = out.loc[prior, "model"] + "|none|base"
    out.loc[~prior, "entity"] = out.loc[~prior, "model"] + "|elicited:all|pooled"
    out.loc[~prior, "constitution"] = "elicited:all"
    out.loc[~prior, "variant"] = None                  # not a prompt cell: add_factors() must skip it
    out["replicate"] = None
    out["repeat"] = 0
    return out


GROUP_COLOUR = {"frontier_closed": "#2a78d6", "open_frontier": "#eb6834", "open_mid": "#1baf7a", "low_safety": "#eda100"}
GROUP_LABEL = {"frontier_closed": "closed frontier", "open_frontier": "open frontier", "open_mid": "open mid-size", "low_safety": "low safety"}


def plot_model_arrows(df: pd.DataFrame, groups: dict[str, str], out_png: Path, seed: int = 0,
                      title: str = "From prior to self-written constitutions, all models (UMAP of W1 distances)",
                      shifts: pd.DataFrame | None = None):
    """One panel. For each model an arrow from its prior (hollow circle) to the pooled state
    under all of its elicited constitutions (filled circle), coloured by model group. The
    layout preserves neighbourhoods rather than distances, so each arrow carries a number:
    with `shifts` (prior_shift_table) the excess W1 over the no-effect floor, with a dagger when
    p >= 0.05; without it the raw W1. `df` should come from pool_models()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = entity_meta(df)
    keys = list(meta.index)
    dists = analysis.item_distributions(df)
    D = analysis.w1_matrix(dists, keys)
    xy = analysis.embed(D, seed)
    pos = {k: xy[i] for i, k in enumerate(keys)}
    per_item = df.groupby(["entity", "item"]).size().groupby(level=0).median()
    a_prior = int(per_item[meta.index[meta["constitution"] == "none"]].median())
    a_elic = int(per_item[meta.index[meta["constitution"] != "none"]].median())

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")
    for model, g in meta.groupby("model"):
        c = GROUP_COLOUR.get(groups.get(model, ""), INK_MUTED)
        prior = g[g["constitution"] == "none"].index
        elic = g[g["constitution"] != "none"].index
        if len(prior) != 1 or len(elic) != 1:
            continue
        (x0, y0), (x1, y1) = pos[prior[0]], pos[elic[0]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=6, shrinkB=6), zorder=3)
        ax.scatter(x0, y0, s=70, facecolors="white", edgecolors=c, linewidths=1.6, zorder=4)
        ax.scatter(x1, y1, s=90, color=c, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(model, (x1, y1), fontsize=9, color=INK, xytext=(7, 5), textcoords="offset points", zorder=6)
        if shifts is not None and model in set(shifts["model"]):
            r = shifts.set_index("model").loc[model]
            text = f"+{r['excess']:.2f}" + ("" if r["p_value"] < 0.05 else " \u2020")
        else:
            text = f"{analysis.w1_distance(dists[prior[0]], dists[elic[0]]):.2f}"
        ax.annotate(text, ((x0 + x1) / 2, (y0 + y1) / 2), fontsize=7, color=INK_MUTED, xytext=(3, -9),
                    textcoords="offset points", zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)
    present = [g for g in GROUP_COLOUR if g in {groups.get(m, "") for m in meta["model"].unique()}]
    handles = [Line2D([], [], marker="o", linestyle="", color=GROUP_COLOUR[g], label=GROUP_LABEL[g]) for g in present]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK_MUTED, label="prior (no constitution)"),
                Line2D([], [], marker="o", linestyle="", color=INK_MUTED, label="under its own constitutions (all pooled)"),
                Line2D([], [], linestyle="", marker="$+.3$", color=INK_MUTED, markersize=14,
                       label="W1 excess over the no-effect floor, rating points" if shifts is not None else "W1 distance, rating points")]
    if shifts is not None:
        handles.append(Line2D([], [], linestyle="", marker="$\u2020$", color=INK_MUTED, markersize=9, label="not distinguishable from prior (p \u2265 0.05)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(title + f"  [answers per item: prior {a_prior}, constitutions {a_elic}]", fontsize=10, color=INK, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return D, pos


def pca_states(profiles: pd.DataFrame, n: int = 2):
    """PCA of a state x dimension profile matrix (mean ratings, 0-10, not standardised so the
    axes stay in rating points). Returns scores, loadings (dimension x component) and the
    share of variance per component."""
    X = profiles.fillna(profiles.mean()).to_numpy(dtype=float)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:n].T
    explained = (S ** 2 / (S ** 2).sum())[:n]
    loadings = pd.DataFrame(Vt[:n].T, index=profiles.columns, columns=[f"PC{i + 1}" for i in range(n)])
    return pd.DataFrame(scores, index=profiles.index, columns=loadings.columns), loadings, explained


def _pole_labels(loadings: pd.Series, k: int = 3) -> tuple[str, str]:
    neg = ", ".join(loadings.sort_values().index[:k])
    pos = ", ".join(loadings.sort_values(ascending=False).index[:k])
    return neg, pos


def plot_model_arrows_pca(df: pd.DataFrame, groups: dict[str, str], out_png: Path, shifts: pd.DataFrame | None = None,
                          title: str = "From prior to self-written constitutions, all models (PCA of value profiles)"):
    """One panel, one arrow per model from its prior (hollow) to the pooled state under all of
    its constitutions (filled), laid out by PCA of the 22 states' dimension profiles. Unlike UMAP
    this is a linear projection: distances on the page are real distances in profile space,
    shortened only by what the two components leave out (stated in the axis labels), and the
    axes are interpretable through their top-loading dimensions. The number on each arrow is
    the W1 excess over the no-effect floor from prior_shift_table(). `df` from pool_models()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = entity_meta(df)
    prof = analysis.dimension_profiles(df).loc[meta.index]
    scores, loadings, explained = pca_states(prof)
    fig, ax = plt.subplots(figsize=(9.5, 7.5), facecolor="white")
    for model, g in meta.groupby("model"):
        c = GROUP_COLOUR.get(groups.get(model, ""), INK_MUTED)
        prior = g[g["constitution"] == "none"].index
        elic = g[g["constitution"] != "none"].index
        if len(prior) != 1 or len(elic) != 1:
            continue
        (x0, y0), (x1, y1) = scores.loc[prior[0]].to_numpy(), scores.loc[elic[0]].to_numpy()
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=6, shrinkB=6), zorder=3)
        ax.scatter(x0, y0, s=70, facecolors="white", edgecolors=c, linewidths=1.6, zorder=4)
        ax.scatter(x1, y1, s=90, color=c, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(model, (x1, y1), fontsize=9, color=INK, xytext=(7, 5), textcoords="offset points", zorder=6)
        if shifts is not None and model in set(shifts["model"]):
            r = shifts.set_index("model").loc[model]
            text = f"+{r['excess']:.2f}" + ("" if r["p_value"] < 0.05 else " \u2020")
            t = 0.4 if sum(map(ord, model)) % 2 else 0.6     # deterministic alternation
            ax.annotate(text, (x0 + t * (x1 - x0), y0 + t * (y1 - y0)), fontsize=7, color=INK_MUTED, xytext=(4, -9),
                        textcoords="offset points", zorder=6,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.7))
    ax.axhline(0, color=GRID, lw=1, zorder=1); ax.axvline(0, color=GRID, lw=1, zorder=1)
    for i, axis_set in enumerate((ax.set_xlabel, ax.set_ylabel)):
        neg, pos = _pole_labels(loadings.iloc[:, i])
        axis_set(f"PC{i + 1} ({explained[i]:.0%})   \u2190 {neg}   |   {pos} \u2192", fontsize=8.5, color=INK)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_color(GRID)
    present = [g for g in GROUP_COLOUR if g in {groups.get(m, "") for m in meta["model"].unique()}]
    handles = [Line2D([], [], marker="o", linestyle="", color=GROUP_COLOUR[g], label=GROUP_LABEL[g]) for g in present]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK_MUTED, label="prior (no constitution)"),
                Line2D([], [], marker="o", linestyle="", color=INK_MUTED, label="under its own constitutions (all pooled)")]
    if shifts is not None:
        handles.append(Line2D([], [], linestyle="", marker="$+.3$", color=INK_MUTED, markersize=14, label="W1 excess over the no-effect floor, rating points"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(title + "\n" + f"axes in rating points; the two components keep {explained.sum():.0%} of the profile variance",
                 fontsize=9.5, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return scores, loadings, explained


def plot_text_vs_behaviour_variance(vd_coding: pd.DataFrame, vd_behaviour: pd.DataFrame, out_png: Path,
                                    title: str = "Where the prompt shows up: in the written principles, not in the behaviour"):
    """Per Schwartz value: the prompt's share of variance in the coded text (left) and in the
    ValueBench behaviour (right), with the model's share beside it. Values named by a priming
    set are marked. Text rows come from variance_decomposition_coding.csv, behaviour rows from
    variance_decomposition.csv restricted to the same PVQ40 values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    values = [v for v in vd_coding["dimension"] if v != "ALL"]
    primed = {v: k for k, vs in PRIMED_VALUES.items() for v in vs}
    c = vd_coding.set_index("dimension"); b = vd_behaviour.set_index("dimension")
    order = sorted(values, key=lambda v: -c.loc[v, "vc_prompt"])
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 2.4), facecolor="white", sharey=True)
    for ax, src, name in ((axes[0], c, "coded text (Schwartz label per principle)"), (axes[1], b, "behaviour (ValueBench, same values)")):
        pr = [src.loc[v, "vc_prompt"] if v in src.index else np.nan for v in order]
        mo = [src.loc[v, "vc_model"] if v in src.index else np.nan for v in order]
        ax.barh(y - 0.18, pr, 0.34, color="#eb6834", label="prompt", zorder=3)
        ax.barh(y + 0.18, mo, 0.34, color="#2a78d6", label="model", zorder=3)
        ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("share of variance", fontsize=9, color=INK); ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        for side in ax.spines.values(): side.set_color(GRID)
    labels = [f"{v}  ({primed[v].replace('set_', 'primed set ')})" if v in primed else v for v in order]
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8.5, color=INK); axes[0].invert_yaxis()
    for i, v in enumerate(order):
        if v in primed:
            axes[0].get_yticklabels()[i].set_fontweight("bold")
    handles = [Line2D([], [], marker="s", linestyle="", color="#eb6834", markersize=9, label="prompt"),
               Line2D([], [], marker="s", linestyle="", color="#2a78d6", markersize=9, label="model")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle(title + "\nbold: values named by a priming prompt", fontsize=10, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
