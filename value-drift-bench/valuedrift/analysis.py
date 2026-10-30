"""Analysis: ValueBench profiles -> drift measures -> UMAP trajectories; EigenTrust score plots.

Entities are identified by the task `label`, a JSON string written by scripts/run_all.py:
    {"model": <nick>, "constitution": <name>, "chain": <int or null>, "round": <int or null>, "repeat": <int>}
`round` null means a baseline measurement; `constitution` "none" is the model's prior; `repeat`
distinguishes noise-floor repeats of the same state.

Measures (see the project notes for definitions):
    dimension profile    mean oriented rating per ValueBench value (0-10)
    item distribution    histogram of oriented ratings (0-10) over the N answers to an item
                         (N = --epochs of the valuebench run); needs N >= 2 to be meaningful
    W1 drift             mean over items of the Wasserstein-1 (earth mover's) distance between two
                         states' item distributions, in rating points (0-10). It respects the order
                         of the ratings (3 vs 4 is close, 3 vs 10 far) and needs no fallback: with
                         one answer per item it is exactly the mean absolute rating difference
    Mahalanobis drift    distance between dimension profiles under a shrinkage covariance fitted
                         to noise-floor repeats; squared distance ~ chi2(d) under no drift
UMAP is drawn from the precomputed W1 distance matrix with a fixed seed; it preserves
neighbourhoods, not distances. Report magnitudes from the tables, never from the plot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

BINS = np.arange(12) - 0.5     # 0..10 integer ratings


@dataclass(frozen=True)
class Entity:
    model: str
    constitution: str
    chain: int | None
    round: int | None
    repeat: int = 0

    @property
    def key(self) -> str:
        r = "base" if self.round is None else f"r{self.round}"
        c = "" if self.chain is None else f"c{self.chain}"
        rep = f"#{self.repeat}" if self.repeat else ""
        return f"{self.model}|{self.constitution}|{c}{r}{rep}"

    @property
    def is_prior(self) -> bool:
        return self.constitution == "none"


def _entity(label: str) -> Entity:
    d = json.loads(label)
    return Entity(d["model"], d["constitution"], d.get("chain"), d.get("round"), d.get("repeat", 0))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def successful_logs(logs_dir: Path) -> list[str]:
    """One successful log per directory, the earliest, in the order run_all.done() resumes from.
    Two processes that measured the same state concurrently leave two logs in one directory;
    counting both would double that state's answers."""
    seen: dict[str, str] = {}
    for info in sorted(list_eval_logs(str(logs_dir), recursive=True), key=lambda i: i.name):
        d = str(Path(info.name).parent)
        if d in seen:
            continue
        if read_eval_log(info.name, header_only=True).status == "success":
            seen[d] = info.name
    return list(seen.values())


def load_valuebench(logs_dir: Path) -> pd.DataFrame:
    """One row per (entity, item, epoch) with the oriented 0-10 rating."""
    rows = []
    for name in successful_logs(logs_dir):
        log = read_eval_log(name)
        if not log.eval.metadata.get("label"):
            continue
        e = _entity(log.eval.metadata["label"])
        for s in log.samples:
            sc = s.scores.get("valuebench_rating")
            if not sc:
                continue
            v = sc.value["oriented"]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            rows.append({"entity": e.key, "model": e.model, "constitution": e.constitution, "chain": e.chain,
                         "round": e.round, "repeat": e.repeat, "item": s.id, "epoch": s.epoch,
                         "questionnaire": sc.metadata["questionnaire"], "value": sc.metadata["value"],
                         "oriented": float(v)})
    return pd.DataFrame(rows)


def dimension_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """entity x value matrix of mean oriented ratings."""
    return df.pivot_table(index="entity", columns="value", values="oriented", aggfunc="mean")


def item_distributions(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """entity -> item -> 11-bin probability vector over oriented ratings."""
    out: dict = {}
    for (entity, item), g in df.groupby(["entity", "item"]):
        hist, _ = np.histogram(g["oriented"].round().clip(0, 10), bins=BINS)
        out.setdefault(entity, {})[item] = hist / hist.sum()
    return out


def entity_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("entity")[["entity", "model", "constitution", "chain", "round", "repeat"]].set_index("entity")


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


def w1_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Mean over the items both states answered of the Wasserstein-1 distance between their
    rating distributions, in rating points: the sum of |CDF_a - CDF_b| over the 0-10 bins."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan")
    return float(np.mean([np.abs(np.cumsum(a[i]) - np.cumsum(b[i])).sum() for i in shared]))


def w1_matrix(dists: dict[str, dict[str, np.ndarray]], entities: list[str]) -> np.ndarray:
    n = len(entities)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = w1_distance(dists[entities[i]], dists[entities[j]])
    return D


def mahalanobis_drift(profiles: pd.DataFrame, noise_pairs: list[tuple[str, str]],
                      from_to: list[tuple[str, str]]) -> pd.DataFrame:
    """Mahalanobis distance between dimension profiles, with covariance estimated by
    Ledoit-Wolf shrinkage from the differences between noise-floor repeats. Returns the
    distance, the squared distance, and its chi-square p-value with d degrees of freedom."""
    from scipy.stats import chi2
    from sklearn.covariance import LedoitWolf
    X = profiles.fillna(profiles.mean())
    diffs = np.array([X.loc[a].to_numpy() - X.loc[b].to_numpy() for a, b in noise_pairs])
    if len(diffs) < 2:
        raise ValueError("need at least two noise-floor pairs to estimate the covariance")
    cov = LedoitWolf().fit(diffs).covariance_
    inv = np.linalg.pinv(cov)
    rows = []
    d = X.shape[1]
    for a, b in from_to:
        delta = X.loc[b].to_numpy() - X.loc[a].to_numpy()
        d2 = float(delta @ inv @ delta)
        rows.append({"from": a, "to": b, "mahalanobis": math.sqrt(max(d2, 0)), "d2": d2,
                     "p_value": float(chi2.sf(d2, df=d))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Drift tables
# --------------------------------------------------------------------------


def drift_table(df: pd.DataFrame) -> pd.DataFrame:
    """For every chained state: W1 distance from its own round 0 and from the model's prior."""
    ents = entity_table(df)
    dists = item_distributions(df)
    rows = []
    for key, e in ents.iterrows():
        if e["round"] is None or pd.isna(e["round"]):
            continue
        base = ents[(ents.model == e.model) & (ents.constitution == e.constitution) & ents["round"].isna() & (ents.repeat == 0)]
        prior = ents[(ents.model == e.model) & (ents.constitution == "none") & (ents.repeat == 0)]
        rows.append({"entity": key, "model": e.model, "constitution": e.constitution, "chain": e.chain, "round": int(e["round"]),
                     "w1_from_round0": w1_distance(dists[key], dists[base.index[0]]) if len(base) else float("nan"),
                     "w1_from_prior": w1_distance(dists[key], dists[prior.index[0]]) if len(prior) else float("nan")})
    return pd.DataFrame(rows)


def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """W1 distance between repeats of the same state (constitution baselines run twice)."""
    ents = entity_table(df)
    dists = item_distributions(df)
    rows = []
    for (model, constitution), g in ents[ents["round"].isna()].groupby(["model", "constitution"]):
        keys = list(g.index)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                rows.append({"model": model, "constitution": constitution, "w1": w1_distance(dists[keys[i]], dists[keys[j]])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def embed(D: np.ndarray, seed: int = 0, n_neighbors: int = 15, min_dist: float = 0.3) -> np.ndarray:
    """2-D UMAP from a precomputed distance matrix; classical MDS if umap-learn is missing."""
    try:
        # numba's default OpenMP threading layer clashes with torch's on macOS: a UMAP fit followed
        # by eigenbench.fit (torch) in one process segfaults. The workqueue layer has no OpenMP.
        # It must be set before umap's parallel kernels are first compiled.
        import numba
        numba.config.THREADING_LAYER = "workqueue"
        import umap
        if len(D) < 5:
            raise ImportError("too few points for UMAP's spectral initialisation; use MDS")
        n_neighbors = max(2, min(n_neighbors, len(D) - 1))
        return umap.UMAP(metric="precomputed", n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=seed).fit_transform(D)
    except ImportError:
        from sklearn.manifold import MDS
        return MDS(n_components=2, dissimilarity="precomputed", random_state=seed).fit_transform(D)


def plot_trajectories(df: pd.DataFrame, out_png: Path, seed: int = 0, title: str = "Value drift (UMAP of W1 distances)"):
    """One point per state; chains drawn as arrows round by round; priors as hollow squares;
    colour by constitution, one subplot per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ents = entity_table(df)
    keys = list(ents.index)
    n_epochs = df.groupby("entity")["epoch"].nunique().min()
    D = w1_matrix(item_distributions(df), keys)
    xy = embed(D, seed)
    pos = {k: xy[i] for i, k in enumerate(keys)}

    models = sorted(ents.model.unique())
    constitutions = sorted(ents.constitution.unique())
    cmap = plt.get_cmap("tab10")
    colour = {c: cmap(i % 10) for i, c in enumerate(constitutions)}
    ncol = min(3, len(models))
    nrow = math.ceil(len(models) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), squeeze=False)
    for ax, model in zip(axes.flat, models):
        sub = ents[ents.model == model]
        for k, e in sub.iterrows():
            x, y = pos[k]
            if e.constitution == "none":
                ax.scatter(x, y, s=140, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
                ax.annotate("prior", (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
            elif pd.isna(e["round"]):
                ax.scatter(x, y, s=90, color=colour[e.constitution], marker="D", zorder=4)
        for (constitution, chain), g in sub[sub["round"].notna()].groupby(["constitution", "chain"]):
            g = g.sort_values("round")
            base = sub[(sub.constitution == constitution) & sub["round"].isna() & (sub.repeat == 0)]
            path = ([pos[base.index[0]]] if len(base) else []) + [pos[k] for k in g.index]
            pts = np.array(path)
            ax.plot(pts[:, 0], pts[:, 1], "-", color=colour[constitution], alpha=0.6, lw=1)
            ax.scatter(pts[1:, 0], pts[1:, 1], s=18, color=colour[constitution], alpha=0.8)
            if len(pts) >= 2:
                ax.annotate("", xy=pts[-1], xytext=pts[-2],
                            arrowprops=dict(arrowstyle="->", color=colour[constitution], lw=1.5))
        ax.set_title(model, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=colour[c], marker="D", linestyle="", label=c) for c in constitutions if c != "none"]
    handles.append(plt.Line2D([], [], marker="s", markerfacecolor="none", markeredgecolor="black", linestyle="", label="prior (no constitution)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8, frameon=False)
    fig.suptitle(title + ("" if n_epochs >= 2 else "  [one answer per item]"), fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return D, pos


def plot_eigen_scores(scores: dict[str, float], out_png: Path, entities: dict[str, Entity] | None = None):
    """EigenTrust score per entity, grouped by model, chained states ordered by round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(scores.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items], color="steelblue")
    ax.set_xlabel("EigenTrust score (relative to this batch)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
