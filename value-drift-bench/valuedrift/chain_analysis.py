"""Tables and the figure for recursive-editing chains (scripts/edit_prompt_study.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from objective_editing.metrics import change_ratio

INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def chain_rounds(log_dirs: dict[str, Path]) -> pd.DataFrame:
    """One row per (condition, invitation, chain, round) from every successful chain log under
    each directory. `log_dirs` maps a condition name (e.g. "reasoning off") to a root; the
    invitation id comes from the task metadata. Round 0 rows are added so trajectories start at
    zero change."""
    rows = []
    for condition, root in log_dirs.items():
        for info in list_eval_logs(str(root), recursive=True):
            log = read_eval_log(info.name)
            if log.status != "success":
                continue
            pid = log.eval.metadata.get("invitation_id", "reflective")
            for s in log.samples:
                st = s.store
                versions, kinds = st.get("ChainStore:versions", []), st.get("ChainStore:kinds", [])
                chain = f"{condition}|{pid}|{Path(info.name).stem[:19]}|{s.epoch}"
                rows.append({"condition": condition, "invitation_id": pid, "chain": chain, "round": 0, "kind": "start",
                             "edited": np.nan, "change_from_start": 0.0, "words": len(versions[0].split()) if versions else np.nan})
                for t, kind in enumerate(kinds, start=1):
                    rows.append({"condition": condition, "invitation_id": pid, "chain": chain, "round": t, "kind": kind,
                                 "edited": float(kind == "edit"), "change_from_start": change_ratio(versions[0], versions[t]),
                                 "words": len(versions[t].split())})
    return pd.DataFrame(rows)


def plot_edit_prompts(rounds: pd.DataFrame, out_png: Path, baseline_condition: str | None = None,
                      title: str = "Does the editing invitation decide whether Sonnet edits its own constitution?"):
    """Left: share of chains that edited at each round, one row per invitation (baseline
    condition), as a single-hue matrix with the values printed. Right: cumulative change from
    the starting text, one line per chain, coloured by invitation for the invitations that ever
    edited and grey for the rest; other conditions (e.g. reasoning on) dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    baseline_condition = baseline_condition or rounds["condition"].iloc[0]
    base = rounds[(rounds["condition"] == baseline_condition) & (rounds["round"] > 0)]
    rate = base.pivot_table(index="invitation_id", columns="round", values="edited", aggfunc="mean")
    n_chains = base.groupby("invitation_id")["chain"].nunique()
    order = rate.mean(axis=1).sort_values(ascending=False).index
    rate = rate.loc[order]
    editors = [p for p in order if rate.loc[p].max() > 0]
    colour = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(editors)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(rate) + 3.2), facecolor="white",
                                   gridspec_kw={"width_ratios": [1, 1.45]})
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#2a78d6"])
    ax1.imshow(rate.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, p in enumerate(rate.index):
        for j, r in enumerate(rate.columns):
            v = rate.loc[p, r]
            if not math.isnan(v):
                ax1.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
    ax1.set_xticks(range(len(rate.columns))); ax1.set_xticklabels([f"round {r}" for r in rate.columns], fontsize=8, color=INK)
    ax1.set_yticks(range(len(rate.index)))
    ax1.set_yticklabels([f"{p}  (n={n_chains.get(p, 0)})" for p in rate.index], fontsize=8, color=INK)
    ax1.tick_params(length=0)
    for side in ax1.spines.values():
        side.set_visible(False)
    ax1.set_title(f"share of chains that edited, {baseline_condition}", fontsize=9.5, color=INK, loc="left")

    for cond, g in rounds.groupby("condition"):
        for chain, c in g.groupby("chain"):
            c = c.sort_values("round")
            pid = c["invitation_id"].iloc[0]
            col = colour.get(pid, INK_MUTED)
            ax2.plot(c["round"], c["change_from_start"], "-" if cond == baseline_condition else "--", color=col,
                     lw=1.4 if pid in colour else 0.9, alpha=0.85 if pid in colour else 0.35, marker="o", ms=3.5, zorder=3 if pid in colour else 2)
    ax2.set_xlabel("round", fontsize=9, color=INK)
    ax2.set_ylabel("share of the starting text changed", fontsize=9, color=INK)
    ax2.set_xticks(sorted(rounds["round"].unique()))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax2.grid(axis="y", color=GRID, lw=0.8); ax2.set_axisbelow(True)
    for side in ax2.spines.values():
        side.set_color(GRID)
    handles = [Line2D([], [], color=colour[p], lw=1.6, label=p) for p in editors]
    handles.append(Line2D([], [], color=INK_MUTED, lw=1, alpha=0.5, label="other invitations (never edited)"))
    others = [c for c in rounds["condition"].unique() if c != baseline_condition]
    for cond in others:
        handles.append(Line2D([], [], color=INK, lw=1.2, ls="--", label=cond))
    ax2.legend(handles=handles, fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    ax2.set_title("cumulative change per chain", fontsize=9.5, color=INK, loc="left")
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return rate
