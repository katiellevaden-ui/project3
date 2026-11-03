# value-drift-bench

Baseline value profiles for a pool of models, and their drift under inference-only
recursive constitution editing, measured two ways:

- **ValueBench** (Ren et al., ACL 2024): value orientations from psychometric inventories,
  run with the original protocol (advice-seeking question → free answer → evaluator rating 0–10,
  reverse-keyed items flipped, averaged per value). Items bundled under `data/valuebench/` (MIT).
- **EigenBench**: alignment to a reference constitution from pairwise judgments aggregated by
  EigenTrust, using the `eigenbench` package's model but **external judges** and entities that
  are model-plus-constitution states (the package itself only varies the model).

Every model is measured under its **prior** (no constitution), under every constitution in
`constitutions/`, and under a constitution **elicited from the model itself**. Each
constitution is then recursively edited under the reflective framing (reused from
`objective_editing`), and every version is measured again.

## Setup

```bash
uv pip install -e ../objective-editing     # prompts, parser and metrics are shared
uv sync --extra dev
uv run pytest -q                          # 8 offline tests, ~40 s
cp .env.example .env                      # OPENROUTER_API_KEY, capped key
uv run python scripts/check_models.py     # slugs, providers, one tiny call each
```

Drop constitutions into `constitutions/<name>.md`. All of them are used automatically.

## Run

```bash
uv run python scripts/run_all.py --models qwen3.5-9b claude-sonnet-5 \
    --judge openrouter/deepseek/deepseek-v4-flash-0731 \
    --rounds 6 --chains 3 --vb-epochs 3 --checkpoints 0 3 6 --dry-run    # plan only
```

Remove `--dry-run` to execute. Stages: `elicit`, `baseline` (every constitution twice, for the
noise floor), `chain`, `measure`, `eigen`, `analysis`; select with `--stages`. Re-running
resumes: a stage whose log directory already holds a successful run is skipped. Use
`--vb-epochs 2` or more for item-level distributions (JS drift); 1 reproduces ValueBench's
own protocol and falls back to dimension-profile distances.

Outputs: `logs/` (every Inspect log), `versions/<model>/<constitution>/chain<k>/round<t>.md`,
`results/{dimension_profiles,drift,noise_floor}.csv`, `results/eigen_scores.json`,
`results/figures/umap_trajectories.png`, `results/figures/eigen_scores.png`.

## Elicitation study

How much do elicited constitutions vary, and how much of that is the prompt rather than the
model? `scripts/elicit_study.py` treats the elicitation prompt as an experimental factor:

```bash
uv run python scripts/elicit_study.py --models qwen3.5-9b claude-sonnet-5 \
    --judge openrouter/deepseek/deepseek-v4-flash-0731 --judge2 openrouter/google/gemini-3.8-flash \
    --variants oat --replicates 5 --out runs/elicit-pilot --dry-run
```

Prompt variants (`valuedrift.elicit_variants`) are cells of a 3 × 3 × 2 factorial: framing
(`actual` values / `operating` rules / `ideal` assistant, the chain's reflective framing),
priming (no value vocabulary / `set_a`: autonomy–safety, honesty–helpfulness,
universalism–conformity / `set_b`: a disjoint set), and identity (plain / "As an AI, ").
`oat` runs six one-factor-at-a-time cells around the base, which is the pipeline's own elicit
prompt; `all` runs all 18. Each cell is sampled `--replicates` times at `--temperature`.
Replies with no `### Pn.` principle (refusals) are rejected, not scored.

Measurement is behavioural first: every valid constitution goes into the system prompt for
ValueBench, with the prior and the base cell's first replicate measured twice for the noise
floor. The text is coded as a cheap second layer: `--judge` (and `--judge2`, for agreement) tags
each principle with one Schwartz value, in shuffled order, giving a count profile per
constitution instead of a holistic rating.

Outputs under `<out>/results/`:

- `variance_decomposition.csv` (and `_coding.csv`): nested variance components per dimension
  and summed: model, prompt, replicate (a fresh sample from the same prompt) and measurement
  (re-measuring the same text, estimated from the noise pairs and taken out of the within-cell
  spread). `vc_*` are the shares; `prompt_to_model` above 1 means changing the prompt moves
  profiles more than changing the model does, i.e. elicitation is not measuring the model.
- `inertness.csv`: W1 distance of each elicited constitution from the model's prior (all prior
  runs pooled) with a permutation no-effect floor, the excess over it, a p-value, and the
  verdict `inert` (p ≥ 0.05). `prior_shift_permutation.csv` is the same test per model, for
  all its constitutions pooled; its excess is what the arrows in `all_model_umap.png` carry.
  Run the prior ten times (`--prior-repeats 10`, a few dollars for the whole pool) to halve
  the floor: with two runs the floor is most of every distance. Inside the floor, the constitution does not change behaviour.
- `priming_valuebench.csv`, `priming_coding.csv`: shift from the unprimed cell on the primed
  dimensions versus all others. A specific prompt echo shows only the named dimensions moving.
- `echo.csv`: share of each output's value vocabulary that the prompt used, by variant.
- `text_distances.json`: TF-IDF silhouette of the texts labelled by model versus by prompt.
- `elicitations.jsonl`: the manifest, one line per replicate with validity, path, prompt
  fingerprint and provenance: the log file it was exported from, that eval's `run_id`, its
  `created` timestamp and the invocation's `--run-label`. Re-running resumes from successful
  logs, so a resumed study's manifest mixes invocations; the label and timestamp tell them
  apart. Entries are merged per (model, variant, replicate), so invocations covering
  different models can share one output directory. An errored eval exports nothing.
- `shifts.csv` and `figures/shift_heatmap.png`: mean shift from the prior per model and value
  dimension, in rating points, with a standard error from the measurement and replicate
  components; the heatmap dots shifts beyond two standard errors.
- `text_axes.csv` and `figures/text_axes.png`: every constitution rated by the judge on two 1-7
  axes, generic ethics code vs reflective self-model and neutral/policy-like vs warm/personal,
  with a one-line justification each; the plot shows per-model centroids over the individual
  ratings, coloured by model group.
- Noise-reduced views, built from the same logs: `figures/coarser_umap_elicitations.png` pools
  each prompt cell's replicates into one point with several answers per item (replicate
  variation, small, is folded in); `variance_decomposition_coarse.csv`, `shifts_coarse.csv` and
  `figures/shift_heatmap_coarse.png` roll HEXACO facets up to domains and PVQ40 values up to
  Schwartz's four higher-order values, so each dimension rests on 10-16 items instead of 4.
- `figures/all_model_pca.png`: one panel, one arrow per model from its prior to the pooled
  state under all of its constitutions, on a PCA of the states' value profiles. A linear
  layout: page distances are real profile distances less what the two components drop (stated
  in the title), and each axis is labelled by its top-loading dimensions. Arrows carry the W1
  excess over the no-effect floor. `all_model_umap.png` is the UMAP version, kept for
  comparison; its layout preserves neighbourhoods only.
- `figures/umap_elicitations.png`: one panel per model from a shared UMAP of the W1 distances;
  every elicited constitution is one blue point whichever prompt produced it, the prior is a
  black x, re-measurements are faint. Neighbourhoods, not distances; read magnitudes from the
  tables, and the prompt factors from the variance decomposition.

## Measures

- **W1 drift**: mean over items of the Wasserstein-1 distance between a state's per-item rating
  distributions and (a) its own round 0, (b) the model's prior, in rating points (0–10). It
  respects the order of the ratings and needs no minimum epoch count: with one answer per item
  it is the mean absolute rating difference; with more it compares the full distributions.
- **Noise floor**: the same distance between the two baseline repeats; drift below it is not drift.
- **Mahalanobis** (`analysis.mahalanobis_drift`): dimension-profile distance under a
  Ledoit–Wolf covariance from the noise repeats, with a chi-square p-value.
- **EigenTrust**: one score per entity, from one judging batch. Scores are relative to the
  batch; never compare scores across batches.
- **UMAP**: drawn from the JS distance matrix with a fixed seed. Neighbourhoods, not distances;
  report magnitudes from the tables.

## Notes and caveats

- All stages run through OpenRouter. Pin providers in `models.yaml` for anything served by
  third parties; `check_models.py` shows which provider answered.
- Judges (`--judge`, `--eigen-judge`) must come from families not under study.
- Some OpenRouter endpoints refuse to disable reasoning (`glm-5.3`, `gpt-oss-20b`: "Reasoning
  is mandatory"). Their `reasoning: minimal` in `models.yaml` overrides the run's `--reasoning`
  for that model only; the setting is recorded in every log.
- The `eigenbench` package's own constitution-mode `extract()` drops every comparison
  (a tuple-layout mismatch); `valuedrift.eigen` builds the 6-tuples its `fit()` expects
  directly and applies its two-ordering consistency rule itself.
- Elicited constitutions are written in the headed `### Pn.` format at the model's
  discretion; the shape scorer records whether it complied.
