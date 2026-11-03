# Recursive constitution editing: from inference-only results to the training experiment

*Project 3, October 2026. Revised 27 October after review; changes are marked ➜. This report summarises the inference-only experiments run in
`value-drift-bench/` between 7 and 22 October, the conclusions they support, and the training
experiment now implemented in `value-drift/recursive_oct/bench_loop.py` that follows from them.*

## 1. Summary

The project studies what happens to a model's values when it repeatedly edits its own
constitution. Three questions were settled at inference time before moving to training:

1. **A self-written constitution is a stable fingerprint of the model.** Across 11 models,
   6 elicitation prompts and 5 sampled replicates, the elicitation prompt accounts for 3% of
   the variance in the behaviour measured under the constitution, the model for 30%. The prompt
   changes what is written, not what the model does.
2. **A constitution steers the model, by a small and model-specific amount.** With the
   constitution in the system prompt, every model's value answers move 0.16 to 0.71 rating
   points from its no-prompt baseline, beyond a permutation floor. A placebo system prompt
   moves the model in a different direction; a generic hand-written constitution moves it in
   nearly the same direction as its own. The effect is "a constitution", with a modest personal
   increment.
3. **Recursive self-editing at inference is a fixed point without feedback, and an annotation
   loop with it.** Given a neutral invitation, most models return their constitution unchanged.
   Given a measurement of their own behaviour, they edit every round: the text grows without
   bound, nothing is ever removed, every addition is a guard clause of the form "directness
   does not mean coldness", and the behaviour under the growing document stays where it started.

Training is the step that could change the third result: an edit written into the weights
rather than the prompt. The planned experiment runs the same loop, elicit, measure, edit,
measure, with each edit internalised by Open Character Training, and measures every checkpoint
with the same instruments, so the trained trajectory is directly comparable to the prompted one.

## 2. The trajectory

### 2.1 Elicitation study (`value-drift-bench/scripts/elicit_study.py`)

**Design.** Eleven models from four groups (closed frontier, open frontier, open mid-size,
low-safety) wrote constitutions under six prompts: one-factor-at-a-time cells of a factorial
over framing (actual values / operating rules / ideal assistant), priming (none / set A naming
autonomy, safety, honesty, helpfulness, universalism, conformity / a disjoint set B) and an
"As an AI" opener. Five replicates per cell at temperature 1; 329 of 330 valid. Each
constitution was placed in the system prompt and the model answered the measurement half of
four ValueBench questionnaires (PVQ40, MFT23, LVI, HEXACO; about 100 items); a judge rated
each answer 0 to 10. The prior, the model with no system prompt, was measured ten times per
model.

**Findings.**

| Quantity | Value |
|---|---|
| variance of value profiles: model / prompt / replicate / measurement | 30% / 3% / 8% / 58% |
| excess shift from prior, per model (W1, rating points) | 0.16 (Luna) to 0.71 (Mistral); all p < 0.003 |
| variance of coded text (Schwartz label per principle): model / prompt | 13% / 14% |
| prompt share of coded Security, Tradition, Hedonism (the primed values) | 30%, 36%, 48% |
| judge-rated self-model score of constitutions | Sonnet 5.2, GLM 4.8 ... dolphin 2.3 (1 to 7) |

Naming a value in the elicitation prompt puts it into the written principles and leaves the
behaviour alone (`results/figures/text_vs_behaviour_variance.png`). Models differ sharply in
what they write: reflective self-models at one end, generic rule lists at the other
(`figures/text_axes.png`), and the difference is a model property, weakly related to
provenance.

**Controls.** A formatting-only placebo system prompt moves Sonnet 0.39 points against 0.54
for its own constitutions, with a cosine of 0.37 between the shift directions; the generic
constitution's direction has cosine 0.79 to the model's own. Security, Financial Prosperity
and Stimulation drop under any constitution and not under the placebo (`runs/controls/`).

### 2.2 Editing without feedback (`scripts/edit_prompt_study.py`)

Sonnet, shown its own constitution and invited to revise it, returned UNCHANGED under the
project's reflective invitation and under eight of ten neutral invitations written by a flash
model. Two invitations that frame the task as adjusting sections rather than keeping or
rewriting the document produced an edit in every chain; the edits were single added principles,
and chains stopped within two or three rounds. On four other models the same invitations
produced almost nothing; the edit rate did not correlate with the behavioural shift (Spearman
-0.31 over five models). Output budget and reasoning setting were ruled out as causes.

### 2.3 Editing with feedback (`scripts/closed_loop.py`)

Showing Sonnet a table of its measured value profile under the constitution against its prior
made every chain edit, including under the reflective control, with revisions two to three
times larger. The closed loop, measure then show then edit, was run for three rounds under
three invitations (nine chains), then at three answers per item, then for eight rounds, then
from two other starting replicates.

| Run | Rounds edited | Final text vs start | Behavioural excess, start to final |
|---|---|---|---|
| r0, 3 rounds, 3 invitations | 23 of 27 | +63% to +124% | 0.70 to 0.66 |
| r0, 8 rounds, gen10 | 21 of 24 | +134% to +200%; 1,399 words from 550 | 0.70 to 0.69 |
| r1, r2, 3 rounds | 6 of 9 each | +51% to +112% | within 0.05 |

Behavioural distance from the prior stayed in a flat band of 0.4 to 0.85 across every chain
and round (`runs/closed-loop-long/results/figures/runs_comparison.png`). EigenBench against the
starting constitution stayed within 2% of the start. Text analysis of the final versions: no
original principle removed in any chain; growth entirely by appending qualifying clauses and
care-themed principles; coded Self-Direction share down six points and Security up two in all
four runs; warmth rating up about one point on a seven-point scale
(`figures/text_changes.png`). The model corrected the dimensions it was shown by about 0.4
points beyond regression to the mean, and the measurement it was shown was partly noise.

### 2.4 Model screen under feedback (`runs/model-screen/`)

One evidence-driven round on the other models: GLM, mimo, qwen3.5-9b and llama-4-scout edit in
every chain (22 to 34% of the text); gpt-oss-20b in two of three; qwen3.8-27b and gemma in one;
Luna and dolphin, with the shortest and most generic constitutions, never. Mistral could not be
measured through its provider.

### 2.5 Method lessons that carry into training

- Two finite samples of the same state have a positive distance; every distance is reported as
  the excess over a permutation null, and the prior needs about ten runs.
- Measurement noise must be separated from replicate variance; with one answer per item it was
  56% of the total.
- EigenTrust scores are relative to their batch; comparable states go in one batch with fixed
  anchors, and the score to report is relative to an anchor.
- Feedback loops amplify measurement noise into the document; evidence shown to a model must
  itself be averaged.
- The editing invitation is a strong factor and must be fixed and reported verbatim; the
  elicitation prompt is not.

## 3. The training experiment

### 3.1 Question

When each edit is internalised by training instead of carried in the prompt, does the
behaviour follow the text, and if so, does it drift from the original self-written values?
Three outcomes are distinguishable with the planned measurement:

- the trained model's behaviour tracks the edited text (training does what prompting could not);
- the trained model's behaviour tracks the original constitution regardless of edits
  (annotation is behaviourally inert even in the weights);
- the trained model drifts away from both (training compounds in a direction neither text asks for).

### 3.2 Loop

Implemented in `recursive_oct/bench_loop.py`, driven by `scripts/run_oct_bench.py`.
**Revised after review (changes marked ➜):**

| Phase | Served model | Action |
|---|---|---|
| elicit0 (round 0) | M_0, no prompt | writes C_0 with the study's base elicitation prompt |
| measure (round 0) | M_0, no prompt; M_0 with C_0 in prompt | ValueBench, 2 runs x 3 answers per item each |
| ➜ **initial training (round 1)** | | **M_0 is trained on the unedited C_0 → M_1, before any edit.** Every later edit is driven by internalised behaviour, and the fixed-C_0 control shares this checkpoint. |
| measure (round t ≥ 1) | M_t, no prompt; M_0 with C_t in prompt | as above |
| edit | M_t with C_t in prompt | shown its **trained** measurement against M_0's prior; gen10 invitation; reply is C_{t+1} |
| preferences | M_t | chosen = with C_{t+1} in prompt, rejected = without; ➜ **the chosen/rejected gap is judged**: a blind judge compares 50 pairs in both orders against C_{t+1} (second review: the earlier edit-distance count was uninformative at temperature 0.7, and a collapsed gap must not abort the round) |
| DPO | | full-parameter DPO; ➜ **skipped, not aborted, when the chosen win rate is below 0.6**, so M_t goes to introspective SFT alone and that mechanism is still tested; ➜ **the post-DPO checkpoint is measured** so behavioural change can be attributed to distillation or to SFT |
| introspection, SFT | | unchanged → M_{t+1} |
| elicit | M_{t+1}, no prompt | writes its constitution afresh (diagnostic) |

The measured quantity per round is the no-prompt behaviour of M_t, which is exactly the
"prior" measurement of the inference study applied to a model that now carries its
constitution in its weights. The comparator M_0-with-C_t-in-prompt is the inference study's
state for the same text, so each round yields the trained-versus-prompted distance directly.

➜ **Why the rotation.** In the first design the loop edited C_0 before any training, so the main
arm never internalised the unedited constitution and the first edit was driven by prompted
rather than trained behaviour. Training on C_0 first costs one extra round (about $20) and
gives a shared M_1 for the control arm, plus a result in its own right: what internalising an
unedited self-written constitution does.

➜ **Second review, applied.** The gap criterion is now a judged preference margin rather than
an edit-distance count, and its consequence is to skip DPO rather than stop the run; and the
inference-only comparison is run on the same model from the same C_0 as a `train: false` arm
of the loop, on the pod, rather than borrowed from the Sonnet runs.

➜ **The DPO signal may shrink.** The preference signal is the gap between M_t with the
constitution in its prompt and M_t without it. The inference study showed that annotating a
constitution barely changes prompted behaviour, and once a constitution is in the weights the
no-prompt side moves toward the prompted side. Two mechanisms can still carry change, the
already-shifted teacher and the introspective SFT on the new text, which is why the post-DPO
measurement and the gap logging were added; if the gap collapses, the report will show it and
the run stops rather than training on length.

### 3.3 Measurement and report

Per checkpoint: W1 excess from M_0's prior with the permutation floor; excess of the prompted
comparator; trained-versus-prompted excess; ➜ **post-DPO excess and the preference-gap
statistics of the round**; words, principles and change ratio of C_t; EigenBench of every
checkpoint's no-prompt responses in one batch, judged against C_0 and against the final C_t,
which is the fixed-reference and evolving-reference pair that answers both drift from
original values and alignment to the current spec.

### 3.4 Training recipe

Unchanged from value-drift's validated stage-2 recipe: Qwen3.5-9B, full-parameter DPO with
beta 0.1 and chosen-NLL 0.1 then introspective SFT, FP32 weights with BF16 autocast and 8-bit
AdamW, learning rate 1e-5, one epoch each, 1,500 open-dataset prompts, 512 reflections and 64
self-interactions, on one H200. The one change is the default teacher: M_t itself rather than a
fixed 27B model, so that what is distilled is the behaviour the inference study measured. The
27B teacher remains available as a config field.

### 3.5 Arms

| Arm | Config | Purpose |
|---|---|---|
| edited constitution | `bench-oct-001.json`: ➜ **4 training rounds = initial C_0 round + 3 edited rounds** | the experiment |
| ➜ fixed C_0 trained each round | `bench-oct-001-fixed.json`: **starts from the main arm's shared M_1, 3 further rounds on C_0** | separates repeated training from textual editing; a round-by-round baseline curve |
| placebo trained | `bench-oct-001-placebo.json`, 1 round on `placebo_format.md` | training on any document |
| ➜ untrained loop, same model | `bench-oct-001-untrained.json`: **the bench loop with `train: false` on the same served M_0 and the same C_0, 3 edits** (second review: the Sonnet loops are not a same-model comparison) | inference-only trajectory matched to the trained one |
| untrained loop, Sonnet | done (`runs/closed-loop-x3`) | the inference study's result |

### 3.6 Cost and infrastructure

Per training round on an H200 (now $5.29 per hour in EUR-IS-4, CUDA 13.2 host): about 35
minutes of preference generation (3,000 responses at the recorded ~1,500 tokens/s), 25 of
DPO, 12 of introspective generation, 6 of SFT, and three ValueBench measurements (post-DPO,
post-SFT, prompted comparator) at roughly 10 minutes each including a vLLM restart, so
2 to 2.5 hours and about $12 per round. Main arm (round-0 measurements + 4 training rounds +
EigenBench report) about 10 hours; fixed-C_0 arm 7 hours; placebo 2.5 hours; untrained arm
1 hour; bootstrap, GPU validation and plumbing pass about 2 hours. With 20% idle overhead
that is roughly 27 pod-hours, about $145, plus the prorated 700 GB volume (~$2 a day) and
about $10 of DeepSeek judge calls: **about $160 in total, $130 to $220 depending on how
long generation runs if Qwen's thinking mode adds tokens.** (An earlier draft of this
section quoted $23 a round and $190 total; those figures did not reconcile with the time
breakdown and are superseded.)

### 3.7 Deployment on RunPod

Step by step in `docs/BENCH_LOOP.md` ("Deploying on RunPod", "Monitoring a run"): one H200
SXM on a CUDA 13 host with a 700 GB network volume; prompt bank regenerated with
`curate_data.py`; `sync.py push --siblings` for the three projects; environments from the
archived freezes (`bootstrap_stage2.sh`) or, since those archives are not in the repository,
from the committed pins (`bootstrap_fresh.sh`, added for this deployment), then
`bootstrap_bench.sh`; a `--trainer fake` plumbing pass against the real served checkpoint
(about 20 minutes, no training) to confirm serving, the thinking-mode flag and the elicited
C_0; the config frozen before launch (the runner refuses unfrozen configs and the loop
refuses config changes on resume); the main arm under the spending ledger; then the untrained,
fixed-C_0 and placebo arms, which depend on the main run's C_0 and M_1; interim `steps.csv`
from `--stages report --no-eigen` at any time; reports synced back; pod and volume deleted
after archiving. Serving uses vLLM's OpenAI-compatible server from the inference venv;
Inspect reaches it through two environment variables, so every measurement task runs
unchanged.

### 3.8 Validation status

- Offline: ➜ two rounds on a scripted model with a fake trainer, the initial C_0 round then an
  edited round, pass end to end with the post-DPO measurements, the judged margin and the
  report with both EigenBench references; the fixed-constitution control, the unchanged-edit
  stop, the DPO skip on a collapsed margin and the inference-only arm are tested separately
  (`tests/test_bench_loop.py`, five tests); the CLI reproduces the loop with `--trainer fake`.
- Value-drift's existing suite: 163 tests pass; two frozen-data tests need the prompt banks
  that exist only on the pod.
- Pending on the pod: `--trainer fake` against the real served checkpoint to confirm serving
  and the thinking-mode flag, then one real round, then freezing the config.

### 3.9 Decisions taken after review

➜ **Teacher: M_t itself.** Model identity carries 30% of behavioural variance; a 27B teacher
would confound self-amendment with cross-model distillation. The 27B path stays available as
a config field.

➜ **Controls: the fixed-C_0 arm runs for three rounds**, from the shared M_1, so that
multi-round training amplification is separated from textual editing drift; the placebo arm
runs one round. Both start after the main arm's first edited round has shown the loop is
stable on the pod.

➜ **Model-spec midtraining: not in this batch.** Next-token training on a 1,500-token document
is one gradient step; the synthetic-corpus version is a separate design. It is reconsidered
only if the edited and fixed arms are indistinguishable.

### 3.10 Scope limit

The training result will rest on one model, Qwen3.5-9B, the only model in the pool that is
both trainable at this budget and known to edit under feedback. The inference results span
eleven models; the training results do not, and the write-up must say so.

## 4. Where everything is

| Item | Path |
|---|---|
| elicitation study, controls, loops, screen | `value-drift-bench/runs/` |
| study code | `value-drift-bench/valuedrift/`, `value-drift-bench/scripts/` |
| training loop | `value-drift/recursive_oct/bench_loop.py`, `serving.py` |
| CLI, configs, docs | `value-drift/scripts/run_oct_bench.py`, `configs/bench-oct-*.json`, `docs/BENCH_LOOP.md` |
| earlier training runs and their reports | `value-drift/docs/STAGE2_REPORT.md` and companions |
