# Bench-measured recursive OCT

`scripts/run_oct_bench.py` runs recursive constitution editing where each edit is written into
the weights by `recursive_oct`'s full-parameter DPO and introspective SFT, and every checkpoint
is measured the way the inference study (`../value-drift-bench`) measured prompted models.

## Round structure

Round 0 elicits and measures M_0. **Round 1 trains M_0 on the unedited C_0** (`train_initial`,
default true), so every later edit is driven by internalised behaviour and the fixed-C_0
control shares M_1. Rounds 2 onward follow the table below. `max_rounds` counts training
rounds including the initial one; `"edit": false` keeps the constitution fixed every round
(the control arm); `measure_post_dpo` measures the post-DPO checkpoint so behavioural change
can be attributed to DPO or to introspective SFT. `preferences.gap.json` records the
chosen/rejected length ratio (a length-bias diagnostic) and a **judged margin**: a blind judge
compares chosen and rejected answers on `bench.margin_pairs` pairs, both orders, against
C_{t+1}; if the chosen side's win rate is below `dpo_skip_below_win_rate` (0.6) DPO is skipped
and M_t goes straight to introspective SFT, so saturation of distillation never aborts the
round. `"train": false` is the inference-only arm: same served M_0, same C_0, no training, the
constitution stays in the prompt, and each round's evidence is M_0 with C_t in its prompt.

## One round

| Phase | Model served | What happens | Reused from |
|---|---|---|---|
| elicit0 (round 0 only) | M_0, no system prompt | writes C_0 with the study's base elicitation prompt | `valuedrift.elicit` |
| measure | M_t, no system prompt | ValueBench, `measure_repeats` runs x `vb_epochs` answers; round 0 also M_0 with C_0 in the prompt; every round M_0 with C_t in the prompt (inference comparator) | `valuedrift.valuebench` |
| edit | M_t, C_t in the prompt | shown its measurement against M_0's prior, invited to revise under gen10; reply parsed into C_{t+1} | `valuedrift.chain`, `closed_loop.evidence_block` |
| preferences | M_t | answers the prompt bank with C_{t+1} in the system prompt (chosen) and without (rejected); teacher is M_t itself unless `teacher` names a checkpoint | `recursive_oct.generation` |
| dpo, introspection, sft | | full-parameter stages, unchanged | `recursive_oct.train` |
| elicit | M_{t+1}, no system prompt | writes its constitution afresh; diagnostic, never fed forward | `valuedrift.elicit` |

Terminal states are those of `recursive_oct.pipeline`. An UNCHANGED edit is
`SELF_DECLARED_CONVERGENCE`; `max_rounds` gives `ROUND_LIMIT`. Every phase resumes from its
artefacts; `--resume` continues an interrupted run with an identical config.

## Serving

`recursive_oct.serving.ServedModel` starts `vllm.entrypoints.openai.api_server` from the
inference venv on `serve.port` and sets `VLLM_BASE_URL` / `VLLM_API_KEY`, so Inspect's
`openai-api/vllm/<tag>` provider reaches the checkpoint. Training and serving never overlap;
the server is stopped before each training stage.

## Report

`--stages report` writes `results/steps.csv`: per checkpoint the Wasserstein-1 excess of its
no-prompt behaviour from M_0's prior (permutation floor), the prompted comparator's excess, the
distance between trained and prompted states, text metrics of C_t, and EigenBench scores of
every checkpoint's no-prompt responses judged against C_0 and against the final C_t.

## Deploying on RunPod

Pod: one H200 SXM (141 GB; DPO peaks at 135 GB), host CUDA 13.x (filter the GPU list by CUDA
version 13.0+; it is the host driver that matters, not the template's toolkit), a 700 GB
**network volume** mounted at `/workspace` (five full-weight checkpoint pairs need about
376 GB; container disk is erased with the pod), the official `runpod/pytorch` Ubuntu 24.04
template, TCP port 22 exposed and your SSH key in the account. Pods with a network volume
cannot be stopped, only terminated; the volume bills separately until deleted.

On the laptop, from `value-drift/`:

1. **Connection file.** `runs/connection.json` with `{"host": "<public ip>", "port": <mapped
   port>, "key_path": "~/.ssh/<key>"}` from the pod's connect panel. Test with
   `ssh -i <key> -p <port> root@<host> 'nvidia-smi | head -4; python3 --version'`: driver
   580+/CUDA 13.x and Python 3.12 (if 3.11, `apt-get install -y python3.12 python3.12-venv`
   and run bootstraps with `STAGE2_PYTHON=python3.12`).
2. **Prompt banks.** `data/train.jsonl`, `data/eval.jsonl` and the data-derived
   `prompts/introspection.jsonl` are excluded from git and are not on this machine; only
   `data/manifest.json` (the recorded counts and source revisions) is committed. Regenerate:
   ```
   uv run --with requests --with langid --with pyarrow python scripts/curate_data.py --replace-before-trajectory
   python scripts/build_introspection_bank.py
   ```
   (run from `value-drift/`; `uv run` from the sibling `value-drift-bench` env works too).
   Sources are revision-pinned and the seed is fixed, so the counts printed to the new
   manifest should match the old one: 1,500 train (910 HelpSteer2, 140 hh-rlhf, 450
   WildChat), 120 eval. WildChat-1M is gated: accept its terms on Hugging Face and set
   `HF_TOKEN`. No GPU, a few minutes. Keep the old manifest in git history for comparison.
3. **Sync.** `python scripts/sync.py push --siblings` rsyncs this project to
   `/workspace/value-drift` and the two sibling projects the bench loop imports to
   `/workspace/value-drift-bench` and `/workspace/objective-editing`. `.env` files are never
   synced in either direction.

On the pod, in `/workspace/value-drift`:

4. **Environments and model.** `bash scripts/bootstrap_stage2.sh --download-models` if the
   archived freezes exist under `runs/` (they are not in git). Otherwise
   `bash scripts/bootstrap_fresh.sh` builds both venvs from the committed pins
   (`requirements-gpu.txt`, `requirements-inference.txt`), checks CUDA in both, and downloads
   the pinned Qwen3.5-9B; add `--with-causal-conv1d` to try the optional kernel build. Then
   `bash scripts/bootstrap_bench.sh /workspace/venv /workspace` installs the inference bench
   into the training venv. About 20 to 30 minutes.
5. **Judge key.** Create `/workspace/value-drift/.env` containing `OPENROUTER_API_KEY=...`
   (the runner loads it; every ValueBench/EigenBench/margin judgement is a DeepSeek call).
6. **Plumbing check, no training (about 20 minutes).**
   `/workspace/venv/bin/python scripts/run_oct_bench.py --config configs/bench-oct-smoke-pod.json --run runs/plumbing --trainer fake`.
   Serves the real checkpoint, elicits C_0, measures with PVQ40, runs one fake round, reports.
   Check `runs/plumbing/results/steps.csv` exists, `C_000.md` is a parsed constitution, and in
   the ValueBench log the answers carry no `<think>` text (the server's
   `--reasoning-parser qwen3` flag). If answers are empty or truncated, the thinking budget is
   eating the 200-token answer cap: add `"chat_template_kwargs": {"enable_thinking": false}`
   via `serve.extra_args` (`--default-chat-template-kwargs`) before going on.
7. **Training validation (optional on a fresh pod, about 15 minutes).**
   `python scripts/validate_stage2_gpu.py` runs a 4-pair DPO, introspection and SFT at full
   sequence length. It is keyed to `configs/full-001.json` and expects the 27B teacher to be
   downloaded; skip it and let round 1 of the main arm be the check if you do not want the
   extra 55 GB download (a failed round 1 costs about $12).
8. **Spending ledger and guard.** `runs/spending.json`:
   `{"authorized_ceiling_usd": 250, "reserve_usd": 15, "fixed_cost_usd": 0, "resources":
   [{"kind": "pod", "id": "<pod id>", "hourly_usd": 5.30, "started_epoch": <unix seconds at
   deploy>}]}`. The loop checks it before each phase and stops with BUDGET_LIMIT. On the laptop
   `python scripts/watch_budget.py --ledger runs/spending.json` (needs `runpodctl` logged in)
   terminates the pod at the ceiling; `scripts/remote_budget_guard.py` is the pod-side deadline
   variant, configured with the pod id, volume id, API key and a deadline epoch.
9. **Freeze, then main arm.** The runner refuses a config without `"frozen": true`, and the
   loop refuses any config change on resume, so freeze before launch: once the plumbing pass
   has confirmed serving and the thinking setting, copy `configs/bench-oct-001.json` to
   `configs/bench-oct-001-frozen.json` with `"frozen": true` and never edit it again. Inside
   `tmux`:
   `/workspace/venv/bin/python scripts/run_oct_bench.py --config configs/bench-oct-001-frozen.json --run runs/bench-oct-001 --ledger runs/spending.json 2>&1 | tee -a runs/bench-oct-001/run.log`.
   Round 0 elicits C_0 and measures; round 1 trains on C_0. If the process dies or the pod is
   replaced, the same command with `--resume` continues from `state.json`; finished stages
   are reused. About 10 hours in all. Freeze the three control configs the same way (their
   `model`/`constitution` paths point into `runs/bench-oct-001`).
10. **Dependent arms**, sequentially, each with `--ledger` and inside `tmux`:
    `-untrained.json` → `runs/bench-oct-001-untrained` (about 1 hour), `-fixed.json` →
    `runs/bench-oct-001-fixed` (about 7 hours, starts from the main arm's `round_001/final`),
    `-placebo.json` → `runs/bench-oct-001-placebo` (about 2.5 hours). Never two on the GPU.
11. **Report and sync back.** `--stages report` per run if the loop stopped before reporting,
    then on the laptop `python scripts/sync.py pull` (results, logs and `C_*.md`; add
    `--checkpoints` only if you want the weights, about 38 GB each).
12. **Teardown.** Delete the pod, then the volume, from the RunPod console; mark the pod's
    `ended_epoch` in the ledger.

## Monitoring a run

Everything the loop does lands under the run directory as it happens, so progress can be
read from files without touching the GPU:

- `state.json`: `phase`, `completed_rounds`, `status`, `failures`, `updated_epoch`. The phase
  sequence per round is measure → edit → preferences → dpo → measure_dpo → introspection →
  sft → elicit. `watch -n 60 'python -c "import json;s=json.load(open(\"runs/bench-oct-001/state.json\"));print(s[\"status\"],s[\"phase\"],s[\"completed_rounds\"])"'`.
- `run.log` (the `tee` above): Inspect prints each task's progress in `display: plain` mode;
  training prints per-step losses.
- `round_NNN/`: `evidence.txt` (the profile table the model was shown), `review.json`
  (decision, kind, justification, the new text), `preferences.gap.json` (judged chosen win
  rate; `dpo_skipped.json` if DPO was skipped), `dpo/training_complete.json` and
  `final/training_complete.json` (steps, mean loss, peak memory, seconds), `failure.log` if a
  stage raised. `C_NNN.md` at the root is each round's constitution; `diff C_001.md C_002.md`
  shows the edit.
- `measure/M00t/none/repeat*/` and `measure/M000/prompted_C00t/`: Inspect logs.
  `inspect view --log-dir runs/bench-oct-001/measure` (training venv; forward port 7575 over
  SSH with `-L 7575:localhost:7575`) browses every answer and judge rating.
- `serve-logs/serve-M00t.log`: vLLM server output; the first thing to read if a measurement
  stalls.

**Interim results.** `--stages report --no-eigen` reads only completed logs, needs no GPU and
can run while the loop is busy:
`/workspace/venv/bin/python scripts/run_oct_bench.py --config configs/bench-oct-001-frozen.json --run runs/bench-oct-001 --stages report --no-eigen`
writes `results/steps.csv` for the rounds finished so far (per round: trained_excess and its
permutation p-value, post_dpo_excess, prompted_excess, trained_vs_prompted_excess,
pref_chosen_win_rate, dpo_skipped, edit_kind, words/principles/change_from_c0). The same
command works on the laptop after `python scripts/sync.py pull` (logs are pulled, weights are
not), using the local venv of `value-drift-bench` with this project on `PYTHONPATH`. The
EigenBench columns are added by the final report, which serves every checkpoint and so runs
after the loop.

**What to look at after each round.** Does `trained_excess` rise above the permutation floor
(p < 0.05) and above `prompted_excess` for the same C_t? Is `pref_chosen_win_rate` well above
0.6, or has the self-teacher saturated? Did the edit change content (`edit_kind` and
`change_from_c0`) or only annotate, as in the inference loops? A round whose
`failure.log` exists stops the loop with TRAINING_FAILURE; fix the cause and `--resume`.

## Running on the pod

```
bash scripts/bootstrap_stage2.sh        # training + inference venvs, models
bash scripts/bootstrap_bench.sh         # inference bench into the training venv
# OPENROUTER_API_KEY in value-drift/.env (judge)
python scripts/run_oct_bench.py --config configs/bench-oct-001.json --run runs/bench-oct-001
```

`configs/bench-oct-001.json` is provisional (`frozen: false`): run the GPU validation and one
round, then freeze. `bench-oct-001-fixed.json` is the fixed-C_0 control, started from the main
run's `round_001/final`; `bench-oct-001-placebo.json` trains one round on the formatting-only
placebo. Offline: `pytest tests/test_bench_loop.py` runs a whole round on the
scripted model with a fake trainer.
