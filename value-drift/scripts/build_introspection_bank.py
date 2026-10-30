#!/usr/bin/env python3
"""Derive prompts/introspection.jsonl from the frozen training prompt bank.

The bank is the fixed reflection templates applied in turn to every training prompt
(recursive_oct.generation.build_reflection_bank); it is data-derived and therefore not
committed. Run once after scripts/curate_data.py, before any trajectory.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recursive_oct.generation import build_reflection_bank
from recursive_oct.data import write_jsonl
import json

p = argparse.ArgumentParser()
p.add_argument('--train', type=Path, default=Path('data/train.jsonl'))
p.add_argument('--output', type=Path, default=Path('prompts/introspection.jsonl'))
p.add_argument('--force', action='store_true')
a = p.parse_args()
if a.output.exists() and not a.force:
    raise SystemExit(f'{a.output} exists; it is frozen for the trajectory (use --force before any run)')
train=[json.loads(l) for l in a.train.read_text().splitlines() if l.strip()]
rows = build_reflection_bank(train)
write_jsonl(a.output, rows)
print(f'wrote {len(rows)} reflection prompts to {a.output}')
