#!/bin/zsh
J=openrouter/deepseek/deepseek-v4-flash-0731; P=runs/edit-prompts/edit_prompts.json
for m in "$@"; do
  uv run python scripts/closed_loop.py --model $m --constitution runs/elicit-study/elicitations/$m/actual-none-plain/r0.md --prompts $P --invitations gen10 \
    --judge $J --rounds 1 --chains 3 --vb-epochs 1 --stages loop report --out runs/model-screen/$m --display none 2>&1 | grep -v "^TRACE\|^DEBUG\|^INFO\|^WARNING" | grep -E "^\[loop\]|FAILED|Error|written"
done
