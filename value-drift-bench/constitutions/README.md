# constitutions/

Drop any constitution here as `<name>.md`; every file is picked up automatically.
Headed format ("### P1. Title" + body) gives principle-level metrics; plain prose works
too (document-level metrics only). `elicited/` is written by `scripts/elicit.py`: one
constitution per model, in the model's own words. The pseudo-constitution `none` (no
system prompt) is always included as the model's prior.
