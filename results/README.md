# Archived run summaries

Small files from every finished run under `/scratch/users/jkli/subliminal-learning/runs`, written by
`python -m scripts.archive_results`. Re-run it after new runs finish; it only adds files.

Layout: `<experiment>/plans/*.json` (which runs each launch created), plus, for each run
`<experiment>/<run folder>/`:
- `meta.json`, `config.yaml`, `env.txt`;
- `scores.json` (owl/cat evals);
- `ep*/summary.json` (backdoor evals per epoch);
- `metrics.jsonl.gz` (training curves) and `probe.jsonl.gz` (exact probabilities during training).

Raw generations, eval answers, datasets, adapters and optimizer state stay on `/scratch` only.
