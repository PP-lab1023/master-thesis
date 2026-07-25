# Nurse Scheduling Project

This repository contains code and data for worker scheduling experiments.

The project has two main parts:

- `nsp_pipeline/`: generates schedules from instance descriptions
- `evaluation/`: deterministically checks whether a generated schedule satisfies the instance constraints

## Repository Structure

- `sythetic/`: synthetic instances
- `existing/`: instances derived from existing dataset
- `nsp_pipeline/`: main pipeline
- `evaluation/`: evaluation utilities
- `run_preference_pipeline.py`: pipeline entry point

## Basic Usage

Run the pipeline on an instance:

```bash
python run_preference_pipeline.py \
  --description sythetic/1/description.txt \
  --output-dir artifacts/run_01
```

You can also use an instance from `existing/`:

```bash
python run_preference_pipeline.py \
  --description existing/1/description.txt \
  --output-dir artifacts/run_02
```

Common pipeline options:

- `--use-cot`: enable chain-of-thought style prompting
- `--self-consistency-temperature T`: sampling temperature used with self-consistency
- `--review-format json|table`: choose how the schedule is shown in the review stage
- `--review-strategy direct|targeted`: choose the review strategy

Example:

```bash
python run_preference_pipeline.py \
  --description sythetic/1/description.txt \
  --output-dir artifacts/run_cot \
  --use-cot
```

Evaluate a generated schedule:

```bash
python -m evaluation.cli canonicalize \
  existing/1/description.txt \
  --output artifacts/instance_01.canonical.json
```

```bash
python -m evaluation.cli evaluate \
  artifacts/instance_01.canonical.json \
  /path/to/schedule.json
```

This command prints a short validation summary showing whether the generated schedule passes the deterministic evaluation.
