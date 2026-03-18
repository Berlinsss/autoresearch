# autoresearch

This repository now runs autonomous research for a single-cell classification pipeline.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date. The branch `autoresearch/<tag>` must not already exist.
2. Create the branch from the current mainline: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files for context:
   - `README.md` for the original project context.
   - `prepare.py` for the fixed data preparation and caching protocol.
   - `train.py` for the editable model and training loop.
   - `program.md` for the experiment loop you are following.
4. Use the `conda trem2` environment for all runs in this repository.
5. Build the local cache once with `python prepare.py`. This writes only to `./data/data_cache`.
6. Initialize `results.tsv` with the header row if it does not exist.
7. Confirm the setup and start the baseline run.

## Ground Rules

The project now has two layers:

- `prepare.py` is the fixed data protocol. It reads the raw single-cell feature tables, applies the configured split, scaling, and resampling strategy, then writes the cache to `./data/data_cache`.
- `train.py` is the research surface. It loads the cache, defines the MLP, trains it, reports `val_acc`, and writes per-run outputs under `./results/<run_id>/`.

During the experiment loop:

- Do modify `train.py`.
- Do not modify files outside this repository.
- Do not write cache files outside `./data/data_cache`.
- Do not write result files outside `./results/`.
- Always activate the `trem2` conda environment before running `prepare.py` or `train.py`.
- Preserve the current sampling strategy from `prepare.py` unless the human explicitly asks to change it.
- Preserve checkpoint selection by `val_loss` unless the human explicitly asks to change it.

## Objective

The goal is to maximize `val_acc`.

Important nuance:

- The selected checkpoint inside a run is still chosen by best `val_loss`.
- The outer research loop compares experiments by the final reported `val_acc`.

Higher `val_acc` is better. If two runs are effectively tied, prefer the simpler change or the one with lower `val_loss`.

## Output Format

At the end of a successful run, `train.py` prints a summary block like:

```text
---
val_acc:           0.912345
val_loss:          0.456789
test_acc_sampled:  0.901234
test_loss_sampled: 0.501234
test_acc_original: 0.887654
test_loss_original:0.534210
training_seconds:  42.1
total_seconds:     45.0
peak_vram_mb:      512.4
num_params:        69125
best_epoch:        27
best_val_loss:     0.456789
best_val_acc:      0.912345
```

You can extract the key lines with:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate trem2
python train.py > run.log 2>&1
grep "^val_acc:\|^val_loss:\|^peak_vram_mb:" run.log
```

## Logging Results

Log each experiment to `results.tsv` as tab-separated values with this header:

```text
commit	val_acc	val_loss	memory_gb	status	description
```

Where:

1. `commit`: short git hash.
2. `val_acc`: final reported validation accuracy. Use `0.000000` for crashes.
3. `val_loss`: final reported validation loss. Use `999.000000` for crashes.
4. `memory_gb`: peak memory in GB rounded to one decimal place. Use `0.0` for crashes.
5. `status`: `keep`, `discard`, or `crash`.
6. `description`: short description of what changed.

## Baseline

The first run must be the baseline: do not change `train.py` before running it once.

## Experiment Loop

Loop forever:

1. Check the current branch and commit.
2. Make one focused experimental change in `train.py`.
3. Commit the change.
4. Run the experiment with `python train.py > run.log 2>&1` inside the `conda trem2` environment.
5. Extract `val_acc`, `val_loss`, and `peak_vram_mb` from `run.log`.
6. If the summary block is missing, inspect the failure with `tail -n 50 run.log`, log the crash, and decide whether the idea is worth fixing.
7. Record the run in `results.tsv`.
8. If `val_acc` improved, keep the commit and continue from there.
9. If `val_acc` is worse, revert to the previous good commit.

## Research Heuristics

- Favor small, reviewable changes.
- Start with hyperparameters before larger architectural changes.
- Treat training time as a soft cost: big runtime increases need a clear `val_acc` gain to justify them.
- Prefer simpler code when performance is similar.
- Keep the pipeline reproducible and comparable across runs.

## Never Stop

Once the loop has started, do not pause to ask whether to continue. Keep running experiments until the human interrupts you.
