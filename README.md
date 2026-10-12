# SAMA-EA

Implementation and recorded experiments for resource-constrained multi-project scheduling. MA-EA combines hierarchical GA/DE control with schedule realignment. SAMA-EA adds XGBoost screening and selective verification. Population selection, policy credit and model labels use verified schedules.

## Contents

| Directory | Contents |
| --- | --- |
| `src/` | Final search engine, policy, scheduling evaluator and surrogate models |
| `baselines/` | Resource-buffer GA and parallel priority-rule comparator implementations |
| `configs/` | Development configurations and the complete 28-configuration budget/entropy study |
| `data/dataset/` | Native RCMP inputs, converted JSON and conversion scripts |
| `results/comparison/` | Six methods, 24 comparison instances and seeds 1–30 |
| `results/development/` | Component, parameter and screening analyses used in the manuscript |
| `results/mechanism/` | Every configuration and run from the expanded budget/entropy study |

Comparison and mechanism records include stored final schedules. Development records retain run summaries and convergence trajectories. Trajectories use lossless `.json.gz` compression; other result files retain their recorded contents. `results/files.csv` provides hashes of the uncompressed files.

## Installation

Use Python 3.10 and a virtual environment:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before running these commands if Python otherwise points to another installation. PyTorch runs on the CPU. Each worker uses one numerical-library thread.

## Reproduce experiments

Run the full six-method comparison with up to 30 parallel workers:

```sh
python run.py comparison --workers 30
```

Run selected development configurations:

```sh
python run.py development --configurations C0_Y2 B2_SAMA_y2 --workers 30
python run.py mechanism --workers 30
```

Default seeds are 1–30. MA-EA and SAMA-EA comparison runs use 50 individuals and 200 generations, with adaptive one- or two-round realignment. SaDE and QLESCA use the same population/epoch limits. B23-GA-CB retains its native population of 250 and uses a 10000-pass budget; the product 50×200 defines that budget without changing its internal population. Development and mechanism runs use one round unless the selected configuration specifies forward-only decoding. MA-EA uses 10000 scheduling passes; SAMA-EA uses 6000, including initialisation and executed backward/forward passes. The full-pool screening diagnostic deliberately performs additional verification.

New runs are written to `outputs/` and resume when a completed result is present. They cannot overwrite the archived `results/` directory. Restrict the instance, seed, population and generation count for a quick installation check:

```sh
python run.py comparison --configurations MA_EA SAMA_EA --instances HHH+3 --seeds 1 --population 8 --generations 10 --workers 2
```

Such reduced runs check execution and feasibility; their results do not reproduce the manuscript protocol. Clear shell variables beginning with `SAMA_` before reproducing the default configuration, as the engine permits environment overrides.

## Inspect results

```sh
python summarise.py
python analyse.py
python verify_results.py
```

The summary reports mean best TPM, sample standard deviation, executed scheduling work and measured time for each instance/configuration. Study identifiers distinguish independent experiment batches. Parameter gaps in the manuscript use equal-instance normalisation; they are not percentage differences between grand means. Runtime values are batch-dependent. The development reference and mechanism study are retained as separate records. Use `n_exact_sgs` for executed work: some earlier comparison records store the common nominal budget in `budget_target_exact`. P-SGS/MINSLK uses parallel construction; its period counter is omitted from the serial-SGS summary column.

Read a compressed trace with standard Python:

```python
import gzip, json
with gzip.open("results/comparison/SAMA_EA/HHH+3/seed_001/convergence.json.gz", "rt", encoding="utf-8") as stream:
    trace = json.load(stream)
```

## Benchmarks and comparators

RCMPSPLIB comes from the [Ghent University scheduling data portal](https://www.projectmanagement.ugent.be/research/data); MPLIB1 and MPLIB2 come from the [MPLIB repository](https://github.com/MarioVanhoucke/MPLIB-Multi-Project-Scheduling-Library/releases/tag/v1.0). Native release dates, durations, precedence and renewable-resource quantities are preserved. The four development instances are `mpj60a2`, `mpj60a4`, `MPLIB1_Set1_1` and `MPLIB2_Set1_1`; the remaining 24 instances form the comparison set.

The resource-buffer comparator reproduces Bredael and Vanhoucke (2024), *A genetic algorithm with resource buffers for the resource-constrained multi-project scheduling problem*, under the stated common-budget protocol. The parallel SGS comparator implements the MINSLK construction of Villafáñez et al. (2019). SaDE and QLESCA use the corresponding mealpy implementations. XGBoost follows Chen and Guestrin (2016). See `REFERENCES.md` for bibliographic details and data access links. Benchmark data and external dependencies remain subject to their respective terms.

This folder is the repository content. Extract the distribution archive before adding the files to GitHub; the archive itself is not the repository source tree.
