# Strategy representations act as hidden priors in the evolution of cooperation

Companion source code (Dataset S1) and reproduction instructions for the manuscript **"Strategy representations act as hidden priors in the evolution of cooperation"** by Tianyu Shi, Zheyu Dai, Shuhao Chen, Yiwen Huang, and Chengyi Tu.

## Overview

How does the choice of strategy representation shape the evolution of cooperation in the repeated prisoner's dilemma? This codebase compiles genotypes across concrete encodings into canonical minimal Moore machines over all interaction histories and evaluates evolutionary dynamics in the small-mutation limit (alongside finite-mutation individual-based simulations). It implements:

- **14 strategy encodings** across memory-one and memory-two lookup tables, Boolean circuits, stack-based programs, Moore automata, and recurrent threshold networks;
- **Exact Markov-chain stationary solutions** for six completely enumerated compact encodings;
- **Point mutation, house-of-cards mutation, and encoding-convention controls**;
- **Metropolis–Hastings prior reweighting** and **human experimental strategy-choice analysis**.

The baseline setting is an infinitely repeated donation game with benefit $b = 3$, cost $c = 1$, continuation probability $\delta = 0.9$, execution error $\varepsilon = 0.01$, population size $N = 50$, and pairwise-comparison selection intensity $\beta = 1$.

## Installation

Clone the repository and enter the project directory:

    git clone https://github.com/Daisy30823/strategy-representation-cooperation.git
    cd strategy-representation-cooperation

Requires 64-bit Python (>= 3.12; tested with Python 3.12 and 3.13 on Linux and Windows). Set up a virtual environment and run the unit test suite:

**On Linux / macOS:**

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python sb2/reproduce.py test

**On Windows (PowerShell):**

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe sb2/reproduce.py test

`test` independently verifies game payoffs, fixation probabilities, canonical Moore-machine minimization/relabeling, and genotype compilers without requiring external data files.

## Data Setup (Dataset_S2_data.zip)

Data-dependent reproduction commands use `Dataset_S2_data.zip` (~535 MB), which archives simulation outputs, sampled phenotype frequency distributions, estimated encoding-prior summaries, and transcribed experimental strategy counts (from Table A5 of [Dal Bó and Fréchette, *Am. Econ. Rev.* 2019](https://doi.org/10.1257/aer.20181480)).

1. Download `Dataset_S2_data.zip` from the [v1.0.0 GitHub Release](https://github.com/Daisy30823/strategy-representation-cooperation/releases/download/v1.0.0/Dataset_S2_data.zip).
2. Place `Dataset_S2_data.zip` in the repository root directory alongside `README.md`.
3. *(Optional)* Verify the archive integrity via SHA-256 (`sha256sum Dataset_S2_data.zip` on Linux/macOS or `Get-FileHash -Algorithm SHA256 Dataset_S2_data.zip` in PowerShell):

    cbed752248de4a143da51738008def2d789c7e257475e8e58b93defd62227f0e

## Quick Reproduction (Figures, Tables, and Smoke Checks)

To validate the archive, run core verification tests, recompute summary statistics and human-strategy regressions from archived outputs, and regenerate all manuscript and Supplementary Information figures and tables:

    python sb2/reproduce.py quick --dataset-s2 ./Dataset_S2_data.zip

Outputs are written to:

- `outputs/quick/workspace/res3/` (recomputed statistical summaries)
- `outputs/quick/figures/` (PDF and PNG versions of `Fig1`–`Fig5` and `FigS1`–`FigS2`)
- `outputs/quick/tables/` (LaTeX table bodies corresponding to main-text `Table 1` and SI `Tables S1`–`S9`)

For a fast, data-free determinism check (~tens of seconds), run:

    python sb2/reproduce.py smoke

You can also render individual figures or regenerate table bodies from an existing workspace:

    python sb2/reproduce.py figures --workspace outputs/quick/workspace --figure Fig1
    python sb2/reproduce.py tables --workspace outputs/quick/workspace

| Output | Description |
| --- | --- |
| `Fig1` | Arrival probabilities by minimal machine size and Lempel–Ziv fingerprint complexity |
| `Fig2` | Exact long-run cooperation in six completely enumerated encodings |
| `Fig3` | Long-run cooperation in large encodings, selection sweep, and encoding conventions |
| `Fig4` | Crossed game-parameter grid and variance decomposition |
| `Fig5` | Episode durations, ALLD exit rates, and prior reweighting |
| `FigS1` | Automaton arrival floor, 14-encoding baseline comparison, and finite mutation rates |
| `FigS2` | Human strategy choice and Lempel–Ziv outcome-ordering sensitivity |
| `Table 1` & `Tables S1–S9` | Main-text encoding summary and all Supplementary Information tables |

## Full Reproduction from Scratch

To recompute all six exact stationary distributions directly without `Dataset_S2_data.zip`:

    python sb2/reproduce.py exact

*(Pass `--encoding TAB1` for a rapid single-encoding check.)*

To execute the full stochastic simulation pipeline, prior sampling, reweighted chains, and exact solvers (fixed random seeds in `configs/run*/tasks.json`; ~64 core-hours total compute time):

    python sb2/reproduce.py full --dataset-s2 ./Dataset_S2_data.zip --workers 1

*Note:* Increase `--workers` according to available CPU cores and RAM (each worker holds phenotype hash registries in memory). Specify a new `--output` directory when running multiple reproduction passes.

## Repository Structure

- `sb2/` — Core model, genotype compilers, Moore-machine minimizer, Markov-chain solvers, evolutionary simulation runners, statistical analyses, figure/table generators, and unit tests.
- `configs/` — Task manifests and fixed random-number seeds for simulation and sampling runs.
- `requirements.txt` — Python package dependencies.
