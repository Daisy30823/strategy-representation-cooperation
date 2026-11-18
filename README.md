# Strategy representations act as hidden priors in the evolution of cooperation

Research code prepared from the revised PNAS submission package. The reviewed manuscript has author and repository placeholders. Its publication status, authorship metadata, and DOI remain unverified.

How does the choice of strategy representation change the evolution of cooperation in a repeated prisoner's dilemma? The model compiles genotypes into minimal Moore machines over all histories, then evolves genotypes in the small-mutation limit. It compares 14 encodings, exact stationary distributions for six small encodings, point and house-of-cards mutation, and Metropolis–Hastings reweighting. The baseline is a donation game with benefit 3, cost 1, continuation probability 0.9, decision error 0.01, population 50, and pairwise-comparison selection intensity 1.

## Install

Clone the public repository and enter the project directory:

```sh
git clone https://github.com/Daisy30823/strategy-representation-cooperation.git representation-prior-code
cd representation-prior-code
```

Public GitHub repository: https://github.com/Daisy30823/strategy-representation-cooperation

Use 64-bit CPython 3.13 and a CPU. The audit used Windows 11 and Python 3.13.9. Linux and macOS are untested. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$py = '.\.venv\Scripts\python.exe'
& $py sb2/reproduce.py test
```

On POSIX systems:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python sb2/reproduce.py test
```

The pins describe versions used in the Windows audit, not a fully verified cross-platform environment. The audit used preinstalled packages because network access prevented a fresh pip install. A clean installation on Windows, Linux, or macOS still needs verification.

## Quick start

The data-dependent commands need `Dataset_S2_data.zip` (about 535 MB), which is **not included in the tracked source repository**. For local use, put the authorized archive in the repository root, alongside this `README.md`. This matches the current local folder layout. The archive is excluded from Git by `.gitignore`.

Dataset S2 is publicly available from [GitHub Release v1.0.0](https://github.com/Daisy30823/strategy-representation-cooperation/releases/download/v1.0.0/Dataset_S2_data.zip).

The reference archive used by the audit has SHA-256:

```text
cbed752248de4a143da51738008def2d789c7e257475e8e58b93defd62227f0e
```

You can check a local copy with `Get-FileHash -Algorithm SHA256 Dataset_S2_data.zip` in PowerShell or `sha256sum Dataset_S2_data.zip` on Linux/macOS. If a separately curated public dataset is released, its checksum and any necessary verifier changes must be documented.

From the repository root, use the interpreter set up above to run:

```sh
python sb2/reproduce.py quick --dataset-s2 ./Dataset_S2_data.zip
python sb2/reproduce.py smoke
```

`quick` validates the archive size and SHA-256, runs the core tests, rebuilds the reweighting priors, refits the human-strategy analysis, recomputes statistics from saved outputs, and generates all seven figures and ten table bodies. Some historical inputs are archived aggregates, including the automaton-floor measurements and the original selection sweep. Outputs go to `outputs/quick/workspace/res3/`, `outputs/quick/figures/` (PDF and PNG), and `outputs/quick/tables/` (LaTeX row fragments plus the zero-count strategy list for Table S4).

`smoke` is data-free. It runs a 2,000-draw sample, a 2,000-proposal baseline chain, and a 2,000-proposal reweighted chain twice. It compares the results for fixed seeds, ignoring timing. It usually takes tens of seconds on the audit machine. `test` runs the payoff, fixation, state-relabeling, and compiler checks independently.

## Figures and tables

After Quick or Full, pass the resulting `workspace` directory to `figures` or `tables`. Omit `--figure` to render every figure. Repeat `--figure` to select multiple figures.

```sh
python sb2/reproduce.py figures --workspace outputs/quick/workspace --figure Fig1
python sb2/reproduce.py tables --workspace outputs/quick/workspace
```

| Output | Subject |
| --- | --- |
| `Fig1` | Arrival probabilities by machine size and fingerprint complexity |
| `Fig2` | Exact stationary cooperation in six small encodings |
| `Fig3` | Baseline, selection sweep, and encoding conventions |
| `Fig4` | Crossed game design and variance decomposition |
| `Fig5` | Episode lengths, ALLD replacement, and prior reweighting |
| `FigS1` | Arrival floor, 14-encoding baseline, and finite mutation |
| `FigS2` | Human strategy choice and Lempel–Ziv ordering sensitivity |
| `Table1.tex` | Six main encodings, prior masses, and long-run cooperation |
| `TableS1.tex` through `TableS9.tex` | Maps, OFAT, mechanisms, human counts, neighborhoods, exact checks, convergence, grid, reweighting |

The table command writes table bodies for inspection and reuse. Some outputs depend on historical aggregates included with Dataset S2 rather than on newly rerun original simulations.

## Full reproduction

```sh
python sb2/reproduce.py exact
python sb2/reproduce.py full --dataset-s2 ./Dataset_S2_data.zip --workers 1
```

`exact` runs all six complete enumerations without a data archive. Use `--encoding TAB1` for a smaller standalone check. `full` computes the revised simulations and samples, builds their reweighting priors, runs the reweighted chains and exact solvers, then produces the same analysis outputs as Quick. It imports the historical aggregates and transcribed human-choice counts from the verified archive. The revision batches are themselves runnable from the included code and manifests. The complete historical workflow still requires archived v1 outputs.

The revised simulation seeds are in `configs/run*/tasks.json`. The archived task durations total about 64 CPU-core hours for the revised stochastic experiment. Start with one worker. Higher concurrency multiplies peak worker memory. The audit machine had 20 logical CPUs and 31.7 GiB RAM. Budget several GB of disk space for the archive, extracted inputs, generated priors, results, and compilation caches. Quick and Full require a fresh output path. Use `--output outputs/quick-2` for a second Quick run. The audited Quick, core tests, smoke checks, and six-encoding exact computation passed. The full formal stochastic workload was **not** executed in that audit; prior experiment results and historical v1 aggregates remain part of the input archive. Do not describe Quick as a complete independent rerun of all underlying simulations.

## Code and data availability

- `sb2/` holds the model, simulation runner, exact solver, analysis, figures, tables, and test entry points.
- `configs/` holds the four revised task manifests. Large data archives and derived outputs are external.

Dataset S2 contains archived simulation outputs, historical aggregate inputs, and a transcription of Table A5 from Dal Bó and Fréchette (2019). Permission to redistribute the transcription and other included material must be checked before publication of a data asset. Original article: [Dal Bó and Fréchette (2019)](https://pubs.aeaweb.org/doi/abs/10.1257/aer.20181480). The repository can be used for data-free `test`, `smoke`, and `exact` commands without Dataset S2.

For now, refer to *Strategy representations act as hidden priors in the evolution of cooperation*, revised submission package, Datasets S1 and S2. Add verified author names, bibliographic metadata, and the approved code/data DOI or commit identifier when available.

## Release and licensing status

The research code and Dataset S2 are publicly available on GitHub.

Release v1.0.0: https://github.com/Daisy30823/strategy-representation-cooperation/releases/tag/v1.0.0

Dataset S2 is provided as a separate downloadable ZIP attachment. No software license is currently declared. Redistribution permissions for included third-party material remain to be confirmed.
