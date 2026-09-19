# TRT-VLBM experiment reproduction

Code and numerical data for **Two-relaxation-time vectorial lattice Boltzmann methods for incompressible viscous flows**. This repository reproduces the experiments in the main paper and supplementary material. The numbered scripts follow the order in which the experiments first appear in the current manuscript; supplement-only experiments follow the main-paper experiments.

All commands below run from **this repository's root**, the directory containing this file and `reproduce.py`. No files from a parent directory are needed.

## 1. Generate figures and tables from the supplied data

Use Python **3.12 or newer**, preferably in a virtual environment, and install the dependencies:

```console
python -m pip install -r requirements.txt
python reproduce.py render
```

This is the shortest reproduction route. It reads the **13 supplied CSV files**, generates **6 PDF figures** in `figure_output/` and **13 LaTeX tables** in `table_output/`, and performs no fluid simulation. The figures require no LaTeX installation. The table files are optional, inspectable text outputs; the repository does not compile the paper or the response letter.

To generate only figures, only tables, or one selected output:

```console
python reproduce.py figures
python reproduce.py tables
python reproduce.py figures --only figure_01_periodic_acoustic_histories
python reproduce.py tables --only table_05_poiseuille_boundary_tuning
python reproduce.py render --figure-output-dir results/figures --table-output-dir results/tables
```

`--only` accepts one or more generator stems, or filenames ending in `.py`. The names are listed below. A generator can also run directly:

```console
python _figures/figure_01_periodic_acoustic_histories.py
python _tables/table_05_poiseuille_boundary_tuning.py --output-dir results/tables
```

The supplied CSV data are sufficient for every figure and table listed here. **Complete `solver_instances/` files are not bundled.** They are created when experiments are recomputed and are unnecessary for this CSV-to-figure route.

## 2. Recompute the experiments

Select experiments using their two-digit IDs:

```console
python reproduce.py experiments --only 01
python reproduce.py experiments --only 02 03
python reproduce.py experiments --only 05 06 07
```

To recompute every published experiment, in manuscript order, and then render its data:

```console
python reproduce.py experiments
python reproduce.py render
```

Experiment commands write the corresponding files in `data_csv/`, replacing the supplied reference values, and save solver states under `solver_instances/`. Keep a separate checkout if you want to preserve an untouched reference dataset. Finish the selected experiments before rendering: several generators reject incomplete CSV datasets.

The full calculation is substantial. Fine-grid convergence runs, the 336-case parameter scan, the three-dimensional problems and the long Kolmogorov runs can require hours and several GiB of memory and disk space. Use the first section for a quick check of the published plots. All numerical calculations use double precision. The optional C accelerator described below speeds up experiments 06 and 07.

All nine scripts accept `--rerun`, which starts fresh cases instead of reusing matching saved states. Without it, compatible saved results are reused or continued where supported:

```console
python reproduce.py experiments --only 01 --rerun
python 05_poiseuille_boundary_tuning.py --rerun
```

Parameters, grids and stopping criteria are fixed in the scripts to the published experiment. The common conventions are `h = 1/N`, `dt = alpha*h**2`, `s_minus = 2*a/(a + alpha*nu)`, SRT `s_plus = s_minus`, and OTRT `s_plus = 2 - s_minus`. The special low-rate TRT comparison in experiment 09 uses its explicitly stated `s_plus` instead.

### 01 — Periodic acoustic stability (main paper, Section 4.1)

```console
python 01_periodic_acoustic_stability.py
```

- Grids: `N = 12, 16, 24, 32, 48, 64, 96, 128`; `nu = 0.01`, `a = 0.2`, `alpha = 0.5`.
- Equilibrium initialization with zero velocity and pressure amplitude `1e-4` at lattice frequency `pi/2`; compare OTRT and SRT at the same viscosity. This is a grid-scale mode test, not refinement of one fixed smooth solution.
- Run up to 800 steps, subject to the prescribed diagnostics and modal-gain limit of 1000. Fit the per-step modal factor over steps 50–250. In the supplied data, OTRT reaches 800 steps and SRT reaches the gain limit at step 435.
- Exports: `data_01_periodic_acoustic_results.csv`, `data_01_periodic_acoustic_modes.csv`, and `data_01_periodic_acoustic_histories.csv`. The history CSV contains the `N = 128` curves used in the figure; full per-case histories are stored with the recomputed solvers.

### 02 — Nonlinear manufactured solution (main paper, Section 4.2; supplement, SM6)

```console
python 02_nonlinear_manufactured_convergence.py
```

Use `N = 16, 24, 32, 48, 64, 96, 128, 192, 256`, `nu = 0.1`, `a = 0.2`, and target time `T = 0.25`. Both SRT and OTRT are run at `alpha = 0.2` and `0.5`, for 36 cases. The manufactured solution and forcing are defined in `d2n5_nonlinearmanufactured.py`. Export: `data_02_nonlinear_manufactured_results.csv`.

The convergence study records the velocity errors only for cases reaching the common target time. Cases stopped by a diagnostic limit retain their actual stopping time and diagnostics; their unavailable target-time errors are left empty in the CSV and shown as dashes in the summary table.

### 03 — Taylor–Green convergence (main paper, Section 4.3; supplement, SM6)

```console
python 03_taylor_green_convergence.py
```

Use the same 36 grid/method/`alpha` combinations and target time as experiment 02, with `U0 = 1`, wave numbers `2*pi`, and phases `-pi/2`. Export: `data_03_taylor_green_results.csv`. The main convergence figure and summary table combine experiments 02 and 03; run both before regenerating those combined outputs from new results.

Experiments 02 and 03 share `_periodic_config.py`. At every step, they monitor density in `[0.5, 1.5]`, density fluctuation at most `0.05`, divergence norm `D2` at most `0.5`, and lattice speed `Mh` at most `0.25`. These diagnostics also define the early-stopping comparison reported in SM6.

### 04 — Taylor–Green parameter scan (main paper, Section 4.3; supplement, SM2)

```console
python 04_taylor_green_parameter_sensitivity.py
```

Use the two grids `N = 96, 128`, `nu = 0.1`, `alpha = 0.2`, `T = 0.05`, and `a = 0.12, 0.16, 0.20, 0.24`. For each `a`, test `s_plus = 0.05, 0.10, ..., 2.00` and the exact SRT and OTRT rates: 42 rates on two grids, totaling 336 cases.

Export: `data_04_taylor_green_parameter_scan.csv`. Each case is marked `failure`, `outlier` or `regular`. A completed case is an outlier if `E_u2 > 2e-3` or `D2 > 3e-4`. The scan summary uses the paired coarse/fine results; missing or failed data must not be replaced by zero. By default, up to four worker processes are used; see the optional worker settings below.

### 05 — Poiseuille boundary tuning (main paper, Section 4.4)

```console
python 05_poiseuille_boundary_tuning.py
```

Use `N = 16, 24, 32, 48, 64, 96, 128`, `nu = 0.1`, `a = alpha = 0.2`, forcing `G = 0.08`, halfway walls `gamma = 0.5`, and `ell = 0`. The five rates are OTRT, `0.8*tuned`, `tuned`, `1.2*tuned`, and SRT, where the tuned rate is `s_plus = 1/3`.

This experiment solves the stationary discrete fixed-point equations with SciPy and checks the full population residual. It does not estimate the steady solution by choosing an arbitrary final integration time. Exports: `data_05_poiseuille_boundary_tuning.csv` (35 cases) and `data_05_poiseuille_profiles.csv` (the OTRT, SRT and tuned profiles at `N = 32`).

### 06 — Periodic D3N7 ABC/Beltrami flow (main paper, Section 4.5; supplement, SM1)

```console
python 06_beltrami_convergence.py
```

Use `N = 12, 16, 24, 32, 48, 64, 96, 128`, velocity amplitude `0.05`, wave number `2*pi`, `nu = 0.05`, `a = alpha = 1/7`, and `T = 3/14`. Compare OTRT `s_plus = 2/21` with SRT `s_plus = 40/21`; both use `s_minus = 40/21`. Export: `data_06_beltrami_results.csv` (16 cases), including component errors, combined velocity error and diagnostic values.

### 07 — D3N7 Ethier–Steinman flow (main paper, Section 4.6; supplement, SM3)

```console
python 07_ethier_steinman_convergence.py
```

Use `N = 16, 32, 64, 128`, flow constants `A = pi/4`, `D = pi/2`, `nu = 0.05`, `a = alpha = 1/7`, and `T = 1/16`. The SRT/OTRT rates are those of experiment 06. Prescribe time-dependent Dirichlet data on all six faces, with the halfway rule and `ell = 0` on every boundary link. Export: `data_07_ethier_steinman_results.csv` (8 cases).

### 08 — Three-group double-shear flow (supplement, SM4)

```console
python 08_double_shear_dynamics.py
```

Use `N = 12, 16, 24, 32, 48, 64, 96, 128`, `nu = 0.01`, `a = 0.2`, `U0 = 0.25`, shear thickness `delta = 0.1`, transverse amplitude `epsilon_v = 0.0125`, and the prescribed initial pressure. Compare OTRT at `alpha = 0.5`, SRT at `alpha = 0.5`, and SRT at `alpha = 0.2`.

The common physical endpoint for each grid is `T_N = ceil(N**2/5)/N**2`. Energy, divergence, density and lattice-speed criteria are evaluated each step. Export: `data_08_double_shear_results.csv` (24 cases). The table reports both the completed runs and the early stopping of the `alpha = 0.5` SRT runs. These nonlinear finite-time diagnostics should be interpreted separately from the paper's rest-state linear stability theorem.

### 09 — Forced Kolmogorov start-up flow (supplement, SM5)

```console
python 09_kolmogorov_startup_refinement.py
```

Use `N = 6, 12, 24, 48`, `nu = 0.01`, `a = alpha = 0.2`, force `(sin(2*pi*y), 0)`, and target time `T = 150`. The entire initial velocity is a vortex of amplitude `1e-3`; no steady shear is added. Initial pressure is zero. Compare TRT `s_plus = 0.02` with SRT `s_plus = s_minus = 200/101`.

Diagnostics are sampled every `0.1` time unit; stationarity is assessed over the last 5 time units with tolerance `1e-8`. The run continues to the common target time, rather than terminating at the first stationarity indication. Exports: `data_09_kolmogorov_startup_results.csv` (8 cases) and `data_09_kolmogorov_startup_fields.csv` (the final velocity fields). Two worker processes are used by default.

## 3. Output-to-manuscript map

The numbers in filenames identify experiments, not figure/table numbers in the paper. Figure generators are in `_figures/`; table generators are in `_tables/`. Add `.py` to a stem below to obtain its script name, `.pdf` for its generated figure, or `.tex` for its generated table.

| Location in paper | Generator stem |
|---|---|
| Section 4.1, acoustic figure | `figure_01_periodic_acoustic_histories` |
| Section 4.1, acoustic table | `table_01_periodic_acoustic_stability` |
| Sections 4.2–4.3, combined convergence figure | `figure_03_taylor_green_periodic_convergence` |
| Sections 4.2–4.3, accuracy summary | `table_03_periodic_accuracy_summary` |
| Section 4.3, scan summary | `table_04_taylor_green_parameter_summary` |
| Section 4.4, boundary-error figure | `figure_05_poiseuille_boundary_error` |
| Section 4.4, boundary-tuning table | `table_05_poiseuille_boundary_tuning` |
| Section 4.5, Beltrami convergence | `table_06_beltrami_convergence` |
| Section 4.6, Ethier–Steinman figure | `figure_07_ethier_steinman_convergence` |
| Section 4.6, Ethier–Steinman convergence | `table_07_ethier_steinman_convergence` |
| SM1, complete Beltrami errors | `table_06_beltrami_errors_full` |
| SM2, parameter-domain figure | `figure_04_taylor_green_parameter_domain` |
| SM3, complete Ethier–Steinman errors | `table_07_ethier_steinman_errors_full` |
| SM4, double-shear comparison | `table_08_double_shear_alpha` |
| SM5, Kolmogorov final fields | `figure_09_kolmogorov_startup_fields` |
| SM5, Kolmogorov refinement | `table_09_kolmogorov_startup_refinement` |
| SM6, manufactured convergence | `table_02_nonlinear_manufactured_convergence_full` |
| SM6, Taylor–Green convergence | `table_03_taylor_green_convergence_full` |
| SM6, early-stopping diagnostics | `table_03_periodic_stopping` |

The tables use ordinary LaTeX commands and `booktabs` rules, with no revision-color macros. Captions may refer to other tables or equations in the paper. Table templates control captions and layout; their numerical rows are generated from CSV, not hard-coded in the templates.

## 4. Optional C accelerator: source and build

`_d3n7_fast.c` implements the zero-force D3N7 collision and transport used by experiments 06 and 07. Boundary corrections remain in Python. The repository distributes the **C source and build helper**, not a prebuilt DLL. Build a library for the same architecture as your Python interpreter.

With GCC available on Windows, or a GCC/Clang-compatible C compiler available as `cc` on Linux/macOS:

```console
python build_accelerator.py
```

To select a compiler executable explicitly, including a path containing spaces:

```console
python build_accelerator.py --cc "C:/path/to/mingw64/bin/gcc.exe"
```

The helper uses `-O3 -std=c11` and native shared-library flags. It writes `_d3n7_fast.dll` on Windows, `_d3n7_fast.so` on Linux, or `_d3n7_fast.dylib` on macOS, beside the source. The default build has no OpenMP requirement. With a compiler and runtime supporting OpenMP, it can optionally be enabled:

```console
python build_accelerator.py --cc gcc --openmp
```

On Windows, keep the compiler/runtime directory available on `PATH`; the loader also checks the directory containing `gcc`. An OpenMP build additionally needs its matching runtime library. The build uses a temporary directory and only replaces the output after successful compilation. Close Python processes that have loaded the library before rebuilding it.

Check the selected implementation in a new Python process:

```console
python -c "import _d3n7_fast as f; print('C accelerator' if f._LIB is not None else 'NumPy fallback')"
```

If the native library is absent or cannot be loaded, experiments 06 and 07 automatically use the NumPy implementation. Compilation is optional; it does not affect the CSV-only rendering route.

## 5. Optional saved-state and worker settings

For experiments 02 and 03, compatible completed cases can be reused and earlier compatible saved states can be continued. The default search is the corresponding directory under `solver_instances/`. To select a saved-state directory explicitly:

```console
python reproduce.py experiments --only 02 --resume-from solver_instances/NonlinearManufacturedConvergence
python 03_taylor_green_convergence.py --resume-from solver_instances/TaylorGreenConvergence
```

`--resume-from` is available only for experiment 02 or 03 and cannot be combined with `--rerun`. Compatibility checks include the physical parameters, initial populations, numerical model source hashes, saved population hash, stopping limits and complete diagnostic history. Unmatched cases start afresh. States beyond the requested endpoint cannot be rewound, and a trajectory already stopped by a diagnostic criterion retains that event.

Experiments 02/03 save at the end of each case; forcefully terminating a process can lose the current unsaved case. Other experiments have their own saved-state or checkpoint logic. Saved files are local pickle files intended to be read by this code. A `.periodic_run.lock` prevents simultaneous 02/03 runs. After an abnormal termination, remove a stale lock only after checking that its process is no longer running.

To reduce simultaneous workers, set environment variables before running experiments 04 and 09. PowerShell:

```powershell
$env:TRTVLBM_SCAN_WORKERS = "1"
$env:TRTVLBM_KOLMOGOROV_WORKERS = "1"
python reproduce.py experiments --only 04 09
```

On Linux/macOS:

```sh
TRTVLBM_SCAN_WORKERS=1 TRTVLBM_KOLMOGOROV_WORKERS=1 python reproduce.py experiments --only 04 09
```

`reproduce.py` defaults numerical-library thread counts to one unless already set in the environment. This avoids multiplying BLAS/OpenMP threads across worker processes.

## 6. Repository contents and license

| Path | Purpose |
|---|---|
| `01_*.py` through `09_*.py` | Published experiment entry points |
| `d2n5_*.py`, `d3n7_*.py` | Models and numerical kernels |
| `_periodic_*.py`, other root `_*.py` files | Shared evolution, CSV, diagnostics and plotting helpers |
| `data_csv/` | 13 supplied numerical CSV files; included in version control |
| `_figures/`, `_tables/`, `_tables/templates/` | The 6 figure and 13 table generators and their layouts |
| `_d3n7_fast.c`, `build_accelerator.py` | Optional native accelerator source and build helper |
| `solver_instances/` | Generated solver states and diagnostics; not distributed |
| `figure_output/`, `table_output/`, `results/` | Generated outputs; ignored by Git |
| `requirements.txt`, `LICENSE`, `.gitignore` | Dependencies, license and generated-file exclusions |

The CSV schemas are defined in `_csv_io.py`. Values are stored without publication rounding; rounding is applied by the table generators. Empty error fields indicate unavailable results after a failed or stopped run. Floating-point details can vary between platforms and numerical-library versions; compare errors, rates, stopping behavior and plotted trends rather than expecting every recomputed byte to match.

The repository was checked on Windows with Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2 and PyMuPDF 1.28.2. `requirements.txt` gives the installation dependency ranges; the environment above is the one used for validation. The plotting helpers retain the dimensions, colors and export routines used to generate the figures; separate figure-audit scripts and manuscript/response compilation tools are not part of this reproduction package.

The code is licensed under **GNU GPL version 3 or, at your option, any later version**; see [LICENSE](LICENSE). Every Python source and the C source carry the same license notice. Existing author copyright notices are preserved. `.gitignore` excludes local environments, caches, solver instances, generated plots/tables and compiled libraries while retaining source files and the reference CSV data.
