# Local Environment Setup

This workspace uses a Python 3.12 virtualenv and an editable `sksfolio`
install. The optional capabilities below can be enabled independently:

- native partial-sort PAVA build support;
- OSQP;
- Gurobi;
- MOSEK;
- Julia wrappers.

Use the doctor first:

```powershell
.\.venv\Scripts\python.exe .\local_env_doctor.py
```

## Core path

The pure-Python fallback path is enough for:

- `full_sort` PAVA;
- `partial_sort` PAVA via NumPy fallback;
- PDHG;
- FISTA with `pava`, `budget`, and `dual_fista` proximal paths;
- benchmark runners that do not require OSQP or commercial solvers.

Refresh the editable install from the packaged source root with:

```powershell
Set-Location .\sksfolio-0.2.0\sksfolio-0.2.0
& "$PWD\..\..\.venv\Scripts\python.exe" -m pip install -e .
```

If PowerShell treats the relative path badly, use the full path from the
workspace root instead:

```powershell
& "$PWD\.venv\Scripts\python.exe" -m pip install -e .\sksfolio-0.2.0\sksfolio-0.2.0
```

## Native PAVA

The extension in `sksfolio/_native_pava.c` is optional, but it accelerates
the public `partial_sort` PAVA path. The current shell does not have
`cl.exe`, so rebuilds cannot compile yet.

1. Install Visual Studio Build Tools 2022.
2. Include the `Desktop development with C++` workload.
3. Open a `Developer PowerShell for VS 2022`.
4. From the workspace root, run:

```powershell
.\rebuild_native_pava.ps1
.\.venv\Scripts\python.exe .\local_env_doctor.py
```

Successful rebuild means `native-pava` reports `available=True`.

## OSQP

OSQP enables the exact weak-majorization QP oracle used by
`prox_oracle="majorization_qp"` and the `sksfolio-four-prox` comparison.

```powershell
Set-Location .\sksfolio-0.2.0\sksfolio-0.2.0
& "$PWD\..\..\.venv\Scripts\python.exe" -m pip install -e ".[qp]"
```

## Gurobi

Install the package only if you want the native commercial backend and you
have a valid license available to this machine or user profile.

```powershell
Set-Location .\sksfolio-0.2.0\sksfolio-0.2.0
& "$PWD\..\..\.venv\Scripts\python.exe" -m pip install -e ".[gurobi]"
```

Typical license-related environment variables:

```powershell
$env:GUROBI_HOME = "C:\gurobi1203\win64"
$env:GRB_LICENSE_FILE = "C:\gurobi\gurobi.lic"
```

The package can import without a usable license, but solves will still fail
or skip until the license is valid.

## MOSEK

Install the package only if you want the native commercial backend and you
have a valid MOSEK license file or server setup.

```powershell
Set-Location .\sksfolio-0.2.0\sksfolio-0.2.0
& "$PWD\..\..\.venv\Scripts\python.exe" -m pip install -e ".[mosek]"
```

Typical environment variable:

```powershell
$env:MOSEKLM_LICENSE_FILE = "C:\Users\<user>\mosek\mosek.lic"
```

## Julia wrappers

The Julia/JuMP wrappers are optional and not required for the native Python
backends.

```powershell
Set-Location .\sksfolio-0.2.0\sksfolio-0.2.0
& "$PWD\..\..\.venv\Scripts\python.exe" -m pip install -e ".[julia]"
```

On the first Julia-backed solve, pass `julia_instantiate=True`.

## Validation

Run the focused readiness checks:

```powershell
.\.venv\Scripts\python.exe .\local_env_doctor.py
& "$PWD\.venv\Scripts\python.exe" -m unittest `
  tests.unit.test_pava `
  tests.unit.test_majorization_qp `
  -v
```

Run a benchmark once the desired optional pieces are installed:

```powershell
Set-Location .\sksfolio-0.2.0\sksfolio-0.2.0
& "$PWD\..\..\.venv\Scripts\python.exe" -m sksfolio.benchmarks.run_relaxations `
  --output ..\..\results\sp500_rank50_k10.csv
```
