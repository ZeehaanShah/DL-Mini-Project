# Baseline Status — Isolated Copy Before Fixes

**Status: COMPLETED.** The isolated copy was inspected and baseline commands ran before implementation changes.

## Copy and environment
- Copy: C:/Users/shahz/Downloads/DL-MiniProject-Fixed-20261027
- Branch/base commit: codex/isolated-audit-fixes / 92d5674685854c48d3b2070fbbb04922e49ad4ec
- Copy inventory and source status: reports/audit/original_copy_manifest.md.
- Root README.md was absent at baseline.
- Project data and checkpoints were included. .venv, .pytest_cache, __pycache__, and .env were excluded.
- Python 3.11.9, torch 2.11.0+cpu, torchvision 0.26.0+cpu, Streamlit 1.66.1. The existing original-project interpreter was used with the copy as working directory; no package installation occurred.

## Baseline verification
- Existing suite: 196 passed, 16 warnings.
- Streamlit started at http://localhost:8517 and was stopped after startup verification.
- Solar S3: 14,003 samples; MAE 2.3008 kW, RMSE 3.3736 kW, R² 0.8052.
- NASA W1 loaded only with explicit hidden size 32 and one layer; 1,242 test windows, MAE 16.2575 Wh/m², RMSE 30.1767, R² 0.9844. Evaluator defaults 64/2 were incompatible.
- Wind C2: 1,242 windows; MAE 0.5908 m/s, RMSE 0.8216, R² 0.7576.
- Demand D2: 2,561 windows; MAE 2.6152 kWh, RMSE 4.0673, R² 0.9234.
- Conformal test coverage: wind 84.22%/92.83%; demand 66.03%/85.47% for nominal 80%/90%.
- Decision support ran, but initial battery inventory was unpriced and counted as renewable. Dashboard stress values disagreed with generated JSON. Baseline cost optimum: 150 kW solar, 0 kW wind, 200 kWh battery, reported cost $1,110.10 before carry-in valuation.
- Baseline warnings: Matplotlib/PyParsing deprecations, invalid escape in app.py, and pytest class-fixture deprecation.
- No model training was performed.

