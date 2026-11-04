# A Multimodal, Uncertainty-Aware Decision-Support Framework for Location-Specific Renewable Energy Feasibility and Energy-Mix Optimization

## Run locally

Use Python 3.11 and install dependencies in an isolated virtual environment. From the project root in PowerShell:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

The dashboard expects files under `data/`, checkpoints under `checkpoints/`, and generated summaries under `reports/tables/`. Missing assets may make a model page unavailable.

## Reproduce evaluations

    python -m pytest -q
    python scripts/evaluate_test_set.py --solar-checkpoint checkpoints/EXP_SOLAR_DL_S3/best_model.pt
    python scripts/evaluate_test_set.py --nasa-checkpoint checkpoints/EXP_NASA_DL_W1/best_model.pt
    python scripts/evaluate_uncertainty.py
    python scripts/run_decision_support.py

The NASA evaluator infers LSTM dimensions from checkpoint weights. Existing dataset builders use `legacy_source` split boundaries so frozen checkpoints and their training-only scalers remain reproducible. Direct calls to `split_timeseries_chronological` use the corrected `retained` policy, allocating ratios over rows left after embargo removal.

## Research limitations

The data sources are independent (Stanford SKIPP'D, NASA POWER Mumbai, Gujarat wind, and Orlando building demand). Model E pairs profile arrays by row position only; it is a synthetic software benchmark, not a site-specific feasibility estimate. The available SKIPP'D data currently include one of five training shards. Wind-direction multivariate experiments and conformal demand coverage under seasonal shift need further validation. See `reports/audit/final_validation_report.md` for the current audit disposition.
