# Risk-Aware Portfolio Construction

**Authors:** Nguyen Van Duy Anh · Pham Dinh Hieu · Cao Pham Minh Dang · Tran Anh Chuong · Ngo Dinh Khanh  
**Repository:** https://github.com/dinhieufam/FinLove  
**License:** MIT

---

## Overview

A reproducible Python portfolio research engine that blends Black–Litterman views, CVaR optimization, and dynamic risk (GARCH/DCC) to produce stable, risk-aware allocations with execution realism (transaction costs, rebalance bands).

- **Motivation:** Naïve 60/40 or equal-weight ignores tail risk, regime shifts, and costs. Classic Markowitz is fragile to estimation error and yields extreme, unstable weights.
- **Approach:** Combine shrinkage covariance (Ledoit–Wolf/GLASSO), Black–Litterman returns, tail-risk (CVaR) optimization, and dynamic volatility/correlation (GARCH/DCC). Add execution constraints and costs.
- **Outcome:** Smoother weights, controlled drawdowns/CVaR, lower turnover, and improved risk-adjusted returns vs. plain Markowitz.

---

## Features

- **Risk models**
  - Ledoit–Wolf shrinkage covariance
  - Graphical LASSO (GLASSO)
  - GARCH(1,1) per asset and DCC correlations (planned)
- **Return models**
  - Black–Litterman with data-driven, uncertainty-weighted views (planned)
- **Objectives**
  - Sharpe maximization, Minimum-Variance, CVaR minimization (planned)
- **Execution realism**
  - Transaction costs, rebalance bands, optional market impact (Almgren–Chriss) (planned)
- **Reporting**
  - Rolling Sharpe/volatility, Max Drawdown, VaR/CVaR, turnover, weight paths, regime labels (planned)

Status: Initial scaffolding in place; core modeling and backtesting components are being implemented.

---

## Data

- **Source:** Yahoo Finance via `yfinance`
- **Default universe:** Sector ETFs — `XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC`
- **Frequency:** Daily prices; compute daily log returns; aggregate to month-end for rebalancing; daily retained for GARCH
- **Fields:** Date, Adj Close, Close, Volume

Example loader: see `src/data.py` for a minimal `yfinance` download example.

---

## Project Structure

```
DATA2010/project/
├─ notebooks/
│  ├─ 01_data_pipeline.ipynb
│  ├─ 02_eda.ipynb
│  ├─ 03_markowitz_backtest.ipynb
│  ├─ 04_black_litterman.ipynb
│  └─ 05_garch_integration.ipynb
├─ src/
│  ├─ data.py          # data acquisition example
│  ├─ features.py      # feature engineering (planned)
│  ├─ risk.py          # covariance/GARCH/DCC estimators (planned)
│  ├─ optimize.py      # MV/BL/CVaR optimizers (planned)
│  ├─ backtest.py      # walk-forward, costs, bands (planned)
│  └─ metrics.py       # performance/risk metrics (planned)
├─ requirements.txt
└─ README.md           # you are here
```

---

## Quickstart

### 1) Environment

Python ≥ 3.10 recommended.

```
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Current requirements:
```
yfinance
```

Planned additions: `numpy`, `pandas`, `scipy`, `cvxpy` (or `scikit-learn`/`skfolio`), `arch`, `matplotlib`, `seaborn`, `plotly`.

### 2) Fetch data (example)

```
python src/data.py
```

This prints sample rows and summary statistics for a demonstration ticker. Replace ticker and dates as needed.

### 3) Run notebooks

Open the notebooks under `notebooks/` in Jupyter or VS Code. Suggested order:
1. `01_data_pipeline.ipynb` — acquisition, cleaning, returns, features
2. `02_eda.ipynb` — exploratory analysis, rolling vol/corr, PCA, clustering
3. `03_markowitz_backtest.ipynb` — baseline Markowitz
4. `04_black_litterman.ipynb` — BL returns and uncertainty-weighted views
5. `05_garch_integration.ipynb` — GARCH(1,1) + DCC and integration with optimizers

---

## Methodology

- **Covariance**
  - Ledoit–Wolf shrinkage for stability
  - Graphical LASSO for sparse precision and robust inverse covariance
- **Dynamic risk**
  - Per-asset GARCH(1,1) volatility forecasts; DCC for time-varying correlations
- **Black–Litterman**
  - Market equilibrium prior; incorporate data-driven sector views with confidence scaling
- **Optimization**
  - Problems: Sharpe-max, Min-Variance, CVaR-min
  - Constraints: long-only or bounded long/short, turnover and band limits, budget constraints
- **Execution realism**
  - Proportional transaction costs, optional market impact, monthly rebalancing with bands

Evaluation: rolling out-of-sample backtest (e.g., 36-month train → 1-month test), reporting Sharpe, volatility, MaxDD, VaR/CVaR, turnover, stability of weights.

---

## Reproducibility Card

- **Universe:** 11 Sector ETFs (see above)
- **Data source:** Yahoo Finance (free)
- **Date range:** configurable; typical ≥ 15 years where available
- **Rebalance:** monthly, end-of-month
- **Costs:** proportional (tunable)
- **Risk settings:** LW/GLASSO; GARCH(1,1) with Gaussian or Student-t innovations; DCC parameters
- **BL settings:** τ (prior uncertainty), λ (risk aversion), view confidence mapping
- **Randomness:** set `numpy`/`pandas` seeds where appropriate

---

## Roadmap

- Implement `risk.py`: LW/GLASSO, GARCH(1,1), DCC
- Implement `optimize.py`: MV/BL/CVaR with constraints and costs
- Implement `backtest.py`: walk-forward with costs, turnover bands
- Expand `requirements.txt` with pinned versions
- Add plotting/reporting utilities and a lightweight dashboard
- Add unit tests and CI

---

## Team & Roles

- **Data & Infrastructure** — acquisition via `yfinance`, caching/versioning, feature store (returns, momentum, volatility, covariance). *(Tran Anh Chuong)*
- **EDA & Research** — crisis case studies, rolling vol/corr, PCA & clustering, BL view logic draft. *(Nguyen Van Duy Anh)*
- **Modeling** — MV baseline → BL → CVaR; risk models (LW/GLASSO/GARCH/DCC). *(Cao Pham Minh Dang)*
- **Execution & Backtesting** — monthly walk-forward (36M train → 1M OOS), costs, bands, impact option; ablation grid & metrics. *(Pham Dinh Hieu)*
- **Visualization & Reporting** — dashboards/notebooks; reproducibility card; write-up of results & insights. *(Ngo Dinh Khanh)*

---

## How to Contribute

1. Create a feature branch from `main`.
2. Keep changes small and well-documented; add docstrings and type hints.
3. Open a PR with a clear description and figures where relevant.

---

## Citation

If you use this work in academic settings:

```
@software{FinLove2025,
  title   = {Risk-Aware Portfolio Construction},
  author  = {Nguyen, Van Duy Anh and Pham, Dinh Hieu and Cao, Pham Minh Dang and Tran, Anh Chuong and Ngo, Dinh Khanh},
  year    = {2025},
  url     = {https://github.com/dinhieufam/FinLove},
  note    = {Black–Litterman, CVaR, GARCH/DCC-based portfolio engine}
}
```

---

## License

MIT — see `LICENSE`.
