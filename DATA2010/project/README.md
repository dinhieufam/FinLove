# Project Proposal — Risk-Aware Portfolio Construction

**Authors:** Nguyen Van Duy Anh · Pham Dinh Hieu · Cao Pham Minh Dang · Tran Anh Chuong · Ngo Dinh Khanh  
**GitHub:** https://github.com/dinhieufam/FinLove

---

## 1) What is the project about?

**Title:** Stable, Cost-Aware Portfolio Construction with Black–Litterman, CVaR, and Dynamic Risk

**Industry:** Finance / Asset & Wealth Management

**Context:** Investors need allocations that are stable, risk-aware, and net of costs. Classic Markowitz is fragile to estimation error; practitioners instead combine shrinkage covariance, Black–Litterman (BL) views, tail-risk (CVaR) optimization, and dynamic volatility/correlation (GARCH/DCC).

---

## 2) Why this project?

- **Motivation & Problem:** Naïve 60/40 or equal-weight ignores tail risk, regime shifts, and costs. Plain Markowitz overfits noisy return estimates → extreme weights and high turnover.
- **Importance:** A robust, reproducible allocator that blends BL + CVaR + dynamic risk can lower drawdowns/CVaR, keep weights smooth, and improve net Sharpe — directly useful for institutional or wealth portfolios.

---

## 3) What is the final product?

A Python-based **portfolio engine** that supports:

1. **Risk models:** Ledoit–Wolf, GLASSO, GARCH(1,1) per asset, DCC correlations  
2. **Return models:** Black–Litterman with data-driven, uncertainty-weighted views  
3. **Objectives:** Sharpe-max / Minimum-Variance / CVaR  
4. **Execution realism:** Transaction costs, rebalance bands, optional impact (Almgren–Chriss)

**Deliverables:** A lightweight dashboard/notebook showing rolling Sharpe/volatility, MaxDD, VaR/CVaR, turnover, weight paths, and regime labels.

---

## 4) Why this data?

- **Primary dataset (used):** Yahoo Finance via `yfinance`
- **Universe (default):** 11 liquid Sector ETFs — `XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC`
- **Provenance:** Free historical prices/dividends/splits
- **Structure & variables:** Date, Adj Close, Close, Volume. We compute daily log returns, aggregate to month-end for rebalancing; keep daily for GARCH.
- **Suitability:** Highly liquid, diversified, long histories → realistic backtests. Works well with our risk models (covariance, GARCH/DCC) and tail-risk scenarios (bootstrap/MC).

---

## 5) Team responsibilities (Data Science Life Cycle)

- **Data & Infrastructure** — Acquire via `yfinance`, caching & versioning; build feature store (returns, momentum, vol, Σ). *(Tran Anh Chuong)*
- **EDA & Research** — Crisis case studies; rolling vol/corr; PCA & clustering; draft BL view logic. *(Nguyen Van Duy Anh)*
- **Modeling** — Implement MV (baseline) → BL (standard & uncertainty-weighted) → CVaR; risk models (LW/GLASSO/GARCH/DCC). *(Cao Pham Minh Dang)*
- **Execution & Backtesting** — Monthly walk-forward (36M train → 1M OOS), costs, bands, impact option; ablation grid & metrics. *(Pham Dinh Hieu)*
- **Visualization & Reporting** — Dashboard/notebook; “Reproducibility Card” (universe, dates, costs, τ/λ settings); write-up of results & insights. *(Ngo Dinh Khanh)*

---

*Repository:* https://github.com/dinhieufam/FinLove
