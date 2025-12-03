"use client";

import React from "react";
import { AppShell } from "@components/layout/AppShell";
import { StatCard } from "@components/ui/StatCard";

export default function DashboardPage() {
  return (
    <AppShell>
      <div className="space-y-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
              FinLove Portfolio Studio
            </h1>
            <p className="mt-1 text-sm text-slate-300">
              Configure assets, risk, and optimization settings. Then run
              walk-forward backtests powered by the FinLove engine.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="finlove-pill">Demo mode · Static sample data</span>
          </div>
        </header>

        {/* Top stats */}
        <section
          aria-label="High-level performance metrics"
          className="grid gap-4 sm:grid-cols-3"
        >
          <StatCard
            label="Cumulative return (backtest)"
            value="+142.7%"
            sublabel="Equal-weight benchmark: +98.3%"
            tone="positive"
          />
          <StatCard
            label="Max drawdown"
            value="-16.4%"
            sublabel="Benchmark: -24.1%"
            tone="positive"
          />
          <StatCard
            label="Sharpe ratio"
            value="1.46"
            sublabel="252-day rolling, rf ≈ 0%"
            tone="positive"
          />
        </section>

        {/* Main layout */}
        <section className="grid gap-5 md:grid-cols-[minmax(0,1.15fr),minmax(0,1.85fr)]">
          {/* Left: controls */}
          <div className="space-y-4" id="portfolio">
            <div className="finlove-card p-4">
              <h2 className="text-sm font-semibold">Universe & period</h2>
              <p className="mt-1 text-xs text-slate-300">
                In the full app, this panel will let you pick sector ETFs or
                custom tickers, and choose the backtest window.
              </p>
              <div className="mt-4 grid gap-3 text-xs sm:grid-cols-2">
                <div className="space-y-2">
                  <p className="font-medium text-slate-200">Universe</p>
                  <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2">
                    <p className="text-[11px] text-slate-400">Preset universe</p>
                    <p className="mt-0.5 text-xs text-slate-100">
                      11 Sector ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU,
                      XLRE, XLC)
                    </p>
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="font-medium text-slate-200">Date range</p>
                  <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2">
                    <p className="text-[11px] text-slate-400">Example</p>
                    <p className="mt-0.5 text-xs text-slate-100">
                      2018-01-01 → 2024-12-31 (daily)
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="finlove-card p-4" id="risk">
              <h2 className="text-sm font-semibold">Risk model & optimization</h2>
              <p className="mt-1 text-xs text-slate-300">
                Configure the core engine: covariance estimation, objective, and
                constraints, aligned with the modes in the README.
              </p>
              <div className="mt-4 grid gap-3 text-xs sm:grid-cols-2">
                <div className="space-y-2">
                  <p className="font-medium text-slate-200">Risk model</p>
                  <ul className="space-y-1 text-slate-300">
                    <li>• Ledoit-Wolf (recommended)</li>
                    <li>• GLASSO (sparse precision)</li>
                    <li>• GARCH volatility</li>
                    <li>• DCC correlations</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <p className="font-medium text-slate-200">Optimization method</p>
                  <ul className="space-y-1 text-slate-300">
                    <li>• Markowitz mean-variance</li>
                    <li>• Min variance / Sharpe max</li>
                    <li>• Black-Litterman</li>
                    <li>• CVaR optimization</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Right: charts and details */}
          <div className="space-y-4" id="backtest">
            <div className="finlove-card p-4">
              <h2 className="text-sm font-semibold">Performance & drawdown</h2>
              <p className="mt-1 text-xs text-slate-300">
                This panel will host interactive charts: cumulative returns vs
                benchmark, drawdowns, and rolling Sharpe.
              </p>
              <div className="mt-4 h-40 rounded-lg border border-dashed border-slate-700/80 bg-slate-950/60" />
              <p className="mt-2 text-[11px] text-slate-400">
                For now this area is a placeholder. You can plug in Plotly,
                Recharts, or another charting library when wiring data from the
                FastAPI backend.
              </p>
            </div>

            <div className="finlove-card p-4" id="metrics">
              <h2 className="text-sm font-semibold">Risk analytics & metrics</h2>
              <p className="mt-1 text-xs text-slate-300">
                Summaries of VaR, CVaR, volatility, and configuration details for
                reproducible research.
              </p>
              <div className="mt-3 grid gap-3 text-xs sm:grid-cols-3">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                    95% VaR
                  </p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">
                    -2.3%
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">1-day horizon</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                    95% CVaR
                  </p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">
                    -3.1%
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">Tail losses</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                    Volatility
                  </p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">
                    12.4%
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">
                    Annualized (252 trading days)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}


