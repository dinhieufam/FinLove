"use client";

import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="finlove-gradient-bg min-h-screen px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-10 lg:flex-row lg:items-center">
        {/* Left: hero content */}
        <section className="max-w-xl space-y-6">
          <span className="finlove-pill">Risk-aware Quant Portfolio Platform</span>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            FinLove:
            <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">
              {" "}
              build portfolios you can trust.
            </span>
          </h1>
          <p className="finlove-subtle">
            Combine advanced risk models, robust optimization, and realistic
            backtesting to design stable portfolios across regimes. Powered by
            Ledoit-Wolf, GLASSO, GARCH, DCC and more.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/dashboard"
              className="inline-flex items-center justify-center rounded-xl bg-emerald-500 px-6 py-2.5 text-sm font-medium text-slate-950 shadow-md transition hover:bg-emerald-400"
            >
              Launch Portfolio Studio
            </Link>
            <a
              href="https://github.com/dinhieufam/FinLove"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-center rounded-xl border border-slate-700 bg-slate-900/60 px-6 py-2.5 text-sm font-medium text-slate-100 transition hover:border-slate-500 hover:bg-slate-900"
            >
              View documentation
            </a>
          </div>
          <div className="mt-4 grid gap-3 text-xs text-slate-300 sm:grid-cols-3">
            <div>
              <p className="font-semibold text-slate-100">Risk Models</p>
              <p>Ledoit-Wolf, GLASSO, GARCH, DCC.</p>
            </div>
            <div>
              <p className="font-semibold text-slate-100">Optimization</p>
              <p>Markowitz, Black-Litterman, CVaR, Sharpe.</p>
            </div>
            <div>
              <p className="font-semibold text-slate-100">Backtesting</p>
              <p>Walk-forward with costs and rebalance bands.</p>
            </div>
          </div>
        </section>

        {/* Right: feature / metrics panel */}
        <section className="flex-1">
          <div className="finlove-card relative overflow-hidden p-5 sm:p-6">
            <div className="pointer-events-none absolute inset-x-0 -top-8 h-32 bg-gradient-to-b from-emerald-400/20 to-transparent blur-3xl" />
            <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-xs font-medium uppercase tracking-[0.14em] text-emerald-300">
                  Strategy snapshot
                </p>
                <p className="mt-1 text-sm text-slate-300">
                  Example walk-forward backtest vs. equal-weight benchmark.
                </p>
              </div>
              <div className="text-right text-xs text-slate-400">
                <p>Universe: Sector ETFs</p>
                <p>Risk model: Ledoit-Wolf</p>
                <p>Method: Markowitz</p>
              </div>
            </div>

            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <div>
                <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                  Cumulative return
                </p>
                <p className="mt-1 text-xl font-semibold text-emerald-400">
                  +142.7%
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  Benchmark: +98.3%
                </p>
              </div>
              <div>
                <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                  Max drawdown
                </p>
                <p className="mt-1 text-xl font-semibold text-emerald-300">
                  -16.4%
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  Benchmark: -24.1%
                </p>
              </div>
              <div>
                <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                  Sharpe ratio
                </p>
                <p className="mt-1 text-xl font-semibold text-emerald-400">
                  1.46
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  252-day rolling, rf ≈ 0%
                </p>
              </div>
            </div>

            <div className="mt-5 flex items-center justify-between border-t border-slate-800/70 pt-3 text-[11px] text-slate-400">
              <p>
                Daily data from Yahoo Finance via{" "}
                <span className="font-medium text-slate-200">yfinance</span>.
              </p>
              <p>Built for research and education purposes.</p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}


