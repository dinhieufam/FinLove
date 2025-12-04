"use client";

import Link from "next/link";

export default function LandingPage() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="finlove-gradient-bg min-h-screen">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b border-slate-800/70 bg-slate-950/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
              <span className="text-lg font-semibold text-emerald-400">ƒ</span>
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">FinLove</p>
              <p className="text-[11px] text-slate-400">Risk-Aware Portfolio Lab</p>
            </div>
          </Link>
          <div className="hidden items-center gap-6 md:flex">
            <button
              onClick={() => scrollToSection("home")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Home
            </button>
            <button
              onClick={() => scrollToSection("about")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection("features")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("team")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Our Team
            </button>
            <button
              onClick={() => scrollToSection("contact")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Contact
            </button>
            <Link
              href="/dashboard"
              className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-emerald-400"
            >
              Launch Dashboard
            </Link>
          </div>
        </div>
      </nav>

      {/* Home/Hero Section */}
      <section id="home" className="px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto flex max-w-6xl flex-col gap-10 lg:flex-row lg:items-center">
          {/* Left: hero content */}
          <div className="max-w-xl space-y-6">
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
                View on GitHub
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
          </div>

          {/* Right: feature / metrics panel */}
          <div className="flex-1">
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
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              About <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">FinLove</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              A comprehensive Python-based portfolio construction engine designed for researchers, 
              students, and practitioners who need robust risk-aware portfolio optimization.
            </p>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">🎯</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Mission</h3>
              <p className="text-sm text-slate-300">
                FinLove combines advanced risk models, optimization methods, and execution realism 
                to create stable, risk-aware portfolios. Our goal is to make sophisticated portfolio 
                construction accessible to everyone.
              </p>
            </div>
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">🔬</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Research-Focused</h3>
              <p className="text-sm text-slate-300">
                Built for research and education purposes, FinLove provides a comprehensive 
                toolkit for exploring different risk models, optimization strategies, and 
                backtesting methodologies in a realistic environment.
              </p>
            </div>
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">⚡</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Performance</h3>
              <p className="text-sm text-slate-300">
                With automatic data caching and optimized algorithms, FinLove delivers fast 
                performance even with large portfolios and extended time periods. Pre-download 
                data for even faster analysis.
              </p>
            </div>
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Interactive Dashboard</h3>
              <p className="text-sm text-slate-300">
                Our Streamlit dashboard provides an intuitive interface for portfolio analysis, 
                visualization, and exploration. No coding required to get started.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Powerful <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">Features</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              Everything you need for comprehensive portfolio construction and analysis
            </p>
          </div>

          {/* Risk Models */}
          <div className="mb-16">
            <h3 className="mb-6 text-xl font-semibold text-emerald-400">Risk Models</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Ledoit-Wolf Shrinkage</h4>
                <p className="text-xs text-slate-400">
                  Reduces estimation error by shrinking sample covariance towards a target matrix. 
                  Recommended for stability.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Graphical LASSO</h4>
                <p className="text-xs text-slate-400">
                  Estimates sparse precision matrix using L1 regularization. Useful for 
                  high-dimensional portfolios.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">GARCH(1,1)</h4>
                <p className="text-xs text-slate-400">
                  Time-varying volatility per asset. Captures volatility clustering and 
                  regime changes.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">DCC</h4>
                <p className="text-xs text-slate-400">
                  Dynamic Conditional Correlation approximation. Models time-varying 
                  correlations between assets.
                </p>
              </div>
            </div>
          </div>

          {/* Optimization Methods */}
          <div className="mb-16">
            <h3 className="mb-6 text-xl font-semibold text-emerald-400">Optimization Methods</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Markowitz Mean-Variance</h4>
                <p className="text-xs text-slate-400">
                  Maximizes return - risk penalty. The classic portfolio optimization approach.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Minimum Variance</h4>
                <p className="text-xs text-slate-400">
                  Minimizes portfolio variance subject to budget constraints. Focuses purely on risk reduction.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Sharpe Maximization</h4>
                <p className="text-xs text-slate-400">
                  Maximizes risk-adjusted returns (Sharpe ratio). Balances return and risk optimally.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Black-Litterman</h4>
                <p className="text-xs text-slate-400">
                  Combines market equilibrium returns with investor views. More stable than pure Markowitz.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">CVaR Optimization</h4>
                <p className="text-xs text-slate-400">
                  Minimizes Conditional Value at Risk. Focuses on tail risk and extreme losses.
                </p>
              </div>
            </div>
          </div>

          {/* Backtesting & Dashboard */}
          <div className="grid gap-6 md:grid-cols-2">
            <div className="finlove-card p-6">
              <h3 className="mb-4 text-xl font-semibold text-emerald-400">Backtesting Engine</h3>
              <ul className="space-y-3 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Simple Backtest:</strong> One-time optimization using all historical data</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Walk-Forward Backtest:</strong> Rolling window backtest for realistic performance</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Transaction Costs:</strong> Proportional costs per rebalancing</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Rebalance Bands:</strong> Drift-based rebalancing to reduce turnover</span>
                </li>
              </ul>
            </div>
            <div className="finlove-card p-6">
              <h3 className="mb-4 text-xl font-semibold text-emerald-400">Dashboard Features</h3>
              <ul className="space-y-3 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Interactive Interface:</strong> Easy-to-use Streamlit web app</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Comprehensive Visualizations:</strong> Returns, Sharpe, drawdowns, weights</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Performance Metrics:</strong> All key metrics in organized tabs</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Company Information:</strong> Detailed data for each ticker</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Our Team Section */}
      <section id="team" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Our <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">Team</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              A collaborative effort by passionate students and researchers
            </p>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <div className="finlove-card p-6 text-center">
              <div className="mb-4 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl font-semibold text-emerald-400">NDA</span>
              </div>
              <h3 className="mb-1 text-lg font-semibold text-slate-100">Nguyen Van Duy Anh</h3>
              <p className="text-sm text-slate-400">Team Member</p>
            </div>
            <div className="finlove-card p-6 text-center">
              <div className="mb-4 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl font-semibold text-emerald-400">PDH</span>
              </div>
              <h3 className="mb-1 text-lg font-semibold text-slate-100">Pham Dinh Hieu</h3>
              <p className="text-sm text-slate-400">Team Member</p>
            </div>
            <div className="finlove-card p-6 text-center">
              <div className="mb-4 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl font-semibold text-emerald-400">CPMD</span>
              </div>
              <h3 className="mb-1 text-lg font-semibold text-slate-100">Cao Pham Minh Dang</h3>
              <p className="text-sm text-slate-400">Team Member</p>
            </div>
            <div className="finlove-card p-6 text-center">
              <div className="mb-4 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl font-semibold text-emerald-400">TAC</span>
              </div>
              <h3 className="mb-1 text-lg font-semibold text-slate-100">Tran Anh Chuong</h3>
              <p className="text-sm text-slate-400">Team Member</p>
            </div>
            <div className="finlove-card p-6 text-center">
              <div className="mb-4 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl font-semibold text-emerald-400">NDK</span>
              </div>
              <h3 className="mb-1 text-lg font-semibold text-slate-100">Ngo Dinh Khanh</h3>
              <p className="text-sm text-slate-400">Team Member</p>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Get in <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">Touch</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              Have questions, suggestions, or want to contribute? We'd love to hear from you!
            </p>
          </div>
          <div className="finlove-card p-8">
            <div className="grid gap-8 md:grid-cols-2">
              <div>
                <h3 className="mb-4 text-lg font-semibold text-slate-100">Project Information</h3>
                <div className="space-y-4 text-sm text-slate-300">
                  <div>
                    <p className="mb-1 font-medium text-slate-200">GitHub Repository</p>
                    <a
                      href="https://github.com/dinhieufam/FinLove"
                      target="_blank"
                      rel="noreferrer"
                      className="text-emerald-400 hover:text-emerald-300 transition"
                    >
                      github.com/dinhieufam/FinLove
                    </a>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">License</p>
                    <p className="text-slate-400">See LICENSE file for details</p>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Documentation</p>
                    <p className="text-slate-400">
                      Comprehensive guides available in the repository, including:
                    </p>
                    <ul className="mt-2 ml-4 list-disc space-y-1 text-slate-400">
                      <li>README.md - Main documentation</li>
                      <li>DATA.md - Data download and caching</li>
                      <li>QUICK_START.md - Quick start guide</li>
                    </ul>
                  </div>
                </div>
              </div>
              <div>
                <h3 className="mb-4 text-lg font-semibold text-slate-100">Support & Resources</h3>
                <div className="space-y-4 text-sm text-slate-300">
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Issues & Bug Reports</p>
                    <p className="text-slate-400">
                      Please report issues on our GitHub repository's Issues page
                    </p>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Data Questions</p>
                    <p className="text-slate-400">
                      Refer to DATA.md for data-related questions and caching information
                    </p>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Usage Help</p>
                    <p className="text-slate-400">
                      Check the README.md and QUICK_START.md for detailed usage instructions
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-4 border-t border-slate-800/70 pt-6">
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center rounded-xl bg-emerald-500 px-6 py-2.5 text-sm font-medium text-slate-950 shadow-md transition hover:bg-emerald-400"
              >
                Launch Dashboard
              </Link>
              <a
                href="https://github.com/dinhieufam/FinLove"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-center rounded-xl border border-slate-700 bg-slate-900/60 px-6 py-2.5 text-sm font-medium text-slate-100 transition hover:border-slate-500 hover:bg-slate-900"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/70 px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-sm font-semibold text-emerald-400">ƒ</span>
              </div>
              <p className="text-sm text-slate-400">
                FinLove — Risk-Aware Portfolio Construction
              </p>
            </div>
            <p className="text-xs text-slate-500">
              Built for research and education purposes · Data from Yahoo Finance
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}


