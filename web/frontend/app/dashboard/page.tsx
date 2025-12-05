"use client";

import React, { useMemo, useState } from "react";
import { AppShell } from "@components/layout/AppShell";
import { StatCard } from "@components/ui/StatCard";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Area,
  AreaChart,
} from "recharts";

type TimeSeriesPayload = {
  dates: string[];
  values: number[];
};

type AnalyzeResponse = {
  ok: boolean;
  error?: string;
  universe?: string[];
  metrics?: {
    annualized_return: number;
    annualized_volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_return: number;
    var_95: number;
    cvar_95: number;
    avg_turnover?: number;
    weight_stability?: number;
  };
  series?: {
    cumulative_portfolio: TimeSeriesPayload;
    cumulative_benchmark: TimeSeriesPayload;
    rolling_sharpe: TimeSeriesPayload;
    drawdown: TimeSeriesPayload;
    rolling_volatility: TimeSeriesPayload;
    returns: TimeSeriesPayload;
  };
  weights?: {
    current: { asset: string; weight: number }[];
  };
};

function toChartData(series?: TimeSeriesPayload) {
  if (!series) return [];
  return series.dates.map((d, i) => ({
    date: d,
    value: series.values[i],
  }));
}

export default function DashboardPage() {
  const [tickers, setTickers] = useState("XLK,XLF,XLV,XLY,XLP,XLE,XLI,XLB,XLU,XLRE,XLC");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [optimizationMethod, setOptimizationMethod] = useState("markowitz");
  const [riskModel, setRiskModel] = useState("ledoit_wolf");
  const [riskAversion, setRiskAversion] = useState(1.0);
  const [backtestType, setBacktestType] = useState<"simple" | "walk_forward">(
    "walk_forward",
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalyzeResponse | null>(null);

  const cumulativeSeries = useMemo(() => {
    const portfolio = toChartData(data?.series?.cumulative_portfolio);
    const benchmark = toChartData(data?.series?.cumulative_benchmark);
    
    // Merge portfolio and benchmark by date for chart display
    const merged = portfolio.map((p) => {
      const b = benchmark.find((b) => b.date === p.date);
      return {
        date: p.date,
        portfolio: p.value,
        benchmark: b?.value ?? null,
      };
    });
    
    return {
      portfolio,
      benchmark,
      merged,
    };
  }, [data]);

  const drawdownSeries = useMemo(
    () => toChartData(data?.series?.drawdown),
    [data],
  );

  const handleRun = async () => {
    setLoading(true);
    setError(null);

    try {
      const body = {
        tickers: tickers
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        use_default_universe: false,
        start_date: startDate,
        end_date: endDate,
        optimization_method: optimizationMethod,
        risk_model: riskModel,
        risk_aversion: riskAversion,
        backtest_type: backtestType,
        transaction_cost: 0.001,
        rebalance_band: 0.05,
        train_window: 36,
        test_window: 1,
      };

      const res = await fetch("http://localhost:8000/api/portfolio/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const json: AnalyzeResponse = await res.json();
      if (!json.ok) {
        setError(json.error || "Backtest failed");
        setData(null);
      } else {
        setData(json);
      }
    } catch (e) {
      setError("Unable to reach backend. Is FastAPI running on :8000?");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const metrics = data?.metrics;
  const sidebarContent = (
    <>
      {/* Universe & Period */}
      <div className="finlove-card p-4" id="portfolio">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
            <span className="text-sm">📊</span>
          </div>
          <h2 className="text-base font-semibold">Universe & Period</h2>
        </div>
        <p className="mb-4 text-xs leading-relaxed text-slate-400">
          Select tickers and define your backtest window. Default values use sector ETFs matching the Streamlit dashboard.
        </p>
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-300">
              Tickers <span className="text-slate-500">(comma-separated)</span>
            </label>
            <input
              value={tickers}
              onChange={(e) => setTickers(e.target.value)}
              className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 outline-none transition-colors focus:border-emerald-400/60 focus:bg-slate-900/80"
              placeholder="AAPL, MSFT, GOOGL"
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none transition-colors focus:border-emerald-400/60 focus:bg-slate-900/80"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none transition-colors focus:border-emerald-400/60 focus:bg-slate-900/80"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Risk & Optimization */}
      <div className="finlove-card p-4" id="risk">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
            <span className="text-sm">⚙️</span>
          </div>
          <h2 className="text-base font-semibold">Risk & Optimization</h2>
        </div>
        <p className="mb-4 text-xs leading-relaxed text-slate-400">
          Configure covariance estimation, portfolio objective, and risk appetite.
        </p>
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">
                Risk Model
              </label>
              <select
                value={riskModel}
                onChange={(e) => setRiskModel(e.target.value)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none transition-colors focus:border-emerald-400/60 focus:bg-slate-900/80"
              >
                <option value="ledoit_wolf">Ledoit-Wolf</option>
                <option value="sample">Sample</option>
                <option value="glasso">GLASSO</option>
                <option value="garch">GARCH</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">
                Optimization
              </label>
              <select
                value={optimizationMethod}
                onChange={(e) => setOptimizationMethod(e.target.value)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none transition-colors focus:border-emerald-400/60 focus:bg-slate-900/80"
              >
                <option value="markowitz">Markowitz</option>
                <option value="min_variance">Min Variance</option>
                <option value="sharpe">Sharpe Max</option>
                <option value="black_litterman">Black-Litterman</option>
                <option value="cvar">CVaR</option>
              </select>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-xs font-medium text-slate-300">
                Risk Appetite (λ)
              </label>
              <span className="text-xs font-semibold text-emerald-400">
                {riskAversion.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={10}
              step={0.1}
              value={riskAversion}
              onChange={(e) => setRiskAversion(parseFloat(e.target.value))}
              className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-emerald-500"
            />
            <p className="text-[11px] text-slate-500">
              Higher values create more defensive portfolios
            </p>
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-300">
              Backtest Type
            </label>
            <select
              value={backtestType}
              onChange={(e) => setBacktestType(e.target.value as "simple" | "walk_forward")}
              className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none transition-colors focus:border-emerald-400/60 focus:bg-slate-900/80"
            >
              <option value="simple">Simple (one-time)</option>
              <option value="walk_forward">Walk-forward (rolling)</option>
            </select>
          </div>
          <button
            type="button"
            onClick={handleRun}
            disabled={loading}
            className="w-full rounded-lg bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-emerald-950 shadow-lg transition-all hover:bg-emerald-400 hover:shadow-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-emerald-500"
          >
            {loading ? "⏳ Running Analysis..." : "🚀 Run Analysis"}
          </button>
          {error && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-3 py-2">
              <p className="text-xs text-rose-400">{error}</p>
            </div>
          )}
        </div>
      </div>
    </>
  );

  return (
    <AppShell sidebarContent={sidebarContent}>
      <div className="space-y-8">
        {/* Header */}
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-1">
            <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
              Portfolio Studio
            </h1>
            <p className="text-sm text-slate-400">
              Configure assets, risk models, and optimization strategies. Run walk-forward backtests powered by the FinLove engine.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="finlove-pill">
              {loading
                ? "⏳ Running analysis…"
                : data
                  ? "✅ Results ready"
                  : "⚡ Ready to analyze"}
            </span>
          </div>
        </header>

        {/* Top metrics cards */}
        {metrics && (
          <section
            aria-label="Key performance metrics"
            className="grid gap-4 sm:grid-cols-3"
          >
            <StatCard
              label="Total Return"
              value={`${(metrics.total_return * 100).toFixed(1)}%`}
              sublabel={
                data?.universe
                  ? `${data.universe.length} assets analyzed`
                  : "Portfolio performance"
              }
              tone={metrics.total_return >= 0 ? "positive" : "negative"}
            />
            <StatCard
              label="Max Drawdown"
              value={`${(metrics.max_drawdown * 100).toFixed(1)}%`}
              sublabel="Peak-to-trough loss"
              tone={metrics.max_drawdown > -0.2 ? "positive" : "negative"}
            />
            <StatCard
              label="Sharpe Ratio"
              value={metrics.sharpe_ratio.toFixed(2)}
              sublabel="Risk-adjusted return"
              tone={metrics.sharpe_ratio >= 1 ? "positive" : "neutral"}
            />
          </section>
        )}

        {/* Main content */}
        <div className="space-y-6" id="backtest">
          {/* Performance Charts */}
          <div className="finlove-card p-5" id="backtest">
              <div className="mb-4 flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
                  <span className="text-sm">📈</span>
                </div>
                <h2 className="text-base font-semibold">Performance & Drawdown</h2>
              </div>
              {data && cumulativeSeries.portfolio.length > 0 ? (
                <div className="space-y-6">
                  {/* Cumulative Returns */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-medium text-slate-400">
                        Cumulative Returns vs Benchmark
                      </p>
                      <div className="flex items-center gap-3 text-[11px] text-slate-500">
                        <span className="flex items-center gap-1.5">
                          <div className="h-2 w-2 rounded-full bg-emerald-400" />
                          Portfolio
                        </span>
                        <span className="flex items-center gap-1.5">
                          <div className="h-2 w-2 rounded-full bg-slate-500" />
                          Benchmark
                        </span>
                      </div>
                    </div>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={cumulativeSeries.merged}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis
                            dataKey="date"
                            tick={{ fontSize: 11, fill: "#94a3b8" }}
                            tickFormatter={(v) => {
                              const d = new Date(v);
                              return `${d.getMonth() + 1}/${d.getFullYear()}`;
                            }}
                          />
                          <YAxis
                            tick={{ fontSize: 11, fill: "#94a3b8" }}
                            tickFormatter={(v) => `${v.toFixed(0)}%`}
                          />
                          <Tooltip
                            formatter={(v: number) => (v != null ? `${v.toFixed(2)}%` : "—")}
                            labelFormatter={(l) => new Date(l).toLocaleDateString()}
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "8px",
                              fontSize: 12,
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="portfolio"
                            stroke="#34d399"
                            strokeWidth={2}
                            dot={false}
                            name="Portfolio"
                          />
                          <Line
                            type="monotone"
                            dataKey="benchmark"
                            stroke="#64748b"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={false}
                            name="Benchmark"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Drawdown */}
                  {drawdownSeries.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-slate-400">
                        Portfolio Drawdown
                      </p>
                      <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={drawdownSeries}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis
                              dataKey="date"
                              tick={{ fontSize: 11, fill: "#94a3b8" }}
                              tickFormatter={(v) => {
                                const d = new Date(v);
                                return `${d.getMonth() + 1}/${d.getFullYear()}`;
                              }}
                            />
                            <YAxis
                              tick={{ fontSize: 11, fill: "#94a3b8" }}
                              tickFormatter={(v) => `${v.toFixed(0)}%`}
                            />
                            <Tooltip
                              formatter={(v: number) => [`${v.toFixed(2)}%`, ""]}
                              labelFormatter={(l) => new Date(l).toLocaleDateString()}
                              contentStyle={{
                                backgroundColor: "#0f172a",
                                borderColor: "#334155",
                                borderRadius: "8px",
                                fontSize: 12,
                              }}
                            />
                            <Area
                              type="monotone"
                              dataKey="value"
                              stroke="#fb7185"
                              fill="#fb7185"
                              fillOpacity={0.2}
                              name="Drawdown"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed border-slate-700/50 bg-slate-950/30">
                  <div className="text-center">
                    <p className="text-sm font-medium text-slate-400">
                      No data available
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      Configure settings and run an analysis to see performance charts
                    </p>
                  </div>
                </div>
              )}
          </div>

          {/* Risk Metrics */}
          <div className="finlove-card p-5" id="metrics">
            <div className="mb-4 flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
                <span className="text-sm">📊</span>
              </div>
              <h2 className="text-base font-semibold">Risk Analytics</h2>
            </div>
            {metrics ? (
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-lg border border-slate-800/50 bg-slate-950/40 p-4">
                  <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
                    95% VaR
                  </p>
                  <p className="mt-2 text-xl font-bold text-slate-100">
                    {(metrics.var_95 * 100).toFixed(2)}%
                  </p>
                  <p className="mt-1 text-xs text-slate-400">1-day horizon</p>
                </div>
                <div className="rounded-lg border border-slate-800/50 bg-slate-950/40 p-4">
                  <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
                    95% CVaR
                  </p>
                  <p className="mt-2 text-xl font-bold text-slate-100">
                    {(metrics.cvar_95 * 100).toFixed(2)}%
                  </p>
                  <p className="mt-1 text-xs text-slate-400">Expected tail loss</p>
                </div>
                <div className="rounded-lg border border-slate-800/50 bg-slate-950/40 p-4">
                  <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
                    Volatility
                  </p>
                  <p className="mt-2 text-xl font-bold text-slate-100">
                    {(metrics.annualized_volatility * 100).toFixed(2)}%
                  </p>
                  <p className="mt-1 text-xs text-slate-400">Annualized</p>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-slate-800/50 bg-slate-950/30 p-8 text-center">
                <p className="text-sm text-slate-400">
                  Risk metrics will appear after running an analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

