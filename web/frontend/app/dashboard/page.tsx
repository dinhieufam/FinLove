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
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Legend,
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
    history?: {
      dates: string[];
      assets: string[];
      matrix: number[][];
    };
  };
};

type PredictionResponse = {
  ok: boolean;
  error?: string;
  series?: {
    historical: TimeSeriesPayload;
    forecast: TimeSeriesPayload;
  };
  metrics?: {
    expected_daily_return: number;
    forecast_volatility: number;
  };
  top_models?: {
    model_id: string;
    sharpe_ratio: number;
    annualized_return: number;
    annualized_volatility: number;
  }[];
  forecast_horizon?: number;
  forecast_method?: string; // Which forecasting method was used (xgboost, lstm, etc.)
};

function toChartData(series?: TimeSeriesPayload) {
  if (!series) return [];
  return series.dates.map((d, i) => ({
    date: d,
    value: series.values[i],
  }));
}

const COLORS = ['#34d399', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa', '#2dd4bf', '#fb7185', '#94a3b8'];

export default function DashboardPage() {
  // General State
  const [activeTab, setActiveTab] = useState<"investment" | "analyze" | "prediction">("investment");

  // Analysis State
  const [tickers, setTickers] = useState("XLK,XLF,XLV,XLY,XLP,XLE,XLI,XLB,XLU,XLRE,XLC");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [optimizationMethod, setOptimizationMethod] = useState("markowitz");
  const [riskModel, setRiskModel] = useState("ledoit_wolf");
  const [riskAversion, setRiskAversion] = useState(1.0);
  const [backtestType, setBacktestType] = useState<"simple" | "walk_forward">("walk_forward");
  const [investmentAmount, setInvestmentAmount] = useState(10000);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalyzeResponse | null>(null);

  // Prediction State
  const [predHorizon, setPredHorizon] = useState(30);
  const [predModel, setPredModel] = useState("ensemble");
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);
  const [predData, setPredData] = useState<PredictionResponse | null>(null);

  // Computed Data for Charts
  const cumulativeSeries = useMemo(() => {
    const portfolio = toChartData(data?.series?.cumulative_portfolio);
    const benchmark = toChartData(data?.series?.cumulative_benchmark);

    const merged = portfolio.map((p) => {
      const b = benchmark.find((b) => b.date === p.date);
      return {
        date: p.date,
        portfolio: p.value,
        benchmark: b?.value ?? null,
      };
    });

    return { portfolio, benchmark, merged };
  }, [data]);

  const drawdownSeries = useMemo(() => toChartData(data?.series?.drawdown), [data]);
  const rollingSharpeSeries = useMemo(() => toChartData(data?.series?.rolling_sharpe), [data]);
  const rollingVolSeries = useMemo(() => toChartData(data?.series?.rolling_volatility), [data]);

  const allocationData = useMemo(() => {
    if (!data?.weights?.current) return [];
    return data.weights.current
      .filter(w => w.weight > 0.001)
      .sort((a, b) => b.weight - a.weight)
      .map(w => ({
        name: w.asset,
        value: w.weight * 100,
        amount: w.weight * investmentAmount
      }));
  }, [data, investmentAmount]);

  const predictionSeries = useMemo<{ hist: { date: string; value: number }[]; fore: { date: string; value: number }[] } | null>(() => {
    if (!predData?.series) return null;
    const hist = toChartData(predData.series.historical);
    const fore = toChartData(predData.series.forecast);
    return { hist, fore };
  }, [predData]);

  // Handlers
  const handleRunAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const body = {
        tickers: tickers.split(",").map((t) => t.trim()).filter(Boolean),
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

  const handleRunPrediction = async () => {
    setPredLoading(true);
    setPredError(null);
    try {
      const body = {
        tickers: tickers.split(",").map((t) => t.trim()).filter(Boolean),
        start_date: startDate,
        end_date: endDate,
        forecast_horizon: predHorizon,
        model: predModel,
        use_top_models: 3
      };

      const res = await fetch("http://localhost:8000/api/portfolio/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const json: PredictionResponse = await res.json();
      if (!json.ok) {
        setPredError(json.error || "Prediction failed");
        setPredData(null);
      } else {
        setPredData(json);
      }
    } catch (e) {
      setPredError("Unable to reach backend.");
      setPredData(null);
    } finally {
      setPredLoading(false);
    }
  };

  const metrics = data?.metrics;

  // Sidebar Content
  const sidebarContent = (
    <>
      <div className="finlove-card p-4 mb-4">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
            <span className="text-sm">⚙️</span>
          </div>
          <h2 className="text-base font-semibold">Configuration</h2>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-300">Tickers</label>
            <input
              value={tickers}
              onChange={(e) => setTickers(e.target.value)}
              className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60"
              placeholder="AAPL, MSFT..."
            />
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">Start Date</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60" />
            </div>
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">End Date</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60" />
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">Risk Model</label>
              <select value={riskModel} onChange={(e) => setRiskModel(e.target.value)} className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60">
                <option value="ledoit_wolf">Ledoit-Wolf</option>
                <option value="sample">Sample</option>
                <option value="glasso">GLASSO</option>
                <option value="garch">GARCH</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-300">Optimization</label>
              <select value={optimizationMethod} onChange={(e) => setOptimizationMethod(e.target.value)} className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60">
                <option value="markowitz">Markowitz</option>
                <option value="min_variance">Min Variance</option>
                <option value="sharpe">Sharpe Max</option>
                <option value="black_litterman">Black-Litterman</option>
                <option value="cvar">CVaR</option>
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-300">Investment Capital ($)</label>
            <input
              type="number"
              value={investmentAmount}
              onChange={(e) => setInvestmentAmount(parseFloat(e.target.value))}
              className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60"
            />
          </div>

          <button
            onClick={handleRunAnalysis}
            disabled={loading}
            className="w-full rounded-lg bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-emerald-950 shadow-lg transition-all hover:bg-emerald-400 disabled:opacity-50"
          >
            {loading ? "⏳ Analyzing..." : "🚀 Run Analysis"}
          </button>
          {error && <p className="text-xs text-rose-400">{error}</p>}
        </div>
      </div>

      <div className="finlove-card p-4">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20">
            <span className="text-sm">🔮</span>
          </div>
          <h2 className="text-base font-semibold">Prediction</h2>
        </div>
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-300">Horizon (Days)</label>
            <input type="number" value={predHorizon} onChange={(e) => setPredHorizon(parseInt(e.target.value))} className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60" />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-300">Model</label>
            <select value={predModel} onChange={(e) => setPredModel(e.target.value)} className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-emerald-400/60">
              <option value="ensemble">Ensemble (Recommended)</option>
              <option value="arima">ARIMA</option>
              <option value="prophet">Prophet</option>
              <option value="lstm">LSTM</option>
              <option value="tcn">TCN (Temporal CNN)</option>
              <option value="xgboost">XGBoost</option>
              <option value="transformer">Transformer</option>
              <option value="ma">Moving Average</option>
              <option value="exponential_smoothing">Exponential Smoothing</option>
            </select>
          </div>
          <button
            onClick={handleRunPrediction}
            disabled={predLoading}
            className="w-full rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-2.5 text-sm font-semibold text-emerald-400 transition-all hover:bg-emerald-500/20 disabled:opacity-50"
          >
            {predLoading ? "⏳ Predicting..." : "🔮 Forecast"}
          </button>
          {predError && <p className="text-xs text-rose-400">{predError}</p>}
        </div>
      </div>
    </>
  );

  return (
    <AppShell sidebarContent={sidebarContent}>
      <div className="space-y-8">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Portfolio Studio</h1>
            <p className="text-sm text-slate-400">Design, analyze, and forecast your portfolio strategies.</p>
          </div>
          <div className="flex gap-2">
            {(["investment", "analyze", "prediction"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === tab
                    ? "bg-emerald-500 text-slate-950"
                    : "bg-slate-800/50 text-slate-400 hover:text-slate-200"
                  }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </header>

        {/* Investment Tab */}
        {activeTab === "investment" && (
          <div className="space-y-6">
            {!data ? (
              <div className="finlove-card p-12 text-center text-slate-400">
                <p>Run an analysis to generate an investment plan.</p>
              </div>
            ) : (
              <>
                <div className="grid gap-4 sm:grid-cols-3">
                  <StatCard label="Total Capital" value={`$${investmentAmount.toLocaleString()}`} sublabel="Planned Investment" tone="neutral" />
                  <StatCard label="Assets" value={data.universe?.length.toString() || "0"} sublabel="Diversified Positions" tone="neutral" />
                  <StatCard label="Strategy" value={optimizationMethod} sublabel={riskModel} tone="neutral" />
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="finlove-card p-6">
                    <h3 className="mb-4 text-lg font-semibold">Allocation Breakdown</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={allocationData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            {allocationData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip
                            formatter={(val: number) => `${val.toFixed(1)}%`}
                            contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "8px" }}
                          />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="finlove-card p-6">
                    <h3 className="mb-4 text-lg font-semibold">Dollar Allocation</h3>
                    <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
                      {allocationData.map((item, idx) => (
                        <div key={item.name} className="flex items-center justify-between p-2 rounded bg-slate-900/50">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                            <span className="font-medium">{item.name}</span>
                          </div>
                          <div className="text-right">
                            <p className="font-mono text-emerald-400">${item.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                            <p className="text-xs text-slate-500">{item.value.toFixed(1)}%</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Analyze Tab */}
        {activeTab === "analyze" && (
          <div className="space-y-6">
            {!data ? (
              <div className="finlove-card p-12 text-center text-slate-400">
                <p>Run an analysis to see performance metrics.</p>
              </div>
            ) : (
              <>
                {metrics && (
                  <div className="grid gap-4 sm:grid-cols-4">
                    <StatCard label="Total Return" value={`${(metrics.total_return * 100).toFixed(1)}%`} tone={metrics.total_return >= 0 ? "positive" : "negative"} />
                    <StatCard label="Sharpe Ratio" value={metrics.sharpe_ratio.toFixed(2)} tone={metrics.sharpe_ratio >= 1 ? "positive" : "neutral"} />
                    <StatCard label="Volatility" value={`${(metrics.annualized_volatility * 100).toFixed(1)}%`} tone="neutral" />
                    <StatCard label="Max Drawdown" value={`${(metrics.max_drawdown * 100).toFixed(1)}%`} tone="negative" />
                  </div>
                )}

                <div className="finlove-card p-6">
                  <h3 className="mb-4 text-lg font-semibold">Cumulative Performance</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={cumulativeSeries.merged}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => new Date(v).getFullYear().toString()} />
                        <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                        <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                        <Line type="monotone" dataKey="portfolio" stroke="#34d399" strokeWidth={2} dot={false} name="Portfolio" />
                        <Line type="monotone" dataKey="benchmark" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Benchmark" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="finlove-card p-6">
                    <h3 className="mb-4 text-lg font-semibold">Drawdown</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={drawdownSeries}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#94a3b8" }} hide />
                          <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                          <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                          <Area type="monotone" dataKey="value" stroke="#fb7185" fill="#fb7185" fillOpacity={0.2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="finlove-card p-6">
                    <h3 className="mb-4 text-lg font-semibold">Rolling Sharpe (252d)</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={rollingSharpeSeries}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#94a3b8" }} hide />
                          <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
                          <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                          <Line type="monotone" dataKey="value" stroke="#fbbf24" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Prediction Tab */}
        {activeTab === "prediction" && (
          <div className="space-y-6">
            {!predData ? (
              <div className="finlove-card p-12 text-center text-slate-400">
                <p>Run a forecast to see future projections.</p>
              </div>
            ) : (
              <>
                <div className="grid gap-4 sm:grid-cols-3">
                  <StatCard label="Exp. Daily Return" value={`${(predData.metrics?.expected_daily_return! * 100).toFixed(3)}%`} tone="positive" />
                  <StatCard label="Forecast Volatility" value={`${(predData.metrics?.forecast_volatility! * 100).toFixed(2)}%`} tone="neutral" />
                  <StatCard label="Horizon" value={`${predData.forecast_horizon} Days`} tone="neutral" />
                </div>

                <div className="finlove-card p-6">
                  <h3 className="mb-4 text-lg font-semibold">Forecast Trajectory</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" type="category" allowDuplicatedCategory={false} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                        <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => v.toFixed(2)} />
                        <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                        {predictionSeries && (
                          <>
                            <Line data={predictionSeries.hist} type="monotone" dataKey="value" stroke="#34d399" strokeWidth={2} dot={false} name="Historical" />
                            <Line data={predictionSeries.fore} type="monotone" dataKey="value" stroke="#f472b6" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Forecast" />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {predData.top_models && (
                  <div className="finlove-card p-6">
                    <div className="mb-4 flex items-center justify-between">
                      <h3 className="text-lg font-semibold">Top Portfolio Models</h3>
                      {predData.forecast_method && (
                        <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400">
                          Forecasted with: {predData.forecast_method.toUpperCase()}
                        </span>
                      )}
                    </div>
                    <p className="mb-4 text-xs text-slate-400">
                      These are the top-performing portfolio optimization models (selected by Sharpe ratio). 
                      Each model's historical returns were forecasted using the {predData.forecast_method || 'selected'} method.
                    </p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-slate-400 uppercase bg-slate-900/50">
                          <tr>
                            <th className="px-4 py-3">Portfolio Model</th>
                            <th className="px-4 py-3">Sharpe</th>
                            <th className="px-4 py-3">Return</th>
                            <th className="px-4 py-3">Volatility</th>
                          </tr>
                        </thead>
                        <tbody>
                          {predData.top_models.map((m) => (
                            <tr key={m.model_id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                              <td className="px-4 py-3 font-medium">{m.model_id}</td>
                              <td className="px-4 py-3">{m.sharpe_ratio.toFixed(2)}</td>
                              <td className="px-4 py-3 text-emerald-400">{(m.annualized_return * 100).toFixed(1)}%</td>
                              <td className="px-4 py-3">{(m.annualized_volatility * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </AppShell>
  );
}
