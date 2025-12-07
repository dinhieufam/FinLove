"use client";

import React, { useMemo, useState, useEffect, useRef } from "react";
import { AppShell } from "@components/layout/AppShell";
import { StatCard } from "@components/ui/StatCard";
import { Heatmap } from "@components/ui/Heatmap";
import { AssetCard } from "@components/ui/AssetCard";
import { ChatBot } from "@components/ui/ChatBot";
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

type AssetMetrics = {
  total_return: number;
  volatility: number;
  sharpe: number;
};

type AssetData = {
  metrics: AssetMetrics;
  cumulative: TimeSeriesPayload;
  drawdown: TimeSeriesPayload;
  rolling_sharpe: TimeSeriesPayload;
};

type CompanyInfo = {
  symbol: string;
  name: string;
  sector?: string;
  market_cap?: number;
  pe_ratio?: number;
  beta?: number;
};

type RiskMatrices = {
  covariance: {
    labels: string[];
    matrix: number[][];
  };
  correlation: {
    labels: string[];
    matrix: number[][];
  };
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
  // New Fields
  risk_matrices?: RiskMatrices;
  company_info?: CompanyInfo[];
  assets_analysis?: Record<string, AssetData>;
  assets_returns?: Record<string, number[]>;
  portfolio_id?: string;
};

type ModelPrediction = {
  forecast: TimeSeriesPayload;
  volatility: TimeSeriesPayload;
  metrics: {
    cumulative_return: number;
    volatility: number;
    mean_return: number;
  };
};

type PredictionResponse = {
  ok: boolean;
  error?: string;
  mode?: "portfolio";
  series?: {
    historical: TimeSeriesPayload;
    forecast: TimeSeriesPayload;
  };
  metrics?: {
    expected_daily_return: number;
    forecast_volatility: number;
  };
  forecast_horizon?: number;
  optimization_method?: string;
  risk_model?: string;
};

function toChartData(series?: TimeSeriesPayload) {
  if (!series) return [];
  return series.dates.map((d, i) => ({
    date: d,
    value: series.values[i],
  }));
}

const COLORS = ['#34d399', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa', '#2dd4bf', '#fb7185', '#94a3b8'];
const MODEL_COLORS: Record<string, string> = {
  lstm: '#d62728', // Red
  tcn: '#2ca02c', // Green
  xgboost: '#ff7f0e', // Orange
  transformer: '#9467bd' // Purple
};

const RISK_MODEL_OPTIONS = [
  {
    id: 'sample',
    label: 'A. "My industry is generally stable and predictable; things don\'t swing too wildly."',
    value: 'sample',
    description: "Sample Covariance - Basic risk estimate; good when data is clean and plentiful."
  },
  {
    id: 'ledoit_wolf',
    label: 'B. "My industry is messy or uncertain — signals are weak, and I just need something reliable."',
    value: 'ledoit_wolf',
    description: "Ledoit-Wolf Shrinkage - More stable results; works well when data is noisy."
  },
  {
    id: 'glasso',
    label: 'C. "I work with many assets, but only a few really move together — I care about the strongest relationships."',
    value: 'glasso',
    description: "Graphical Lasso (GLASSO) - Highlights only the strongest links between assets; good for many-asset portfolios."
  },
  {
    id: 'garch',
    label: 'D. "My market changes fast — volatility spikes matter a lot."',
    value: 'garch',
    description: "GARCH - Captures changing market volatility; useful in fast-moving markets."
  }
];

const OPTIMIZATION_OPTIONS = [
  {
    id: 'markowitz',
    label: 'A. "Balanced — I want a reasonable mix of risk and return."',
    value: 'markowitz',
    description: "Markowitz - Balanced mix of risk and return; good all-around choice."
  },
  {
    id: 'min_variance',
    label: 'B. "Very cautious — I just want the lowest risk possible."',
    value: 'min_variance',
    description: "Minimum Variance - Focuses on lowest risk; ideal for very cautious investors."
  },
  {
    id: 'sharpe',
    label: 'C. "Return-focused — I want the best reward for the risk I take."',
    value: 'sharpe',
    description: "Max Sharpe - Aims for the best return for the risk taken; great for growth."
  },
  {
    id: 'black_litterman',
    label: 'D. "Strategic — I want to blend market trends with my own views."',
    value: 'black_litterman',
    description: "Black-Litterman - Combines market data with your views; good for strategic thinking."
  },
  {
    id: 'cvar',
    label: 'E. "Risk-aware — I want protection against big losses."',
    value: 'cvar',
    description: "CVaR - Reduces extreme losses; useful in turbulent or risky markets."
  }
];

export default function DashboardPage() {
  // Ref for scrolling to top
  const topRef = useRef<HTMLDivElement>(null);

  // General State
  const [activeTab, setActiveTab] = useState<"investment" | "analyze" | "prediction">("investment");

  // Available tickers from backend
  const [availableTickers, setAvailableTickers] = useState<string[]>([]);
  const [tickersWithNames, setTickersWithNames] = useState<Array<{ ticker: string, name: string }>>([]);
  const [selectedTickers, setSelectedTickers] = useState<string[]>(["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]);

  // Analysis State
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
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);
  const [predData, setPredData] = useState<PredictionResponse | null>(null);

  // Fetch available tickers on mount
  useEffect(() => {
    const fetchAvailableTickers = async () => {
      try {
        const res = await fetch("/api/portfolio/available-tickers");
        const json = await res.json();
        if (json.tickers) {
          setAvailableTickers(json.tickers);
          // Store tickers with names if available
          if (json.tickers_with_names) {
            setTickersWithNames(json.tickers_with_names);
          }
          // Filter selected tickers to only include available ones
          setSelectedTickers(prev => prev.filter(t => json.tickers.includes(t)));
        }
      } catch (e) {
        console.error("Failed to fetch available tickers:", e);
      }
    };
    fetchAvailableTickers();
  }, []);

  // Helper function to get company name for a ticker
  const getTickerName = (ticker: string): string => {
    const tickerInfo = tickersWithNames.find(t => t.ticker === ticker);
    return tickerInfo?.name || ticker;
  };

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

  const predictionSeries = useMemo(() => {
    if (!predData?.series) return { hist: [], fore: [] };
    const hist = toChartData(predData.series.historical);
    const fore = toChartData(predData.series.forecast);
    return { hist, fore };
  }, [predData]);

  // Histogram Data
  const histogramData = useMemo(() => {
    if (!data?.series?.returns) return [];
    const returns = data.series.returns.values.map(v => v * 100); // Convert to %
    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const bins = 40;
    const step = (max - min) / bins;

    const hist = new Array(bins).fill(0);
    const binLabels = new Array(bins).fill(0);

    returns.forEach(val => {
      const binIdx = Math.min(Math.floor((val - min) / step), bins - 1);
      hist[binIdx]++;
    });

    return hist.map((count, i) => ({
      bin: (min + i * step).toFixed(1) + '%',
      count
    }));
  }, [data]);

  // Handlers
  const handleRunAnalysis = async () => {
    // Scroll to top of page
    if (topRef.current) {
      topRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      document.documentElement.scrollTo({ top: 0, behavior: 'smooth' });
    }

    setLoading(true);
    setError(null);
    try {
      const body = {
        tickers: selectedTickers,
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

      const res = await fetch("/api/portfolio/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const json: AnalyzeResponse = await res.json();
      if (!json.ok) {
        setError(json.error || "Backtest failed");
        setData(null);
      } else {
        // Store portfolio for RAG
        try {
          const storeRes = await fetch("/api/qa/store", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ analysis_result: json }),
          });
          const storeJson = await storeRes.json();
          if (storeJson.ok && storeJson.portfolio_id) {
            json.portfolio_id = storeJson.portfolio_id;
          }
        } catch (e) {
          console.error("Failed to store portfolio for RAG:", e);
        }
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
    // Scroll to top of page
    if (topRef.current) {
      topRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      document.documentElement.scrollTo({ top: 0, behavior: 'smooth' });
    }

    setPredLoading(true);
    setPredError(null);
    try {
      const body = {
        tickers: selectedTickers,
        start_date: startDate,
        end_date: endDate,
        forecast_horizon: predHorizon,
        optimization_method: optimizationMethod,
        risk_model: riskModel,
        risk_aversion: riskAversion
        // forecast_method removed - backend now always uses SARIMAX
      };

      const res = await fetch("/api/portfolio/predict", {
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

  // Combined loading state
  const isLoading = loading || predLoading;
  const loadingMessage = loading ? "Analyzing portfolio..." : predLoading ? "Forecasting portfolio..." : "";

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
            <label className="block text-xs font-medium text-slate-300">Tickers ({selectedTickers.length} selected)</label>
            <div className="w-full rounded-lg border border-slate-700/60 bg-slate-950/60 p-2 text-sm max-h-48 overflow-y-auto">
              {availableTickers.length === 0 ? (
                <p className="text-slate-500 text-xs py-2">Loading available tickers...</p>
              ) : (
                <div className="space-y-1">
                  {availableTickers.map((ticker) => (
                    <label
                      key={ticker}
                      className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-slate-800/50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedTickers.includes(ticker)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedTickers([...selectedTickers, ticker]);
                          } else {
                            setSelectedTickers(selectedTickers.filter(t => t !== ticker));
                          }
                        }}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
                      />
                      <span className="text-slate-100 text-sm">
                        {ticker} <span className="text-slate-400">({getTickerName(ticker)})</span>
                      </span>
                    </label>
                  ))}
                </div>
              )}
            </div>
            {selectedTickers.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {selectedTickers.map((ticker) => (
                  <span
                    key={ticker}
                    className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    title={getTickerName(ticker)}
                  >
                    {ticker}
                    <button
                      onClick={() => setSelectedTickers(selectedTickers.filter(t => t !== ticker))}
                      className="hover:text-emerald-300 ml-1"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
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

          <div className="space-y-3">
            <label className="block text-xs font-medium text-slate-300">
              How would you describe the market environment you operate in?
            </label>
            <div className="space-y-2">
              {RISK_MODEL_OPTIONS.map((option) => (
                <label
                  key={option.id}
                  className={`flex items-start gap-2 p-2 rounded-lg border cursor-pointer transition-colors ${riskModel === option.value
                      ? 'border-emerald-500/60 bg-emerald-500/10'
                      : 'border-slate-700/60 bg-slate-950/60 hover:bg-slate-900/60'
                    }`}
                >
                  <input
                    type="radio"
                    name="riskModel"
                    value={option.value}
                    checked={riskModel === option.value}
                    onChange={(e) => setRiskModel(e.target.value)}
                    className="mt-1 w-4 h-4 text-emerald-500 border-slate-600 focus:ring-emerald-500 focus:ring-offset-0"
                  />
                  <div className="flex-1">
                    <span className="text-xs text-slate-100">{option.label}</span>
                    {riskModel === option.value && (
                      <p className="text-xs text-slate-400 mt-1">{option.description}</p>
                    )}
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <label className="block text-xs font-medium text-slate-300">
              What is your investing style?
            </label>
            <div className="space-y-2">
              {OPTIMIZATION_OPTIONS.map((option) => (
                <label
                  key={option.id}
                  className={`flex items-start gap-2 p-2 rounded-lg border cursor-pointer transition-colors ${optimizationMethod === option.value
                      ? 'border-emerald-500/60 bg-emerald-500/10'
                      : 'border-slate-700/60 bg-slate-950/60 hover:bg-slate-900/60'
                    }`}
                >
                  <input
                    type="radio"
                    name="optimizationMethod"
                    value={option.value}
                    checked={optimizationMethod === option.value}
                    onChange={(e) => setOptimizationMethod(e.target.value)}
                    className="mt-1 w-4 h-4 text-emerald-500 border-slate-600 focus:ring-emerald-500 focus:ring-offset-0"
                  />
                  <div className="flex-1">
                    <span className="text-xs text-slate-100">{option.label}</span>
                    {optimizationMethod === option.value && (
                      <p className="text-xs text-slate-400 mt-1">{option.description}</p>
                    )}
                  </div>
                </label>
              ))}
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
            <p className="text-xs text-slate-400">Uses SARIMAX model with portfolio strategy from config (Risk Model & Optimization)</p>
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
      {/* Loading Bar */}
      {isLoading && (
        <div className="fixed top-0 left-0 right-0 z-50 bg-slate-950/95 backdrop-blur-sm border-b border-emerald-500/30 shadow-lg">
          <div className="h-1 bg-slate-800/50 overflow-hidden relative">
            <div className="h-full bg-gradient-to-r from-emerald-500 via-cyan-400 to-emerald-500 relative" style={{
              width: '100%',
              background: 'linear-gradient(90deg, #10b981 0%, #22d3ee 50%, #10b981 100%)',
              backgroundSize: '200% 100%',
              animation: 'loading 2s ease-in-out infinite'
            }}>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
            </div>
          </div>
          <div className="px-4 py-2 flex items-center justify-center gap-2">
            <div className="flex gap-1">
              <div className="h-2 w-2 animate-bounce rounded-full bg-emerald-400 [animation-delay:-0.3s]"></div>
              <div className="h-2 w-2 animate-bounce rounded-full bg-emerald-400 [animation-delay:-0.15s]"></div>
              <div className="h-2 w-2 animate-bounce rounded-full bg-emerald-400"></div>
            </div>
            <p className="text-sm text-slate-300 font-medium">{loadingMessage}</p>
          </div>
        </div>
      )}

      <div ref={topRef} className="space-y-8" style={{ paddingTop: isLoading ? '60px' : '0' }}>
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

                {/* Detailed Metrics Table */}
                <div className="finlove-card p-6">
                  <h3 className="mb-4 text-lg font-semibold">Detailed Metrics</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-slate-400 uppercase bg-slate-900/50">
                        <tr>
                          <th className="px-4 py-3">Metric</th>
                          <th className="px-4 py-3">Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics && Object.entries(metrics).map(([key, value]) => (
                          <tr key={key} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                            <td className="px-4 py-3 font-medium capitalize">{key.replace(/_/g, ' ')}</td>
                            <td className="px-4 py-3">
                              {key.includes('ratio') ? value.toFixed(3) :
                                key.includes('stability') ? value.toFixed(3) :
                                  `${(value * 100).toFixed(2)}%`}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
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

                {/* Risk Matrices */}
                {data.risk_matrices && (
                  <div className="grid gap-6 md:grid-cols-2">
                    <Heatmap
                      title="Correlation Matrix"
                      data={data.risk_matrices.correlation.matrix}
                      labels={data.risk_matrices.correlation.labels}
                      colorScale="diverging"
                    />
                    <Heatmap
                      title="Covariance Matrix"
                      data={data.risk_matrices.covariance.matrix}
                      labels={data.risk_matrices.covariance.labels}
                      colorScale="red"
                    />
                  </div>
                )}

                {/* Return Distribution */}
                <div className="finlove-card p-6">
                  <h3 className="mb-4 text-lg font-semibold">Return Distribution</h3>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={histogramData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="bin" tick={{ fontSize: 10, fill: "#94a3b8" }} />
                        <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
                        <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                        <Bar dataKey="count" fill="#34d399" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Per-Asset Performance Table */}
                {data.assets_analysis && (
                  <div className="finlove-card p-6">
                    <h3 className="mb-4 text-lg font-semibold">Per-Asset Performance</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-slate-400 uppercase bg-slate-900/50">
                          <tr>
                            <th className="px-4 py-3">Asset</th>
                            <th className="px-4 py-3">Total Return</th>
                            <th className="px-4 py-3">Volatility</th>
                            <th className="px-4 py-3">Sharpe</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(data.assets_analysis).map(([ticker, assetData]) => (
                            <tr key={ticker} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                              <td className="px-4 py-3 font-medium">{ticker}</td>
                              <td className={`px-4 py-3 ${assetData.metrics.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {(assetData.metrics.total_return * 100).toFixed(2)}%
                              </td>
                              <td className="px-4 py-3">{(assetData.metrics.volatility * 100).toFixed(2)}%</td>
                              <td className="px-4 py-3">{assetData.metrics.sharpe.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Asset Deep Dive */}
                {data.assets_analysis && (
                  <div>
                    <h3 className="mb-4 text-xl font-bold">Asset Deep Dive</h3>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                      {Object.entries(data.assets_analysis).map(([ticker, assetData]) => (
                        <AssetCard
                          key={ticker}
                          ticker={ticker}
                          data={assetData}
                          info={data.company_info?.find(c => c.symbol === ticker)}
                        />
                      ))}
                    </div>
                  </div>
                )}
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
                {/* Portfolio Mode View */}
                {predData.mode === 'portfolio' && (
                  <>
                    <div className="grid gap-4 sm:grid-cols-3">
                      <StatCard label="Exp. Daily Return" value={`${(predData.metrics?.expected_daily_return! * 100).toFixed(3)}%`} tone="positive" />
                      <StatCard label="Forecast Volatility" value={`${(predData.metrics?.forecast_volatility! * 100).toFixed(2)}%`} tone="neutral" />
                      <StatCard label="Horizon" value={`${predData.forecast_horizon} Days`} tone="neutral" />
                    </div>

                    <div className="finlove-card p-6">
                      <div className="mb-4 flex items-center justify-between">
                        <h3 className="text-lg font-semibold">Forecast Trajectory</h3>
                        {predData.optimization_method && predData.risk_model && (
                          <div className="text-xs text-slate-400">
                            Strategy: {predData.optimization_method.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} + {predData.risk_model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </div>
                        )}
                      </div>
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="date" type="category" allowDuplicatedCategory={false} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                            <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => v.toFixed(2)} />
                            <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                            <Line data={predictionSeries.hist} type="monotone" dataKey="value" stroke="#34d399" strokeWidth={2} dot={false} name="Historical" />
                            <Line data={predictionSeries.fore} type="monotone" dataKey="value" stroke="#f472b6" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Forecast" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>
      <ChatBot
        portfolioData={{
          optimizationMethod: optimizationMethod,
          riskModel: riskModel,
          metrics: data?.metrics,
          tickers: selectedTickers,
          portfolio_id: (data as any)?.portfolio_id,
        }}
      />
    </AppShell>
  );
}
