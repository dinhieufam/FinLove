import React from 'react';
import {
    Line,
    LineChart,
    ResponsiveContainer,
    Area,
    AreaChart,
} from "recharts";

type AssetMetrics = {
    total_return: number;
    volatility: number;
    sharpe: number;
};

type AssetSeries = {
    dates: string[];
    values: number[];
};

type AssetData = {
    metrics: AssetMetrics;
    cumulative: AssetSeries;
    drawdown: AssetSeries;
    rolling_sharpe: AssetSeries;
};

type CompanyInfo = {
    symbol: string;
    name: string;
    sector?: string;
    market_cap?: number;
    pe_ratio?: number;
    beta?: number;
};

type AssetCardProps = {
    ticker: string;
    data: AssetData;
    info?: CompanyInfo;
};

function toChartData(series: AssetSeries) {
    return series.dates.map((d, i) => ({
        date: d,
        value: series.values[i],
    }));
}

export function AssetCard({ ticker, data, info }: AssetCardProps) {
    const cumData = toChartData(data.cumulative);
    const ddData = toChartData(data.drawdown);

    // Determine color based on return
    const isPositive = data.metrics.total_return >= 0;
    const trendColor = isPositive ? "#34d399" : "#fb7185";

    return (
        <div className="finlove-card p-4 transition-all hover:border-emerald-500/30 hover:shadow-lg hover:shadow-emerald-500/5">
            {/* Header */}
            <div className="mb-3 flex items-start justify-between">
                <div>
                    <div className="flex items-center gap-2">
                        <h3 className="text-lg font-bold text-slate-100">{ticker}</h3>
                        <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium text-slate-400">
                            {info?.sector || "Equity"}
                        </span>
                    </div>
                    <p className="text-xs text-slate-500 truncate max-w-[180px]" title={info?.name}>
                        {info?.name || ticker}
                    </p>
                </div>
                <div className="text-right">
                    <p className={`text-lg font-bold ${isPositive ? "text-emerald-400" : "text-rose-400"}`}>
                        {(data.metrics.total_return * 100).toFixed(1)}%
                    </p>
                    <p className="text-[10px] text-slate-500">Total Return</p>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="mb-4 grid grid-cols-3 gap-2 rounded-lg bg-slate-950/50 p-2">
                <div className="text-center">
                    <p className="text-[10px] text-slate-500">Vol</p>
                    <p className="text-xs font-semibold text-slate-200">
                        {(data.metrics.volatility * 100).toFixed(1)}%
                    </p>
                </div>
                <div className="text-center border-l border-slate-800">
                    <p className="text-[10px] text-slate-500">Sharpe</p>
                    <p className="text-xs font-semibold text-slate-200">
                        {data.metrics.sharpe.toFixed(2)}
                    </p>
                </div>
                <div className="text-center border-l border-slate-800">
                    <p className="text-[10px] text-slate-500">Beta</p>
                    <p className="text-xs font-semibold text-slate-200">
                        {info?.beta?.toFixed(2) || "-"}
                    </p>
                </div>
            </div>

            {/* Sparklines */}
            <div className="space-y-3">
                {/* Cumulative Return Sparkline */}
                <div className="h-16 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={cumData}>
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke={trendColor}
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Drawdown Sparkline */}
                <div className="h-12 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={ddData}>
                            <Area
                                type="monotone"
                                dataKey="value"
                                stroke="#fb7185"
                                fill="#fb7185"
                                fillOpacity={0.2}
                                strokeWidth={1}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
