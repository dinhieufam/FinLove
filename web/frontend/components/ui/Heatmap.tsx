import React from 'react';

type HeatmapProps = {
    data: number[][];
    labels: string[];
    title: string;
    colorScale?: 'blue' | 'red' | 'diverging';
};

export function Heatmap({ data, labels, title, colorScale = 'blue' }: HeatmapProps) {
    const size = labels.length;

    // Helper to get color based on value
    const getColor = (value: number) => {
        if (colorScale === 'diverging') {
            // -1 to 1 (Red to Blue)
            if (value < 0) {
                const intensity = Math.min(Math.abs(value), 1);
                return `rgba(244, 63, 94, ${intensity})`; // Rose-500
            } else {
                const intensity = Math.min(value, 1);
                return `rgba(59, 130, 246, ${intensity})`; // Blue-500
            }
        } else if (colorScale === 'red') {
            // 0 to max (Red scale)
            // Normalize assuming max covariance is roughly 0.1 for daily or 0.5 for annual
            // But covariance is unbounded. Let's use a log scale or simple cap.
            const intensity = Math.min(Math.sqrt(Math.abs(value)) * 5, 1);
            return `rgba(244, 63, 94, ${intensity})`;
        } else {
            // 0 to 1 (Blue scale) - default for correlation
            const intensity = Math.abs(value);
            return `rgba(16, 185, 129, ${intensity})`; // Emerald-500
        }
    };

    return (
        <div className="finlove-card p-4">
            <h3 className="mb-4 text-sm font-semibold text-slate-300">{title}</h3>
            <div
                className="grid gap-1"
                style={{
                    gridTemplateColumns: `auto repeat(${size}, minmax(0, 1fr))`,
                }}
            >
                {/* Header Row */}
                <div className="h-6"></div> {/* Empty corner */}
                {labels.map((label, i) => (
                    <div key={`h-${i}`} className="flex items-center justify-center text-[10px] font-medium text-slate-500">
                        {label}
                    </div>
                ))}

                {/* Data Rows */}
                {data.map((row, i) => (
                    <React.Fragment key={`row-${i}`}>
                        {/* Row Label */}
                        <div className="flex items-center justify-end pr-2 text-[10px] font-medium text-slate-500">
                            {labels[i]}
                        </div>
                        {/* Cells */}
                        {row.map((val, j) => (
                            <div
                                key={`cell-${i}-${j}`}
                                className="group relative aspect-square w-full rounded-sm transition-all hover:scale-110 hover:z-10 hover:shadow-lg"
                                style={{ backgroundColor: getColor(val) }}
                            >
                                {/* Tooltip */}
                                <div className="absolute bottom-full left-1/2 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded bg-slate-900 px-2 py-1 text-xs text-white shadow-xl group-hover:block z-20 border border-slate-700">
                                    <div className="font-semibold">{labels[i]} x {labels[j]}</div>
                                    <div>{val.toFixed(4)}</div>
                                </div>
                            </div>
                        ))}
                    </React.Fragment>
                ))}
            </div>
        </div>
    );
}
