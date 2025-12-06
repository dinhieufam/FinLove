"""
Standalone EDA script for the OHLCV CSVs in `data/`.
Run from repo root:
    python notebooks/financial_data_eda.py --output eda_report.html
Produces an HTML report with interactive Plotly charts, no nbformat needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


def resolve_data_dir() -> Path:
    """Find the data directory whether run from repo root or notebooks/."""
    cwd = Path.cwd().resolve()
    candidates = [cwd / "data", cwd.parent / "data"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find data directory next to this script.")


def clean_price_file(csv_path: Path) -> pd.DataFrame:
    """Load a single ticker CSV, fix header, and index by Date."""
    raw = pd.read_csv(csv_path, header=[0, 1, 2])

    cleaned_cols: List[Tuple[str, str]] = []
    for col in raw.columns:
        if col[0] == "Price" and col[1] == "Ticker":
            cleaned_cols.append(("Date", "Date"))
        else:
            cleaned_cols.append((col[0], col[1]))
    raw.columns = pd.MultiIndex.from_tuples(cleaned_cols, names=["field", "ticker"])

    df = raw.copy()
    df[("Date", "Date")] = pd.to_datetime(df[("Date", "Date")])
    df = df.set_index(("Date", "Date")).sort_index()
    df.index.name = "Date"
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_all_price_data(data_dir: Path) -> pd.DataFrame:
    """Load and merge all ticker CSV files into a single MultiIndex DataFrame."""
    frames = [clean_price_file(p) for p in sorted(data_dir.glob("*.csv"))]
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def choose_focus(tickers: Iterable[str]) -> List[str]:
    preferred = ["AAPL", "MSFT", "NVDA", "AMZN"]
    available = [t for t in preferred if t in tickers]
    if available:
        return available
    tickers_list = list(tickers)
    return tickers_list[: min(4, len(tickers_list))]


def build_figures(price_data: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return list of (section title, html snippet) with Plotly figures."""
    close = price_data.xs("Close", axis=1, level=0)
    volume = price_data.xs("Volume", axis=1, level=0)
    tickers = close.columns.tolist()
    focus = choose_focus(tickers)

    figs: List[Tuple[str, str]] = []

    # Price trends
    fig_prices = px.line(
        close.reset_index(),
        x="Date",
        y=focus,
        title="Closing prices (selected tickers)",
        labels={"value": "Close", "variable": "Ticker"},
    )
    figs.append(("Closing prices", fig_prices.to_html(full_html=False, include_plotlyjs=False)))

    # Return stats
    returns = close.pct_change().dropna()
    fig_hist = px.histogram(
        returns,
        x=focus[0],
        nbins=60,
        title=f"Daily returns (%) for {focus[0]}",
        labels={focus[0]: "Return"},
    )
    fig_hist.update_xaxes(tickformat=".2%")
    figs.append(("Return distribution", fig_hist.to_html(full_html=False, include_plotlyjs=False)))

    # Rolling volatility
    window = 21
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    fig_vol = px.line(
        rolling_vol.reset_index(),
        x="Date",
        y=focus,
        title=f"{window}-day annualized volatility",
        labels={"value": "Volatility", "variable": "Ticker"},
    )
    figs.append(("Rolling volatility", fig_vol.to_html(full_html=False, include_plotlyjs=False)))

    # Correlation
    corr = returns.corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        origin="lower",
        title="Correlation of daily returns",
        zmin=-1,
        zmax=1,
    )
    figs.append(("Correlation heatmap", fig_corr.to_html(full_html=False, include_plotlyjs=False)))

    # Volume ranks
    avg_vol = volume.mean().sort_values(ascending=False).reset_index()
    avg_vol.columns = ["Ticker", "avg_volume"]
    fig_vol_bar = px.bar(
        avg_vol,
        x="Ticker",
        y="avg_volume",
        title="Average daily volume",
        labels={"avg_volume": "Shares"},
    )
    fig_vol_bar.update_layout(xaxis_tickangle=-45)
    figs.append(("Volume snapshot", fig_vol_bar.to_html(full_html=False, include_plotlyjs=False)))

    return figs


def render_report(figs: List[Tuple[str, str]], output: Path, data_dir: Path, price_data: pd.DataFrame) -> None:
    """Write a self-contained HTML file with embedded PlotlyJS."""
    plotly_js = pio.to_html(px.line(pd.DataFrame({"x": [0, 1], "y": [0, 1]})), include_plotlyjs=True, full_html=False)
    plotly_js_only = plotly_js.split("</body>")[0].split("<body>")[-1]

    coverage = []
    close = price_data.xs("Close", axis=1, level=0)
    for ticker in close.columns:
        s = close[ticker]
        coverage.append(
            f"<tr><td>{ticker}</td><td>{s.first_valid_index().date()}</td>"
            f"<td>{s.last_valid_index().date()}</td><td>{s.isna().sum()}</td></tr>"
        )
    coverage_html = "\n".join(coverage)

    sections = []
    for title, html_snippet in figs:
        sections.append(f"<section><h2>{title}</h2>{html_snippet}</section>")
    sections_html = "\n".join(sections)

    template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Financial Data EDA</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0 auto; padding: 24px; max-width: 1200px; }}
    section {{ margin-bottom: 48px; }}
    h1, h2 {{ margin-bottom: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  {plotly_js_only}
  <h1>Financial Price Data EDA</h1>
  <p>Data directory: {data_dir}</p>
  <h2>Coverage (Close prices)</h2>
  <table>
    <tr><th>Ticker</th><th>Start</th><th>End</th><th>Missing Close Count</th></tr>
    {coverage_html}
  </table>
  {sections_html}
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA report for financial CSV data.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing CSV files (default: auto-detect).")
    parser.add_argument("--output", type=Path, default=Path("notebooks/eda_report.html"), help="HTML output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or resolve_data_dir()
    price_data = load_all_price_data(data_dir)
    figs = build_figures(price_data)
    render_report(figs, args.output, data_dir, price_data)
    print(f"EDA report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
