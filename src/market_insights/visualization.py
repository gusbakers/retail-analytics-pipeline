"""One consistent visual system, with source and scope on every figure."""

from pathlib import Path
import textwrap
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from .sentiment import CLASSES

NAVY = "#142D4E"
BLUE = "#347CCA"
TEAL = "#13A7A8"
MUTED = "#61758B"
GRID = "#E3EBF4"
CMAP = LinearSegmentedColormap.from_list(
    "retail", ["#F2F7FC", "#B7D5EE", "#4489BD", "#154575"]
)


def figure(title, subtitle, size=(10.5, 5.8)):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, ax = plt.subplots(figsize=size, facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.93, top=0.77, bottom=0.18)
    fig.text(0.06, 0.94, title, fontsize=21, weight="bold", color=NAVY)
    fig.text(0.06, 0.86, subtitle, fontsize=10.5, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8, labelcolor=MUTED)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    return fig, ax


def save(fig, path, note):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.06, 0.035, note, fontsize=8.2, color=MUTED)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def plot_weekly(weekly, cutoff, scope, out):
    sales = weekly.groupby("week_start").units_sold.sum().sort_index()
    fig, ax = figure("Weekly recorded sales", scope)
    ax.axvspan(pd.Timestamp(cutoff), sales.index.max(), color="#EDF5FD")
    ax.plot(sales.index, sales.values, lw=2.4, color=BLUE)
    ax.fill_between(sales.index, 0, sales.values, color=BLUE, alpha=0.06)
    ax.axvline(pd.Timestamp(cutoff), color=TEAL, ls="--", lw=1.4)
    ax.set_ylim(0, sales.max() * 1.12)
    ax.set_xlim(sales.index.min(), sales.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1000:,.0f}k"))
    ax.set_ylabel("Recorded units", color=MUTED)
    save(
        fig,
        out / "weekly_sales.png",
        "Shaded period: final holdout. Recorded sales are not a measure of unconstrained demand.",
    )


def plot_forecast(
    predictions, example, scope, out, country=None, weights=None, baseline="rolling_13"
):
    d = predictions.loc[predictions.segment.eq(example)].sort_values("week_start")
    if country is not None:
        d = d.loc[d.country.eq(country)]
    if d.empty or d.week_start.duplicated().any():
        raise ValueError("Choose one nonempty country/segment forecast series.")
    fig, ax = figure(
        "Observed sales vs. one-week forecast",
        f"{scope} | {d.country.iloc[0]} / {example}",
    )
    ax.plot(d.week_start, d.units_sold, color=NAVY, lw=2.3, label="Observed")
    ax.plot(
        d.week_start,
        d.ensemble,
        color=TEAL,
        lw=2.2,
        marker="o",
        ms=3,
        label="Selected ensemble",
    )
    ax.plot(
        d.week_start,
        d[baseline],
        color=MUTED,
        lw=1.5,
        ls="--",
        label=f"{baseline.removeprefix('rolling_')}-week mean",
    )
    ax.set_ylim(0, max(d.units_sold.max(), d.ensemble.max()) * 1.15)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("Units", color=MUTED)
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=9)
    save(
        fig,
        out / "forecast_vs_actual.png",
        "One country/segment; observed history updates weekly. Ensemble weights selected using earlier validation periods.",
    )


def plot_models(metrics, scope, out):
    labels = {
        "naive": "Last-week baseline",
        "rolling_4": "4-week mean",
        "rolling_8": "8-week mean",
        "rolling_13": "13-week mean",
        "rolling_26": "26-week mean",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "ridge": "Ridge + seasonal terms",
        "ensemble": "Selected ensemble",
    }
    d = metrics.set_index("model").reindex(list(labels))
    fig, ax = figure(
        "Forecast error against stronger baselines", scope, size=(10.5, 7.5)
    )
    ax.grid(False)
    ax.grid(axis="x", color=GRID)
    fig.subplots_adjust(left=0.25)
    bars = ax.barh(
        [labels[k] for k in d.index],
        d.mae,
        color=[MUTED] * 5 + [BLUE, BLUE, BLUE, TEAL],
        height=0.52,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, d.mae.max() * 1.23)
    for bar, v in zip(bars, d.mae):
        ax.text(
            v + d.mae.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,.1f}",
            va="center",
            color=NAVY,
        )
    ax.set_xlabel("Mean absolute error (units) | lower is better", color=MUTED)
    save(
        fig,
        out / "model_comparison.png",
        "Same target and dates for every model. Ensemble configuration and weights use only earlier temporal validation.",
    )


def plot_errors(predictions, scope, out):
    d = predictions.assign(
        absolute_error=(predictions.units_sold - predictions.ensemble).abs()
    )
    by = (
        d.groupby(["country", "segment"])
        .absolute_error.mean()
        .sort_values(ascending=False)
        .head(12)
    )
    fig, ax = figure("Where the ensemble misses", scope, size=(10.5, 6.5))
    fig.subplots_adjust(left=0.35)
    ax.grid(False)
    ax.grid(axis="x", color=GRID)
    labels = [
        textwrap.fill(f"{country} / {segment}", 30)
        if d.country.nunique() > 1
        else str(segment)
        for country, segment in by.index
    ]
    ax.barh(labels, by.values, color=BLUE, height=0.56)
    ax.invert_yaxis()
    ax.set_xlim(0, by.max() * 1.22)
    for i, v in enumerate(by):
        ax.text(v + by.max() * 0.025, i, f"{v:,.0f}", va="center", color=NAVY)
    ax.set_xlabel("Mean absolute error (units)", color=MUTED)
    save(
        fig,
        out / "error_by_series.png",
        "Final holdout errors; series ordered by MAE. Investigating a high error does not establish its cause.",
    )


def plot_sql(results, scope, out):
    top = results["01_top_products"].head(8).copy()
    fig, ax = figure("Highest-volume products", scope, size=(11, 6.5))
    fig.subplots_adjust(left=0.40)
    names = ["\n".join(textwrap.wrap(str(v), 31)) for v in top.product_name]
    ax.grid(False)
    ax.grid(axis="x", color=GRID)
    ax.barh(names, top.units_sold, color=[TEAL] + [BLUE] * (len(top) - 1), height=0.58)
    ax.invert_yaxis()
    ax.set_xlim(0, top.units_sold.max() * 1.22)
    for i, v in enumerate(top.units_sold):
        ax.text(
            v + top.units_sold.max() * 0.02,
            i,
            f"{v / 1000:,.1f}k",
            va="center",
            fontsize=10,
            color=NAVY,
        )
    ax.set_xlabel("Recorded units | thousands", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1000:,.0f}k"))
    save(
        fig,
        out / "top_products.png",
        "All retained paid positive product sales. Products grouped by SKU; returns and nonproduct codes excluded.",
    )
    timing = results["05_dow_x_hour_heatmap"]
    if timing.empty or timing.total_orders.sum() <= 0:
        raise ValueError("Cannot plot an empty order-timing result.")
    heat = (
        timing.pivot(index="order_dow", columns="order_hour", values="total_orders")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )
    fig, ax = figure("When orders are recorded", scope, size=(11, 5.5))
    ax.grid(False)
    im = ax.imshow(
        heat.values, aspect="auto", cmap=CMAP, interpolation="nearest", vmin=0
    )
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{v:02d}:00" for v in range(0, 24, 2)])
    ax.set_xlabel("Hour recorded in the source export", color=MUTED)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cb.set_label("Distinct orders", color=MUTED)
    cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(
        fig,
        out / "orders_heatmap.png",
        "Monday = 0; actual hour labels. Empty cells mean no retained orders recorded in the export, not unobserved demand.",
    )


def plot_pricing(results, out):
    labels = {
        "revenue_usd": "Revenue",
        "units_sold": "Units",
        "margin_usd": "Contribution margin",
    }
    fig, ax = figure(
        "A simulated discount: revenue, volume and margin",
        "SIMULATION | 800 independent experimental units | Week 4 generator",
    )
    fig.subplots_adjust(left=0.27)
    ax.grid(False)
    ax.grid(axis="x", color=GRID)
    for i, row in enumerate(results.itertuples(index=False)):
        low = row.ci95_low / row.control_mean * 100
        high = row.ci95_high / row.control_mean * 100
        ax.plot([low, high], [i, i], color=BLUE, lw=4, solid_capstyle="round")
        ax.scatter([row.lift_pct], [i], s=100, color=TEAL, zorder=3)
        ax.text(high + 1.2, i, f"{row.lift_pct:+.1f}%", va="center", color=NAVY)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(results)), [labels[m] for m in results.metric])
    ax.invert_yaxis()
    ax.margins(x=0.28, y=0.35)
    ax.set_xlabel("Difference as % of the observed control mean", color=MUTED)
    save(
        fig,
        out / "pricing_effects.png",
        "95% Welch difference intervals scaled by control mean (not ratio CIs). Costs are assumed; Holm-adjusted p-values in CSV.",
    )


def plot_sentiment(metrics, matrix, out):
    fig, ax = figure(
        "Sentiment on unseen text templates",
        "SIMULATION | word + character TF-IDF / logistic regression | template-group holdout",
        size=(8.5, 6.6),
    )
    fig.subplots_adjust(left=0.21, right=0.87, bottom=0.19)
    ax.grid(False)
    ax.imshow(matrix, cmap=CMAP, interpolation="nearest", vmin=0)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted class", color=MUTED)
    ax.set_ylabel("Label", color=MUTED)
    for r in range(3):
        for c in range(3):
            ax.text(
                c,
                r,
                str(matrix[r, c]),
                ha="center",
                va="center",
                fontsize=19,
                color="white" if matrix[r, c] > 0.55 * matrix.max() else NAVY,
            )
    save(
        fig,
        out / "sentiment_confusion.png",
        "Unique full texts may still share template phrases. This additional holdout evaluates unseen template families, not real customer reviews.",
    )


def plot_shap(importance, scope, out):
    d = importance.head(10).iloc[::-1]
    fig, ax = figure("What drives the ensemble prediction", scope, size=(11, 6.5))
    fig.subplots_adjust(left=0.34)
    ax.grid(False)
    ax.grid(axis="x", color=GRID)
    names = [
        textwrap.fill(
            v.replace("numeric__", "")
            .replace("category__", "")
            .replace("num__", "")
            .replace("dow__dow_", "weekday "),
            32,
        )
        for v in d.feature
    ]
    ax.barh(names, d.mean_abs_shap, color=BLUE, height=0.57)
    ax.set_xlabel("Mean absolute SHAP contribution (units)", color=MUTED)
    save(
        fig,
        out / "shap_global.png",
        "Tree-path and linear training references. Weighted RF/XGBoost/Ridge contributions reconstruct the ensemble.",
    )


def plot_daily(predictions, metrics, out):
    scope = f"REAL DATA | UCI Online Retail II | UK | {predictions.date.min():%d %b} - {predictions.date.max():%d %b %Y}"
    fig, ax = figure("Next-day paid-order forecasts", scope, size=(12, 6.5))
    ax.plot(
        predictions.date,
        predictions.orders,
        color=NAVY,
        lw=1.8,
        label="Recorded orders",
    )
    ax.plot(
        predictions.date,
        predictions.ensemble,
        color=TEAL,
        lw=1.7,
        label="Selected ensemble",
    )
    ax.set_ylim(0, max(predictions.orders.max(), predictions.ensemble.max()) * 1.12)
    ax.set_xlim(predictions.date.min(), predictions.date.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Distinct retained orders per day", color=MUTED)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    save(
        fig,
        out / "daily_forecast.png",
        f"{len(predictions)} chronological test days. Each next-day forecast uses prior observed days. This target differs from weekly SKU units.",
    )
    labels = {
        "weekly_naive": "Previous same weekday",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "ridge": "Ridge + seasonal terms",
        "ensemble": "Selected ensemble",
    }
    fig, ax = figure(
        "Daily forecast error: ensemble vs. baseline", scope, size=(10.5, 6.2)
    )
    fig.subplots_adjust(left=0.29)
    ax.grid(False)
    ax.grid(axis="x", color=GRID)
    d = metrics.set_index("model").reindex(list(labels))
    bars = ax.barh(
        [labels[k] for k in d.index],
        d.mae,
        color=[MUTED, BLUE, BLUE, BLUE, TEAL],
        height=0.52,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, d.mae.max() * 1.27)
    for bar, value in zip(bars, d.mae):
        ax.text(
            value + d.mae.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            color=NAVY,
        )
    ax.set_xlabel("Mean absolute error (orders per day) | lower is better", color=MUTED)
    save(
        fig,
        out / "model_comparison.png",
        f"All models forecast the same {len(predictions)} days. Configuration and blend weights use three earlier 28-day validation folds.",
    )
