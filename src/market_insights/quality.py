"""Transaction contracts and auditable cleaning rules."""

import numpy as np
import pandas as pd

ALIASES = {
    "orderid": "order_id",
    "productid": "product_id",
    "orderdate": "order_date",
    "productname": "product_name",
    "invoiceno": "order_id",
    "invoice": "order_id",
    "stockcode": "product_id",
    "description": "product_name",
    "quantity": "quantity",
    "invoicedate": "order_date",
    "unitprice": "unit_price",
    "price": "unit_price",
    "customerid": "customer_id",
    "country": "country",
}


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep paid, positive product sales; report every exclusion."""
    df = raw.copy()
    names = df.columns.str.strip().str.lower().str.replace(r"[\s_]+", "", regex=True)
    df.columns = [ALIASES.get(c, c) for c in names]
    required = {
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
        "order_date",
        "country",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing transaction columns: {', '.join(missing)}")
    if df.columns.duplicated().any():
        raise ValueError("Ambiguous duplicate column names after normalization.")
    audit = {"input_rows": len(df)}
    audit["exact_duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    for c in ["order_id", "product_id", "country"]:
        df[c] = df[c].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df.quantity, errors="coerce")
    df["unit_price"] = pd.to_numeric(df.unit_price, errors="coerce")
    df["order_date"] = pd.to_datetime(df.order_date, errors="coerce")
    invalid = (
        df[list(required)].isna().any(axis=1)
        | df.order_id.eq("")
        | df.product_id.eq("")
        | df.country.eq("")
        | ~np.isfinite(df.quantity)
        | ~np.isfinite(df.unit_price)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        invalid |= ~np.isfinite(df.quantity * df.unit_price)
    audit["invalid_required_rows"] = int(invalid.sum())
    df = df.loc[~invalid].copy()
    cancelled = df.order_id.str.upper().str.startswith("C") | df.quantity.le(0)
    audit["cancelled_or_nonpositive_quantity_rows"] = int(cancelled.sum())
    df = df.loc[~cancelled].copy()
    free = df.unit_price.le(0)
    audit["nonpositive_price_rows"] = int(free.sum())
    df = df.loc[~free].copy()
    nonproduct = ~df.product_id.str.fullmatch(r"\d{5}[A-Za-z]{0,2}").fillna(False)
    audit["nonproduct_code_rows"] = int(nonproduct.sum())
    df = df.loc[~nonproduct].copy()
    if df.empty:
        raise ValueError("No paid product sales remain after cleaning.")
    if "product_name" not in df:
        df["product_name"] = df.product_id
    df["product_name"] = (
        df.product_name.astype("string").fillna(df.product_id).str.strip()
    )
    if "customer_id" not in df:
        df["customer_id"] = pd.NA
    audit["retained_rows_without_customer_id"] = int(df.customer_id.isna().sum())
    df["order_dow"] = df.order_date.dt.dayofweek  # Monday = 0
    df["order_hour"] = df.order_date.dt.hour
    df["gross_sales"] = df.quantity * df.unit_price
    audit["retained_rows"] = len(df)
    audit["date_min"] = str(df.order_date.min())
    audit["date_max"] = str(df.order_date.max())
    assert df.order_dow.between(0, 6).all()
    assert df.order_hour.between(0, 23).all()
    assert df.quantity.gt(0).all() and df.unit_price.gt(0).all()
    return df.reset_index(drop=True), audit


def validate_weekly(df: pd.DataFrame) -> dict:
    required = {"week_start", "country", "segment", "units_sold", "avg_price"}
    if not required.issubset(df):
        raise ValueError(f"Weekly data must include {sorted(required)}.")
    if df.empty or df.columns.duplicated().any():
        raise ValueError("Weekly data must be nonempty with unique column names.")
    if df.duplicated(["week_start", "country", "segment"]).any():
        raise ValueError("Duplicate country/segment/week keys.")
    if df[["week_start", "country", "segment", "units_sold"]].isna().any().any():
        raise ValueError("Missing weekly keys or target.")
    if (
        not pd.api.types.is_numeric_dtype(df.units_sold)
        or not np.isfinite(df.units_sold).all()
    ):
        raise ValueError("Weekly targets must be finite numeric values.")
    if (df.units_sold < 0).any():
        raise ValueError("Negative weekly sales.")
    price = pd.to_numeric(df.avg_price, errors="coerce")
    if (
        df.avg_price.notna() & (price.isna() | ~np.isfinite(price) | price.le(0))
    ).any():
        raise ValueError("Observed weekly prices must be finite and positive.")
    if (
        df[["country", "segment"]]
        .astype("string")
        .apply(lambda col: col.str.strip().eq(""))
        .any()
        .any()
    ):
        raise ValueError("Weekly country/segment keys cannot be blank.")
    dates = pd.to_datetime(df.week_start, errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid weekly dates.")
    if not dates.dt.dayofweek.eq(0).all():
        raise ValueError("week_start must be Monday.")
    gaps = (
        df.assign(week_start=dates)
        .sort_values("week_start")
        .groupby(["country", "segment"])
        .week_start.diff()
        .dropna()
    )
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("The panel must have contiguous weekly records.")
    coverage = (
        df.assign(week_start=dates)
        .groupby(["country", "segment"])
        .week_start.agg(["min", "max", "count"])
    )
    if coverage.nunique().gt(1).any():
        raise ValueError("All series must share the same covered calendar weeks.")
    return {
        "rows": len(df),
        "series": int(df.groupby(["country", "segment"]).ngroups),
        "weeks": int(dates.nunique()),
        "missing_by_column": df.isna().sum().to_dict(),
    }
