"""Public UCI loader and separate reproducible simulation sources."""

from pathlib import Path
import hashlib
import json
import logging
import urllib.request
import zipfile
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"


def download_retail(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "online_retail_ii.zip"
    if path.exists() and zipfile.is_zipfile(path):
        return path
    temporary = path.with_suffix(".part")
    LOGGER.info("Downloading public UCI Online Retail II (no account required)")
    try:
        with (
            urllib.request.urlopen(DATASET_URL, timeout=30) as response,
            temporary.open("wb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("The response is not a valid dataset ZIP.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_transactions(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    path = download_retail(raw_dir)
    archive_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = raw_dir / "transactions_cached.csv.gz"
    manifest = raw_dir / "transactions_cached.json"
    if cache.exists() and manifest.exists():
        provenance = json.loads(manifest.read_text())
        if (
            provenance.get("zip_sha256") == archive_hash
            and provenance.get("cache_sha256")
            == hashlib.sha256(cache.read_bytes()).hexdigest()
        ):
            LOGGER.info("Reading verified cached transaction export")
            return pd.read_csv(
                cache, dtype={"Invoice": "string", "StockCode": "string"}
            ), provenance
    LOGGER.info("Reading the two-year Excel workbook")
    with zipfile.ZipFile(path) as archive:
        workbooks = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        if len(workbooks) != 1:
            raise ValueError("Expected exactly one Excel workbook in the UCI archive.")
        with archive.open(workbooks[0]) as source:
            sheets = pd.read_excel(
                source,
                sheet_name=None,
                dtype={"Invoice": "string", "StockCode": "string"},
            )
    raw = pd.concat(sheets.values(), ignore_index=True)
    provenance = {
        "source": "UCI Online Retail II",
        "source_url": DATASET_PAGE,
        "license": "CC BY 4.0",
        "currency": "GBP",
        "zip_sha256": archive_hash,
        "raw_rows": len(raw),
        "sheets": list(sheets),
    }
    raw.to_csv(cache, index=False, compression="gzip")
    provenance["cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(provenance, indent=2))
    return raw, provenance


def real_weekly(
    clean: pd.DataFrame,
    holdout_weeks: int = 16,
    top_n: int = 12,
    selection_validation_weeks: int = 24,
) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """UK top SKUs selected on training sales; complete calendar weeks only."""
    minimum = clean.order_date.min().normalize()
    maximum = clean.order_date.max().normalize()
    first = minimum + pd.Timedelta(days=(7 - minimum.dayofweek) % 7)
    last = maximum - pd.Timedelta(days=maximum.dayofweek)
    if maximum.dayofweek != 6:
        last -= pd.Timedelta(days=7)
    weeks = pd.date_range(first, last, freq="W-MON")
    if len(weeks) < holdout_weeks + 32:
        raise ValueError("Insufficient complete weeks for this forecasting case.")
    cutoff = weeks[-holdout_weeks]
    uk = clean.loc[clean.country.eq("United Kingdom")].copy()
    uk["week_start"] = uk.order_date.dt.to_period("W-SUN").dt.start_time
    uk = uk.loc[uk.week_start.between(first, last)]
    selection_cutoff = cutoff - pd.Timedelta(weeks=selection_validation_weeks)
    # Fix the business cohort before the FIRST validation fold, not after it.
    history = uk.loc[uk.week_start < selection_cutoff]
    activity = history.groupby("product_id").week_start.nunique()
    eligible = activity[activity >= 26].index
    ranked = (
        history.loc[history.product_id.isin(eligible)]
        .groupby("product_id")
        .quantity.sum()
        .sort_values(ascending=False)
    )
    selected = ranked.head(top_n).index.tolist()
    if len(selected) < 2:
        raise ValueError("Not enough active training SKUs.")
    table = (
        uk.loc[uk.product_id.isin(selected)]
        .groupby(["product_id", "week_start"])
        .agg(units_sold=("quantity", "sum"), revenue=("gross_sales", "sum"))
        .reset_index()
    )
    table["avg_price"] = table.revenue / table.units_sold
    dense = pd.MultiIndex.from_product(
        [selected, weeks], names=["product_id", "week_start"]
    )
    panel = table.set_index(["product_id", "week_start"]).reindex(dense).reset_index()
    # Zero denotes no paid sales recorded in this export, not zero latent demand.
    panel["units_sold"] = panel.units_sold.fillna(0)
    panel["country"] = "United Kingdom"
    panel = panel.rename(columns={"product_id": "segment"}).drop(columns="revenue")
    descriptions = (
        history.dropna(subset=["product_name"])
        .groupby("product_id")
        .product_name.last()
    )
    metadata = {
        "forecast_country": "United Kingdom",
        "selected_series": selected,
        "selection": "Top-volume SKUs active in at least 26 weeks before the first validation origin",
        "selection_cutoff": str(selection_cutoff.date()),
        "example_series": selected[0],
        "example_country": "United Kingdom",
        "labels": {s: str(descriptions.get(s, s)) for s in selected},
        "first_complete_week": str(first.date()),
        "last_complete_week": str(last.date()),
        "cutoff": str(cutoff.date()),
        "holdout_weeks": holdout_weeks,
        "target": "Paid positive units sold; cancellations/returns excluded",
        "zero_policy": "No paid sales recorded for a selected SKU in a covered week",
        "availability_limit": "Inventory and stockouts are not observed",
    }
    return panel, cutoff, metadata


def demo_demand() -> pd.DataFrame:
    """Week 2 generator, unchanged equations and seed 42."""
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2021-01-04", periods=104, freq="W-MON")
    country_factors = {
        "Dominican Republic": 1.28,
        "Colombia": 1.13,
        "Mexico": 1.05,
        "Peru": 0.91,
        "Argentina": 0.84,
        "Chile": 0.79,
    }
    category_units = {
        "Frozen Foods": 450,
        "Personal Care": 360,
        "Household Cleaning": 260,
        "Dairy": 410,
        "Snacks": 370,
        "Beverages": 500,
    }
    category_prices = {
        "Frozen Foods": 6.20,
        "Personal Care": 5.10,
        "Household Cleaning": 4.10,
        "Dairy": 3.00,
        "Snacks": 2.50,
        "Beverages": 1.70,
    }
    records = []

    for country, factor in country_factors.items():
        for category, category_level in category_units.items():
            base_units = max(60.0, category_level * factor + rng.normal(0, 95))
            prior_units = float(max(18, round(base_units + rng.normal(0, 25))))
            prior_units_2 = prior_units
            phase = rng.uniform(0, 2 * np.pi)
            for i, week in enumerate(weeks):
                promo = float(rng.uniform(0, 1))
                holiday = int(
                    week.month == 12 or (week.month in [1, 4, 11] and week.day <= 7)
                )
                price = round(
                    category_prices[category]
                    * (1 - 0.12 * promo)
                    * rng.uniform(0.98, 1.02),
                    2,
                )
                competitor = float(np.clip(rng.normal(1.0, 0.06), 0.75, 1.30))
                economic = float(
                    100 + 4 * np.sin(2 * np.pi * i / 104) + rng.normal(0, 1)
                )
                market_level = base_units * (
                    1 + 0.25 * np.sin(2 * np.pi * i / 52 + phase)
                )
                demand = (
                    0.45 * (prior_units + prior_units_2) / 2
                    + 0.55 * market_level
                    + 70 * promo
                    + 130 * promo * holiday
                    - 80 * (competitor - 1)
                    + 3 * (economic - 100)
                    - 18 * (price - category_prices[category])
                    + rng.normal(0, 35)
                )
                units = float(np.clip(round(demand), 18, 1584))
                # Shocks sintéticos puntuales con los extremos de referencia del PDF.
                if country == "Chile" and category == "Household Cleaning" and i == 17:
                    units = 18.0
                if (
                    country == "Dominican Republic"
                    and category == "Beverages"
                    and i == 80
                ):
                    units = 1584.0
                records.append(
                    {
                        "week_start": week.strftime("%Y-%m-%d"),
                        "country": country,
                        "product_category": category,
                        "month": week.month,
                        "units_sold_prior_week": prior_units,
                        "avg_price_usd": price,
                        "promo_intensity": round(promo, 4),
                        "holiday_flag": holiday,
                        "competitor_price_index": round(competitor, 4),
                        "economic_index": round(economic, 4),
                        "units_sold": units,
                    }
                )
                prior_units_2, prior_units = prior_units, units

    df_synthetic = pd.DataFrame(records)
    # Nulos repartidos por país; las demás columnas se mantienen completas.
    missing_counts = {
        "Dominican Republic": 10,
        "Colombia": 9,
        "Mexico": 9,
        "Peru": 8,
        "Argentina": 11,
        "Chile": 9,
    }
    for country, count in missing_counts.items():
        indices = df_synthetic.index[df_synthetic["country"] == country]
        missing_indices = rng.choice(indices, size=count, replace=False)
        df_synthetic.loc[missing_indices, "competitor_price_index"] = np.nan

    assert df_synthetic.shape == (3744, 11)
    assert df_synthetic["competitor_price_index"].isna().sum() == 56
    assert not df_synthetic.duplicated(
        ["week_start", "country", "product_category"]
    ).any()

    return df_synthetic.rename(
        columns={"product_category": "segment", "avg_price_usd": "avg_price"}
    )


def simulate_pricing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Week 4 scenario; not an experiment on UCI customers."""
    pricing_rng = np.random.default_rng(2427)
    base_price_by_category = {
        "Frozen Foods": 6.20,
        "Personal Care": 5.10,
        "Household Cleaning": 4.10,
        "Dairy": 3.00,
        "Snacks": 2.50,
        "Beverages": 1.70,
    }
    cost_synthetic = pd.DataFrame(
        {
            "product_category": list(base_price_by_category),
            "unit_cost_usd": [
                round(0.45 * p, 3) for p in base_price_by_category.values()
            ],
        }
    )
    groups = pricing_rng.permutation(np.repeat(["control", "treatment"], 400))
    categories = pricing_rng.choice(list(base_price_by_category), size=800)
    regular_price = np.array([base_price_by_category[c] for c in categories])
    # Una unidad experimental independiente con una compra agregada por fila.
    base_quantity = pricing_rng.lognormal(mean=np.log(80), sigma=0.30, size=800)
    units = np.maximum(
        1, np.round(base_quantity * np.where(groups == "treatment", 1.22, 1))
    ).astype(int)
    paid_price = regular_price * np.where(groups == "treatment", 0.92, 1)
    ab_synthetic = pd.DataFrame(
        {
            "group": groups,
            "revenue_usd": np.round(units * paid_price, 2),
            "product_category": categories,
            "units_sold": units,
        }
    )

    ab_synthetic["experimental_unit_id"] = [
        f"EXP-{i:04d}" for i in range(1, len(ab_synthetic) + 1)
    ]
    return ab_synthetic, cost_synthetic


def simulate_feedback() -> pd.DataFrame:
    """Week 4 text generator, plus template IDs for group-held-out evaluation."""
    country_factors = dict.fromkeys(
        ["Dominican Republic", "Colombia", "Mexico", "Peru", "Argentina", "Chile"]
    )
    category_units = dict.fromkeys(
        [
            "Frozen Foods",
            "Personal Care",
            "Household Cleaning",
            "Dairy",
            "Snacks",
            "Beverages",
        ]
    )
    import re

    feedback_rng = np.random.default_rng(2240)
    counts = {"positive": 273, "negative": 215, "neutral": 162}
    labels = feedback_rng.permutation(
        np.concatenate([np.repeat(label, count) for label, count in counts.items()])
    )
    phrases = {
        "positive": [
            "Great product, muy buena calidad!",
            "Lo recomiendo y lo volvería a comprar.",
            "Excellent quality and good value.",
            "Me encantó, cumplió mis expectativas.",
            "Very happy with this purchase.",
            "La calidad es buena y el precio razonable.",
            "Fresh and reliable, I recommend it.",
            "Superó mis expectativas, excelente.",
            "Not bad at all, I really like it.",
            "Buena compra, vale la pena.",
        ],
        "negative": [
            "Nunca más compro esto.",
            "Poor quality and disappointing service.",
            "No me gustó y no lo recomiendo.",
            "Bad experience, not worth the price.",
            "Llegó dañado y la calidad es mala.",
            "Very disappointed with this purchase.",
            "I expected better, the product failed.",
            "Demasiado caro para esta calidad.",
            "Not good, I would not buy again.",
            "El empaque falló y el contenido decepciona.",
        ],
        "neutral": [
            "It's ok, nothing special.",
            "No está mal, pero tampoco excelente.",
            "Average product, it does the job.",
            "Es aceptable, sin nada que destacar.",
            "Fine I guess, neither great nor bad.",
            "Cumple lo básico, nada más.",
            "Good packaging but average quality.",
            "Some things are good and some are bad.",
            "Ni lo recomiendo ni lo rechazo.",
            "Me parece normal por este precio.",
        ],
    }
    ambiguous = [
        "Ok.",
        "Fine I guess.",
        "Better than I feared.",
        "No está mal.",
        "Nothing special.",
        "Está bien, supongo.",
    ]
    contexts = [
        "Lo compré para la familia.",
        "It was a weekend purchase.",
        "Fue una compra para el trabajo.",
        "I bought it online.",
        "Lo encontré en el supermercado.",
        "I tried it after a friend mentioned it.",
        "Lo probé durante un viaje.",
        "It was part of my weekly shopping.",
        "Lo usé después de la entrega.",
        "I compared it with my usual brand.",
        "Lo elegí por el empaque.",
        "I received it with another order.",
    ]
    details = [
        "Tamaño pequeño.",
        "Tamaño familiar.",
        "Standard size.",
        "Presentación nueva.",
        "Regular package.",
        "Formato individual.",
        "The usual size.",
        "Nuevo formato.",
        "Same packaging.",
        "Edición habitual.",
    ]
    aspects = [
        "La entrega fue el lunes.",
        "Delivery was on Friday.",
        "Lo abrí por la mañana.",
        "I opened it after lunch.",
        "Lo revisé al llegar a casa.",
        "I used it during the weekend.",
        "Lo compartí con otra persona.",
        "I stored it on a shelf.",
        "Lo guardé para después.",
        "It arrived in a box.",
    ]
    rows = []
    seen = set()
    for i, label in enumerate(labels):
        for attempt in range(1000):
            # Ambigüedad intencional: el texto no siempre revela la etiqueta simulada.
            phrase = feedback_rng.choice(
                ambiguous if feedback_rng.random() < 0.16 else phrases[label]
            )
            review = " ".join(
                [
                    phrase,
                    feedback_rng.choice(contexts),
                    feedback_rng.choice(details),
                    feedback_rng.choice(aspects),
                ]
            )
            key = re.sub(r"[^\w\s]", "", review.lower()).strip()
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError("No se pudo generar un texto único.")
        rows.append(
            {
                "feedback_id": f"FB-{i + 1:04d}",
                "template_id": str(phrase),
                "country": feedback_rng.choice(list(country_factors)),
                "product_category": feedback_rng.choice(list(category_units)),
                "source_channel": feedback_rng.choice(
                    ["survey", "online_review", "customer_support"]
                ),
                "review_text": review,
                "sentiment_label": label,
            }
        )
    fb_synthetic = pd.DataFrame(rows)

    return fb_synthetic
