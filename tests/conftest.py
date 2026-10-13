import pandas as pd
import pytest


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "Invoice": ["1", "2", "3"],
            "StockCode": ["10001", "10002", "10001"],
            "Quantity": [2, 1, 3],
            "Price": [4, 5, 4],
            "InvoiceDate": ["2024-01-08 10:00", "2024-01-09 11:00", "2024-01-15 10:00"],
            "Country": ["United Kingdom"] * 3,
            "Description": ["A", "B", "A"],
        }
    )


@pytest.fixture
def weekly():
    dates = pd.date_range("2021-01-04", periods=64, freq="W-MON")
    return pd.DataFrame(
        [
            {
                "week_start": d,
                "country": "UK",
                "segment": s,
                "units_sold": float(10 + i + (s == "B") * 5),
                "avg_price": 2.0,
            }
            for s in ["A", "B"]
            for i, d in enumerate(dates)
        ]
    )
