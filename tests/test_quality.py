import pandas as pd
import pytest
from market_insights.quality import clean_transactions, validate_weekly


def test_cleaning_normalizes_before_validation(transactions):
    normalized = transactions.rename(
        columns={
            "Invoice": " Order_ID ",
            "StockCode": "Product_ID",
            "Price": "Unit_Price",
            "InvoiceDate": "Order_Date",
            "Description": "Product_Name",
        }
    )
    clean, audit = clean_transactions(normalized)
    assert len(clean) == 3 and clean.order_dow.tolist() == [0, 1, 0]
    assert audit["retained_rows"] == 3


def test_cleaning_audits_returns_duplicates_and_invalid_values(transactions):
    extra = transactions.iloc[[0, 0, 0, 0]].copy()
    extra["Invoice"] = ["C9", "10", "11", "12"]
    extra["Quantity"] = [2, -1, 1, 1]
    extra["Price"] = [4, 4, 0, 4]
    extra.iloc[3, extra.columns.get_loc("InvoiceDate")] = "invalid"
    data = pd.concat([transactions, transactions.iloc[[0]], extra], ignore_index=True)
    clean, audit = clean_transactions(data)
    assert len(clean) == 3
    assert audit["exact_duplicate_rows"] == 1
    assert audit["cancelled_or_nonpositive_quantity_rows"] == 2
    assert audit["nonpositive_price_rows"] == 1
    assert audit["invalid_required_rows"] == 1
    assert sum(
        audit[k]
        for k in [
            "retained_rows",
            "exact_duplicate_rows",
            "invalid_required_rows",
            "cancelled_or_nonpositive_quantity_rows",
            "nonpositive_price_rows",
            "nonproduct_code_rows",
        ]
    ) == len(data)


def test_missing_customer_is_retained_for_aggregate_sales(transactions):
    transactions["Customer ID"] = None
    clean, audit = clean_transactions(transactions)
    assert len(clean) == 3 and audit["retained_rows_without_customer_id"] == 3


def test_missing_contract_column_gives_clear_error(transactions):
    with pytest.raises(ValueError, match="unit_price"):
        clean_transactions(transactions.drop(columns="Price"))


def test_weekly_keys_and_gaps(weekly):
    assert validate_weekly(weekly)["series"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        validate_weekly(pd.concat([weekly, weekly.iloc[[0]]]))
    with pytest.raises(ValueError, match="contiguous"):
        validate_weekly(weekly.drop(index=5))
