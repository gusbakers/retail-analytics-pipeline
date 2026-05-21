import pandas as pd
import pytest
from src.transform import clean

def test_removes_duplicates():
    df = pd.DataFrame({
        "order_id": [1, 1, 2],
        "product_id": [10, 10, 20],
        "sales_amount": [100, 100, 200]
    })
    result = clean(df)
    assert len(result) == 2

def test_removes_negative_sales():
    df = pd.DataFrame({
        "order_id": [1, 2],
        "product_id": [10, 20],
        "sales_amount": [-50, 200]
    })
    result = clean(df)
    assert all(result["sales_amount"] > 0)

def test_columns_lowercase():
    df = pd.DataFrame({
        "Order_ID": [1],
        "Product_ID": [10],
        "Sales_Amount": [100]
    })
    result = clean(df)
    assert all(c == c.lower() for c in result.columns)
