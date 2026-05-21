import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/retail.db")

def run_query(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def sales_by_category() -> pd.DataFrame:
    query = """
        SELECT category, SUM(sales_amount) AS total_sales,
        COUNT(order_id) AS total_orders
        FROM sales_clean GROUP BY category ORDER BY total_sales DESC
    """
    return run_query(query)

def sales_by_month() -> pd.DataFrame:
    query = """
        SELECT year, month, SUM(sales_amount) AS total_sales
        FROM sales_clean GROUP BY year, month ORDER BY year, month
    """
    return run_query(query)

def top_products(n: int = 10) -> pd.DataFrame:
    query = f"""
        SELECT product_id, SUM(sales_amount) AS total_revenue
        FROM sales_clean GROUP BY product_id
        ORDER BY total_revenue DESC LIMIT {n}
    """
    return run_query(query)

def run():
    print(sales_by_category())
    print(sales_by_month())
    print(top_products())

if __name__ == "__main__":
    run()
