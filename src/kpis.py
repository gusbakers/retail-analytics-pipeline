import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/retail.db")

def load_clean() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sales_clean", conn)
    conn.close()
    return df

def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {
        "total_revenue": round(df["sales_amount"].sum(), 2),
        "total_orders": df["order_id"].nunique(),
        "avg_order_value": round(df["sales_amount"].mean(), 2),
    }
    logger.info("KPIs computed")
    return kpis

def run():
    df = load_clean()
    kpis = compute_kpis(df)
    for k, v in kpis.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run()
