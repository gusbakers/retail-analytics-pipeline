import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/retail.db")

def load_raw(table_name: str = "sales_raw") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Shape before cleaning: {df.shape}")
    df = df.drop_duplicates()
    df = df.dropna(subset=["order_id", "product_id", "sales_amount"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["year"] = df["order_date"].dt.year
        df["month"] = df["order_date"].dt.month
    if "sales_amount" in df.columns:
        df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce")
        df = df[df["sales_amount"] > 0]
    logger.info(f"Shape after cleaning: {df.shape}")
    return df

def save_clean(df: pd.DataFrame, table_name: str = "sales_clean"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.info(f"Saved {len(df)} rows to '{table_name}'")
    conn.close()

def run():
    df = load_raw()
    df_clean = clean(df)
    save_clean(df_clean)

if __name__ == "__main__":
    run()
