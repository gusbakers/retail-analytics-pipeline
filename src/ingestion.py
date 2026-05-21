import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw")
DB_PATH = Path("data/retail.db")

def load_csv(filename: str) -> pd.DataFrame:
    filepath = RAW_DATA_PATH / filename
    logger.info(f"Loading {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filename}")
    return df

def load_to_sqlite(df: pd.DataFrame, table_name: str, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.info(f"Loaded {len(df)} rows into table '{table_name}'")
    conn.close()

def run():
    df = load_csv("sales.csv")
    load_to_sqlite(df, "sales_raw")

if __name__ == "__main__":
    run()
