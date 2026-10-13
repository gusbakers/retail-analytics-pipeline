"""The same five SQL files used by the pipeline and notebook."""

from pathlib import Path
import duckdb


def export_queries(transactions, database: Path, sql_dir: Path, destination: Path):
    database.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    with duckdb.connect(str(database)) as connection:
        connection.register("input_sales", transactions)
        connection.execute(
            "CREATE OR REPLACE TABLE sales_clean AS SELECT * FROM input_sales"
        )
        connection.unregister("input_sales")
        for path in sorted(sql_dir.glob("*.sql")):
            result = connection.execute(path.read_text(encoding="utf-8")).fetchdf()
            count = connection.execute("SELECT COUNT(*) FROM sales_clean").fetchone()[0]
            if count != len(transactions) or result.empty:
                raise ValueError(
                    f"Query {path.name} must preserve input rows and return analytics."
                )
            result.to_csv(destination / f"{path.stem}.csv", index=False)
            results[path.stem] = result
    return results
