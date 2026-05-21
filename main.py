import logging
from src.ingestion import run as ingest
from src.transform import run as transform
from src.queries import run as queries
from src.kpis import run as kpis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== PIPELINE START ===")
    ingest()
    transform()
    queries()
    kpis()
    logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
