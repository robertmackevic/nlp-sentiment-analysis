from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DATASET_CSV_PATH = ROOT_DIR / "IMDB.csv"
PROCESSED_DATASET_CSV_PATH = ROOT_DIR / "IMDB-processed.csv"
