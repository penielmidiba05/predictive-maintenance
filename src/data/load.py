from pathlib import Path
import pandas as pd

PROJECT_ROOT= Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

OP_COLS = [f"op_setting_{i}" for i in range (1,4)]
SENSOR_COLS = [f"s_{i}" for i in range (1,22)]
COLUMNS = ["unit_id","cycle"] + OP_COLS + SENSOR_COLS

CONSTANT_SENSORS = ["s1","s5","s6","s10","s16","s18","s19"]
FEATURE_SENSORS = [c for c in SENSOR_COLS if c not in CONSTANT_SENSORS]

def load_cmapss(split: str, subset: str = "FD001", raw_dir: Path = RAW_DIR ) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError(f"split doit etre train ou test, reçu: {slpit!r}")
    path = Path(raw_dir)/ f"{split}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header= None, names = COLUMNS)

def load_test_rul(subset: str = "FD001", raw_dir: Path = RAW_DIR) -> pd.Series:
    path = Path(raw_dir)/ f"RUL_{subset}.txt"
    return pd.read_csv (path, sep=r"s+", header=None, names= ["RUL"]) ["RUL"]
