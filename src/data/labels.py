import pandas as pd


def make_labels(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Ajoute deux colonnes à une copie de df :
    - RUL : cycles restants avant la panne = dernier cycle du moteur - cycle courant
    - target : 1 si RUL <= horizon, sinon 0
    """
    if horizon < 0:
        raise ValueError(f"horizon doit être positif ou nul, reçu : {horizon}")

    out = df.copy()

    last_cycle = out.groupby("unit_id")["cycle"].transform("max")

    out["RUL"] = last_cycle - out["cycle"]
    out["target"] = (out["RUL"] <= horizon).astype(int)

    return out