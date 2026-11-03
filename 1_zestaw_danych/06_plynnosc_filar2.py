"""Etap 6. Miary płynności dla filaru 2 selekcji spółek: lata z brakiem obrotu i udział kwartałów z
obrotem poniżej 50 tys. PLN.
"""

import numpy as np
import pandas as pd

from konfiguracja import DANE, POSREDNIE


def px(sh):
    d = pd.read_excel(DANE / "yahoo_wszystkie_ceny.xlsx", sh, index_col=0)
    d.index = pd.to_datetime(d.index)
    d.columns = [c.replace(".WA", "") for c in d.columns]
    return d


def filar2():
    close, vol = px("Close"), px("Volume")
    qe = pd.date_range("2017-03-31", "2026-03-31", freq="QE")
    rows = []
    for t in close.columns:
        c = close[t]
        v = vol[t].reindex(c.index)
        f, l = c.first_valid_index(), c.last_valid_index()
        if f is None:
            continue
        w = (c.index >= f) & (c.index <= l)
        _cw, vw = c[w], v[w]
        zero = vw.fillna(0) == 0
        yr = zero.groupby(zero.index.year).agg(["mean", "size"])
        lata = int((yr["mean"] > 0.05).sum())
        maxy = float(yr["mean"].max())
        to = c * v
        med21 = to.rolling(21, min_periods=15).median()
        mean21 = to.rolling(21, min_periods=15).mean()
        med63 = to.rolling(63, min_periods=40).median()
        out = {}
        for nm, s in [("med21", med21), ("mean21", mean21), ("med63", med63)]:
            vals = [
                (
                    s[:q].iloc[-1]
                    if len(s[:q].dropna()) and s[:q].dropna().index[-1] >= q - pd.Timedelta(days=10)
                    else np.nan
                )
                for q in qe
                if q >= f
            ]
            vals = pd.Series(vals).dropna()
            out[nm] = float((vals < 50000).mean()) if len(vals) else np.nan
            out["n_" + nm] = len(vals)
        rows.append(
            dict(
                Ticker=t,
                Lata_zero_ponad5=lata,
                Max_rok_zero=maxy,
                Udzial_kw_pod50k=out["mean21"],
                Udz_med21=out["med21"],
                Udz_med63=out["med63"],
                N_kw=out["n_mean21"],
            )
        )
    d = pd.DataFrame(rows)
    d["F2_war11"] = (d.Lata_zero_ponad5 == 0).astype(int)
    d["F2_war12"] = (d.Udzial_kw_pod50k <= 0.10).astype(int)
    d["Filar2_Pass"] = d.F2_war11 * d.F2_war12
    return d


if __name__ == "__main__":
    filar2().to_csv(POSREDNIE / "filar2_tickery.csv", index=False)
