"""Etap 5. Wskaźniki makro: Makro_A (średnia z-score) i Makro_B (wagi z regresji grzbietowej na
oknie 500 sesji, bez wykorzystania przyszłości).
"""

import numpy as np
import pandas as pd

from konfiguracja import POSREDNIE
from makro_wczytanie import D

W = 252
idx = D["WIG"].dropna().index
X = D.drop(columns="WIG").reindex(D.index.union(idx)).ffill().reindex(idx)
Z = (X - X.rolling(W, min_periods=W).mean()) / X.rolling(W, min_periods=W).std()
Z["EPOL"] = -Z["EPOL"]
Z = Z.rename(columns={"EPOL": "EPOL_inv"})
Z = Z.dropna()
A = Z.mean(axis=1)
w = D["WIG"].dropna()
fut = w.shift(-64) / w.shift(-1) - 1
H = 64
N = 500
lam = 50.0
Zv = Z.values
fv = fut.reindex(Z.index).values
Bv = np.full(len(Z), np.nan)
Wt = np.full((len(Z), Z.shape[1]), np.nan)
for i in range(N + H, len(Z)):
    e = i - H
    sl = slice(e - N + 1, e + 1)
    xa = Zv[sl]
    ya = fv[sl]
    m = ~np.isnan(ya)
    xa = xa[m]
    ya = ya[m] - ya[m].mean()
    xc = xa - xa.mean(0)
    wts = np.linalg.solve(xc.T @ xc + lam * len(xc) * np.eye(xc.shape[1]), xc.T @ ya)
    Wt[i] = wts
    Bv[i] = Zv[i] @ wts
B = pd.Series(Bv, Z.index)
Bz = (B - B.rolling(W, min_periods=126).mean()) / B.rolling(W, min_periods=126).std()
M = pd.DataFrame({"Makro_A": A, "Makro_B": Bz, "Makro_A_s10": A.rolling(10).mean()}).join(Z)
M["fut63"] = fut.reindex(M.index)
M.to_pickle(POSREDNIE / "makro_daily.pkl")
pd.DataFrame(Wt, index=Z.index, columns=Z.columns).dropna().to_pickle(POSREDNIE / "makro_wagi.pkl")
