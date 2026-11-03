"""Etap 4. Cechy cenowe i rynkowe (momentum, RSI, SMA, zmienność, beta, Amihud, GARCH), wskaźniki
wyceny, F-Score Piotroskiego oraz zmienna celu (63 sesje).
"""

import math
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from konfiguracja import DANE, POSREDNIE

close, adj, vol, div, cech = pickle.load(open(POSREDNIE / "prep.pkl", "rb"))
idx = close.index
H = 63
fxq = cech.groupby("Data").fx.median().rename("FX_USDPLN_QE")
cech["FX_USDPLN_QE"] = cech.Data.map(fxq)
w = pd.read_csv(DANE / "wig_d.csv", parse_dates=["Data"]).set_index("Data").Zamkniecie
wig = w.reindex(idx)
cf = close.ffill(limit=10)
af = adj.ffill(limit=10)
ret = af.pct_change(fill_method=None)
lret = np.log(af / af.shift(1))
wr = wig.pct_change(fill_method=None)
mom = {k: af / af.shift(k) - 1 for k in (21, 63, 126, 252)}
mom12_1 = af.shift(21) / af.shift(252) - 1
d = af.diff()
up = d.clip(lower=0)
dn = (-d).clip(lower=0)
au = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
ad = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
rsi = 100 - 100 / (1 + au / ad)
rsi = rsi.where(ad > 0, 100).where(au.notna())
sma50 = af / af.rolling(50, min_periods=50).mean() - 1
sma200 = af / af.rolling(200, min_periods=200).mean() - 1
v63 = lret.rolling(63, min_periods=40).std() * np.sqrt(252)
v252 = lret.rolling(252, min_periods=126).std() * np.sqrt(252)
cov = ret.rolling(252, min_periods=126).cov(wr)
varw = wr.rolling(252, min_periods=126).var()
beta = cov.div(varw, axis=0)
hi252 = af / af.rolling(252, min_periods=126).max() - 1
turn = close * vol
amih = (ret.abs() / turn.where(turn > 0)).rolling(63, min_periods=30).mean() * 1e6
dsum = div.rolling(252, min_periods=1).sum()
cech["pos_t"] = idx.searchsorted(pd.to_datetime(cech.sesja_t), side="right") - 1
assert (
    idx[cech.pos_t.values] == pd.to_datetime(cech.sesja_t).values
).mean() > 0.99, "sesja_t nie jest sesja"


def look(df):
    out = []
    for t, p in zip(cech.Ticker, cech.pos_t):
        out.append(df[t].iloc[p] if t in df.columns else np.nan)
    return np.array(out)


f = {}
for k, v in mom.items():
    f[f"Mom_{k}d"] = look(v)
f["Mom_12_1"] = look(mom12_1)
f["RSI_14"] = look(rsi)
f["SMA50_dist"] = look(sma50)
f["SMA200_dist"] = look(sma200)
f["Vol_63d"] = look(v63)
f["Vol_252d"] = look(v252)
f["Beta_WIG_252d"] = look(beta)
f["Dist_52W_High"] = look(hi252)
f["Amihud_63d"] = look(amih)
cech["Close_t"] = look(cf)
f["Dywidenda_12M_PLN"] = look(dsum)
for k, v in f.items():
    cech[k] = v
cech["Div_Yield_12M"] = cech.Dywidenda_12M_PLN / cech.Close_t
cech["Cap_t_PLN_mln"] = cech.Close_t * cech.Akcje / 1e6
cech["Log_Cap"] = np.log(cech.Cap_t_PLN_mln)
cech["DolVol_63d_PLN"] = cech.Mediana_wartosci_obrotu_63sesje_PLN
cech["Log_DolVol"] = np.log(cech.DolVol_63d_PLN.where(cech.DolVol_63d_PLN > 0))


def garch(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 250:
        return np.nan
    r = (r - r.mean()) * 100
    v0 = r.var()
    if v0 <= 0:
        return np.nan
    rl = r.tolist()

    def run(a, b):
        wv = v0 * (1 - a - b)
        s = v0
        ll = 0.0
        for x in rl:
            ll += math.log(s) + x * x / s
            s = wv + a * x * x + b * s
        return ll, s

    def nll(p):
        a, b = p
        if a + b >= 0.999:
            return 1e10
        return run(a, b)[0]

    res = minimize(nll, [0.08, 0.88], bounds=[(1e-4, 0.5), (0.3, 0.998)], method="L-BFGS-B")
    if not res.success and res.fun >= 1e9:
        return np.nan
    s = run(*res.x)[1]
    return math.sqrt(s) * math.sqrt(252) / 100


g = []
for t, p in zip(cech.Ticker, cech.pos_t):
    g.append(garch(lret[t].iloc[max(0, p - 499) : p + 1].values))
cech["GARCH_Vol"] = g
fx = cech.FX_USDPLN_QE
cap = cech.Cap_t_PLN_mln


def yld(x):
    return x * fx / cap


cech["EP"] = yld(cech.Zysk_Netto_TTM)
cech["BP"] = yld(cech.Kapital_Dominujacej.fillna(cech.Kapital_Wlasny))
cech["SP"] = yld(cech.Przychody_TTM)
cech["OCFP"] = yld(cech.OCF_TTM)
cech["FCFP"] = yld(cech.FCF_Calc_TTM)
cech["TangBP"] = yld(cech.Wartosc_Ksiegowa_Tang)
ev = cap + cech.Dlug_Netto * fx
cech["EV_PLN_mln"] = ev
cech["EV_EBITDA"] = (ev / (cech.EBITDA_TTM * fx)).where(cech.EBITDA_TTM > 0)
cech["EV_Sales"] = ev / (cech.Przychody_TTM * fx)
cech["Log_Aktywa_PLN"] = np.log((cech.Aktywa * fx).where(cech.Aktywa > 0))
cech = cech.sort_values(["Ticker", "Data"]).reset_index(drop=True)
prev = cech.groupby("Ticker").Akcje.shift(4)
cech["F7_Brak_Emisji"] = np.where(prev.notna(), (cech.Akcje <= prev).astype(float), np.nan)
g4 = cech.groupby("Ticker")


def l4(x):
    return x.groupby(cech.Ticker).shift(4)


q3 = cech.Zysk_Netto / cech.Aktywa
q9 = cech.Przychody / cech.Aktywa
d3 = q3 - l4(q3)
d9 = q9 - l4(q9)
d8 = None
for num in ["EBIT", "EBITDA", "Zysk_Brutto", "Zysk_Netto"]:
    m_ = cech[num] / cech.Przychody
    d = m_ - l4(m_)
    d8 = d if d8 is None else d8.fillna(d)
for col, dq in [
    ("F3_Wzrost_ROA", d3),
    ("F8_Wzrost_Marzy_EBIT", d8),
    ("F9_Wzrost_Obrotu_Aktywow", d9),
]:
    fb = (dq > 0).astype(float).where(dq.notna())
    cech["Zrodlo_" + col[:2]] = np.where(
        cech[col].notna(), "TTM", np.where(fb.notna(), "kwartal_YoY", "brak")
    )
    cech[col] = cech[col].fillna(fb)
Fs = [
    "F1_Zysk_Dodatni",
    "F2_OCF_Dodatni",
    "F3_Wzrost_ROA",
    "F4_OCF_ponad_Zysk",
    "F5_Spadek_Zadluzenia_DT",
    "F6_Wzrost_Plynnosci",
    "F7_Brak_Emisji",
    "F8_Wzrost_Marzy_EBIT",
    "F9_Wzrost_Obrotu_Aktywow",
]
cech["F_Score_9"] = cech[Fs].sum(axis=1, min_count=9)
cech["F_n"] = cech[Fs].notna().sum(axis=1)
cech["F_Score_dost"] = (cech[Fs].mean(axis=1) * 9).where(cech.F_n >= 7)
fin = cech.Spolka_finansowa == 1
for c in Fs + ["F_Score_8", "F_Score_9", "F_Score_dost", "Altman_Z_EM"]:
    cech.loc[fin, c] = np.nan
ent = cech.pos_t + 1
ex = cech.pos_t + 1 + H
ok = ex < len(idx)
A = af.values
col = {t: i for i, t in enumerate(af.columns)}


def px(pos, t):
    return A[pos, col[t]] if pos < len(idx) else np.nan


ps = np.array([px(e, t) for e, t in zip(ent, cech.Ticker)])
pe = np.array([px(e, t) for e, t in zip(ex, cech.Ticker)])
W = wig.values
ws = np.array([W[e] if e < len(idx) else np.nan for e in ent])
we = np.array([W[e] if e < len(idx) else np.nan for e in ex])
cech["Data_wejscia"] = [idx[e] if e < len(idx) else pd.NaT for e in ent]
cech["Data_wyjscia"] = [idx[e] if e < len(idx) else pd.NaT for e in ex]
cech["Zwrot_Spolki_63"] = pe / ps - 1
cech["Zwrot_WIG_63"] = we / ws - 1
cech["Y_excess"] = cech.Zwrot_Spolki_63 - cech.Zwrot_WIG_63
cech["Znany_cel"] = cech.Y_excess.notna().astype(int)
pickle.dump(cech, open(POSREDNIE / "stage1.pkl", "wb"))
