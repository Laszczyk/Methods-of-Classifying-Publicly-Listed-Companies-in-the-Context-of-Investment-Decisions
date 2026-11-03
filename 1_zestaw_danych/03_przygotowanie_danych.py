"""Etap 3. Łączy ceny, liczbę akcji, cechy fundamentalne, płynność i kapitalizację EMIS w jedną
tabelę spółka-kwartał.
"""

import pickle
import re

import numpy as np
import pandas as pd

from konfiguracja import DANE, POSREDNIE, WYNIKI
from korekta_ebitda import korekta_ebitda

QE = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}


def px(sh):
    d = pd.read_excel(DANE / "yahoo_wszystkie_ceny.xlsx", sh, index_col=0)
    d.index = pd.to_datetime(d.index)
    d.columns = [c.replace(".WA", "") for c in d.columns]
    return d


close, adj, vol, div = px("Close"), px("Adj_Close"), px("Volume"), px("Dividends")
st = pd.read_excel(DANE / "liczba_akcji_status.xlsx", "Status")
tb = pd.read_excel(DANE / "liczba_akcji_status.xlsx", "Tabela_uzytkownika")
zd = pd.read_excel(DANE / "liczba_akcji_status.xlsx", "Zdarzenia")
cech = pd.read_excel(DANE / "cechy_fundamentalne_v1.xlsx", "Cechy")
cech = cech[~cech.Ticker_Yahoo.isin(["NEU", "ATR"])].copy()
cech["Ticker"] = cech.Ticker_Yahoo
cech["Data"] = pd.to_datetime(cech.Rok.astype(str) + "-" + cech.Okres.map(QE))
shares = {}
for _, r in tb.iterrows():
    for c in tb.columns[1:]:
        m = re.match(r"(\d{4})/Q(\d)", c)
        q = pd.Timestamp(f"{m[1]}-" + QE["Q" + m[2]])
        if pd.notna(r[c]):
            shares[(r["Ticker"], q)] = r[c]
zd["Data"] = pd.to_datetime(zd.Data)
for t, g in zd[~zd.Ticker.isin(tb.Ticker)].groupby("Ticker"):
    g = g.sort_values("Data")
    for q in pd.to_datetime(sorted(cech.Data.unique())):
        e = g[g.Data <= q]
        if len(e):
            shares[(t, q)] = e.iloc[-1].Liczba_akcji_baza_Close
        else:
            shares[(t, q)] = g.iloc[0].Liczba_akcji_baza_Close
auto = st[~st.Ticker.isin(tb.Ticker) & ~st.Ticker.isin(zd.Ticker) & (st.Ticker != "NEU")]
for _, r in auto.iterrows():
    for q in pd.to_datetime(sorted(cech.Data.unique())):
        shares[(r.Ticker, q)] = (
            r.Liczba_biezaca_wyemitowanych
            if pd.notna(r.Liczba_biezaca_wyemitowanych)
            else r.Yahoo_obecna
        )
sh = pd.Series(shares).rename_axis(["Ticker", "Data"]).rename("Akcje").reset_index()
cech = cech.merge(sh, on=["Ticker", "Data"], how="left")
wl = pd.read_excel(DANE / "plynnosc_dynamiczna.xlsx", "Wiersze")
wl["Data"] = pd.to_datetime(wl.Data)
cech = cech.merge(wl, on=["Ticker", "Data"], how="left")
em = pd.read_excel(
    DANE / "emis_polaczone.xlsx",
    usecols=["Num", "Rok", "Okres", "Kapitalizacja rynkowa", "Cena zamknięcia"],
)
em = em.rename(
    columns={
        "Okres": "Okres_E",
        "Kapitalizacja rynkowa": "EMIS_cap_USD",
        "Cena zamknięcia": "EMIS_cena_USD",
    }
)
cech["Okres_E"] = cech.Okres.map({"Q1": "Q1", "Q2": "Q2C", "Q3": "Q3C", "Q4": "Y"})
cech = cech.merge(em, on=["Num", "Rok", "Okres_E"], how="left")
cf = close.ffill(limit=10)


def at(df, t, d):
    s = df[t]
    i = s.index.searchsorted(d, side="right") - 1
    return s.iloc[i] if i >= 0 else np.nan


cech["Close_QE"] = [at(cf, t, d) for t, d in zip(cech.Ticker, cech.Data)]
cech["Cap_QE_PLN"] = cech.Close_QE * cech.Akcje / 1e6
cech["fx"] = cech.Cap_QE_PLN / cech.EMIS_cap_USD
fx_kwartalny = cech.groupby("Data").fx.median()
cech["fx_med"] = cech.Data.map(fx_kwartalny)
cech["dev"] = cech.fx / cech.fx_med - 1
cech, log = korekta_ebitda(
    cech, fx_kwartalny, DANE / "kgh_business_radar.xlsx", WYNIKI / "kgh_porownanie.csv"
)
(WYNIKI / "korekta_ebitda.log").write_text("\n".join(log))
cech = cech.reset_index(drop=True)
pickle.dump((close, adj, vol, div, cech), open(POSREDNIE / "prep.pkl", "wb"))
