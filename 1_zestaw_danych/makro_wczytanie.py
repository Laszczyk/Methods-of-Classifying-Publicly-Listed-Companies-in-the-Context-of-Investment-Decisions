"""Wczytanie dziennych szeregów makro: VIX, EPOL, USDPLN, WIBOR 3M, rentowność obligacji 5-letnich
i WIG.
"""

import pandas as pd

from konfiguracja import DANE


def s(f, dc, vc, **k):
    d = pd.read_csv(DANE / f, **k)
    d[dc] = pd.to_datetime(d[dc])
    return d.set_index(dc)[vc].sort_index().astype(float)


vix = s("vix_d.csv", "Date", "Zamkniecie")
epol = s("epol_d.csv", "Date", "Zamkniecie")
wibor = s("wibor3m_d.csv", "Data", "Zamkniecie")
usd = s("usdpln_d.csv", "Data", "Zamkniecie")
wig = s("wig_d.csv", "Data", "Zamkniecie")
b = pd.read_csv(DANE / "obligacje_pl_5l.csv", encoding="utf-8-sig")
b["Date"] = pd.to_datetime(b["Date"], format="%m/%d/%Y")
bond = b.set_index("Date")["Price"].astype(float).sort_index()
D = pd.DataFrame(
    {"VIX": vix, "EPOL": epol, "USDPLN": usd, "WIBOR3M": wibor, "BOND5Y": bond, "WIG": wig}
)
