"""Etap 2 (pomocniczy). Pobranie dziennych szeregów VIX, EPOL i USDPLN z Yahoo Finance.

Pozostałe szeregi makro (WIBOR 3M, rentowność obligacji 5-letnich, WIG) pochodzą z plików
CSV wymienionych w README i są wczytywane w makro_wczytanie.py.
"""

import pandas as pd
import yfinance as yf

from konfiguracja import DANE

START, END = "2016-01-01", "2026-10-13"
SERIE = {"VIX": "^VIX", "EPOL": "EPOL", "USDPLN": "USDPLN=X"}

with pd.ExcelWriter(DANE / "makro_yahoo.xlsx") as zapis:
    for nazwa, symbol in SERIE.items():
        d = yf.download(symbol, start=START, end=END, auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        wynik = d[["Close"]].rename(columns={"Close": "Zamkniecie"}).dropna()
        wynik.index.name = "Data"
        wynik.to_excel(zapis, sheet_name=nazwa)
        print(nazwa, len(wynik), wynik.index.min().date(), wynik.index.max().date())
