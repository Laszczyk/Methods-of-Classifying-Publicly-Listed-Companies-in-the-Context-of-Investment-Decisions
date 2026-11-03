"""Etap 1. Pobranie dziennych notowań 98 spółek GPW z Yahoo Finance (Close, Adj Close, wolumen,
dywidendy, splity) wraz z diagnostyką.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from konfiguracja import DANE

START = "2016-01-01"
END = None
PROG_SKOKU = 0.35
PLIK_WYJSCIOWY = DANE / "yahoo_wszystkie_ceny.xlsx"

tickers = [
    "11B.WA",
    "1AT.WA",
    "ABE.WA",
    "ABS.WA",
    "ACP.WA",
    "AGO.WA",
    "AMB.WA",
    "AMC.WA",
    "APR.WA",
    "APT.WA",
    "ARH.WA",
    "ASE.WA",
    "ATC.WA",
    "ATR.WA",
    "ATT.WA",
    "BDX.WA",
    "BFT.WA",
    "BIO.WA",
    "BLO.WA",
    "BMC.WA",
    "BRS.WA",
    "CAR.WA",
    "CBF.WA",
    "CDR.WA",
    "CIG.WA",
    "CLN.WA",
    "CMP.WA",
    "COG.WA",
    "CPS.WA",
    "DAT.WA",
    "DCR.WA",
    "DIG.WA",
    "DNP.WA",
    "DOM.WA",
    "DVL.WA",
    "EAT.WA",
    "ECH.WA",
    "ELT.WA",
    "ENA.WA",
    "ENT.WA",
    "ERB.WA",
    "EUR.WA",
    "FRO.WA",
    "GPW.WA",
    "JSW.WA",
    "KGH.WA",
    "KGN.WA",
    "KTY.WA",
    "LBW.WA",
    "LPP.WA",
    "LWB.WA",
    "MBR.WA",
    "MCI.WA",
    "MDG.WA",
    "MLG.WA",
    "MNC.WA",
    "MRB.WA",
    "MRC.WA",
    "MSZ.WA",
    "NEU.WA",
    "NWG.WA",
    "OPL.WA",
    "OPN.WA",
    "PCR.WA",
    "PGE.WA",
    "PKN.WA",
    "PLW.WA",
    "PXM.WA",
    "QRS.WA",
    "RBW.WA",
    "RVU.WA",
    "SCP.WA",
    "SEL.WA",
    "SGN.WA",
    "SKA.WA",
    "SNK.WA",
    "SNT.WA",
    "STP.WA",
    "STX.WA",
    "SVE.WA",
    "TAR.WA",
    "TEN.WA",
    "TOA.WA",
    "TOR.WA",
    "TPE.WA",
    "UNI.WA",
    "UNT.WA",
    "VGO.WA",
    "VOT.WA",
    "VOX.WA",
    "VRG.WA",
    "WLT.WA",
    "WPL.WA",
    "WTN.WA",
    "WWL.WA",
    "XTB.WA",
    "ZEP.WA",
    "ZRE.WA",
]

POLA = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
NAZWY_ARKUSZY = {
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Adj Close": "Adj_Close",
    "Volume": "Volume",
    "Dividends": "Dividends",
    "Stock Splits": "Splits",
}

print(f"Pobieranie {len(tickers)} tickerów od {START}...")
raw = yf.download(
    tickers=tickers,
    start=START,
    end=END,
    auto_adjust=False,
    actions=True,
    group_by="column",
    threads=True,
    progress=True,
)

fields = {}
dostepne_pola = set(raw.columns.get_level_values(0))
for pole in POLA:
    if pole in dostepne_pola:
        fields[pole] = raw[pole].copy()
    else:
        fields[pole] = pd.DataFrame(index=raw.index)

for pole in fields:
    fields[pole].index = pd.to_datetime(fields[pole].index).tz_localize(None).normalize()


def brak_danych(t):
    c = fields["Close"]
    return t not in c.columns or c[t].dropna().empty


brakujace = [t for t in tickers if brak_danych(t)]
if brakujace:
    print(f"\nPonawiam pojedynczo dla {len(brakujace)} tickerów: {brakujace}")
for t in brakujace:
    try:
        h = yf.Ticker(t).history(start=START, end=END, auto_adjust=False, actions=True)
    except Exception as e:
        print(f"  {t}: błąd {e}")
        continue
    if h is None or h.empty:
        print(f"  {t}: brak danych w Yahoo")
        continue
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    for pole in POLA:
        if pole in h.columns:
            fields[pole] = fields[pole].drop(columns=[t], errors="ignore")
            fields[pole] = fields[pole].join(h[pole].rename(t), how="outer")
    print(f"  {t}: pobrano {len(h)} wierszy")

close, vol = fields["Close"], fields["Volume"]
maska_sesji = close.notna().any(axis=1) & (vol.fillna(0).sum(axis=1) > 0)
dni_sesyjne = close.index[maska_sesji]

for pole in fields:
    fields[pole] = fields[pole].reindex(dni_sesyjne).sort_index()
    fields[pole] = fields[pole].reindex(columns=tickers)

for pole in ["Dividends", "Stock Splits"]:
    fields[pole] = fields[pole].fillna(0.0)

close, adj = fields["Close"], fields["Adj Close"]
div, spl = fields["Dividends"], fields["Stock Splits"]

ret = close.pct_change(fill_method=None)

wiersze = []
for t in tickers:
    if close[t].dropna().empty:
        wiersze.append({"Ticker": t, "Dane": "BRAK"})
        continue
    c = close[t].dropna()
    a = adj[t].reindex(c.index)
    wsp = (a / c).dropna()
    r = ret[t].dropna()
    skok = r.abs()
    wiersze.append(
        {
            "Ticker": t,
            "Dane": "OK",
            "Pierwsza_data": c.index[0].strftime("%Y-%m-%d"),
            "Ostatnia_data": c.index[-1].strftime("%Y-%m-%d"),
            "N_sesji": len(c),
            "Adj_do_Close_na_starcie": wsp.iloc[0] if len(wsp) else np.nan,
            "Adj_do_Close_na_koncu": wsp.iloc[-1] if len(wsp) else np.nan,
            "Udzial_dni_Adj_rozne_od_Close": (
                (wsp.sub(1).abs() > 0.001).mean() if len(wsp) else np.nan
            ),
            "N_dywidend": int((div[t] > 0).sum()),
            "Suma_dywidend_PLN": float(div[t].sum()),
            "N_splitow": int((spl[t] > 0).sum()),
            "Max_dzienny_ruch_Close": skok.max() if len(skok) else np.nan,
            "Data_max_ruchu": skok.idxmax().strftime("%Y-%m-%d") if len(skok) else None,
        }
    )
diag = pd.DataFrame(wiersze)

lista_splitow = []
for t in tickers:
    s = spl[t]
    for d, v in s[s > 0].items():
        lista_splitow.append({"Ticker": t, "Data": d.strftime("%Y-%m-%d"), "Wspolczynnik": v})
df_splity = pd.DataFrame(lista_splitow, columns=["Ticker", "Data", "Wspolczynnik"])

lista_skokow = []
for t in tickers:
    r = ret[t].dropna()
    for d, v in r[r.abs() > PROG_SKOKU].items():
        lista_skokow.append(
            {
                "Ticker": t,
                "Data": d.strftime("%Y-%m-%d"),
                "Zwrot_Close": v,
                "Close_przed": close[t].shift(1).get(d),
                "Close_po": close[t].get(d),
                "Split_tego_dnia": float(spl[t].get(d, 0.0)),
            }
        )
df_skoki = pd.DataFrame(
    lista_skokow,
    columns=["Ticker", "Data", "Zwrot_Close", "Close_przed", "Close_po", "Split_tego_dnia"],
)


def do_zapisu(df):
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "Data"
    return out


with pd.ExcelWriter(PLIK_WYJSCIOWY, engine="openpyxl") as writer:
    for pole in POLA:
        do_zapisu(fields[pole]).to_excel(writer, sheet_name=NAZWY_ARKUSZY[pole])
    diag.to_excel(writer, sheet_name="Diagnostyka", index=False)
    df_splity.to_excel(writer, sheet_name="Splity_lista", index=False)
    df_skoki.to_excel(writer, sheet_name="Duze_skoki", index=False)

print("\n" + "-" * 60)
print(f"Zapisano: {PLIK_WYJSCIOWY}")
print(f"Sesji: {len(dni_sesyjne)}  ({dni_sesyjne.min().date()} -> {dni_sesyjne.max().date()})")
print(f"Tickery bez danych: {diag.loc[diag['Dane'] == 'BRAK', 'Ticker'].tolist()}")
