"""Korekta EBITDA i amortyzacji.

1. KGH: amortyzacja i EBITDA z Business Radar (tys. PLN) przeliczone na USD
   kwartalnym kursem implikowanym z kapitalizacji EMIS.
2. Pozostałe spółki: wiersze, w których EBITDA równa się amortyzacji mimo
   niezerowego EBIT, są przeliczane jako EBIT + amortyzacja.
3. Ponowne wyliczenie sum kroczących z 4 kwartałów (TTM) i pochodnych wskaźników.
"""

import re

import numpy as np
import pandas as pd

KONCE_KWARTALOW = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}


def _liczba(x):
    return float(str(x).replace(" ", "").replace("\xa0", ""))


def _wczytaj_kgh(plik, fx):
    d = pd.read_excel(plik, header=None)
    kwartaly = d.iloc[0, 3:40].tolist()
    ebit = d.iloc[1, 3:40].map(_liczba).values
    amort = d.iloc[2, 3:40].map(_liczba).values
    daty = []
    for q in kwartaly:
        numer = re.search(r"Q(\d)", q).group(1)
        rok = re.search(r"(\d{4})", q).group(1)
        daty.append(pd.Timestamp(f"{rok}-{KONCE_KWARTALOW[numer]}"))
    br = pd.DataFrame({"Data": daty, "EBIT_PLN": ebit / 1000, "AM_PLN": amort / 1000}).set_index(
        "Data"
    )
    br["fx"] = fx.reindex(br.index).values
    br["EBIT_USD"] = br.EBIT_PLN / br.fx
    br["AM_USD"] = br.AM_PLN / br.fx
    return br


def korekta_ebitda(cech, fx_kwartalny, plik_kgh, plik_porownanie):
    """Zwraca poprawioną tabelę oraz listę komunikatów do dziennika."""
    log = []
    br = _wczytaj_kgh(plik_kgh, fx_kwartalny)

    kgh = cech.Ticker == "KGH"
    biezace = cech[kgh].set_index("Data")
    porownanie = pd.DataFrame(
        {
            "EBIT_EMIS_patch": biezace.EBIT,
            "EBIT_BR_USD": br.EBIT_USD,
            "AM_EMIS": biezace.Amortyzacja,
            "AM_BR_USD": br.AM_USD,
        }
    ).dropna(how="all")
    porownanie["EBIT_roznica_%"] = (porownanie.EBIT_BR_USD / porownanie.EBIT_EMIS_patch - 1) * 100
    porownanie["AM_roznica_%"] = (porownanie.AM_BR_USD / porownanie.AM_EMIS - 1) * 100
    porownanie.to_csv(plik_porownanie)
    log.append("KGH: zapisano porownanie z Business Radar")

    for i in cech.index[kgh]:
        data = cech.at[i, "Data"]
        if data in br.index:
            cech.at[i, "Amortyzacja"] = br.at[data, "AM_USD"]
            ebit = cech.at[i, "EBIT"]
            cech.at[i, "EBITDA"] = ebit + br.at[data, "AM_USD"] if pd.notna(ebit) else np.nan

    zla = (
        (cech.Ticker != "KGH")
        & cech.EBITDA.notna()
        & cech.Amortyzacja.notna()
        & cech.EBIT.notna()
        & ((cech.EBITDA - cech.Amortyzacja).abs() <= 1e-3 * cech.Amortyzacja.abs().clip(lower=1e-9))
        & (cech.EBIT.abs() > 1e-3 * cech.Amortyzacja.abs())
    )
    log.append(
        "Naprawione wiersze (EBITDA == Amortyzacja): %d, spolki: %s"
        % (zla.sum(), cech[zla].Ticker.value_counts().to_dict())
    )
    cech.loc[zla, "EBITDA"] = cech.loc[zla, "EBIT"] + cech.loc[zla, "Amortyzacja"]

    cech = cech.sort_values(["Ticker", "Data"])
    g = cech.groupby("Ticker")
    cech["EBITDA_TTM"] = g.EBITDA.transform(lambda s: s.rolling(4, min_periods=4).sum())
    cech["Amortyzacja_TTM"] = g.Amortyzacja.transform(lambda s: s.rolling(4, min_periods=4).sum())
    cech["Marza_EBITDA_TTM"] = cech.EBITDA_TTM / cech.Przychody_TTM
    cech["Dlug_Netto_EBITDA"] = cech.Dlug_Netto / cech.EBITDA_TTM.where(cech.EBITDA_TTM > 0)
    return cech, log
