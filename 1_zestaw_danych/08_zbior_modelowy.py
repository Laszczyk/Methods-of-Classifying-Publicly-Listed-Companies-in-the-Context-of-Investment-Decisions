"""Etap 8. Wybór cech i zapis zbioru modelowego (zbior_modelowy_v1.csv) wraz z opisem zmiennych.
"""

import pandas as pd

from konfiguracja import WYNIKI

m = pd.read_csv(WYNIKI / "panel_finalny_v1_model.csv")
c = list(m.columns)
i = c.index("Fold_Rekomendacja")
base = c[: i + 1]
CECHY = [
    ("EP_rank", "Wycena", "P/E", "odwrotność P/E (zysk/cena), ranga przekrojowa w kwartale [-1,1]"),
    ("BP_rank", "Wycena", "P/BV", "kapitał własny/cena, ranga przekrojowa"),
    (
        "EV_EBITDA_rank",
        "Wycena",
        "EV/EBITDA",
        "EV/EBITDA, ranga przekrojowa; NaN gdy EBITDA_TTM <= 0",
    ),
    (
        "ROE_TTM",
        "Jakość",
        "ROE",
        "zwrot z kapitału własnego z 4 kwartałów, winsoryzowany 1/99 per kwartał",
    ),
    (
        "Marza_OCF_TTM",
        "Jakość",
        "Marża OCF",
        "przepływy operacyjne / przychody z 4 kwartałów, winsoryzowane",
    ),
    (
        "F_Score_dost",
        "Jakość",
        "F-Score",
        "Piotroski: średnia dostępnych składników × 9 przy co najmniej 7 dostępnych",
    ),
    ("Altman_Z_EM", "Ryzyko upadłości", "Z-Score", "Altman Z dla rynków wschodzących"),
    (
        "Growth_Przychody_YoY",
        "Wzrost",
        "ΔRev YoY",
        "zmiana przychodów kwartał do kwartału rok wcześniej, winsoryzowana",
    ),
    (
        "Delta_ZyskNetto_Aktywa",
        "Wzrost",
        "ΔEPS YoY (zamiennik)",
        "zmiana zysku netto r/r odniesiona do aktywów; zastępuje Delta_EPS_YoY (usunięta: dzielenie przez EPS bliski zera)",
    ),
    ("Mom_63d", "Momentum", "Mom 3M", "zwrot z Adj Close z 63 sesji do dnia t"),
    ("Mom_12_1", "Momentum", "Mom 12M", "zwrot z 252 sesji z pominięciem ostatnich 21 sesji"),
    ("RSI_14", "Momentum", "RSI(14)", "RSI Wildera z 14 sesji"),
    ("SMA200_dist", "Trend", "D-SMA200", "Adj Close / średnia z 200 sesji - 1"),
    (
        "GARCH_Vol",
        "Ryzyko",
        "GARCH",
        "prognoza zannualizowanej zmienności GARCH(1,1), okno 500 sesji",
    ),
    ("Beta_WIG_252d", "Ryzyko", "Beta", "beta wobec WIG z 252 sesji (min. 126)"),
    ("Log_Cap_rank", "Wielkość", "dodana", "log kapitalizacji, ranga przekrojowa"),
    ("Dlug_Netto_Aktywa", "Zadłużenie", "dodana", "dług netto / aktywa, winsoryzowany"),
    (
        "Div_Yield_12M_rank",
        "Dywidenda",
        "dodana",
        "stopa dywidendy z 12 miesięcy, ranga przekrojowa",
    ),
    ("Amihud_63d_rank", "Płynność", "dodana", "miara Amihuda z 63 sesji, ranga przekrojowa"),
    (
        "Growth_Aktywa_YoY",
        "Wzrost aktywów",
        "dodana",
        "zmiana aktywów r/r (anomalia inwestycyjna), winsoryzowana",
    ),
]
MAKRO = [
    (
        "Rynek_Vol_63d",
        "Makro",
        "zmienność WIG z 63 sesji (annualizowana), wspólna dla wszystkich spółek w dniu t",
    ),
    (
        "Makro_A",
        "Makro",
        "średnia z-score (VIX, EPOL odwrócony, USDPLN, WIBOR 3M, rentowność obligacji 5Y); wyższa = stres",
    ),
    (
        "Makro_B",
        "Makro",
        "jak A, z wagami z kroczącej regresji grzbietowej (okno kończy się w t-64)",
    ),
    ("Makro_A_x_Beta", "Makro", "Makro_A × Beta_WIG_252d"),
    ("Makro_B_x_Beta", "Makro", "Makro_B × Beta_WIG_252d"),
]
f20 = [a for a, *_ in CECHY]
fm = [a for a, *_ in MAKRO]
out = m[base + f20 + fm]
opis = pd.DataFrame(
    [(a, g, n, d, "bez makro i z makro") for a, g, n, d in CECHY]
    + [(a, g, "-", d, "tylko wariant z makro") for a, g, d in MAKRO],
    columns=["Kolumna", "Grupa", "Odpowiednik z listy autora", "Definicja", "Zestaw"],
)
zest = pd.DataFrame(
    {
        "Zestaw": ["A: bez makro (20 cech)"] * 20 + ["B: z makro (25 cech)"] * 25,
        "Cecha": f20 + f20 + fm,
    }
)
fold = pd.read_excel(WYNIKI / "panel_finalny_v1.xlsx", sheet_name="Foldy")
with pd.ExcelWriter(WYNIKI / "zbior_modelowy_v1.xlsx", engine="openpyxl") as w:
    out.to_excel(w, sheet_name="Model", index=False)
    opis.to_excel(w, sheet_name="Opis_cech", index=False)
    zest.to_excel(w, sheet_name="Zestawy", index=False)
    fold.to_excel(w, sheet_name="Foldy", index=False)
out.to_csv(WYNIKI / "zbior_modelowy_v1.csv", index=False)
