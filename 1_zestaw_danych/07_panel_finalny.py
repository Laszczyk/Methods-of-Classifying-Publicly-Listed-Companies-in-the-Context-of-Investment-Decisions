"""Etap 7. Winsoryzacja i rangi per kwartał, cechy rynkowe i makro, podział na zbiory (z
oczyszczeniem okna zwrotów) i zapis panelu finalnego.
"""

import pickle

import numpy as np
import pandas as pd

from konfiguracja import DANE, POSREDNIE, WYNIKI

c = pickle.load(open(POSREDNIE / "stage1.pkl", "rb"))
c = c.drop(columns=["Delta_EPS_YoY"], errors="ignore")
st = pd.read_excel(DANE / "liczba_akcji_status.xlsx", "Status")
tb = pd.read_excel(DANE / "liczba_akcji_status.xlsx", "Tabela_uzytkownika")
zd = pd.read_excel(DANE / "liczba_akcji_status.xlsx", "Zdarzenia")
c["Zrodlo_akcji"] = np.where(
    c.Ticker.isin(tb.Ticker),
    "tabela_uzytkownika",
    np.where(c.Ticker.isin(zd.Ticker), "zdarzenia", "stala_Yahoo_bez_weryfikacji"),
)
last = c.sort_values("Data").groupby("Ticker").Akcje.last()
yh = st.set_index("Ticker").Yahoo_obecna
fl = (yh / last - 1).abs() > 0.01
c["Flaga_akcje_wlasne_mozliwe"] = c.Ticker.map(fl).fillna(False).astype(int)
c["Dev_vs_EMIS_cap"] = c.dev
c["Flaga_EMIS_dev5"] = (c.dev.abs() > 0.05).astype(int)
c["Data_t"] = pd.to_datetime(c.sesja_t)
f2 = pd.read_csv(POSREDNIE / "filar2_tickery.csv").rename(
    columns={
        "Lata_zero_ponad5": "Filar2_Lata_zero_ponad5",
        "Max_rok_zero": "Filar2_Max_rok_zero",
        "Udzial_kw_pod50k": "Filar2_Udzial_kw_pod50k",
    }
)
c = c.merge(
    f2[
        [
            "Ticker",
            "Filar2_Lata_zero_ponad5",
            "Filar2_Max_rok_zero",
            "Filar2_Udzial_kw_pod50k",
            "Filar2_Pass",
        ]
    ],
    on="Ticker",
    how="left",
)


def pod(r):
    if r.Rok < 2018:
        return "burn_in_2017"
    if r.Znany_cel == 0:
        return "bez_celu"
    return "train" if r.Data <= pd.Timestamp("2023-09-30") else "test"


c["Podzial"] = c.apply(pod, axis=1)
tr = c[c.Podzial == "train"]
te = c[c.Podzial == "test"]
fund = [
    "ROE_TTM",
    "ROA_TTM",
    "Marza_Operacyjna_TTM",
    "Marza_EBITDA_TTM",
    "Marza_Netto_TTM",
    "Marza_OCF_TTM",
    "Marza_FCF_TTM",
    "Wskaznik_Zadluzenia",
    "Zadluzenie_Bilans",
    "Dlug_Netto_Aktywa",
    "Dlug_Netto_EBITDA",
    "Plynnosc_Biezaca",
    "Gotowka_Aktywa",
    "WNiP_Aktywa",
    "Zyski_Zatrzymane_Aktywa",
    "CapEx_Przychody_TTM",
    "Obrot_Aktywow_TTM",
    "Growth_Przychody_YoY",
    "Growth_Przychody_TTM_YoY",
    "Delta_ZyskNetto_Aktywa",
    "Delta_OCF_Aktywa",
    "Delta_ZyskNetto_TTM_Aktywa",
    "Growth_Aktywa_YoY",
    "Altman_Z_EM",
]
tech = [
    "Mom_21d",
    "Mom_63d",
    "Mom_126d",
    "Mom_252d",
    "Mom_12_1",
    "RSI_14",
    "SMA50_dist",
    "SMA200_dist",
    "Vol_63d",
    "Vol_252d",
    "Beta_WIG_252d",
    "Dist_52W_High",
    "GARCH_Vol",
    "Udzial_sesji_zero_wolumen_252",
]
winz = fund + tech
rank = [
    "Log_Cap",
    "Log_DolVol",
    "Amihud_63d",
    "Log_Aktywa_PLN",
    "EP",
    "BP",
    "SP",
    "OCFP",
    "FCFP",
    "TangBP",
    "EV_EBITDA",
    "EV_Sales",
    "Div_Yield_12M",
]
bin_ = [
    "Turnaround_ZyskNetto",
    "F1_Zysk_Dodatni",
    "F2_OCF_Dodatni",
    "F3_Wzrost_ROA",
    "F4_OCF_ponad_Zysk",
    "F5_Spadek_Zadluzenia_DT",
    "F6_Wzrost_Plynnosci",
    "F7_Brak_Emisji",
    "F8_Wzrost_Marzy_EBIT",
    "F9_Wzrost_Obrotu_Aktywow",
    "F_Score_9",
    "F_Score_dost",
    "F_n",
]
gq = c.groupby(["Data", "Spolka_finansowa", "Filar2_Pass"])


def wz(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)


def rk(s):
    n = s.notna().sum()
    r = s.rank(method="average")
    return (2 * (r - 1) / (n - 1) - 1) if n > 1 else s * np.nan


M = pd.DataFrame(index=c.index)
for k in winz:
    M[k] = gq[k].transform(wz)
for k in rank:
    M[k + "_rank"] = gq[k].transform(rk)
for k in bin_:
    M[k] = c[k]
_w = pd.read_csv(DANE / "wig_d.csv", parse_dates=["Data"]).set_index("Data").Zamkniecie.sort_index()
_lr = np.log(_w / _w.shift(1))
_rf = pd.DataFrame(
    {
        "Rynek_Mom_21d": _w / _w.shift(21) - 1,
        "Rynek_Mom_63d": _w / _w.shift(63) - 1,
        "Rynek_Mom_126d": _w / _w.shift(126) - 1,
        "Rynek_Mom_252d": _w / _w.shift(252) - 1,
        "Rynek_Vol_63d": _lr.rolling(63, min_periods=40).std() * np.sqrt(252),
        "Rynek_SMA200_dist": _w / _w.rolling(200, min_periods=200).mean() - 1,
    }
)
_pos = _rf.index.searchsorted(pd.to_datetime(c.Data_t), side="right") - 1
for k in _rf.columns:
    c[k] = _rf[k].values[_pos]
    M[k] = c[k]
rynek = list(_rf.columns)
_mk = pd.read_pickle(POSREDNIE / "makro_daily.pkl")
_pm = _mk.index.searchsorted(pd.to_datetime(c.Data_t), side="right") - 1
c["Makro_A"] = _mk["Makro_A"].values[_pm]
c["Makro_B"] = _mk["Makro_B"].values[_pm]
_bcol = "Beta_WIG_252d" if "Beta_WIG_252d" in M.columns else None
c["Makro_A_x_Beta"] = c["Makro_A"] * M[_bcol]
c["Makro_B_x_Beta"] = c["Makro_B"] * M[_bcol]
makro = ["Makro_A", "Makro_B", "Makro_A_x_Beta", "Makro_B_x_Beta"]
for k in makro:
    M[k] = c[k]


c["Y_demean"] = c.Y_excess - gq["Y_excess"].transform("mean")
c["Do_modelu_bez_F2"] = ((c.Spolka_finansowa == 0) & (c.Rok >= 2018) & (c.Znany_cel == 1)).astype(
    int
)
c["Do_modelu"] = (c.Do_modelu_bez_F2 * c.Filar2_Pass).astype(int)
qs = sorted(c.Data.unique())
trq = [q for q in qs if q <= pd.Timestamp("2023-09-30") and q >= pd.Timestamp("2018-03-31")]
vq = [q for q in trq if q >= pd.Timestamp("2021-03-31")]
foldcols = []
finfo = []
for k, v in enumerate(vq, 1):
    col = f"Fold_{k:02d}"
    foldcols.append(col)
    c[col] = ""
    val = (c.Data == v) & (c.Do_modelu == 1)
    ent = c.loc[val, "Data_wejscia"].min()
    older = (c.Data < v) & (c.Data >= pd.Timestamp("2018-03-31")) & (c.Do_modelu == 1)
    purge = older & (c.Data_wyjscia >= ent)
    train = older & ~purge
    c.loc[train, col] = "train"
    c.loc[purge, col] = "purge"
    c.loc[val, col] = "val"
    finfo.append((col, v.date(), int(train.sum()), int(purge.sum()), int(val.sum()), ent.date()))
_qlast = c.loc[(c.Do_modelu == 1), "Data"].max()
_val = (c.Data == _qlast) & (c.Do_modelu == 1)
_ent = c.loc[_val, "Data_wejscia"].min()
_old = (c.Data < _qlast) & (c.Data >= pd.Timestamp("2018-03-31")) & (c.Do_modelu == 1)
_pg = _old & (c.Data_wyjscia >= _ent)
_tr = _old & ~_pg
c["Fold_Rekomendacja"] = ""
c.loc[_tr, "Fold_Rekomendacja"] = "train"
c.loc[_pg, "Fold_Rekomendacja"] = "purge"
c.loc[_val, "Fold_Rekomendacja"] = "rekomendacja"
foldcols.append("Fold_Rekomendacja")
finfo = pd.DataFrame(
    finfo,
    columns=[
        "Fold",
        "Kwartal_walidacji",
        "Wiersze_train",
        "Wiersze_purge",
        "Wiersze_val",
        "Wejscie_walidacji",
    ],
)
c["Y_excess_w"] = gq["Y_excess"].transform(wz)
c["Y_rank"] = gq["Y_excess"].transform(rk)
c["Y_bin"] = np.where(c.Y_excess.notna(), (c.Y_excess > 0).astype(float), np.nan)
c["Zwrot_Spolki_63_w"] = gq["Zwrot_Spolki_63"].transform(wz)
c["Y_klasa"] = np.where(
    c.Y_rank.isna(), np.nan, np.where(c.Y_rank < -1 / 3, 0.0, np.where(c.Y_rank > 1 / 3, 2.0, 1.0))
)
ident = ["Ticker", "Firma", "Data", "Rok", "Okres", "Data_t", "Data_wejscia", "Data_wyjscia"]
flags = [
    "Do_modelu",
    "Do_modelu_bez_F2",
    "Filar2_Pass",
    "Filar2_Lata_zero_ponad5",
    "Filar2_Max_rok_zero",
    "Filar2_Udzial_kw_pod50k",
    "Podzial",
    "Znany_cel",
    "W_Probie_v3",
    "Spolka_finansowa",
    "Plynna_A_50k",
    "Plynna_B_100k",
    "Plynna_C_250k",
    "Flaga_Okno_TTM",
    "Flaga_Jakosci",
    "Flaga_Reczna",
    "Flaga_EMIS_dev5",
    "Flaga_akcje_wlasne_mozliwe",
    "Zrodlo_akcji",
]
targ = [
    "Zwrot_Spolki_63",
    "Zwrot_Spolki_63_w",
    "Zwrot_WIG_63",
    "Y_excess",
    "Y_excess_w",
    "Y_rank",
    "Y_demean",
    "Y_bin",
    "Y_klasa",
]
model = pd.concat([c[ident + flags + targ + foldcols], M], axis=1).rename(
    columns={"Data": "Koniec_kwartalu"}
)
rawcols = (
    rynek
    + makro
    + [
        "Close_t",
        "Akcje",
        "Cap_t_PLN_mln",
        "FX_USDPLN_QE",
        "EV_PLN_mln",
        "Dev_vs_EMIS_cap",
        "Mediana_wartosci_obrotu_63sesje_PLN",
    ]
    + winz
    + rank
    + ["Altman_Z_EM", "F_Score_8"]
)
raw = pd.concat([c[ident], c[rawcols + bin_]], axis=1).rename(columns={"Data": "Koniec_kwartalu"})
raw = raw.loc[:, ~raw.columns.duplicated()]
kt = (
    c.groupby("Ticker")
    .agg(
        Zrodlo=("Zrodlo_akcji", "first"),
        Akcje_pierwsza=("Akcje", "first"),
        Akcje_ostatnia=("Akcje", "last"),
        Zmian_liczby_akcji=("Akcje", lambda s: int((s.diff().fillna(0) != 0).sum())),
        EMIS_dev_mediana=("Dev_vs_EMIS_cap", "median"),
        EMIS_dev_max_abs=("Dev_vs_EMIS_cap", lambda s: s.abs().max()),
        Kwartaly_dev_ponad5=("Flaga_EMIS_dev5", "sum"),
        Flaga_akcje_wlasne=("Flaga_akcje_wlasne_mozliwe", "max"),
        W_Probie_v3=("W_Probie_v3", "max"),
    )
    .reset_index()
)
kt["Yahoo_obecna"] = kt.Ticker.map(yh)
fxt = (
    c.groupby("Koniec_kwartalu" if "Koniec_kwartalu" in c else "Data")
    .FX_USDPLN_QE.first()
    .reset_index()
)
nan = model.isna().mean().rename("Udzial_NaN").reset_index().rename(columns={"index": "Kolumna"})
dq = (
    c.groupby("Data")
    .agg(
        Data_t=("Data_t", "first"),
        N=("Ticker", "size"),
        Znany_cel=("Znany_cel", "sum"),
        Y_mean=("Y_excess", "mean"),
        Y_median=("Y_excess", "median"),
        WIG_63=("Zwrot_WIG_63", "first"),
    )
    .reset_index()
)
opis = [
    (
        "Makro_A / Makro_B",
        "cecha makro",
        "Makro_A: srednia z 252-sesyjnych z-score (VIX, EPOL odwrocony, USDPLN, WIBOR 3M, rentownosc 5-letnich obligacji PL); wyzsza wartosc = wiekszy stres makro. Makro_B: wagi z kroczacej regresji grzbietowej (okno 500 sesji konczace sie 64 sesje przed t, wiec bez wycieku) zastosowane do biezacych z-score, standaryzowane wstecznie. Wartosc na dzien t, wspolna dla wszystkich spolek",
    ),
    (
        "Makro_A_x_Beta / Makro_B_x_Beta",
        "cecha makro",
        "Makro_A (B) razy Beta_WIG_252d (po winsoryzacji): pozwala modelowi uzaleznic wplyw makro od wrazliwosci spolki na rynek",
    ),
    (
        "Rynek_Mom_21d / 63d / 126d / 252d",
        "cecha rynku",
        "zwrot WIG z ostatnich 21/63/126/252 sesji do dnia decyzji t (wspólny dla wszystkich spółek w kwartale, bez normalizacji)",
    ),
    (
        "Rynek_Vol_63d",
        "cecha rynku",
        "zmienność WIG z 63 sesji (annualizowane odchylenie log-zwrotów) do dnia t",
    ),
    ("Rynek_SMA200_dist", "cecha rynku", "odległość WIG od średniej z 200 sesji do dnia t"),
    (
        "Zwrot_Spolki_63_w",
        "cel",
        "Zwrot_Spolki_63 (surowy zwrot spółki z 63 sesji) winsoryzowany 1/99 per kwartał w grupie [Spolka_finansowa, Filar2_Pass]; wariant celu bez odejmowania WIG",
    ),
    (
        "Fold_Rekomendacja",
        "walidacja",
        "podział do rekomendacji dla ostatniego kwartału ze znanym celem (2026 Q1): train = obserwacje, których okno celu zakończyło się przed wejściem w ten kwartał; purge = obserwacje z oknem nakładającym się; rekomendacja = 54 spółki z 2026 Q1 (zbiór do przypisania do grup i porównania z faktycznym zwrotem)",
    ),
    (
        "Y_klasa",
        "cel",
        "kategoria rekomendacji z tercyli Y_rank w kwartale (w grupie [Spolka_finansowa, Filar2_Pass]): 0 = niezalecana (dolna 1/3), 1 = neutralna, 2 = atrakcyjna (górna 1/3); wersja alternatywna celu do klasyfikacji",
    ),
    (
        "Filar2_Pass",
        "flaga",
        "1 = spółka przechodzi statyczny filtr płynności (Filar II): brak roku kalendarzowego z >5% sesji z zerowym wolumenem ORAZ średni dzienny obrót z 21 sesji (Close×Wolumen) poniżej 50 tys. PLN na koniec nie więcej niż 10% kwartałów. Filtr liczony na całej historii 2017–2026 (uwaga: wykorzystuje także okres testowy)",
    ),
    (
        "Do_modelu_bez_F2",
        "flaga",
        "jak Do_modelu, ale bez filtra Filar II (80 spółek niefinansowych); Do_modelu = Do_modelu_bez_F2 × Filar2_Pass",
    ),
    (
        "Filar2_Lata_zero_ponad5 / Filar2_Max_rok_zero / Filar2_Udzial_kw_pod50k",
        "kontrola",
        "liczba lat z >5% sesji bez wolumenu; największy roczny udział takich sesji; udział kwartałów ze średnim 21-sesyjnym obrotem poniżej 50 tys. PLN",
    ),
    (
        "Normalizacja po Filarze II",
        "uwaga",
        "winsoryzacja, rangi i Y_rank liczone per kwartał w obrębie grupy [Spolka_finansowa, Filar2_Pass], czyli spółki z filtrem są porównywane tylko między sobą",
    ),
    (
        "Do_modelu",
        "flaga",
        "1 = wiersz do głównego modelu: spółka niefinansowa, od 2018, ze znanym celem (finansowe zostają w pliku, ale poza treningiem)",
    ),
    (
        "Fold_01..Fold_NN",
        "walidacja",
        "dla każdego foldu: train / purge (usunąć z uczenia – okno zwrotu zachodzi na okres walidacji) / val; foldy kwartalne od 2021 Q1 do 2023 Q3, uczenie rozszerzające od 2018 Q1; tylko wiersze Podzial=train",
    ),
    ("Y_demean", "cel", "Y_excess minus średnia z kwartału (w grupie niefinansowych)"),
    ("Koniec_kwartalu", "id", "koniec kwartału, którego dotyczą dane fundamentalne"),
    (
        "Data_t",
        "id",
        "data decyzji t = pierwsza sesja po terminie publikacji (GPW max, przesunięty na dzień roboczy); cechy tylko z danych do t",
    ),
    ("Data_wejscia", "id", "zamknięcie pierwszej sesji po t (wariant B)"),
    ("Data_wyjscia", "id", "63 sesje po wejściu"),
    (
        "Podzial",
        "flaga",
        "burn_in_2017 / train (Koniec_kwartalu ≤ 2023-09-30) / test (≥ 2023-12-31) / bez_celu; sprawdzone, że wyjście ostatniego train < wejście pierwszego test",
    ),
    ("Znany_cel", "flaga", "1 jeśli okno 63 sesji mieści się w danych"),
    (
        "W_Probie_v3",
        "flaga",
        "0 = spółki usunięte w v3 (BMC, COG, DAT, MDG, PKN, SGN, SVE) – przywrócone do pliku, ale oflagowane",
    ),
    (
        "Plynna_A_50k / B_100k / C_250k",
        "flaga",
        "mediana 63-sesyjnego obrotu (Close×Wolumen) na t powyżej progu",
    ),
    ("Spolka_finansowa", "flaga", "GPW, XTB, MCI, QRS, VOT: Altman i Piotroski ustawione na NaN"),
    (
        "Flaga_EMIS_dev5",
        "kontrola",
        "kapitalizacja z moją liczbą akcji różni się >5% od kapitalizacji EMIS (po przeliczeniu kursem) – możliwy błąd liczby akcji/akcje własne",
    ),
    (
        "Flaga_akcje_wlasne_mozliwe",
        "kontrola",
        "Yahoo (liczba netto) różni się >1% od liczby wyemitowanych – możliwe akcje własne",
    ),
    ("Zwrot_Spolki_63", "cel", "Adj Close(wyjście)/Adj Close(wejście)−1"),
    ("Zwrot_WIG_63", "cel", "to samo dla WIG"),
    ("Y_excess", "cel", "zwrot spółki − zwrot WIG (główna zmienna celu)"),
    ("Y_excess_w", "cel", "Y_excess winsoryzowany 1/99 per kwartał"),
    (
        "Y_rank",
        "cel",
        "ranga Y_excess per kwartał w [−1,1] (w grupie niefinansowych; główny cel modelu)",
    ),
    ("Y_bin", "cel", "1 gdy Y_excess>0"),
    (
        "Mom_*",
        "cecha",
        "momentum z Adj Close (21/63/126/252 sesji; Mom_12_1 pomija ostatni miesiąc)",
    ),
    ("RSI_14", "cecha", "RSI Wildera 14 sesji"),
    ("SMA50_dist/SMA200_dist", "cecha", "Adj/średnia−1"),
    ("Vol_63d/Vol_252d", "cecha", "zannualizowane odchylenie log-zwrotów"),
    ("Beta_WIG_252d", "cecha", "beta wobec WIG, 252 sesje (min 126)"),
    ("GARCH_Vol", "cecha", "prognoza zannualizowanej zmienności z GARCH(1,1), okno 500 sesji"),
    ("Dist_52W_High", "cecha", "Adj/maks 252 sesji −1"),
    ("Amihud_63d", "cecha", "średnia |zwrot|/obrót PLN × 1e6"),
    (
        "*_rank",
        "cecha",
        "ranga per kwartał w [−1,1] (cechy wrażliwe na poziom rynku: kapitalizacja, obroty, wyceny)",
    ),
    ("Log_Cap", "cecha", "ln(Close_t × liczba akcji)"),
    (
        "EP,BP,SP,OCFP,FCFP,TangBP",
        "cecha",
        "odwrotności wskaźników wyceny (zysk, kapitał dominujących, przychody, OCF, FCF, WK tang. / kapitalizacja); fundamenty USD×kurs implikowany z kapitalizacji EMIS",
    ),
    (
        "EV_EBITDA,EV_Sales",
        "cecha",
        "EV = kapitalizacja + dług netto; EV_EBITDA NaN gdy EBITDA_TTM ≤ 0",
    ),
    (
        "F1..F9, F_Score_9",
        "cecha",
        "Piotroski; F7 = liczba akcji (koniec kwartału) nie wzrosła względem 4 kwartałów wcześniej; NaN gdy któryś składnik brak; F3/F8/F9 gdy brak TTM (np. 2018 Q1–Q3) liczone z kwartału r/r; F_Score_dost = średnia dostępnych składników × 9 przy ≥7 dostępnych (F_n = liczba dostępnych składników)",
    ),
    (
        "pozostałe (ROE_TTM, marże, zadłużenie, growth, delta)",
        "cecha",
        "winsoryzowane 1/99 per kwartał, bez rang",
    ),
    (
        "Kurs FX",
        "uwaga",
        "FX_USDPLN_QE = mediana po spółkach z (kapitalizacja PLN / kapitalizacja EMIS USD) na koniec kwartału; po stronie przekroju rangi eliminują wpływ kursu",
    ),
    ("Liczba akcji", "uwaga", "stan na koniec kwartału (nie na t); akcje własne nieuwzględnione"),
]
with pd.ExcelWriter(
    WYNIKI / "panel_finalny_v1.xlsx", engine="openpyxl", datetime_format="YYYY-MM-DD"
) as wr_:
    model.to_excel(wr_, sheet_name="Model", index=False)
    raw.to_excel(wr_, sheet_name="Surowe", index=False)
    pd.DataFrame(opis, columns=["Kolumna", "Typ", "Opis"]).to_excel(
        wr_, sheet_name="Opis", index=False
    )
    kt.to_excel(wr_, sheet_name="Kontrola_akcji", index=False)
    finfo.to_excel(wr_, sheet_name="Foldy", index=False)
    fxt.to_excel(wr_, sheet_name="FX_kwartalny", index=False)
    dq.to_excel(wr_, sheet_name="Kwartaly", index=False)
    nan.to_excel(wr_, sheet_name="NaN_kolumny", index=False)
model.to_csv(WYNIKI / "panel_finalny_v1_model.csv", index=False)
