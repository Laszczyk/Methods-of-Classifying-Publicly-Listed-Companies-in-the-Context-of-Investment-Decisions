"""Etap 9. Kontrola spójności panelu: klucze, daty, wartości skrajne, braki, skoki cen, jakość celu
i korelacje cech z celem.
"""

import pickle

import numpy as np
import pandas as pd

from konfiguracja import DANE, POSREDNIE, WYNIKI

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
m = pd.read_excel(WYNIKI / "panel_finalny_v1.xlsx", "Model")
r = pd.read_excel(WYNIKI / "panel_finalny_v1.xlsx", "Surowe")
close, adj, vol, div, c = pickle.load(open(POSREDNIE / "prep.pkl", "rb"))
c1 = pickle.load(open(POSREDNIE / "stage1.pkl", "rb"))
print("== klucze")
print(
    m.duplicated(["Ticker", "Koniec_kwartalu"]).sum(),
    m.groupby("Ticker").size().value_counts().to_dict(),
    m.Ticker.nunique(),
)
print("== t > koniec kwartalu, wejscie>t, wyjscie>wejscie")
print(
    (m.Data_t <= m.Koniec_kwartalu).sum(),
    (m.Data_wejscia <= m.Data_t).sum(),
    (m.Data_wyjscia <= m.Data_wejscia).sum(),
)
print("== inf / skrajne w Surowe")
num = r.select_dtypes("number")
print(np.isinf(num).sum().loc[lambda s: s > 0].to_dict())
q = num.quantile([0, 0.01, 0.5, 0.99, 1]).T
q["ratio"] = q[1.0].abs() / (q[0.99].abs() + 1e-9)
print(q[(q.ratio > 5)].round(3).head(25))
print("== skoki cen niewyjasnione splitem (sample)")
ret = close.pct_change(fill_method=None)
sp = pd.read_excel(DANE / "yahoo_wszystkie_ceny.xlsx", "Splity_lista")
sp["Ticker"] = sp.Ticker.str.replace(".WA", "")
sample = set(m.Ticker)
rows = []
for t in sample:
    s = ret[t].dropna()
    b = s[s.abs() > 0.35]
    for d, v in b.items():
        near = (
            (sp.Ticker == t)
            & (pd.to_datetime(sp.Data).between(d - pd.Timedelta(days=5), d + pd.Timedelta(days=5)))
        ).any()
        rows.append((t, d.date(), round(v, 2), close[t].shift(1)[d], close[t][d], near))
j = pd.DataFrame(rows, columns=["T", "Data", "Zwrot", "Przed", "Po", "split_obok"])
print(len(j), (~j.split_obok).sum())
print(j[~j.split_obok].sort_values("Data").to_string(index=False))
print("== Adj/Close na t (ostatnia vs pierwsza)")
ratio = adj / close
print(ratio.iloc[0].loc[list(sample)].nsmallest(5).round(3).to_dict())
print("== braki cech wg kwartalu (Model, % NaN, kolumny >20%)")
feats = [
    x
    for x in m.columns
    if x
    not in [
        "Ticker",
        "Firma",
        "Koniec_kwartalu",
        "Rok",
        "Okres",
        "Data_t",
        "Data_wejscia",
        "Data_wyjscia",
        "Podzial",
        "Zrodlo_akcji",
    ]
]
nanq = m.groupby("Koniec_kwartalu")[feats].apply(lambda g: g.isna().mean())
print(nanq.mean().sort_values(ascending=False).head(12).round(2).to_dict())
tr = m[m.Podzial.isin(["train", "test"])]
print(
    "NaN w train/test (>15%):", tr[feats].isna().mean().loc[lambda s: s > 0.15].round(2).to_dict()
)
print("== ticker bez F_Score_9 w calym train/test")
print(
    tr.groupby("Ticker")
    .F_Score_9.apply(lambda s: s.notna().mean())
    .loc[lambda s: s < 0.5]
    .round(2)
    .to_dict()
)
print("== flagi jakosci w train/test")
print(
    tr[
        [
            "Flaga_Jakosci",
            "Flaga_Reczna",
            "Flaga_Okno_TTM",
            "Flaga_EMIS_dev5",
            "Flaga_akcje_wlasne_mozliwe",
            "W_Probie_v3",
        ]
    ]
    .mean()
    .round(3)
    .to_dict()
)
print("== plynnosc: udzial B w train/test per kwartal (min,med,max)")
g = tr.groupby("Koniec_kwartalu").Plynna_B_100k.sum()
print(g.min(), g.median(), g.max(), "wierszy B:", int(tr.Plynna_B_100k.sum()), "/", len(tr))
print("== cel: skrajne")
print(tr.Y_excess.describe().round(3).to_dict())
print(
    tr.nlargest(5, "Y_excess")[
        ["Ticker", "Koniec_kwartalu", "Y_excess", "Plynna_B_100k"]
    ].to_string(index=False)
)
print("Y_excess>2 w plynnych B:", ((tr.Y_excess > 2) & (tr.Plynna_B_100k == 1)).sum())
print("== cel stale price: wejscie/wyjscie bez handlu tego dnia (wolumen=0)")
st = []
for _, x in c1.iterrows():
    t = x.Ticker
    try:
        ve = vol[t].loc[x.Data_wejscia]
        vx = vol[t].loc[x.Data_wyjscia]
    except Exception:
        ve = vx = np.nan
    st.append((ve == 0 or pd.isna(ve), vx == 0 or pd.isna(vx)))
st = np.array(st)
print("wejscie bez wolumenu:", st[:, 0].mean().round(3), "wyjscie:", st[:, 1].mean().round(3))
bb = c1.assign(we=st[:, 0], wy=st[:, 1]).groupby("Plynna_B_100k")[["we", "wy"]].mean().round(3)
print(bb)
print("== korelacja przekrojowa cech z celem (rank IC, train, top 10 abs)")
fe = [
    x
    for x in m.columns
    if x.endswith("_rank")
    or x
    in [
        "Mom_21d",
        "Mom_63d",
        "Mom_126d",
        "Mom_252d",
        "Mom_12_1",
        "Vol_63d",
        "Beta_WIG_252d",
        "ROE_TTM",
        "F_Score_9",
        "GARCH_Vol",
        "RSI_14",
    ]
]
ic = tr.groupby("Koniec_kwartalu").apply(lambda g: g[fe].corrwith(g.Y_rank, method="spearman"))
print(
    ic.mean().sort_values().round(3).head(5).to_dict(),
    ic.mean().sort_values().round(3).tail(5).to_dict(),
)
