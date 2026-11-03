"""XGBoost: strojenie hiperparametrów, walidacja i ocena na zbiorze testowym.

Dane:    zbior_modelowy_v1.csv (wynik etapu 8 w 1_zestaw_danych)
Cel:     Zwrot_Spolki_63_w (zwrot spółki z 63 sesji, winsoryzowany 1/99 per kwartał)
Zestawy: definicje w słowniku SETS; w pracy zestaw DS (główny dla XGBoost i Random Forest).
         Wybór zestawów: --features, np. --features DS

Przebieg dla każdego zestawu cech:
  0. Kontrole spójności: brak nakładania okien celu między zbiorem uczącym a walidacją i testem,
     brak zmiennych celu wśród cech.
  1. Strojenie hiperparametrów (Optuna, TPE) na 11 kwartalnych foldach walidacyjnych z oczyszczeniem
     okna zwrotów. Kryterium: średni rank IC (korelacja Spearmana prognozy z realizowanym zwrotem
     w obrębie kwartału).
  2. Ponowna ocena najlepszych kandydatów z kilkoma ziarnami losowości.
  3. Model końcowy: średnia z kilku ziaren, uczona na całym zbiorze uczącym.
  4. Jednorazowa ocena na zbiorze testowym (2023 Q4 – 2025 Q4) i porównanie z punktami odniesienia:
     stała (średnia z uczenia), momentum Mom_12_1, regresja Ridge.
  5. Testy kontrolne: test placebo (cel przetasowany w obrębie kwartału) oraz ważność cech
     (permutacyjna na foldach walidacyjnych i na teście, SHAP, gain).
  6. Prognoza dla 2026 Q1 (zbiór wstrzymany) wyłącznie z flagą --final; bez niej ten kwartał nie jest
     używany.
  7. Porównanie kolejnych zestawów z pierwszym podanym (parami po kwartalach).

Opcja --base-momentum: model startuje od prognozy opartej na randze Mom_12_1 w obrębie kwartału
i uczy się tylko reszty.

Przykład uruchomienia:
    python model_xgboost.py --features DS --trials 100 --jobs 4

Zbiór testowy jest używany tylko raz, po zakończeniu strojenia.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import rankdata

TARGET = "Zwrot_Spolki_63_w"
EVAL_COL = "Zwrot_Spolki_63_w"
FOLD_COLS = [f"Fold_{i:02d}" for i in range(1, 12)]
REC_COL = "Fold_Rekomendacja"

FEATURES_A = [
    "EP_rank",
    "BP_rank",
    "EV_EBITDA_rank",
    "ROE_TTM",
    "Marza_OCF_TTM",
    "F_Score_dost",
    "Altman_Z_EM",
    "Growth_Przychody_YoY",
    "Delta_ZyskNetto_Aktywa",
    "Mom_63d",
    "Mom_12_1",
    "RSI_14",
    "SMA200_dist",
    "GARCH_Vol",
    "Beta_WIG_252d",
    "Log_Cap_rank",
    "Dlug_Netto_Aktywa",
    "Div_Yield_12M_rank",
    "Amihud_63d_rank",
    "Growth_Aktywa_YoY",
]
FEATURES_MAKRO = ["Rynek_Vol_63d", "Makro_A", "Makro_B", "Makro_A_x_Beta", "Makro_B_x_Beta"]

GROUPS = {
    "wycena": ["EP_rank", "BP_rank", "EV_EBITDA_rank", "Div_Yield_12M_rank"],
    "jakosc": ["ROE_TTM", "Marza_OCF_TTM", "F_Score_dost", "Altman_Z_EM"],
    "wzrost": ["Growth_Przychody_YoY", "Delta_ZyskNetto_Aktywa", "Growth_Aktywa_YoY"],
    "momentum_trend": ["Mom_63d", "Mom_12_1", "RSI_14", "SMA200_dist"],
    "ryzyko": ["GARCH_Vol", "Beta_WIG_252d", "Dlug_Netto_Aktywa"],
    "wielkosc_plynnosc": ["Log_Cap_rank", "Amihud_63d_rank"],
    "makro": FEATURES_MAKRO,
}

SKLAD_C = [f for f in FEATURES_A if f not in GROUPS["wielkosc_plynnosc"]]
MARTWE_CV = [
    "Div_Yield_12M_rank",
    "Delta_ZyskNetto_Aktywa",
    "Growth_Przychody_YoY",
    "F_Score_dost",
    "RSI_14",
    "Growth_Aktywa_YoY",
]
SKLAD_D = [f for f in SKLAD_C if f not in MARTWE_CV]
SKLAD_E = [
    "EP_rank",
    "BP_rank",
    "EV_EBITDA_rank",
    "ROE_TTM",
    "Marza_OCF_TTM",
    "F_Score_dost",
    "Altman_Z_EM",
    "Growth_Przychody_YoY",
    "Delta_ZyskNetto_Aktywa",
    "Dlug_Netto_Aktywa",
    "Div_Yield_12M_rank",
    "Growth_Aktywa_YoY",
]
SKLAD_F = [
    "Mom_63d",
    "Mom_12_1",
    "RSI_14",
    "SMA200_dist",
    "GARCH_Vol",
    "Beta_WIG_252d",
    "Log_Cap_rank",
    "Amihud_63d_rank",
]
SKLAD_G = [f for f in SKLAD_D if f != "Mom_12_1"]
SKLAD_H = [f for f in SKLAD_D if f not in ("Mom_12_1", "Mom_63d", "SMA200_dist")]
SKLAD_DM = SKLAD_D + FEATURES_MAKRO
SKLAD_D2 = [f for f in SKLAD_D if f not in ("GARCH_Vol", "EP_rank")]
SKLAD_D3 = [
    f for f in SKLAD_D if f not in ("GARCH_Vol", "EP_rank", "SMA200_dist", "Dlug_Netto_Aktywa")
]
SKLAD_DS = SKLAD_D + ["Log_Cap_rank"]
SETS = {
    "A": FEATURES_A,
    "B": FEATURES_A + FEATURES_MAKRO,
    "C": SKLAD_C,
    "D": SKLAD_D,
    "E": SKLAD_E,
    "F": SKLAD_F,
    "G": SKLAD_G,
    "H": SKLAD_H,
    "DM": SKLAD_DM,
    "D2": SKLAD_D2,
    "D3": SKLAD_D3,
    "DS": SKLAD_DS,
}

BASE_MOM = False
BM_COL = "_bm"
MAX_ROUNDS = 1000
CHECK_STEP = 20
SMOOTH = 3

log = logging.getLogger("xgb")
ZBIOR_DOMYSLNY = (
    Path(__file__).resolve().parent.parent / "1_zestaw_danych" / "wyniki" / "zbior_modelowy_v1.csv"
)


def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, parse_dates=["Koniec_kwartalu", "Data_t", "Data_wejscia", "Data_wyjscia"]
    )
    df = df[df["Do_modelu"] == 1].sort_values(["Koniec_kwartalu", "Ticker"]).reset_index(drop=True)
    for c in (target, EVAL_COL):
        if df[c].isna().any():
            raise ValueError(f"Kolumna {c} ma braki w próbie Do_modelu")
    log.info(
        "Wczytano %d wierszy, %d spółek, %d kwartałów",
        len(df),
        df.Ticker.nunique(),
        df.Koniec_kwartalu.nunique(),
    )
    return df


def get_features(df: pd.DataFrame, name: str) -> list[str]:
    feats = list(SETS[name])
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w danych: {missing}")
    bad = [
        f for f in feats if f.startswith(("Y_", "Zwrot_", "Fold_")) or f in ("Podzial", "Znany_cel")
    ]
    if bad:
        raise ValueError(f"Cecha wygląda na zmienną celu lub identyfikator: {bad}")
    return feats


def check_splits(df: pd.DataFrame) -> None:
    """Sprawdza, czy okna celu uczenia nie zachodzą na okres walidacji / testu."""
    tr = df[df["Podzial"] == "train"]
    te = df[(df["Podzial"] == "test") & (df[REC_COL] != "rekomendacja")]
    assert len(te) > 0 and len(tr) > 0
    assert (
        df.loc[df[REC_COL] == "rekomendacja", "Koniec_kwartalu"].min() > te["Koniec_kwartalu"].max()
    )
    assert tr["Data_wyjscia"].max() < te["Data_wejscia"].min(), "Okna celu train zachodzą na test"
    assert tr["Koniec_kwartalu"].max() < te["Koniec_kwartalu"].min()
    for fc in FOLD_COLS:
        a, v = df[df[fc] == "train"], df[df[fc] == "val"]
        assert len(a) and len(v), fc
        assert set(a.index).isdisjoint(v.index), fc
        assert a["Podzial"].eq("train").all() and v["Podzial"].eq("train").all(), fc
        assert (
            a["Data_wyjscia"].max() < v["Data_wejscia"].min()
        ), f"{fc}: okno celu train zachodzi na val"
        assert a["Koniec_kwartalu"].max() < v["Koniec_kwartalu"].min(), fc
    r = df[df[REC_COL] == "train"]
    rr = df[df[REC_COL] == "rekomendacja"]
    assert r["Data_wyjscia"].max() < rr["Data_wejscia"].min(), "Fold_Rekomendacja: nakładanie okien"
    log.info(
        "Kontrole podziałów przeszły: %d foldów CV, test od %s, rekomendacja %s",
        len(FOLD_COLS),
        te["Koniec_kwartalu"].min().date(),
        rr["Koniec_kwartalu"].iloc[0].date(),
    )


def momentum_score(df: pd.DataFrame) -> np.ndarray:
    """Ranga Mom_12_1 w obrębie kwartału, wyśrodkowana do [-0,5; 0,5]; brak danych = 0 (neutralnie)."""
    r = df.groupby("Koniec_kwartalu")["Mom_12_1"].rank(method="average")
    n = df.groupby("Koniec_kwartalu")["Mom_12_1"].transform("count")
    s = ((r - 1) / (n - 1).clip(lower=1) - 0.5).fillna(0.0)
    return s.to_numpy(dtype=float)


def fold_indices(df: pd.DataFrame):
    return [
        (np.flatnonzero(df[fc] == "train"), np.flatnonzero(df[fc] == "val")) for fc in FOLD_COLS
    ]


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.nanstd(p) == 0 or np.nanstd(y) == 0:
        return np.nan
    ry, rp = rankdata(y), rankdata(p)
    ry, rp = ry - ry.mean(), rp - rp.mean()
    den = np.sqrt((ry**2).sum() * (rp**2).sum())
    return float(ry @ rp / den) if den > 0 else np.nan


def ic_curve(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Rank IC dla każdej kolumny macierzy predykcji P (n x K) względem y."""
    ry = rankdata(y)
    ry = ry - ry.mean()
    RP = rankdata(P, axis=0)
    RP = RP - RP.mean(axis=0)
    den = np.sqrt((ry**2).sum() * (RP**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ry @ RP) / den
    out[den == 0] = np.nan
    return out


def nw_tstat(x, lags: int = 2) -> float:
    """Statystyka t dla średniej szeregu z odpornym błędem Newey-West (nakładające się okna celu)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return np.nan
    d = x - x.mean()
    s = (d @ d) / n
    for l in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - l / (lags + 1)) * (d[l:] @ d[:-l]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def block_bootstrap_ci(x, block: int = 2, B: int = 5000, seed: int = 0):
    """95% przedział dla średniej szeregu (kołowy bootstrap blokowy po kwartałach)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = x[idx.reshape(B, -1)[:, :n]].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def quarter_table(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Metryki przekrojowe w każdym kwartale: IC, rozpiętość tercyli, trafność górnego tercyla."""
    rows = []
    d = pd.DataFrame(
        {"q": df["Koniec_kwartalu"].to_numpy(), "y": df[EVAL_COL].to_numpy(), "p": pred}
    )
    for q, g in d.groupby("q", sort=True):
        y, p, n = g["y"].to_numpy(), g["p"].to_numpy(), len(g)
        k = max(1, int(round(n / 3)))
        order = np.argsort(-p, kind="mergesort")
        top, bot = y[order[:k]], y[order[-k:]]
        real_top = set(np.argsort(-y, kind="mergesort")[:k])
        rows.append(
            {
                "Koniec_kwartalu": q,
                "n": n,
                "ic": spearman(y, p),
                "top": top.mean(),
                "bottom": bot.mean(),
                "spread": top.mean() - bot.mean(),
                "precision_top": len(real_top & set(order[:k])) / k,
                "rynek": y.mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    df: pd.DataFrame,
    pred: np.ndarray,
    y_train_mean: float,
    target: str | None = None,
    is_score: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Podsumowanie predykcji. is_score=True dla wskaźników bez skali zwrotu (np. momentum):
    wtedy RMSE, MAE i R2 nie mają sensu i są puste."""
    target = target or TARGET
    qt = quarter_table(df, pred)
    ic = qt["ic"].to_numpy()
    y = df[target].to_numpy()
    lo, hi = block_bootstrap_ci(ic)
    ok = ~np.isnan(ic)
    sd = float(np.std(ic[ok], ddof=1)) if ok.sum() > 1 else np.nan
    s = {
        "n_kwartalow": int(len(qt)),
        "ic_srednia": float(np.mean(ic[ok])) if ok.any() else np.nan,
        "ic_odch": sd,
        "icir": float(np.mean(ic[ok]) / sd) if ok.sum() > 1 and sd > 0 else np.nan,
        "ic_udzial_dodatnich": float(np.mean(ic[ok] > 0)) if ok.any() else np.nan,
        "ic_nw_t": nw_tstat(ic),
        "ic_ci95_lo": lo,
        "ic_ci95_hi": hi,
        "spread_gora_dol": float(qt["spread"].mean()),
        "spread_udzial_dodatnich": float((qt["spread"] > 0).mean()),
        "precision_gora_tercyl": float(qt["precision_top"].mean()),
        "rmse": np.nan,
        "mae": np.nan,
        "r2_oos_vs_srednia_train": np.nan,
    }
    if not is_score:
        sse, sse0 = ((y - pred) ** 2).sum(), ((y - y_train_mean) ** 2).sum()
        s.update(
            rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
            mae=float(np.mean(np.abs(y - pred))),
            r2_oos_vs_srednia_train=float(1 - sse / sse0),
        )
    return s, qt


def _xgb():
    import xgboost as xgb

    return xgb


class MomBase:
    """Opakowanie XGBRegressor: margines startowy = c + b*s, gdzie s = kolumna BM_COL w X (wynik z momentum),
    b z regresji y na s na danych uczących, c = średnia (lub mediana) y. Kolumna BM_COL nie trafia do drzew.
    """

    def __init__(self, inner, center_fn):
        self.inner, self.center_fn, self.b, self.c = inner, center_fn, 0.0, 0.0

    def _split(self, X):
        return X.drop(columns=[BM_COL]), X[BM_COL].to_numpy(dtype=float)

    def fit(self, X, y):
        Xf, s = self._split(X)
        self.c = float(self.center_fn(y))
        sc = s - s.mean()
        self.b = float(np.sum(sc * (y - y.mean())) / np.sum(sc**2)) if np.sum(sc**2) > 0 else 0.0
        self.inner.fit(Xf, y, base_margin=self.c + self.b * s)
        return self

    def predict(self, X, iteration_range=None):
        Xf, s = self._split(X)
        kw = {} if iteration_range is None else {"iteration_range": iteration_range}
        return self.inner.predict(Xf, base_margin=self.c + self.b * s, **kw)

    def get_booster(self):
        return self.inner.get_booster()


def build_model(params: dict, seed: int, n_jobs: int, y_train: np.ndarray, n_estimators: int):
    xgb = _xgb()
    obj = params["objective"]
    kw = dict(
        n_estimators=int(n_estimators),
        learning_rate=params["learning_rate"],
        max_depth=int(params["max_depth"]),
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        colsample_bynode=params["colsample_bynode"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        max_bin=int(params["max_bin"]),
        tree_method="hist",
        objective=obj,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
        base_score=float(np.mean(y_train) if obj == "reg:squarederror" else np.median(y_train)),
    )
    if obj == "reg:pseudohubererror":
        kw["huber_slope"] = params["huber_slope"]
    model = xgb.XGBRegressor(**kw)
    if BASE_MOM:
        return MomBase(model, np.mean if obj == "reg:squarederror" else np.median)
    return model


SPACE_VERSION = "v3"


def finalize_params(raw: dict, ysd: float) -> dict:
    """Zamienia parametry względne na bezwzględne. gamma, reg_alpha i huber_slope zależą od skali
    zmiennej celu (zysk z podziału jest proporcjonalny do jej wariancji), więc strojone są w jednostkach
    względnych. Bez tego stała górna granica gamma (np. 2) blokowałaby wszystkie podziały przy zwrotach
    o odchyleniu 0,25 i model byłby prawie stały."""
    p = dict(raw)
    if "gamma_rel" in p:
        p["gamma"] = p.pop("gamma_rel") * 6.0 * ysd**2
    if "reg_alpha_rel" in p:
        p["reg_alpha"] = p.pop("reg_alpha_rel") * ysd
    if "huber_slope_rel" in p:
        p["huber_slope"] = p.pop("huber_slope_rel") * ysd
    return p


def suggest(trial, fixed_objective: str | None, ysd: float) -> dict:
    raw = {
        "objective": fixed_objective
        or trial.suggest_categorical("objective", ["reg:squarederror", "reg:pseudohubererror"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 150.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 500.0, log=True),
        "reg_alpha_rel": trial.suggest_float("reg_alpha_rel", 1e-3, 2.0, log=True),
        "gamma_rel": trial.suggest_float("gamma_rel", 0.0, 3.0),
        "max_bin": trial.suggest_categorical("max_bin", [32, 64, 128, 256]),
    }
    if raw["objective"] == "reg:pseudohubererror":
        raw["huber_slope_rel"] = trial.suggest_float("huber_slope_rel", 0.2, 3.0, log=True)
    return finalize_params(raw, ysd)


def train_sd(y: np.ndarray, folds) -> float:
    """Odchylenie standardowe celu w okresie uczącym (wiersze użyte w foldach CV), bez testu."""
    idx = np.unique(np.concatenate([np.concatenate(f) for f in folds]))
    return float(np.std(y[idx]))


def checkpoints() -> np.ndarray:
    return np.arange(CHECK_STEP, MAX_ROUNDS + 1, CHECK_STEP)


def fold_ic_curves(params, X, y, ysel, folds, seed, n_jobs, cps, on_fold=None) -> np.ndarray:
    """IC na foldach walidacyjnych dla każdej liczby drzew z cps. Zwraca macierz (foldy x punkty)."""
    out = []
    for i, (tr, va) in enumerate(folds):
        m = build_model(params, seed, n_jobs, y[tr], cps[-1])
        m.fit(X.iloc[tr], y[tr])
        P = np.column_stack([m.predict(X.iloc[va], iteration_range=(0, int(k))) for k in cps])
        out.append(ic_curve(ysel[va], P))
        if on_fold is not None:
            on_fold(i, np.vstack(out))
    return np.vstack(out)


def best_from_curves(curves: np.ndarray):
    """Najlepszy punkt krzywej IC. IC = NaN (stała predykcja, brak informacji) liczymy jako 0,
    a nie pomijamy: inaczej model, który w większości foldów nic nie przewiduje, dostawałby
    zawyżoną średnią z kilku foldów."""
    mean_curve = np.nan_to_num(curves, nan=0.0).mean(axis=0)
    sm = uniform_filter1d(mean_curve, size=SMOOTH, mode="nearest")
    j = int(np.argmax(sm))
    return j, float(sm[j])


def run_study(X, y, ysel, folds, args, name: str, out: Path):
    import optuna

    for cat in (getattr(optuna.exceptions, "ExperimentalWarning", Warning),):
        warnings.filterwarnings("ignore", category=cat)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cps = checkpoints()
    ysd = train_sd(y, folds)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed, multivariate=True, n_startup_trials=min(30, args.trials)
    )
    pruner = (
        optuna.pruners.NopPruner()
        if args.no_prune
        else optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3)
    )
    key = hashlib.md5(
        f"{Path(args.data).stat().st_size}{args.target}{name}{MAX_ROUNDS}{CHECK_STEP}{SPACE_VERSION}{BASE_MOM}".encode()
    ).hexdigest()[:8]
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"xgb_{name}_{key}",
        storage=f"sqlite:///{out / ('optuna_' + name + '.db')}",
        load_if_exists=True,
    )
    todo = max(
        0, args.trials - len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    )
    log.info(
        "[%s] Optuna: %d prób do wykonania (po %d foldach, do %d drzew)",
        name,
        todo,
        len(folds),
        MAX_ROUNDS,
    )

    def objective(trial):
        params = suggest(trial, args.objective, ysd)

        def on_fold(i, curves):
            if i >= 2:
                _, v = best_from_curves(curves)
                trial.report(v, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        curves = fold_ic_curves(params, X, y, ysel, folds, args.seed, args.jobs, cps, on_fold)
        j, v = best_from_curves(curves)
        trial.set_user_attr("n_estimators", int(cps[j]))
        trial.set_user_attr("fold_ic", [float(a) for a in curves[:, j]])
        return v

    t0 = time.time()
    if todo > 0:
        study.optimize(objective, n_trials=todo, gc_after_trial=True)
    log.info("[%s] strojenie zakończone w %.1f min", name, (time.time() - t0) / 60)
    return study, cps


def refine(study, X, y, ysel, folds, args, cps, name: str):
    """Najlepsze kandydatki liczone ponownie z kilkoma ziarnami (średnia krzywa IC).
    Wynik: zespół ensemble_k najlepszych konfiguracji, a nie jedna. Przy tak słabym sygnale wybór jednej
    konfiguracji jest w dużej mierze losowy; uśrednienie kilku najlepszych zmniejsza ten szum."""
    done = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)
    cands = done[: args.top]
    seeds = [args.seed + 100 + s for s in range(args.refine_seeds)]
    ysd = train_sd(y, folds)
    res, rows = [], []
    for t in cands:
        params = finalize_params(dict(t.params), ysd)
        if args.objective:
            params["objective"] = args.objective
        curves = np.mean(
            [fold_ic_curves(params, X, y, ysel, folds, s, args.jobs, cps) for s in seeds], axis=0
        )
        j, v = best_from_curves(curves)
        params["n_estimators"] = int(cps[j])
        res.append((v, params, np.nan_to_num(curves, nan=0.0).mean(axis=0), t.number))
        rows.append(
            {
                "trial": t.number,
                "ic_wyszukiwanie": t.value,
                "ic_ponowne_ziarna": v,
                "n_estimators": int(cps[j]),
            }
        )
    res.sort(key=lambda r: r[0], reverse=True)
    chosen = res[: max(1, args.ensemble_k)]
    tab = pd.DataFrame(rows)
    tab["w_zespole"] = tab["trial"].isin([r[3] for r in chosen])
    log.info(
        "[%s] Ponowna ocena %d kandydatek (w_zespole = wchodzi do modelu końcowego):\n%s",
        name,
        len(cands),
        tab.round(4).to_string(index=False),
    )
    return [dict(r[1]) for r in chosen], chosen[0][0], chosen[0][2], tab


def fit_ensemble(params, X, y, seeds, n_jobs):
    """params: jedna konfiguracja (dict) albo zespół konfiguracji (lista dict). Każda x każde ziarno."""
    cfgs = params if isinstance(params, list) else [params]
    models = []
    for cfg in cfgs:
        for s in seeds:
            m = build_model(cfg, s, n_jobs, y, cfg["n_estimators"])
            m.fit(X, y)
            models.append(m)
    return models


def predict_ens(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def cv_predictions(params, X, y, folds, seeds, n_jobs) -> np.ndarray:
    pred = np.full(len(X), np.nan)
    for tr, va in folds:
        pred[va] = predict_ens(fit_ensemble(params, X.iloc[tr], y[tr], seeds, n_jobs), X.iloc[va])
    return pred


def ridge_predict(Xtr, ytr, Xte, alpha):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def baselines(df, X, y, folds, tr_idx, te_idx):
    """Ridge (alfa z tych samych foldów CV) i momentum Mom_12_1. Zwraca CV i test."""
    alphas = [1, 10, 100, 1000, 10000]
    cv = {}
    for a in alphas:
        pred = np.full(len(X), np.nan)
        for tr, va in folds:
            pred[va] = ridge_predict(X.iloc[tr], y[tr], X.iloc[va], a)
        mask = ~np.isnan(pred)
        cv[a] = (np.nanmean(quarter_table(df[mask], pred[mask])["ic"]), pred)
    a_best = max(cv, key=lambda a: cv[a][0])
    mom = df["Mom_12_1"].fillna(df["Mom_12_1"].median()).to_numpy()
    out = {
        "ridge_alpha": a_best,
        "ridge_cv_pred": cv[a_best][1],
        "ridge_test_pred": ridge_predict(X.iloc[tr_idx], y[tr_idx], X.iloc[te_idx], a_best),
        "mom_pred": mom,
    }
    return out


def momentum_diagnostics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Czy model dodaje coś ponad momentum Mom_12_1?
    blend = IC średniej rang modelu i momentum (w kwartale); partial = IC rangi modelu po usunięciu
    (regresją w kwartale) składowej tłumaczonej momentum."""
    d = pd.DataFrame(
        {
            "q": df["Koniec_kwartalu"].to_numpy(),
            "y": df[EVAL_COL].to_numpy(),
            "p": pred,
            "m": df["Mom_12_1"].fillna(df["Mom_12_1"].median()).to_numpy(),
        }
    )
    blend, partial = [], []
    for _, g in d.groupby("q", sort=True):
        rp, rm = rankdata(g["p"].to_numpy()), rankdata(g["m"].to_numpy())
        blend.append(spearman(g["y"].to_numpy(), rp + rm))
        b = np.cov(rp, rm)[0, 1] / np.var(rm, ddof=1) if np.var(rm) > 0 else 0.0
        partial.append(spearman(g["y"].to_numpy(), rp - b * rm))
    return {"blend": float(np.nanmean(blend)), "partial": float(np.nanmean(partial))}


def permutation_importance(
    models,
    X: pd.DataFrame,
    df: pd.DataFrame,
    units: dict,
    n_rep: int,
    seed: int,
    per_fold_models=None,
) -> pd.DataFrame:
    """Spadek średniego IC po przetasowaniu cechy (grupy cech) między spółkami w obrębie kwartału."""
    rng = np.random.default_rng(seed)
    q_idx = [
        np.flatnonzero(df["Koniec_kwartalu"].to_numpy() == q)
        for q in sorted(df["Koniec_kwartalu"].unique())
    ]

    def mean_ic(Xp):
        p = predict_ens(models, Xp)
        return np.nanmean([spearman(df[EVAL_COL].to_numpy()[ix], p[ix]) for ix in q_idx])

    base = mean_ic(X)
    rows = []
    for name, cols in units.items():
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            vals = X[cols].to_numpy().copy()
            for ix in q_idx:
                vals[ix] = X[cols].to_numpy()[rng.permutation(ix)]
            Xp[cols] = vals
            drops.append(base - mean_ic(Xp))
        rows.append(
            {
                "jednostka": name,
                "spadek_ic": float(np.mean(drops)),
                "odch": float(np.std(drops, ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("spadek_ic", ascending=False).reset_index(drop=True)


def cv_permutation_importance(params, X, y, df, folds, units, n_rep, seed, n_jobs) -> pd.DataFrame:
    """Ważność permutacyjna liczona na foldach walidacyjnych (podstawa do decyzji o cechach)."""
    parts = []
    for tr, va in folds:
        models = fit_ensemble(params, X.iloc[tr], y[tr], [seed], n_jobs)
        parts.append(
            permutation_importance(
                models,
                X.iloc[va].reset_index(drop=True),
                df.iloc[va].reset_index(drop=True),
                units,
                n_rep,
                seed,
            ).set_index("jednostka")["spadek_ic"]
        )
    t = pd.concat(parts, axis=1)
    res = pd.DataFrame(
        {
            "spadek_ic_srednia": t.mean(axis=1),
            "spadek_ic_odch_foldow": t.std(axis=1, ddof=1),
            "foldy_ze_spadkiem": (t > 0).sum(axis=1),
        }
    )
    return res.sort_values("spadek_ic_srednia", ascending=False)


def shap_and_gain(models, X: pd.DataFrame) -> pd.DataFrame:
    xgb = _xgb()
    X = X.drop(columns=[BM_COL], errors="ignore")
    sh = np.mean(
        [
            np.abs(m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(axis=0)
            for m in models
        ],
        axis=0,
    )
    gains = []
    for m in models:
        g = m.get_booster().get_score(importance_type="total_gain")
        gains.append(pd.Series({f: g.get(f, 0.0) for f in X.columns}))
    gain = pd.concat(gains, axis=1).mean(axis=1)
    out = pd.DataFrame(
        {"shap_srednia_abs": sh, "gain_udzial": (gain / gain.sum()).to_numpy()}, index=X.columns
    )
    return out.sort_values("shap_srednia_abs", ascending=False)


def placebo_test(params, X, y, df, folds, seed, n_jobs, n=5) -> dict:
    """Przetasowanie celu w obrębie kwartału w zbiorze uczącym. IC powinno wynosić około 0."""
    rng = np.random.default_rng(seed)
    q = df["Koniec_kwartalu"].to_numpy()
    res = []
    for _ in range(n):
        ics = []
        for tr, va in folds:
            ysh = y[tr].copy()
            for qq in np.unique(q[tr]):
                ix = np.flatnonzero(q[tr] == qq)
                ysh[ix] = y[tr][rng.permutation(ix)]
            models = fit_ensemble(params, X.iloc[tr], ysh, [seed], n_jobs)
            ics.append(spearman(df[EVAL_COL].to_numpy()[va], predict_ens(models, X.iloc[va])))
        res.append(np.nanmean(ics))
    return {
        "placebo_ic_srednia": float(np.mean(res)),
        "placebo_ic_odch": float(np.std(res, ddof=1)),
        "placebo_proby": [float(r) for r in res],
    }


def make_plots(out: Path, name: str, cps, curve, cv_qt, te_qt, te_df, te_pred, imp_cv, hist):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning("Brak matplotlib, pomijam wykresy: %s", e)
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].plot(cps, curve)
    ax[0].set_xlabel("liczba drzew")
    ax[0].set_ylabel("średni rank IC (CV)")
    ax[0].set_title("Krzywa IC w walidacji")
    ax[0].grid(alpha=0.3)
    ax[1].plot(hist, color="C1")
    ax[1].set_xlabel("próba")
    ax[1].set_ylabel("najlepsze IC dotąd")
    ax[1].set_title("Przebieg strojenia")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}_strojenie.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, qt, t in ((ax[0], cv_qt, "CV (walidacja)"), (ax[1], te_qt, "Test")):
        a.bar(range(len(qt)), qt["ic"], color=np.where(qt["ic"] >= 0, "C0", "C3"))
        a.axhline(qt["ic"].mean(), color="k", ls="--", lw=1)
        a.set_xticks(range(len(qt)))
        a.set_xticklabels([str(d)[:7] for d in qt["Koniec_kwartalu"]], rotation=70, fontsize=7)
        a.set_title(f"{t}: IC w kwartale (średnia {qt['ic'].mean():.3f})")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}_ic_kwartaly.png", dpi=130)
    plt.close(fig)

    d = pd.DataFrame(
        {"q": te_df["Koniec_kwartalu"].to_numpy(), "y": te_df[EVAL_COL].to_numpy(), "p": te_pred}
    )
    d["kwintyl"] = (
        d.groupby("q")["p"]
        .rank(pct=True, method="first")
        .pipe(lambda r: np.minimum((r * 5).astype(int), 4))
        + 1
    )
    d["y_rel"] = d["y"] - d.groupby("q")["y"].transform("mean")
    g = d.groupby("kwintyl")["y_rel"].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(g.index, g.values * 100, color="C0")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("kwintyl predykcji w kwartale (5 = najwyższa predykcja)")
    ax.set_ylabel("zwrot względem średniej kwartału, p.p.")
    ax.set_title("Test: zwrot wg kwintyli predykcji")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}_kwintyle_test.png", dpi=130)
    plt.close(fig)

    s = imp_cv["spadek_ic_srednia"].sort_values()
    fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(s))))
    ax.barh(s.index, s.values)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("spadek średniego IC po przetasowaniu (CV)")
    ax.set_title("Ważność permutacyjna cech")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}_waznosc.png", dpi=130)
    plt.close(fig)


def recommendation(df, X, y, params, seeds, n_jobs, out: Path, name: str):
    tr = np.flatnonzero(df[REC_COL] == "train")
    rc = np.flatnonzero(df[REC_COL] == "rekomendacja")
    models = fit_ensemble(params, X.iloc[tr], y[tr], seeds, n_jobs)
    pred = predict_ens(models, X.iloc[rc])
    r = df.iloc[rc][
        ["Ticker", "Firma", "Koniec_kwartalu", "Data_t", "Data_wejscia", "Data_wyjscia", EVAL_COL]
    ].copy()
    r["predykcja_zwrotu_63s"] = pred
    r["ranga_predykcji"] = (
        r["predykcja_zwrotu_63s"].round(9).rank(ascending=False, method="min").astype(int)
    )
    r["spolek_z_ta_sama_predykcja"] = r.groupby(r["predykcja_zwrotu_63s"].round(9))[
        "Ticker"
    ].transform("count")
    r["ranga_realizacji"] = r[EVAL_COL].rank(ascending=False, method="min").astype(int)
    r = r.sort_values(["ranga_predykcji", "Ticker"])
    n_unique = r["predykcja_zwrotu_63s"].round(9).nunique()
    if n_unique < 0.8 * len(r):
        log.warning(
            "[%s] Rekomendacja: tylko %d różnych predykcji na %d spółek. Model jest zbyt słaby/płytki, "
            "więc ranking w dużej części jest nierozstrzygnięty (remisy). Nie interpretuj go jako rekomendacji.",
            name,
            n_unique,
            len(r),
        )
    r.to_csv(out / f"{name}_rekomendacja_2026Q1.csv", index=False, encoding="utf-8-sig")
    ic = spearman(r[EVAL_COL].to_numpy(), r["predykcja_zwrotu_63s"].to_numpy())
    log.info(
        "[%s] Rekomendacja 2026 Q1: uczenie na %d wierszach, IC względem realizacji = %.3f",
        name,
        len(tr),
        ic,
    )
    return float(ic)


def run_set(df, name: str, args, out: Path) -> dict:
    feats = get_features(df, name)
    X0 = df[feats].astype(float)
    X = X0.copy()
    if BASE_MOM:
        X[BM_COL] = momentum_score(df)
    y = df[args.target].to_numpy()
    ysel = df[EVAL_COL].to_numpy()
    folds = fold_indices(df)
    tr_idx = np.flatnonzero(df["Podzial"] == "train")
    te_idx = np.flatnonzero((df["Podzial"] == "test") & (df[REC_COL] != "rekomendacja"))
    log.info(
        "=== Zestaw %s: %d cech, %d wierszy uczących, %d testowych ===",
        name,
        len(feats),
        len(tr_idx),
        len(te_idx),
    )
    log.info(
        "Braki w cechach (największe): %s",
        (X.iloc[tr_idx].isna().mean().sort_values(ascending=False).head(4) * 100)
        .round(1)
        .to_dict(),
    )

    cps = checkpoints()
    pfile = out / f"{name}_params.json"
    if args.skip_tune and pfile.exists():
        params = json.loads(pfile.read_text())
        params = params if isinstance(params, list) else [params]
        curve = np.full(len(cps), np.nan)
        hist = [np.nan]
        refine_tab = pd.DataFrame()
        tune_score = np.nan
        log.info("[%s] Użyto zapisanych parametrów z %s", name, pfile)
    else:
        study, cps = run_study(X, y, ysel, folds, args, name, out)
        params, tune_score, curve, refine_tab = refine(study, X, y, ysel, folds, args, cps, name)
        pfile.write_text(json.dumps(params, indent=2, ensure_ascii=False))
        tdf = pd.DataFrame(
            [
                {
                    "trial": t.number,
                    "stan": t.state.name,
                    "ic": t.value,
                    **t.params,
                    "n_estimators": t.user_attrs.get("n_estimators"),
                }
                for t in study.trials
            ]
        )
        tdf.sort_values("ic", ascending=False).to_csv(out / f"{name}_proby_optuna.csv", index=False)
        refine_tab.to_csv(out / f"{name}_ponowna_ocena.csv", index=False)
        vals = [
            t.value if (t.state.name == "COMPLETE" and t.value is not None) else np.nan
            for t in study.trials
        ]
        hist = pd.Series(vals).cummax().to_numpy()
    if BASE_MOM:
        sc = X[BM_COL].to_numpy()[tr_idx]
        sc = sc - sc.mean()
        log.info(
            "[%s] Start od momentum: b na całym zbiorze uczącym = %.4f (zwrot na przejście z najsłabszego do najsilniejszego momentum)",
            name,
            float(np.sum(sc * (y[tr_idx] - y[tr_idx].mean())) / np.sum(sc**2)),
        )
    log.info("[%s] Wybrane parametry: %s", name, json.dumps(params))
    ks = [c["n_estimators"] for c in params]
    log.info("[%s] Zespół %d konfiguracji, liczby drzew: %s", name, len(params), ks)
    if min(ks) <= cps[0]:
        log.warning(
            "[%s] Część konfiguracji ma liczbę drzew na dolnej granicy zakresu (%d): taki model prawie nic się nie uczy, "
            "co sugeruje bardzo słaby sygnał.",
            name,
            cps[0],
        )
    if max(ks) >= cps[-1]:
        log.warning(
            "[%s] Część konfiguracji ma liczbę drzew na górnej granicy zakresu (%d): zwiększ --max-rounds.",
            name,
            cps[-1],
        )

    seeds = [args.seed + i for i in range(args.final_seeds)]
    cv_pred = cv_predictions(params, X, y, folds, seeds, args.jobs)
    mask = ~np.isnan(cv_pred)
    cv_df = df[mask].reset_index(drop=True)
    cv_sum, cv_qt = summarize(cv_df, cv_pred[mask], float(np.mean(y)))
    pd.DataFrame(
        {
            "Ticker": cv_df["Ticker"],
            "Koniec_kwartalu": cv_df["Koniec_kwartalu"],
            "zwrot": cv_df[EVAL_COL],
            "predykcja": cv_pred[mask],
        }
    ).to_csv(out / f"{name}_predykcje_cv.csv", index=False)
    base = baselines(df, X0, y, folds, tr_idx, te_idx)
    bm = ~np.isnan(base["ridge_cv_pred"])
    ridge_cv_sum, ridge_cv_qt = summarize(
        df[bm].reset_index(drop=True), base["ridge_cv_pred"][bm], float(np.mean(y))
    )
    mom_cv_sum, _ = summarize(
        df[mask].reset_index(drop=True), base["mom_pred"][mask], float(np.mean(y)), is_score=True
    )

    final = fit_ensemble(params, X.iloc[tr_idx], y[tr_idx], seeds, args.jobs)
    te_df = df.iloc[te_idx].reset_index(drop=True)
    te_pred = predict_ens(final, X.iloc[te_idx])
    log.info(
        "[%s] Udział unikalnych predykcji na teście: %.0f%%",
        name,
        100 * len(np.unique(np.round(te_pred, 9))) / len(te_pred),
    )
    ymean = float(np.mean(y[tr_idx]))
    te_sum, te_qt = summarize(te_df, te_pred, ymean)
    ridge_te_sum, _ = summarize(te_df, base["ridge_test_pred"], ymean)
    mom_te_sum, _ = summarize(te_df, base["mom_pred"][te_idx], ymean, is_score=True)
    const_sum, _ = summarize(te_df, np.full(len(te_df), ymean), ymean)
    pd.DataFrame(
        {
            "Ticker": te_df["Ticker"],
            "Koniec_kwartalu": te_df["Koniec_kwartalu"],
            "zwrot": te_df[EVAL_COL],
            "predykcja": te_pred,
        }
    ).to_csv(out / f"{name}_predykcje_test.csv", index=False)

    mom_diag_cv = momentum_diagnostics(cv_df, cv_pred[mask])
    mom_diag_te = momentum_diagnostics(te_df, te_pred)

    plc = placebo_test(params, X, y, df, folds, args.seed, args.jobs, n=args.placebo)
    units = {f: [f] for f in feats}
    units.update(
        {
            f"[grupa] {g}": [c for c in cols if c in feats]
            for g, cols in GROUPS.items()
            if any(c in feats for c in cols)
        }
    )
    imp_cv = cv_permutation_importance(
        params, X, y, df, folds, units, args.perm_rep, args.seed, args.jobs
    )
    imp_te = permutation_importance(
        final, X.iloc[te_idx].reset_index(drop=True), te_df, units, args.perm_rep, args.seed
    )
    sg = shap_and_gain(final, X.iloc[te_idx])
    imp_cv.to_csv(out / f"{name}_waznosc_perm_cv.csv")
    imp_te.to_csv(out / f"{name}_waznosc_perm_test.csv", index=False)
    sg.to_csv(out / f"{name}_waznosc_shap_gain.csv")
    cv_qt.to_csv(out / f"{name}_ic_kwartaly_cv.csv", index=False)
    te_qt.to_csv(out / f"{name}_ic_kwartaly_test.csv", index=False)

    rec_ic = recommendation(df, X, y, params, seeds, args.jobs, out, name) if args.final else None
    if not args.final:
        log.info(
            "[%s] Prognoza 2026 Q1 pominięta (uruchom z --final dopiero po wyborze modelu)", name
        )

    make_plots(
        out,
        name,
        cps,
        curve,
        cv_qt,
        te_qt,
        te_df,
        te_pred,
        imp_cv[~imp_cv.index.str.startswith("[grupa]")],
        hist,
    )

    res = {
        "zestaw": name,
        "n_cech": len(feats),
        "parametry": params,
        "ic_strojenie": tune_score,
        "cv": cv_sum,
        "test": te_sum,
        "ridge_cv": ridge_cv_sum,
        "ridge_test": ridge_te_sum,
        "ridge_alfa": base["ridge_alpha"],
        "momentum_cv": mom_cv_sum,
        "momentum_test": mom_te_sum,
        "stala_test": const_sum,
        "placebo": plc,
        "momentum_dodatek_cv": mom_diag_cv,
        "momentum_dodatek_test": mom_diag_te,
        "rekomendacja_ic_2026Q1": rec_ic,
    }
    (out / f"{name}_wyniki.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float)
    )
    res["_cv_qt"], res["_te_qt"] = cv_qt, te_qt
    return res


def log_to_registry(
    path: str, args, name: str, res: dict, started: str, duration_min: float
) -> None:
    """Dopisuje jeden wiersz na (przebieg, zestaw) do zbiorczego pliku CSV. Plik nigdy nie jest nadpisywany,
    więc po wszystkich próbach zostaje pełna lista do opisu w pracy (liczba i wyniki wszystkich wariantów).
    """
    cv, te, pl = res["cv"], res["test"], res["placebo"]
    ks = (
        [c["n_estimators"] for c in res["parametry"]]
        if isinstance(res["parametry"], list)
        else [res["parametry"]["n_estimators"]]
    )
    row = {
        "start": started,
        "czas_min": round(duration_min, 1),
        "folder": args.out,
        "wersja_przestrzeni": SPACE_VERSION,
        "zestaw": name,
        "n_cech": res["n_cech"],
        "cel": args.target,
        "funkcja_straty_stala": args.objective or "strojona",
        "baza_momentum": bool(BASE_MOM),
        "proby_optuna": args.trials,
        "max_drzew": args.max_rounds,
        "krok": args.step,
        "top": args.top,
        "zespol_k": args.ensemble_k,
        "ziarna_ponowna_ocena": args.refine_seeds,
        "ziarna_koncowe": args.final_seeds,
        "seed": args.seed,
        "plik_danych": Path(args.data).name,
        "ic_strojenie": res["ic_strojenie"],
        "cv_ic": cv["ic_srednia"],
        "cv_icir": cv["icir"],
        "cv_nw_t": cv["ic_nw_t"],
        "cv_ci_lo": cv["ic_ci95_lo"],
        "cv_ci_hi": cv["ic_ci95_hi"],
        "test_ic": te["ic_srednia"],
        "test_icir": te["icir"],
        "test_nw_t": te["ic_nw_t"],
        "test_ci_lo": te["ic_ci95_lo"],
        "test_ci_hi": te["ic_ci95_hi"],
        "ridge_cv_ic": res["ridge_cv"]["ic_srednia"],
        "ridge_test_ic": res["ridge_test"]["ic_srednia"],
        "momentum_cv_ic": res["momentum_cv"]["ic_srednia"],
        "momentum_test_ic": res["momentum_test"]["ic_srednia"],
        "cv_oczyszczone_z_momentum": res["momentum_dodatek_cv"]["partial"],
        "test_oczyszczone_z_momentum": res["momentum_dodatek_test"]["partial"],
        "placebo_ic": pl["placebo_ic_srednia"],
        "placebo_odch": pl["placebo_ic_odch"],
        "ic_rekomendacji_2026Q1": res["rekomendacja_ic_2026Q1"],
        "test_bez_2026Q1": True,
        "n_kwartalow_test": te["n_kwartalow"],
        "cechy": ";".join(SETS[name]),
        "liczby_drzew_zespolu": ";".join(map(str, ks)),
        "parametry_json": json.dumps(res["parametry"], ensure_ascii=False),
    }
    df_row = pd.DataFrame([row])
    p = Path(path)
    if p.exists():
        old = pd.read_csv(p)
        df_row = pd.concat([old, df_row], ignore_index=True)
    df_row.to_csv(p, index=False)
    log.info(
        "Dopisano przebieg [%s] do rejestru %s (łącznie wierszy: %d)",
        name,
        p.resolve(),
        len(df_row),
    )


def report(results: dict, out: Path) -> str:
    cols = [
        "ic_srednia",
        "icir",
        "ic_nw_t",
        "ic_ci95_lo",
        "ic_ci95_hi",
        "ic_udzial_dodatnich",
        "spread_gora_dol",
        "precision_gora_tercyl",
        "rmse",
        "mae",
        "r2_oos_vs_srednia_train",
    ]
    rows = []
    for n, r in results.items():
        rows.append({"model": f"XGB {n}", "okres": "CV", **{c: r["cv"][c] for c in cols}})
        rows.append({"model": f"Ridge ({n})", "okres": "CV", **{c: r["ridge_cv"][c] for c in cols}})
        rows.append(
            {"model": f"Mom_12_1 ({n})", "okres": "CV", **{c: r["momentum_cv"][c] for c in cols}}
        )
        rows.append({"model": f"XGB {n}", "okres": "TEST", **{c: r["test"][c] for c in cols}})
        rows.append(
            {"model": f"Ridge ({n})", "okres": "TEST", **{c: r["ridge_test"][c] for c in cols}}
        )
        rows.append(
            {
                "model": f"Mom_12_1 ({n})",
                "okres": "TEST",
                **{c: r["momentum_test"][c] for c in cols},
            }
        )
        rows.append(
            {"model": f"stała ({n})", "okres": "TEST", **{c: r["stala_test"][c] for c in cols}}
        )
    t = pd.DataFrame(rows).round(4)
    t.to_csv(out / "porownanie_modeli.csv", index=False)
    txt = ["PODSUMOWANIE", "=" * 70, t.to_string(index=False), ""]
    for n, r in results.items():
        p = r["placebo"]
        txt.append(
            f"[{n}] placebo (przetasowany cel): średnie IC = {p['placebo_ic_srednia']:.4f} "
            f"(odch. {p['placebo_ic_odch']:.4f}); rzeczywiste IC w CV = {r['cv']['ic_srednia']:.4f}"
        )
        d1, d2 = r["momentum_dodatek_cv"], r["momentum_dodatek_test"]
        txt.append(
            f"[{n}] model + momentum (średnia rang), IC: CV {d1['blend']:.4f}, TEST {d2['blend']:.4f}; "
            f"IC modelu po oczyszczeniu z momentum: CV {d1['partial']:.4f}, TEST {d2['partial']:.4f}"
        )
        if r["rekomendacja_ic_2026Q1"] is not None:
            txt.append(
                f"[{n}] IC rekomendacji 2026 Q1 względem realizacji: {r['rekomendacja_ic_2026Q1']:.3f}"
            )
        txt.append(f"[{n}] parametry: {json.dumps(r['parametry'])}")
    names = list(results)
    for ref, other in ((names[0], n) for n in names[1:]):
        txt += ["", f"PORÓWNANIE zestawów, różnica IC w kwartałach ({other} − {ref}):"]
        for lab, key in (("CV", "_cv_qt"), ("TEST", "_te_qt")):
            a = results[ref][key].set_index("Koniec_kwartalu")["ic"]
            b = results[other][key].set_index("Koniec_kwartalu")["ic"]
            d = (b - a).dropna()
            lo, hi = block_bootstrap_ci(d.to_numpy())
            txt.append(
                f"  {lab}: średnia różnica {d.mean():+.4f}, NW t = {nw_tstat(d.to_numpy()):+.2f}, "
                f"95% przedział [{lo:+.4f}, {hi:+.4f}], {other} lepsze w {int((d > 0).sum())}/{len(d)} kwartałach"
            )
        txt.append("  Jeżeli przedział obejmuje 0, różnicy nie da się odróżnić od szumu.")
    s = "\n".join(txt)
    (out / f"podsumowanie_{out.name}.txt").write_text(s, encoding="utf-8")
    return s


def main():
    global MAX_ROUNDS, CHECK_STEP, TARGET, BASE_MOM
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=str(ZBIOR_DOMYSLNY))
    ap.add_argument("--out", default="wyniki_xgb")
    ap.add_argument(
        "--registry",
        default="rejestr_przebiegow.csv",
        help="plik zbiorczy, do którego dopisywany jest każdy przebieg",
    )
    ap.add_argument(
        "--features",
        default="A,C,D",
        help="zestawy po przecinku, np. A,C,D (dostępne: A-H; both = A,B)",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="policz też prognozę dla 2026 Q1 (dopiero po wyborze modelu)",
    )
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--trials", type=int, default=150, help="liczba prób Optuna na zestaw")
    ap.add_argument(
        "--objective",
        default=None,
        help="stała funkcja straty zamiast strojenia, np. reg:squarederror",
    )
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--step", type=int, default=CHECK_STEP)
    ap.add_argument("--top", type=int, default=8, help="ile najlepszych prób ponownie ocenić")
    ap.add_argument("--refine-seeds", type=int, default=4)
    ap.add_argument(
        "--ensemble-k",
        type=int,
        default=5,
        help="ile najlepszych konfiguracji uśrednić w modelu końcowym",
    )
    ap.add_argument("--final-seeds", type=int, default=5)
    ap.add_argument("--perm-rep", type=int, default=10)
    ap.add_argument("--placebo", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--base-momentum",
        action="store_true",
        help="start modelu od prognozy z momentum Mom_12_1 (base_margin)",
    )
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--skip-tune", action="store_true", help="użyj zapisanych {A,B}_params.json")
    args = ap.parse_args()
    MAX_ROUNDS, CHECK_STEP, TARGET, BASE_MOM = (
        args.max_rounds,
        args.step,
        args.target,
        args.base_momentum,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "log.txt", encoding="utf-8"),
        ],
    )
    try:
        import optuna
        import xgboost

        log.info(
            "xgboost %s, optuna %s, pandas %s, numpy %s",
            xgboost.__version__,
            optuna.__version__,
            pd.__version__,
            np.__version__,
        )
    except ImportError as e:
        sys.exit(
            f"Brak biblioteki: {e}. Zainstaluj: pip install xgboost optuna scikit-learn scipy matplotlib"
        )

    df = load_data(args.data, args.target)
    check_splits(df)
    (out / "konfiguracja.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    names = (
        ["A", "B"]
        if args.features == "both"
        else [x.strip().upper() for x in args.features.split(",")]
    )
    bad = [x for x in names if x not in SETS]
    if bad:
        sys.exit(f"Nieznany zestaw cech: {bad}. Dostępne: {list(SETS)}")
    results = {}
    for n in names:
        t0, started = time.time(), time.strftime("%Y-%m-%d %H:%M:%S")
        results[n] = run_set(df, n, args, out)
        log_to_registry(args.registry, args, n, results[n], started, (time.time() - t0) / 60)
    print("\n" + report(results, out))
    log.info("Wyniki zapisano w %s", out.resolve())


if __name__ == "__main__":
    main()
