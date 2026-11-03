# Predykcja 63-sesyjnych zwrotów spółek GPW

Kod do pracy magisterskiej **„Predykcja 63-sesyjnych zwrotów spółek GPW: metodyka, wyniki i wnioski”**.

**Autor:** [Imię i nazwisko] · **Uczelnia / kierunek:** [uzupełnij] · **Rok:** 2026

## Cel pracy

Sprawdzenie, czy modele uczenia maszynowego (XGBoost, LightGBM, Random Forest) potrafią przewidywać, które spółki giełdowe osiągną w kolejnym kwartale (63 sesje) relatywnie wyższe stopy zwrotu niż pozostałe. Modele porównano z benchmarkiem momentum.

## Dane i metoda

- **Próba:** 54 niefinansowe spółki GPW, 2018Q1–2025Q4.
- **Walidacja:** 11 kwartałów walidacji krzyżowej (2021Q1–2023Q3), 9 kwartałów testu (2023Q4–2025Q4), zbiór wstrzymany 2026Q1. Okna zwrotów są oczyszczane (purging), aby nie nachodziły na siebie.
- **Strojenie:** Optuna (estymator TPE), kryterium: średni rank IC w walidacji.
- **Ocena:** współczynnik informacyjny (IC), statystyka t z estymatorem Newey–West, bootstrap blokowy, portfel 18 spółek z kosztami transakcyjnymi, Sharpe, Information Ratio, Deflated Sharpe Ratio.

## Struktura

```
.
├── dane/              gotowy zbiór modelowy (zbior_modelowy_v1.csv)
├── 1_zestaw_danych/   budowa zbioru modelowego (etapy 01–09)
├── 2_modele/          XGBoost, LightGBM, Random Forest
├── requirements.txt   wymagane biblioteki
└── README.md
```

## Instalacja

```bash
git clone https://github.com/[login]/[nazwa-repozytorium].git
cd [nazwa-repozytorium]
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dane wejściowe

Folder `1_zestaw_danych/dane_wejsciowe/` (inny: zmienna środowiskowa `GPW_DANE`). Wyniki trafiają do `1_zestaw_danych/wyniki/` (inny: `GPW_WYNIKI`). Foldery tworzone są automatycznie.

Surowe dane **nie są częścią repozytorium** (licencje dostawców). Źródła: Yahoo Finance (ceny, VIX, EPOL, USDPLN), Stooq (WIG, WIBOR 3M), Investing.com (rentowność obligacji 5-letnich), EMIS (sprawozdania i kapitalizacja, dostęp licencjonowany), Business Radar (EBIT i amortyzacja KGH).

| Plik | Zawartość |
|---|---|
| `yahoo_wszystkie_ceny.xlsx` | notowania z Yahoo Finance (tworzy je etap 01) |
| `cechy_fundamentalne_v1.xlsx` | cechy fundamentalne spółek (arkusz `Cechy`) |
| `liczba_akcji_status.xlsx` | liczba akcji (arkusze `Status`, `Tabela_uzytkownika`, `Zdarzenia`) |
| `plynnosc_dynamiczna.xlsx` | miary płynności (arkusz `Wiersze`) |
| `emis_polaczone.xlsx` | EMIS: kapitalizacja i cena zamknięcia w USD |
| `kgh_business_radar.xlsx` | EBIT i amortyzacja KGH (tys. PLN) |
| `wig_d.csv` | dzienne notowania WIG (kolumny `Data`, `Zamkniecie`) |
| `vix_d.csv`, `epol_d.csv` | VIX i EPOL (kolumny `Date`, `Zamkniecie`) |
| `usdpln_d.csv`, `wibor3m_d.csv` | USDPLN i WIBOR 3M (kolumny `Data`, `Zamkniecie`) |
| `obligacje_pl_5l.csv` | rentowność obligacji 5-letnich (kolumny `Date`, `Price`) |

Pliki `cechy_fundamentalne_v1`, `liczba_akcji_status`, `plynnosc_dynamiczna`, `emis_polaczone` i `kgh_business_radar` powstały poza skryptami (dane źródłowe i ich ręczna weryfikacja); skrypty zakładają je jako gotowe wejście.

## Gotowy zbiór modelowy (szybki start)

Plik `dane/zbior_modelowy_v1.csv` (3145 wierszy, 73 kolumny; panel spółka-kwartał z cechami, zmienną celu i podziałem na foldy) to wynik etapu 8. Dzięki niemu modele (etap 2) można uruchomić bez dostępu do danych EMIS i bez powtarzania etapu 1:

```bash
cd 2_modele
python model_xgboost.py --features DS --seed 0 --data ../dane/zbior_modelowy_v1.csv --out wyniki_xgboost
```

Zbiór zawiera kolumny pomocnicze oraz spółki spoza próby końcowej; do modelowania używane są wiersze z `Do_modelu = 1` (54 spółki), a definicje cech opisuje słownik `SETS` w skryptach i rozdział 4 pracy.

## Uruchamianie

Etap 1: zbiór modelowy (folder `1_zestaw_danych`)

```bash
python 01_pobranie_cen_yahoo.py             # ceny, dywidendy, splity z Yahoo Finance
python 02_pobranie_makro_yahoo.py           # (pomocniczy) VIX, EPOL, USDPLN
python 03_przygotowanie_danych.py           # łączenie źródeł, korekta EBITDA
python 04_cechy_rynkowe_i_zmienna_celu.py   # cechy cenowe, GARCH, wyceny, F-Score, zmienna celu
python 05_wskazniki_makro.py                # Makro_A i Makro_B
python 06_plynnosc_filar2.py                # miary płynności do selekcji spółek
python 07_panel_finalny.py                  # rangi, winsoryzacja, foldy, panel finalny
python 08_zbior_modelowy.py                 # zbior_modelowy_v1.csv (wejście do modeli)
python 09_kontrola_spojnosci.py             # kontrole spójności (tylko wydruk)
```

Etap 2: modele (folder `2_modele`)

```bash
python model_xgboost.py       --features DS --seed 0 --out wyniki_xgboost
python model_random_forest.py --features DS --seed 0 --out wyniki_random_forest
python model_lightgbm.py      --features D2 --seed 0 --out wyniki_lightgbm
```

Zestaw DS jest zestawem głównym dla XGBoost i Random Forest, a D2 dla LightGBM (definicje: słownik `SETS` w skryptach). Domyślny plik danych: `../1_zestaw_danych/wyniki/zbior_modelowy_v1.csv` (inny: `--data`). Pozostałe argumenty (liczba prób Optuna, ziaren, permutacji) opisuje `--help`. Prognoza dla 2026Q1 liczona jest tylko z flagą `--final` po zakończeniu strojenia (z `--skip-tune`). Każdy przebieg dopisuje wiersz do rejestru (`--registry`).

## Odtwarzalność

Wyniki zależą od ziarna (`--seed`), wersji bibliotek i liczby wątków (`--jobs`). Strojenie Optuna przy innej liczbie wątków może dać nieco inne próby. W pracy główne ziarno to 42, a ocenę stabilności wykonano na ziarnach 0, 1 i 2.

## Cytowanie

[Nazwisko, I.] (2026). *Predykcja 63-sesyjnych zwrotów spółek GPW: metodyka, wyniki i wnioski*. Praca magisterska, [uczelnia].

## Licencja

[np. MIT dla kodu]. Praca magisterska i dane podlegają zasadom uczelni i dostawców danych.
