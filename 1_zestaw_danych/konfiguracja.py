"""Wspólna konfiguracja ścieżek.

Katalogi można zmienić zmiennymi środowiskowymi GPW_DANE (pliki wejściowe)
i GPW_WYNIKI (pliki wynikowe); domyślnie są to podkatalogi folderu ze skryptami.
"""

import os
from pathlib import Path

KATALOG = Path(__file__).resolve().parent
DANE = Path(os.environ.get("GPW_DANE", KATALOG / "dane_wejsciowe"))
WYNIKI = Path(os.environ.get("GPW_WYNIKI", KATALOG / "wyniki"))
POSREDNIE = WYNIKI / "posrednie"

for _katalog in (DANE, WYNIKI, POSREDNIE):
    _katalog.mkdir(parents=True, exist_ok=True)
