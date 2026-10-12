"""
Rechenkern des Momentum-Dashboards (Schritt 2).
Hier stehen nur Berechnungen, keine Webseite. Zahlen aendern = hier oder in app.py oben.

Idee des Scores (Standard aus der Forschung):
  - 12-1-Momentum: Rendite der letzten 12 Monate OHNE den letzten Monat
  - 6-Monats-Rendite
  - beide gemittelt und durch die jaehrliche Schwankung (Volatilitaet) geteilt
  - Trendfilter: Kurs liegt ueber dem 200-Tage-Durchschnitt
"""

import numpy as np
import pandas as pd

HOLD_MONTHS = [1, 3, 6, 12]  # Haltedauern, die im Backtest verglichen werden


def score_frame(P: pd.DataFrame):
    """P = Tabelle mit Schlusskursen (Zeilen = Tage, Spalten = Ticker).
    Gibt Score, Trend (True/False), 12-1-Rendite und 6-Monats-Rendite fuer JEDEN Tag zurueck."""
    R = P.pct_change(fill_method=None)
    vol = R.rolling(252, min_periods=200).std() * np.sqrt(252)
    r121 = P.shift(21) / P.shift(252) - 1
    r6 = P / P.shift(126) - 1
    score = (0.5 * r121 + 0.5 * r6) / vol
    score = score.replace([np.inf, -np.inf], np.nan)
    trend = P > P.rolling(200, min_periods=150).mean()
    return score, trend, r121, r6


def month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Position (Zeilennummer) des letzten Handelstags jedes Monats."""
    pos = pd.Series(np.arange(len(index)), index=index)
    return pos.groupby([index.year, index.month]).last().values


def backtest(P: pd.DataFrame, top_frac: float = 0.2, cost: float = 0.005,
             min_names: int = 10, min_periods: int = 6) -> pd.DataFrame:
    """Testet fuer jede Haltedauer: Jeweils zum Monatsende die staerksten Titel
    (obere top_frac mit Trendfilter) kaufen und X Monate halten.
    cost = Kosten je Kauf+Verkauf (0.005 = 0,5 %), wird von jeder Periode abgezogen."""
    S, T, _, _ = score_frame(P)
    me = month_end_positions(P.index)
    rows = []
    for H in HOLD_MONTHS:
        rets = []
        for k in range(len(me) - H):
            a, b = me[k], me[k + H]
            s = S.iloc[a].dropna()
            if len(s) < min_names:
                continue
            cut = s.quantile(1 - top_frac)
            t = T.iloc[a].reindex(s.index).fillna(False).astype(bool)
            sel = s[(s >= cut) & t].index
            if len(sel) == 0:
                continue
            fwd = (P.iloc[b][sel] / P.iloc[a][sel] - 1).dropna()
            if len(fwd) == 0:
                continue
            rets.append(float(fwd.mean()) - cost)
        if len(rets) >= min_periods:
            avg = float(np.mean(rets))
            per_month = (1 + avg) ** (1 / H) - 1 if avg > -1 else np.nan
            rows.append({
                "Haltedauer (Monate)": H,
                "Zeitraeume getestet": len(rets),
                "Rendite je Zeitraum (netto)": avg,
                "Rendite je Monat (netto)": per_month,
                "Trefferquote": float(np.mean([r > 0 for r in rets])),
            })
    return pd.DataFrame(rows)


def best_hold(table: pd.DataFrame):
    """Haltedauer (Monate) mit der besten Netto-Rendite je Monat, sonst None."""
    if table is None or table.empty:
        return None
    t = table.dropna(subset=["Rendite je Monat (netto)"])
    if t.empty:
        return None
    return int(t.loc[t["Rendite je Monat (netto)"].idxmax(), "Haltedauer (Monate)"])
