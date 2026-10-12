"""
Momentum-Dashboard - Schritt 2
Universum (Indizes, Rohstoffe, Krypto-ETFs), Kurse, Momentum-Score, Signal und Haltedauer.
Benoetigt zusaetzlich die Datei momentum.py im selben Ordner.

Starten lokal:   streamlit run app.py
Aenderungen:     Einstellungen ganz oben (REFRESH_MIN, INDIZES, ROHSTOFFE, KRYPTO_ETFS)
"""

import io
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from momentum import HOLD_MONTHS, backtest, best_hold, score_frame

# ----------------------------------------------------------------------------
# EINSTELLUNGEN (hier aenderst du spaeter am meisten)
# ----------------------------------------------------------------------------
REFRESH_MIN = 15  # Kurse alle X Minuten neu laden (10 bis 20 empfohlen)
BATCH_SIZE = 150  # wie viele Ticker pro Abruf bei Yahoo (Kurse)
HIST_BATCH = 100  # wie viele Ticker pro Abruf bei Yahoo (5 Jahre Historie)

# Signal-Regeln (Anteil der besten Titel, jeweils zusaetzlich ueber dem 200-Tage-Schnitt)
KAUFEN_TOP = 0.10   # obere 10 % = Kaufen
HALTEN_TOP = 0.30   # obere 30 % = Halten, alles darunter = Verkaufen
BACKTEST_TOP = 0.20  # der Backtest testet die oberen 20 %
KOSTEN_PRO_RUNDE = 0.005  # 0,5 % je Kauf+Verkauf (Gebuehren + Spread), anpassbar
MIN_GRUPPE = 30  # eine Gruppe braucht so viele Titel fuer einen eigenen Backtest

# Jeder Index: Wikipedia-Seite (oder None), Boersen-Endung fuer Yahoo,
# erwartete Anzahl (nur fuer die Statusanzeige) und optionale eigene CSV-Datei.
# Eine CSV in Ordner "listen" hat IMMER Vorrang vor Wikipedia.
INDIZES = {
    "S&P 500": dict(
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        suffix="", dot_to_dash=True, erwartet=500, min_rows=400, csv="listen/sp500.csv"),
    "Nasdaq 100": dict(
        url="https://en.wikipedia.org/wiki/Nasdaq-100",
        suffix="", dot_to_dash=True, erwartet=100, min_rows=80, csv="listen/nasdaq100.csv"),
    "Nikkei 225": dict(
        url=None,  # Wikipedia liefert keine Boersencodes -> eigene CSV noetig
        suffix=".T", dot_to_dash=False, erwartet=225, min_rows=150, csv="listen/nikkei225.csv"),
    "CAC 40": dict(
        url="https://en.wikipedia.org/wiki/CAC_40",
        suffix=".PA", dot_to_dash=False, erwartet=40, min_rows=30, csv="listen/cac40.csv"),
    "IBEX 35": dict(
        url="https://en.wikipedia.org/wiki/IBEX_35",
        suffix=".MC", dot_to_dash=False, erwartet=35, min_rows=30, csv="listen/ibex35.csv"),
    "FTSE MIB": dict(
        url="https://en.wikipedia.org/wiki/FTSE_MIB",
        suffix=".MI", dot_to_dash=False, erwartet=40, min_rows=30, csv="listen/ftsemib.csv"),
    "MDAX": dict(
        url="https://en.wikipedia.org/wiki/MDAX",
        suffix=".DE", dot_to_dash=False, erwartet=50, min_rows=40, csv="listen/mdax.csv"),
    "SDAX": dict(
        url="https://en.wikipedia.org/wiki/SDAX",
        suffix=".DE", dot_to_dash=False, erwartet=70, min_rows=50, csv="listen/sdax.csv"),
    "TecDAX": dict(
        url="https://en.wikipedia.org/wiki/TecDAX",
        suffix=".DE", dot_to_dash=False, erwartet=30, min_rows=25, csv="listen/tecdax.csv"),
}

# Rohstoffe als Futures-Ticker bei Yahoo (beliebig erweiterbar)
ROHSTOFFE = {
    "GC=F": "Gold", "SI=F": "Silber", "PL=F": "Platin", "PA=F": "Palladium",
    "HG=F": "Kupfer", "CL=F": "Rohoel WTI", "BZ=F": "Rohoel Brent",
    "NG=F": "Erdgas", "ZW=F": "Weizen", "ZC=F": "Mais", "ZS=F": "Sojabohnen",
    "SB=F": "Zucker", "KC=F": "Kaffee", "CC=F": "Kakao", "CT=F": "Baumwolle",
}

# Krypto-ETFs (Ticker pruefen, falls du einen anderen Ethereum-ETF willst)
KRYPTO_ETFS = {
    "IBIT": "iShares Bitcoin Trust ETF",
    "ETHA": "iShares Ethereum Trust ETF",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (privates Momentum-Dashboard)"}
ZEITZONE = ZoneInfo("Europe/Zurich")

st.set_page_config(page_title="Momentum-Dashboard", layout="wide")
st_autorefresh(interval=REFRESH_MIN * 60 * 1000, key="auto_refresh")


# ----------------------------------------------------------------------------
# UNIVERSUM LADEN
# ----------------------------------------------------------------------------
def _flatten(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            c = " ".join(str(x) for x in c if "Unnamed" not in str(x))
        out.append(str(c).strip())
    return out


def _clean(df, ticker_col, name_col, cfg):
    t = (
        df[ticker_col].astype(str)
        .str.replace(r"\[.*?\]", "", regex=True)  # Wikipedia-Fussnoten
        .str.split(":").str[-1]                    # "EPA: AI" -> "AI"
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    if cfg["dot_to_dash"]:
        t = t.str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (Yahoo-Schreibweise)
    t = t[(t != "") & (t.str.lower() != "nan")]
    names = df.loc[t.index, name_col] if name_col else t
    out = pd.DataFrame({"Ticker": t, "Name": names.astype(str)})
    # Boersen-Endung ergaenzen, falls noch keine da ist
    if cfg["suffix"]:
        mask = ~out["Ticker"].str.contains(".", regex=False)
        out.loc[mask, "Ticker"] = out.loc[mask, "Ticker"] + cfg["suffix"]
    return out.drop_duplicates("Ticker")


def _find_col(columns, keys):
    return next((c for c in columns if any(k in c.lower() for k in keys)), None)


def _from_csv(path, cfg):
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    tcol = _find_col(df.columns, ("ticker", "symbol", "code"))
    ncol = _find_col(df.columns, ("name", "company", "unternehmen"))
    if tcol is None:
        raise ValueError("CSV braucht eine Spalte 'ticker'")
    return _clean(df, tcol, ncol, cfg)


def _from_wikipedia(cfg):
    html = requests.get(cfg["url"], headers=HEADERS, timeout=30).text
    best = None
    for t in pd.read_html(io.StringIO(html)):
        t = t.copy()
        t.columns = _flatten(t.columns)
        tcol = _find_col(t.columns, ("ticker", "symbol"))
        if tcol is None or len(t) < cfg["min_rows"]:
            continue
        if best is None or len(t) > len(best[0]):
            best = (t, tcol)
    if best is None:
        raise ValueError("Keine passende Tabelle auf der Wikipedia-Seite gefunden")
    t, tcol = best
    ncol = _find_col([c for c in t.columns if c != tcol],
                     ("company", "security", "name", "unternehmen"))
    return _clean(t, tcol, ncol, cfg)


@st.cache_data(ttl=24 * 3600, show_spinner="Index-Listen werden geladen ...")
def load_universe():
    frames, status = [], []
    for index_name, cfg in INDIZES.items():
        quelle, fehler, df = "", "", None
        try:
            if os.path.exists(cfg["csv"]):
                df, quelle = _from_csv(cfg["csv"], cfg), "eigene CSV"
            elif cfg["url"]:
                df, quelle = _from_wikipedia(cfg), "Wikipedia"
            else:
                fehler = f"Bitte CSV anlegen: {cfg['csv']}"
        except Exception as e:  # bewusst breit: ein Index darf die Seite nicht kippen
            fehler = str(e)[:120]
        if df is not None and not df.empty:
            df["Index"] = index_name
            frames.append(df)
        status.append({
            "Index": index_name, "Erwartet": cfg["erwartet"],
            "Geladen": 0 if df is None else len(df),
            "Quelle": quelle or "-", "Hinweis": fehler,
        })

    for ticker, name in ROHSTOFFE.items():
        frames.append(pd.DataFrame({"Ticker": [ticker], "Name": [name], "Index": ["Rohstoffe"]}))
    for ticker, name in KRYPTO_ETFS.items():
        frames.append(pd.DataFrame({"Ticker": [ticker], "Name": [name], "Index": ["Krypto-ETFs"]}))
    status.append({"Index": "Rohstoffe", "Erwartet": len(ROHSTOFFE), "Geladen": len(ROHSTOFFE),
                   "Quelle": "fest im Code", "Hinweis": ""})
    status.append({"Index": "Krypto-ETFs", "Erwartet": len(KRYPTO_ETFS), "Geladen": len(KRYPTO_ETFS),
                   "Quelle": "fest im Code", "Hinweis": ""})

    alle = pd.concat(frames, ignore_index=True)
    # Ein Titel kann in mehreren Indizes sein -> eine Zeile, Indizes zusammengefasst
    universe = (
        alle.groupby("Ticker", as_index=False)
        .agg(Name=("Name", "first"), Indizes=("Index", lambda s: ", ".join(sorted(set(s)))))
    )
    return universe, pd.DataFrame(status)


# ----------------------------------------------------------------------------
# KURSE LADEN
# ----------------------------------------------------------------------------
def _close_series(df, ticker):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            return df[ticker]["Close"].dropna()
        return df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=REFRESH_MIN * 60, show_spinner="Kurse werden geladen ...")
def load_prices(tickers: tuple):
    result = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = list(tickers[i:i + BATCH_SIZE])
        try:
            df = yf.download(batch, period="5d", interval="1d", group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for t in batch:
            s = _close_series(df, t)
            if len(s) >= 2:
                result[t] = (float(s.iloc[-1]), float(s.iloc[-2]))
            elif len(s) == 1:
                result[t] = (float(s.iloc[-1]), None)
    return result, datetime.now(ZEITZONE).strftime("%d.%m.%Y %H:%M:%S")


# ----------------------------------------------------------------------------
# HISTORIE, SCORE, BACKTEST (laeuft nur alle paar Stunden, nicht bei jedem Refresh)
# ----------------------------------------------------------------------------
REIHENFOLGE = list(INDIZES) + ["Rohstoffe", "Krypto-ETFs"]


def primaere_gruppe(indizes: str) -> str:
    ids = indizes.split(", ")
    return next(g for g in REIHENFOLGE if g in ids)


def _close_frame(df, batch):
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(1):
            return df.xs("Close", axis=1, level=1)
        if "Close" in df.columns.get_level_values(0):
            return df.xs("Close", axis=1, level=0)
        return pd.DataFrame()
    if "Close" in df.columns:
        return df[["Close"]].rename(columns={"Close": batch[0]})
    return pd.DataFrame()


def load_history(tickers):
    frames = []
    for i in range(0, len(tickers), HIST_BATCH):
        batch = list(tickers[i:i + HIST_BATCH])
        df = None
        for _ in range(2):  # ein zweiter Versuch, falls Yahoo kurz hakt
            try:
                df = yf.download(batch, period="5y", interval="1d", group_by="ticker",
                                 auto_adjust=True, progress=False, threads=True)
                break
            except Exception:
                time.sleep(3)
        if df is None or df.empty:
            continue
        frames.append(_close_frame(df, batch))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, axis=1)
    P = P.loc[:, ~P.columns.duplicated()].sort_index()
    if P.index.tz is not None:
        P.index = P.index.tz_localize(None)
    P = P.ffill(limit=5)  # Feiertage unterschiedlicher Boersen ueberbruecken
    return P.dropna(axis=1, how="all")


@st.cache_data(ttl=12 * 3600, show_spinner=(
    "Historie wird geladen und Momentum berechnet. Beim ersten Mal dauert das "
    "etwa 3 bis 6 Minuten, danach ist es 12 Stunden gespeichert ..."))
def compute_all(tickers: tuple, gruppen: tuple):
    P = load_history(tickers)
    if P.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    S, T, r121, r6 = score_frame(P)
    scores = pd.DataFrame({
        "Score": S.iloc[-1],
        "12-1 %": r121.iloc[-1] * 100,
        "6M %": r6.iloc[-1] * 100,
        "Trend": T.iloc[-1].astype(object),
    })
    scores.loc[scores["Score"].isna(), "Trend"] = None

    # Backtest: einmal fuer alle, dazu je Gruppe (Index), falls genug Titel
    gesamt = backtest(P, BACKTEST_TOP, KOSTEN_PRO_RUNDE)
    best_gesamt = best_hold(gesamt)
    gruppen_map = dict(gruppen)
    infos, best_je_gruppe = [], {}
    for g in REIHENFOLGE:
        cols = [t for t in P.columns if gruppen_map.get(t) == g]
        valide = int(S.iloc[-1][cols].notna().sum()) if cols else 0
        if valide >= MIN_GRUPPE:
            tab = backtest(P[cols], BACKTEST_TOP, KOSTEN_PRO_RUNDE)
            b = best_hold(tab)
            quelle = "eigener Backtest"
        else:
            b, quelle = None, ""
        if b is None:
            b, quelle = best_gesamt, "Gesamt-Backtest (zu wenig Titel/Daten in der Gruppe)"
        best_je_gruppe[g] = b
        infos.append({"Gruppe": g, "Titel mit Score": valide,
                      "Beste Haltedauer (Monate)": b, "Grundlage": quelle})
    return scores, gesamt, pd.DataFrame(infos), best_je_gruppe


def signal(rang, n_valid, trend):
    if pd.isna(rang) or n_valid == 0:
        return "n/a"
    anteil = rang / n_valid
    if trend == True:  # noqa: E712 (Trend kann True/False/None sein)
        if anteil <= KAUFEN_TOP:
            return "Kaufen"
        if anteil <= HALTEN_TOP:
            return "Halten"
    return "Verkaufen"


# ----------------------------------------------------------------------------
# SEITE
# ----------------------------------------------------------------------------
st.title("Momentum-Dashboard")

universe, status = load_universe()
universe["Gruppe"] = universe["Indizes"].apply(primaere_gruppe)
alle_ticker = tuple(sorted(universe["Ticker"]))
prices, stand = load_prices(alle_ticker)
scores, bt_gesamt, gruppen_info, best_je_gruppe = compute_all(
    alle_ticker, tuple(sorted(zip(universe["Ticker"], universe["Gruppe"]))))

df = universe.copy()
df["Kurs"] = df["Ticker"].map(lambda t: prices.get(t, (None, None))[0])
vortag = df["Ticker"].map(lambda t: prices.get(t, (None, None))[1])
df["Aenderung %"] = (df["Kurs"] / vortag - 1) * 100

if scores.empty:
    st.error("Die Kurs-Historie konnte nicht geladen werden (Yahoo antwortet nicht). "
             "Bitte in ein paar Minuten neu laden.")
    df["Score"], df["12-1 %"], df["6M %"], df["Trend"] = None, None, None, None
else:
    df = df.merge(scores, left_on="Ticker", right_index=True, how="left")

n_valid = int(df["Score"].notna().sum())
df["Rang"] = df["Score"].rank(ascending=False, method="min")
df["Signal"] = [signal(r, n_valid, t) for r, t in zip(df["Rang"], df["Trend"])]
df["Trend"] = df["Trend"].map({True: "Ja", False: "Nein"}).fillna("")
df["Haltedauer"] = [
    (f"ca. {best_je_gruppe.get(g)} Monate" if s in ("Kaufen", "Halten") and best_je_gruppe.get(g) else "-")
    for g, s in zip(df["Gruppe"], df["Signal"])
]

st.caption(
    f"Stand der Kurse: {stand} (Zeitzone Zuerich). Kurse alle {REFRESH_MIN} Minuten neu, "
    "Gratis-Daten sind ca. 15 Minuten verzoegert, Kurse in Landeswaehrung. "
    "Score und Signal beruhen auf Tagesschlusskursen und werden alle 12 Stunden neu berechnet."
)

with st.sidebar:
    st.header("Filter")
    alle_indizes = sorted({i for s in df["Indizes"] for i in s.split(", ")})
    auswahl = st.multiselect("Index", alle_indizes, default=alle_indizes)
    signale = st.multiselect("Signal", ["Kaufen", "Halten", "Verkaufen", "n/a"],
                             default=["Kaufen", "Halten", "Verkaufen", "n/a"])
    suche = st.text_input("Suche (Name oder Ticker)")
    sortierung = st.radio("Sortieren nach", ["Rang (Momentum)", "Tagesaenderung"])

mask = df["Indizes"].apply(lambda s: any(i in auswahl for i in s.split(", "))) & df["Signal"].isin(signale)
if suche:
    q = suche.lower()
    mask &= df["Name"].str.lower().str.contains(q, regex=False) | df["Ticker"].str.lower().str.contains(q, regex=False)
if sortierung.startswith("Rang"):
    view = df[mask].sort_values("Rang", na_position="last")
else:
    view = df[mask].sort_values("Aenderung %", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Titel im Universum", len(df))
c2.metric("Mit Kurs / mit Score", f"{int(df['Kurs'].notna().sum())} / {n_valid}")
c3.metric("Signal Kaufen", int((df["Signal"] == "Kaufen").sum()))
c4.metric("Angezeigt", len(view))

st.dataframe(
    view[["Rang", "Ticker", "Name", "Indizes", "Kurs", "Aenderung %", "Score",
          "12-1 %", "6M %", "Trend", "Signal", "Haltedauer"]],
    width="stretch", hide_index=True, height=650,
    column_config={
        "Rang": st.column_config.NumberColumn(format="%d"),
        "Kurs": st.column_config.NumberColumn(format="%.2f"),
        "Aenderung %": st.column_config.NumberColumn(format="%+.2f %%"),
        "Score": st.column_config.NumberColumn(format="%.2f"),
        "12-1 %": st.column_config.NumberColumn(format="%+.1f %%"),
        "6M %": st.column_config.NumberColumn(format="%+.1f %%"),
    },
)

with st.expander("Wie entstehen Score, Signal und Haltedauer?"):
    st.markdown(
        f"""
**Score:** Mittelwert aus der 12-1-Monats-Rendite (letzte 12 Monate ohne den letzten Monat) und der
6-Monats-Rendite, geteilt durch die jaehrliche Schwankung. Hoeher = staerkerer, ruhigerer Aufwaertstrend.

**Signal:** Kaufen = oberste {KAUFEN_TOP:.0%} und Kurs ueber dem 200-Tage-Schnitt. Halten = oberste {HALTEN_TOP:.0%}
und Trend ok. Alles andere = Verkaufen (das ist auch die Ausstiegsregel). n/a = zu wenig Kurs-Historie.

**Haltedauer:** Ergebnis eines Backtests ueber die letzten 5 Jahre: Welche Haltedauer ({", ".join(map(str, HOLD_MONTHS))} Monate)
brachte bei den staerksten {BACKTEST_TOP:.0%} nach Kosten ({KOSTEN_PRO_RUNDE:.1%} je Kauf+Verkauf) die beste Rendite je Monat?
Das ist eine Schaetzung je Gruppe (Index), keine Vorhersage fuer die einzelne Aktie.

**Grenzen:** Der Backtest nutzt nur die heutigen Index-Mitglieder (zu optimistisch, "Survivorship Bias"),
5 Jahre sind kurz, die Zeitraeume ueberlappen, Steuern und Waehrungseffekte fehlen. Analysewerkzeug, keine Anlageberatung.
"""
    )

with st.expander("Backtest-Ergebnis im Detail"):
    if bt_gesamt.empty:
        st.write("Noch keine Backtest-Daten.")
    else:
        st.subheader("Alle Titel zusammen")
        zeigen = bt_gesamt.copy()
        for spalte in ("Rendite je Zeitraum (netto)", "Rendite je Monat (netto)", "Trefferquote"):
            zeigen[spalte] = (zeigen[spalte] * 100).round(2)
        zeigen = zeigen.rename(columns={
            "Rendite je Zeitraum (netto)": "Rendite je Zeitraum netto (%)",
            "Rendite je Monat (netto)": "Rendite je Monat netto (%)",
            "Trefferquote": "Trefferquote (%)"})
        st.dataframe(zeigen, width="stretch", hide_index=True)
        st.subheader("Je Gruppe")
        st.dataframe(gruppen_info, width="stretch", hide_index=True)

with st.expander("Datenstatus der Listen (hier siehst du, was fehlt)"):
    st.dataframe(status, width="stretch", hide_index=True)
    st.caption("Fehlt ein Index (z. B. Nikkei 225), lege im Ordner 'listen' eine CSV mit den Spalten "
               "ticker,name an. Japanische Codes (z. B. 7203) bekommen automatisch '.T' dazu.")
