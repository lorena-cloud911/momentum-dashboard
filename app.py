"""
Momentum-Dashboard - Schritt 1
Universum (Indizes, Rohstoffe, Krypto-ETFs) laden und Kurse automatisch anzeigen.

Starten lokal:   streamlit run app.py
Aenderungen:     Einstellungen ganz oben (REFRESH_MIN, INDIZES, ROHSTOFFE, KRYPTO_ETFS)
"""

import io
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------------------------------
# EINSTELLUNGEN (hier aenderst du spaeter am meisten)
# ----------------------------------------------------------------------------
REFRESH_MIN = 15  # Kurse alle X Minuten neu laden (10 bis 20 empfohlen)
BATCH_SIZE = 150  # wie viele Ticker pro Abruf bei Yahoo

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
# SEITE
# ----------------------------------------------------------------------------
st.title("Momentum-Dashboard")

universe, status = load_universe()
prices, stand = load_prices(tuple(sorted(universe["Ticker"])))

df = universe.copy()
df["Kurs"] = df["Ticker"].map(lambda t: prices.get(t, (None, None))[0])
vortag = df["Ticker"].map(lambda t: prices.get(t, (None, None))[1])
df["Aenderung %"] = (df["Kurs"] / vortag - 1) * 100

st.caption(
    f"Stand der Kurse: {stand} (Zeitzone Zuerich). Aktualisierung alle {REFRESH_MIN} Minuten. "
    "Gratis-Daten sind ca. 15 Minuten verzoegert. Kurse in Landeswaehrung."
)

with st.sidebar:
    st.header("Filter")
    alle_indizes = sorted({i for s in df["Indizes"] for i in s.split(", ")})
    auswahl = st.multiselect("Index", alle_indizes, default=alle_indizes)
    suche = st.text_input("Suche (Name oder Ticker)")

mask = df["Indizes"].apply(lambda s: any(i in auswahl for i in s.split(", ")))
if suche:
    q = suche.lower()
    mask &= df["Name"].str.lower().str.contains(q, regex=False) | df["Ticker"].str.lower().str.contains(q, regex=False)
view = df[mask].sort_values("Aenderung %", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("Titel im Universum", len(df))
c2.metric("Davon mit Kurs", int(df["Kurs"].notna().sum()))
c3.metric("Angezeigt", len(view))

st.dataframe(
    view[["Ticker", "Name", "Indizes", "Kurs", "Aenderung %"]],
    use_container_width=True, hide_index=True, height=650,
    column_config={
        "Kurs": st.column_config.NumberColumn(format="%.2f"),
        "Aenderung %": st.column_config.NumberColumn(format="%+.2f %%"),
    },
)

with st.expander("Datenstatus der Listen (hier siehst du, was fehlt)"):
    st.dataframe(status, use_container_width=True, hide_index=True)
    st.caption("Fehlt ein Index (z. B. Nikkei 225), lege im Ordner 'listen' eine CSV mit den Spalten "
               "ticker,name an. Japanische Codes (z. B. 7203) bekommen automatisch '.T' dazu.")
