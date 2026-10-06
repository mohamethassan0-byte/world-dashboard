"""
🌍 World Today — a one-file Streamlit dashboard.
Only free, no-API-key sources. Deploy on Streamlit Community Cloud with app.py + requirements.txt.

Sources: RSS (BBC, Al Jazeera, NPR, Guardian, DW, France24), USGS earthquakes, NASA EONET,
Open-Meteo weather, CoinGecko, open.er-api.com FX, Yahoo Finance (yfinance), wheretheiss.at,
Open Notify, NASA APOD, Wikimedia pageviews, Hacker News (Algolia).
"""

import calendar
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import feedparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="World Today", page_icon="🌍", layout="wide")

HEADERS = {"User-Agent": "WorldTodayDashboard/1.0 (Streamlit Community Cloud)"}
ERRORS: list[str] = []

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.2);
        padding: 12px 16px; border-radius: 12px;
    }
    .src-tag {font-size: 0.75rem; opacity: 0.65;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe(fn, default, *args):
    """Run a loader; on failure record the error and return a default."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        ERRORS.append(f"{fn.__name__}: {type(e).__name__}: {e}")
        return default


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def fmt_big(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:,.0f}"


# ----------------------------------------------------------------------------
# Data loaders (cached)
# ----------------------------------------------------------------------------
NEWS_FEEDS = {
    "World": {
        "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "NPR": "https://feeds.npr.org/1004/rss.xml",
        "The Guardian": "https://www.theguardian.com/world/rss",
        "DW": "https://rss.dw.com/rdf/rss-en-all",
        "France 24": "https://www.france24.com/en/rss",
    },
    "Business": {
        "BBC": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "NPR": "https://feeds.npr.org/1006/rss.xml",
        "The Guardian": "https://www.theguardian.com/business/rss",
    },
    "Technology": {
        "BBC": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "NPR": "https://feeds.npr.org/1019/rss.xml",
        "The Guardian": "https://www.theguardian.com/technology/rss",
    },
    "Science & Environment": {
        "BBC Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "NPR Science": "https://feeds.npr.org/1007/rss.xml",
        "The Guardian": "https://www.theguardian.com/science/rss",
    },
    "Health": {
        "BBC": "http://feeds.bbci.co.uk/news/health/rss.xml",
        "NPR": "https://feeds.npr.org/1128/rss.xml",
    },
}


def _fetch_feed(source: str, url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
    except Exception:  # noqa: BLE001
        return []
    items = []
    for e in parsed.entries[:25]:
        ts = e.get("published_parsed") or e.get("updated_parsed")
        dt = datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc) if ts else None
        items.append(
            {
                "source": source,
                "title": strip_html(e.get("title", "")),
                "link": e.get("link", ""),
                "summary": strip_html(e.get("summary", ""))[:280],
                "published": dt,
            }
        )
    return items


@st.cache_data(ttl=600, show_spinner=False)
def load_news(category: str, sources: tuple) -> pd.DataFrame:
    feeds = {s: u for s, u in NEWS_FEEDS[category].items() if s in sources}
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda kv: _fetch_feed(*kv), feeds.items()))
    rows = [item for res in results for item in res]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="title")
    return df.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_quakes() -> pd.DataFrame:
    data = get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson")
    rows = []
    for f in data["features"]:
        p, (lon, lat, depth) = f["properties"], f["geometry"]["coordinates"]
        rows.append(
            {
                "time": datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc),
                "magnitude": p["mag"],
                "place": p["place"],
                "depth_km": depth,
                "lat": lat,
                "lon": lon,
                "tsunami": bool(p.get("tsunami")),
                "url": p.get("url"),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("time", ascending=False).reset_index(drop=True) if not df.empty else df


@st.cache_data(ttl=900, show_spinner=False)
def load_eonet() -> pd.DataFrame:
    data = get_json("https://eonet.gsfc.nasa.gov/api/v3/events", {"status": "open", "limit": 100})
    rows = []
    for ev in data.get("events", []):
        geoms = ev.get("geometry", [])
        if not geoms:
            continue
        g = geoms[-1]
        coords = g.get("coordinates")
        if g.get("type") != "Point" or not coords or len(coords) < 2:
            continue
        rows.append(
            {
                "event": ev["title"],
                "category": ev["categories"][0]["title"] if ev.get("categories") else "Other",
                "date": g.get("date", "")[:10],
                "lon": coords[0],
                "lat": coords[1],
                "link": ev.get("link"),
            }
        )
    return pd.DataFrame(rows)


CITIES = {
    "New York": (40.71, -74.01), "London": (51.51, -0.13), "Paris": (48.86, 2.35),
    "Berlin": (52.52, 13.40), "Moscow": (55.76, 37.62), "Cairo": (30.04, 31.24),
    "Lagos": (6.52, 3.38), "Nairobi": (-1.29, 36.82), "Johannesburg": (-26.20, 28.05),
    "Dubai": (25.20, 55.27), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
    "Bangkok": (13.76, 100.50), "Kuala Lumpur": (3.14, 101.69), "Singapore": (1.35, 103.82),
    "Beijing": (39.90, 116.41), "Shanghai": (31.23, 121.47), "Tokyo": (35.68, 139.69),
    "Seoul": (37.57, 126.98), "Sydney": (-33.87, 151.21), "Auckland": (-36.85, 174.76),
    "Los Angeles": (34.05, -118.24), "Mexico City": (19.43, -99.13), "São Paulo": (-23.55, -46.63),
    "Buenos Aires": (-34.60, -58.38), "Reykjavik": (64.15, -21.94),
}


def weather_label(code: int) -> str:
    if code == 0: return "☀️ Clear"
    if code in (1, 2): return "🌤️ Mostly clear"
    if code == 3: return "☁️ Overcast"
    if code in (45, 48): return "🌫️ Fog"
    if 51 <= code <= 57: return "🌦️ Drizzle"
    if 61 <= code <= 67: return "🌧️ Rain"
    if 71 <= code <= 77: return "❄️ Snow"
    if 80 <= code <= 82: return "🌧️ Showers"
    if code in (85, 86): return "🌨️ Snow showers"
    if code >= 95: return "⛈️ Thunderstorm"
    return "—"


@st.cache_data(ttl=900, show_spinner=False)
def load_weather() -> pd.DataFrame:
    names = list(CITIES)
    data = get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": ",".join(str(CITIES[n][0]) for n in names),
            "longitude": ",".join(str(CITIES[n][1]) for n in names),
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        },
    )
    if isinstance(data, dict):
        data = [data]
    rows = []
    for n, d in zip(names, data):
        c = d["current"]
        rows.append(
            {
                "city": n,
                "lat": CITIES[n][0],
                "lon": CITIES[n][1],
                "temp_c": c["temperature_2m"],
                "humidity_%": c["relative_humidity_2m"],
                "wind_kmh": c["wind_speed_10m"],
                "conditions": weather_label(int(c["weather_code"])),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def load_crypto() -> pd.DataFrame:
    data = get_json(
        "https://api.coingecko.com/api/v3/coins/markets",
        {
            "vs_currency": "usd", "order": "market_cap_desc", "per_page": 15, "page": 1,
            "sparkline": "false", "price_change_percentage": "24h",
        },
    )
    df = pd.DataFrame(data)
    return df[["name", "symbol", "current_price", "price_change_percentage_24h", "market_cap", "total_volume"]]


@st.cache_data(ttl=3600, show_spinner=False)
def load_fx(base: str) -> pd.Series:
    data = get_json(f"https://open.er-api.com/v6/latest/{base}")
    return pd.Series(data["rates"]).sort_index()


MARKETS = {
    "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC", "FTSE 100": "^FTSE",
    "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI", "Gold": "GC=F",
    "Crude Oil (WTI)": "CL=F", "US 10Y Yield": "^TNX",
}


@st.cache_data(ttl=600, show_spinner=False)
def load_markets():
    import yfinance as yf

    def one(item):
        name, sym = item
        try:
            s = yf.Ticker(sym).history(period="1mo")["Close"].dropna()
            return name, sym, s
        except Exception:  # noqa: BLE001
            return name, sym, pd.Series(dtype=float)

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one, MARKETS.items()))

    rows, hist = [], {}
    for name, sym, s in results:
        if len(s) < 2:
            continue
        rows.append(
            {
                "name": name, "symbol": sym, "price": float(s.iloc[-1]),
                "change_pct": float((s.iloc[-1] / s.iloc[-2] - 1) * 100),
                "month_pct": float((s.iloc[-1] / s.iloc[0] - 1) * 100),
            }
        )
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        hist[name] = (s / s.iloc[0] - 1) * 100
    return pd.DataFrame(rows), pd.DataFrame(hist)


@st.cache_data(ttl=10, show_spinner=False)
def load_iss() -> dict:
    return get_json("https://api.wheretheiss.at/v1/satellites/25544")


@st.cache_data(ttl=3600, show_spinner=False)
def load_astronauts() -> pd.DataFrame:
    data = get_json("http://api.open-notify.org/astros.json")
    return pd.DataFrame(data["people"]).rename(columns={"craft": "spacecraft", "name": "astronaut"})


@st.cache_data(ttl=3600, show_spinner=False)
def load_apod() -> dict:
    return get_json("https://api.nasa.gov/planetary/apod", {"api_key": "DEMO_KEY"})


@st.cache_data(ttl=3600, show_spinner=False)
def load_wikipedia_top() -> tuple[pd.DataFrame, str]:
    skip_prefixes = ("Special:", "Wikipedia:", "Portal:", "Help:", "File:", "Category:", "Talk:", "Template:")
    for back in (1, 2, 3):
        d = datetime.now(timezone.utc) - timedelta(days=back)
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/"
            f"{d:%Y/%m/%d}"
        )
        try:
            items = get_json(url)["items"][0]["articles"]
        except Exception:  # noqa: BLE001
            continue
        rows = [
            {"article": a["article"].replace("_", " "), "views": a["views"],
             "url": "https://en.wikipedia.org/wiki/" + a["article"]}
            for a in items
            if a["article"] != "Main_Page" and not a["article"].startswith(skip_prefixes) and a["article"] != "-"
        ][:25]
        return pd.DataFrame(rows), f"{d:%Y-%m-%d}"
    raise RuntimeError("Wikimedia top pages unavailable")


@st.cache_data(ttl=600, show_spinner=False)
def load_hn() -> pd.DataFrame:
    data = get_json("https://hn.algolia.com/api/v1/search", {"tags": "front_page", "hitsPerPage": 20})
    rows = []
    for h in data["hits"]:
        rows.append(
            {
                "title": h.get("title"),
                "points": h.get("points", 0),
                "comments": h.get("num_comments", 0),
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🌍 World Today")
now_utc = datetime.now(timezone.utc)
st.sidebar.caption(f"Loaded: {now_utc:%a %d %b %Y, %H:%M UTC}")
if st.sidebar.button("🔄 Refresh all data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data refreshes automatically on a timer (news 10 min, quakes 5 min, markets 10 min, "
    "weather 15 min). Press the button above to force a refresh."
)
error_box = st.sidebar.empty()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌍 What's happening in the world today")
st.caption(f"{now_utc:%A, %d %B %Y} · live data from open public sources")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_over, tab_news, tab_mkts, tab_wx, tab_earth, tab_space, tab_trend = st.tabs(
    ["🏠 Overview", "📰 News", "📈 Markets", "🌦️ Weather", "🌋 Earth & Events", "🚀 Space", "🔥 Trending"]
)

# ----------------------------- Overview -------------------------------------
with tab_over:
    quakes = safe(load_quakes, pd.DataFrame())
    events = safe(load_eonet, pd.DataFrame())
    crypto = safe(load_crypto, pd.DataFrame())
    mk_df, mk_hist = safe(load_markets, (pd.DataFrame(), pd.DataFrame()))

    c1, c2, c3, c4, c5 = st.columns(5)
    if not quakes.empty:
        c1.metric("Earthquakes M2.5+ (24h)", len(quakes))
        top = quakes.loc[quakes["magnitude"].idxmax()]
        c2.metric("Strongest quake", f"M{top['magnitude']:.1f}", top["place"], delta_color="off")
    if not events.empty:
        c3.metric("Active natural events", len(events))
    if not crypto.empty:
        btc = crypto[crypto["symbol"] == "btc"]
        if not btc.empty:
            b = btc.iloc[0]
            c4.metric("Bitcoin", f"${b['current_price']:,.0f}", f"{b['price_change_percentage_24h']:.2f}%")
    if not mk_df.empty:
        sp = mk_df[mk_df["name"] == "S&P 500"]
        if not sp.empty:
            s = sp.iloc[0]
            c5.metric("S&P 500", f"{s['price']:,.0f}", f"{s['change_pct']:.2f}%")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Top world headlines")
        news = safe(load_news, pd.DataFrame(), "World", ("BBC", "Al Jazeera", "NPR", "The Guardian"))
        if news.empty:
            st.info("Headlines unavailable right now.")
        else:
            for _, r in news.head(8).iterrows():
                st.markdown(f"**[{r['title']}]({r['link']})**  \n<span class='src-tag'>{r['source']}</span>",
                            unsafe_allow_html=True)
    with right:
        st.subheader("Earthquakes — last 24h")
        if not quakes.empty:
            fig = px.scatter_geo(
                quakes, lat="lat", lon="lon", size="magnitude", color="magnitude",
                hover_name="place", color_continuous_scale="YlOrRd", projection="natural earth",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=320)
            st.plotly_chart(fig, use_container_width=True)

    if not mk_hist.empty:
        st.subheader("Markets — 1 month performance (%)")
        fig = px.line(mk_hist, labels={"value": "% change", "index": "", "variable": ""})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------- News -----------------------------------------
with tab_news:
    f1, f2, f3 = st.columns([1, 2, 2])
    category = f1.selectbox("Category", list(NEWS_FEEDS))
    available = list(NEWS_FEEDS[category])
    chosen = f2.multiselect("Sources", available, default=available)
    query = f3.text_input("Filter by keyword", placeholder="e.g. climate, election, AI")

    if not chosen:
        st.info("Pick at least one source.")
    else:
        with st.spinner("Fetching headlines…"):
            df = safe(load_news, pd.DataFrame(), category, tuple(chosen))
        if df.empty:
            st.warning("No articles could be loaded.")
        else:
            if query:
                mask = df["title"].str.contains(query, case=False, na=False) | df["summary"].str.contains(
                    query, case=False, na=False
                )
                df = df[mask]
            st.caption(f"{len(df)} articles")
            for _, r in df.head(60).iterrows():
                when = r["published"].strftime("%d %b %H:%M UTC") if pd.notna(r["published"]) else ""
                st.markdown(f"#### [{r['title']}]({r['link']})")
                st.markdown(f"<span class='src-tag'>{r['source']} · {when}</span>", unsafe_allow_html=True)
                if r["summary"]:
                    st.write(r["summary"])
                st.divider()

# ----------------------------- Markets --------------------------------------
with tab_mkts:
    mk_df, mk_hist = safe(load_markets, (pd.DataFrame(), pd.DataFrame()))
    st.subheader("Indices, commodities & rates")
    if mk_df.empty:
        st.warning("Market data unavailable (Yahoo Finance may be rate-limiting). Try refreshing in a minute.")
    else:
        cols = st.columns(5)
        for i, r in mk_df.iterrows():
            cols[i % 5].metric(r["name"], f"{r['price']:,.2f}", f"{r['change_pct']:+.2f}%")
        fig = px.bar(
            mk_df.sort_values("month_pct"), x="month_pct", y="name", orientation="h",
            color="month_pct", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            labels={"month_pct": "1-month change (%)", "name": ""},
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top cryptocurrencies")
    crypto = safe(load_crypto, pd.DataFrame())
    if crypto.empty:
        st.warning("Crypto data unavailable (CoinGecko free tier may be rate-limiting).")
    else:
        st.dataframe(
            crypto.rename(columns={
                "name": "Coin", "symbol": "Ticker", "current_price": "Price (USD)",
                "price_change_percentage_24h": "24h %", "market_cap": "Market cap", "total_volume": "24h volume",
            }),
            hide_index=True, use_container_width=True,
            column_config={
                "Price (USD)": st.column_config.NumberColumn(format="$%.4f"),
                "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                "Market cap": st.column_config.NumberColumn(format="$%d"),
                "24h volume": st.column_config.NumberColumn(format="$%d"),
            },
        )

    st.subheader("Currency exchange rates")
    base = st.selectbox("Base currency", ["USD", "EUR", "GBP", "JPY", "MYR", "SGD", "AUD", "CAD", "CNY", "INR"])
    fx = safe(load_fx, pd.Series(dtype=float), base)
    if fx.empty:
        st.warning("FX data unavailable.")
    else:
        majors = [c for c in ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD", "CHF", "SGD", "MYR",
                              "KRW", "BRL", "MXN", "ZAR", "AED", "TRY", "THB"] if c != base and c in fx.index]
        cols = st.columns(6)
        for i, c in enumerate(majors):
            cols[i % 6].metric(f"1 {base} →", f"{fx[c]:,.4f}", c, delta_color="off")
        with st.expander("All rates"):
            st.dataframe(fx.rename(f"per 1 {base}").to_frame(), use_container_width=True)

# ----------------------------- Weather --------------------------------------
with tab_wx:
    wx = safe(load_weather, pd.DataFrame())
    if wx.empty:
        st.warning("Weather data unavailable.")
    else:
        st.subheader("Current temperatures around the world (°C)")
        fig = px.scatter_geo(
            wx, lat="lat", lon="lon", color="temp_c", size=[14] * len(wx), hover_name="city",
            hover_data={"temp_c": True, "conditions": True, "wind_kmh": True, "lat": False, "lon": False},
            color_continuous_scale="RdYlBu_r", projection="natural earth", range_color=(-10, 40),
        )
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        hot, cold = wx.loc[wx["temp_c"].idxmax()], wx.loc[wx["temp_c"].idxmin()]
        m1, m2, m3 = st.columns(3)
        m1.metric("Hottest city", hot["city"], f"{hot['temp_c']:.1f}°C", delta_color="off")
        m2.metric("Coldest city", cold["city"], f"{cold['temp_c']:.1f}°C", delta_color="off")
        windy = wx.loc[wx["wind_kmh"].idxmax()]
        m3.metric("Windiest city", windy["city"], f"{windy['wind_kmh']:.0f} km/h", delta_color="off")

        st.dataframe(
            wx[["city", "conditions", "temp_c", "humidity_%", "wind_kmh"]].sort_values("temp_c", ascending=False)
            .rename(columns={"temp_c": "Temp °C", "wind_kmh": "Wind km/h"}),
            hide_index=True, use_container_width=True,
        )

# ----------------------------- Earth & Events -------------------------------
with tab_earth:
    quakes = safe(load_quakes, pd.DataFrame())
    st.subheader("🌋 Earthquakes M2.5+ — past 24 hours (USGS)")
    if quakes.empty:
        st.warning("Earthquake data unavailable.")
    else:
        min_mag = st.slider("Minimum magnitude", 2.5, float(max(3.0, quakes["magnitude"].max())), 2.5, 0.1)
        q = quakes[quakes["magnitude"] >= min_mag]
        fig = px.scatter_geo(
            q, lat="lat", lon="lon", size="magnitude", color="magnitude", hover_name="place",
            hover_data={"depth_km": ":.0f", "lat": False, "lon": False},
            color_continuous_scale="YlOrRd", projection="natural earth",
        )
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        qd = q[["time", "magnitude", "place", "depth_km", "tsunami", "url"]].copy()
        qd["time"] = qd["time"].dt.strftime("%d %b %H:%M UTC")
        st.dataframe(
            qd, hide_index=True, use_container_width=True,
            column_config={"url": st.column_config.LinkColumn("Details", display_text="open")},
        )

    st.subheader("🔥 Active natural events (NASA EONET)")
    ev = safe(load_eonet, pd.DataFrame())
    if ev.empty:
        st.warning("Natural event data unavailable.")
    else:
        cats = sorted(ev["category"].unique())
        pick = st.multiselect("Event types", cats, default=cats)
        e = ev[ev["category"].isin(pick)]
        l, r = st.columns([3, 2])
        with l:
            fig = px.scatter_geo(e, lat="lat", lon="lon", color="category", hover_name="event",
                                 projection="natural earth")
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with r:
            counts = e["category"].value_counts().reset_index()
            counts.columns = ["category", "count"]
            fig = px.bar(counts, x="count", y="category", orientation="h")
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            e[["event", "category", "date", "link"]].sort_values("date", ascending=False),
            hide_index=True, use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("Source", display_text="open")},
        )

# ----------------------------- Space ----------------------------------------
with tab_space:
    l, r = st.columns([3, 2])
    with l:
        st.subheader("🛰️ International Space Station — live")
        iss = safe(load_iss, None)
        if iss:
            fig = go.Figure(
                go.Scattergeo(lat=[iss["latitude"]], lon=[iss["longitude"]], mode="markers+text",
                              text=["ISS"], textposition="top center",
                              marker=dict(size=14, color="red", symbol="diamond"))
            )
            fig.update_geos(projection_type="natural earth", showland=True, showocean=True,
                            landcolor="rgb(70,110,70)", oceancolor="rgb(25,60,110)")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("Altitude", f"{iss['altitude']:.0f} km")
            b.metric("Speed", f"{iss['velocity']:,.0f} km/h")
            c.metric("Visibility", str(iss.get("visibility", "—")).title())
        else:
            st.warning("ISS position unavailable.")
    with r:
        st.subheader("👩‍🚀 People in space now")
        ppl = safe(load_astronauts, pd.DataFrame())
        if ppl.empty:
            st.info("Crew data unavailable.")
        else:
            st.metric("Total in orbit", len(ppl))
            st.dataframe(ppl, hide_index=True, use_container_width=True)

    st.subheader("🌌 NASA Astronomy Picture of the Day")
    apod = safe(load_apod, None)
    if apod:
        st.markdown(f"**{apod.get('title', '')}** · {apod.get('date', '')}")
        if apod.get("media_type") == "image":
            st.image(apod.get("url"), use_container_width=True)
        elif apod.get("url"):
            st.video(apod["url"])
        st.write(apod.get("explanation", ""))
    else:
        st.info("APOD unavailable (the shared DEMO_KEY is rate-limited). Try again later.")

# ----------------------------- Trending -------------------------------------
with tab_trend:
    l, r = st.columns(2)
    with l:
        st.subheader("📚 Most-read on Wikipedia")
        res = safe(load_wikipedia_top, (pd.DataFrame(), ""))
        wk, day = res
        if wk.empty:
            st.info("Wikipedia trends unavailable.")
        else:
            st.caption(f"Top English Wikipedia articles on {day}")
            fig = px.bar(wk.head(15).iloc[::-1], x="views", y="article", orientation="h")
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                wk, hide_index=True, use_container_width=True,
                column_config={"url": st.column_config.LinkColumn("Link", display_text="open"),
                               "views": st.column_config.NumberColumn(format="%d")},
            )
    with r:
        st.subheader("💻 Hacker News front page")
        hn = safe(load_hn, pd.DataFrame())
        if hn.empty:
            st.info("Hacker News unavailable.")
        else:
            for _, row in hn.iterrows():
                st.markdown(f"**[{row['title']}]({row['url']})**  \n"
                            f"<span class='src-tag'>▲ {row['points']} · 💬 {row['comments']}</span>",
                            unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Footer / diagnostics
# ----------------------------------------------------------------------------
if ERRORS:
    with error_box.expander(f"⚠️ {len(ERRORS)} source(s) failed"):
        for e in ERRORS:
            st.caption(e)
else:
    error_box.success("All sources loaded")

st.markdown("---")
st.caption(
    "Sources: BBC, Al Jazeera, NPR, The Guardian, DW, France 24 · USGS · NASA EONET & APOD · Open-Meteo · "
    "CoinGecko · open.er-api.com · Yahoo Finance · wheretheiss.at · Open Notify · Wikimedia · Hacker News. "
    "For information only — not financial advice."
)