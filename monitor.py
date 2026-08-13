"""
Watchlist 50 — weekly price monitor.

Reads tickers from Notion, pulls daily closes, computes drawdown relative to a
sector benchmark, and emails a short digest of names worth a second look.

Deliberately dumb: it knows tickers and benchmarks and nothing else. It does not
read thesis pages, does not call an LLM, and does not judge anything. All
judgment happens when you open the name.
"""

import json
import os
import smtplib
import sys
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------- config

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DB_ID = os.environ["NOTION_DB_ID"]
NOTION_VERSION = "2022-06-28"

# Gmail's server never changes, so it stays here rather than in a secret.
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_USER = os.environ["EMAIL_USER"]   # your Gmail address
EMAIL_PASS = os.environ["EMAIL_PASS"]   # 16-character Gmail app password
EMAIL_TO = os.environ["EMAIL_TO"]

STATE_PATH = Path("alert_state.json")

# Trigger thresholds
EXCESS_3M_TRIGGER = -0.15      # 3-month return vs benchmark, in percentage points
SHOCK_TRIGGER = -0.10          # single-day move vs benchmark
DEPTH_TRIGGER = -0.30          # drawdown from 52-week high
DEPTH_BENCH_MAX = -0.10        # only if the benchmark has NOT fallen this far

COOLDOWN_DAYS = 90
MAX_DIGEST_ROWS = 5


# ---------------------------------------------------------------- notion

def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def plain_text(prop):
    """Pull a plain string out of a Notion rich_text or title property."""
    if not prop:
        return ""
    items = prop.get("rich_text") or prop.get("title") or []
    return "".join(i.get("plain_text", "") for i in items).strip()


def fetch_watchlist():
    """Return every row that is not marked Dropped."""
    url = f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query"
    rows, cursor = [], None

    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        resp = requests.post(url, headers=notion_headers(), json=body, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        for page in payload["results"]:
            props = page["properties"]
            status_prop = props.get("Status", {}).get("select")
            status = status_prop["name"] if status_prop else ""
            if status == "Dropped":
                continue

            ticker = plain_text(props.get("Ticker"))
            if not ticker:
                continue

            rows.append({
                "page_id": page["id"],
                "company": plain_text(props.get("Company")),
                "ticker_raw": ticker,
                "symbol": to_yahoo_symbol(ticker),
                "benchmark": plain_text(props.get("Sector ETF")) or "SPY",
                "status": status,
            })

        if not payload.get("has_more"):
            break
        cursor = payload["next_cursor"]

    return rows


def write_prices(row, metrics):
    """Push price and trailing returns back onto the Notion row."""
    props = {
        "Price": {"number": round(metrics["price"], 2)},
        "3M Return": {"number": metrics["ret_3m"]},
        "6M Return": {"number": metrics["ret_6m"]},
        "12M Return": {"number": metrics["ret_12m"]},
    }
    # Notion rejects NaN; drop anything we could not compute.
    props = {k: v for k, v in props.items() if v["number"] is not None}

    resp = requests.patch(
        f"https://api.notion.com/v1/pages/{row['page_id']}",
        headers=notion_headers(),
        json={"properties": props},
        timeout=30,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------- symbols

def to_yahoo_symbol(ticker):
    """
    Normalise the ticker as typed in Notion into what Yahoo expects.

      '7839 (Tokyo)' -> '7839.T'
      'MOG.A'        -> 'MOG-A'
      'RLI'          -> 'RLI'
    """
    t = ticker.strip()
    if "(Tokyo)" in t:
        return t.split("(")[0].strip() + ".T"
    if "." in t and not t.endswith((".T", ".L", ".ST", ".TO")):
        return t.replace(".", "-")
    return t


# ---------------------------------------------------------------- prices

def fetch_closes(symbols):
    """One batched download. Returns a DataFrame of adjusted closes."""
    data = yf.download(
        tickers=list(symbols),
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(data.columns, pd.MultiIndex):
        closes.columns = list(symbols)
    return closes.ffill()


def trailing_return(series, days):
    """Return over the trailing N calendar days, or None if history is short."""
    if series.empty:
        return None
    cutoff = series.index[-1] - timedelta(days=days)
    window = series[series.index <= cutoff]
    if window.empty:
        return None
    start = window.iloc[-1]
    if pd.isna(start) or start == 0:
        return None
    return float(series.iloc[-1] / start - 1)


def compute_metrics(symbol, benchmark, closes):
    if symbol not in closes.columns or benchmark not in closes.columns:
        return None

    px = closes[symbol].dropna()
    bench = closes[benchmark].dropna()
    if len(px) < 30 or len(bench) < 30:
        return None

    ret_3m = trailing_return(px, 91)
    ret_6m = trailing_return(px, 182)
    ret_12m = trailing_return(px, 365)
    bench_3m = trailing_return(bench, 91)

    excess_3m = None
    if ret_3m is not None and bench_3m is not None:
        excess_3m = ret_3m - bench_3m

    # Single-day move relative to the benchmark
    shock = None
    if len(px) >= 2 and len(bench) >= 2:
        px_1d = float(px.iloc[-1] / px.iloc[-2] - 1)
        bench_1d = float(bench.iloc[-1] / bench.iloc[-2] - 1)
        shock = px_1d - bench_1d

    # Drawdown from the 52-week high
    year = px[px.index >= px.index[-1] - timedelta(days=365)]
    drawdown = float(px.iloc[-1] / year.max() - 1) if not year.empty else None

    bench_year = bench[bench.index >= bench.index[-1] - timedelta(days=365)]
    bench_dd = float(bench.iloc[-1] / bench_year.max() - 1) if not bench_year.empty else None

    return {
        "price": float(px.iloc[-1]),
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_12m": ret_12m,
        "excess_3m": excess_3m,
        "shock": shock,
        "drawdown": drawdown,
        "bench_drawdown": bench_dd,
    }


# ---------------------------------------------------------------- triggers

def evaluate(metrics):
    """Return the list of signal names that fired."""
    fired = []

    if metrics["excess_3m"] is not None and metrics["excess_3m"] <= EXCESS_3M_TRIGGER:
        fired.append("Excess drawdown")

    if metrics["shock"] is not None and metrics["shock"] <= SHOCK_TRIGGER:
        fired.append("Shock gap")

    if (
        metrics["drawdown"] is not None
        and metrics["bench_drawdown"] is not None
        and metrics["drawdown"] <= DEPTH_TRIGGER
        and metrics["bench_drawdown"] > DEPTH_BENCH_MAX
    ):
        fired.append("Depth")

    return fired


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def in_cooldown(state, ticker, today):
    last = state.get(ticker)
    if not last:
        return False
    return (today - date.fromisoformat(last)).days < COOLDOWN_DAYS


def prune_state(state, today):
    """Drop entries past their cooldown. They no longer suppress anything."""
    return {
        ticker: stamped
        for ticker, stamped in state.items()
        if (today - date.fromisoformat(stamped)).days < COOLDOWN_DAYS
    }


# ---------------------------------------------------------------- email

def pct(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def build_email(alerts, checked, errors):
    rows = "".join(
        f"""<tr>
              <td style="padding:8px 12px;border-bottom:1px solid #eee">
                <strong>{a['company']}</strong><br>
                <span style="color:#888;font-size:12px">{a['ticker_raw']} · vs {a['benchmark']}</span>
              </td>
              <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:right">${a['price']:.2f}</td>
              <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:right">{pct(a['ret_3m'])}</td>
              <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:#c00">
                <strong>{pct(a['excess_3m'])}</strong>
              </td>
              <td style="padding:8px 12px;border-bottom:1px solid #eee;font-size:12px">{', '.join(a['signals'])}</td>
            </tr>"""
        for a in alerts
    )

    if alerts:
        body = f"""
        <p style="font-family:system-ui,sans-serif">
          {len(alerts)} name{'s' if len(alerts) != 1 else ''} worth opening this week.
          Read the thesis-breaker note before anything else.
        </p>
        <table style="border-collapse:collapse;font-family:system-ui,sans-serif;font-size:14px">
          <tr style="text-align:left;color:#666;font-size:12px">
            <th style="padding:8px 12px">Company</th>
            <th style="padding:8px 12px;text-align:right">Price</th>
            <th style="padding:8px 12px;text-align:right">3M</th>
            <th style="padding:8px 12px;text-align:right">vs sector</th>
            <th style="padding:8px 12px">Signal</th>
          </tr>
          {rows}
        </table>"""
    else:
        body = """<p style="font-family:system-ui,sans-serif">
                    No names triggered this week.
                  </p>"""

    footer = f"""<p style="font-family:system-ui,sans-serif;color:#888;font-size:12px;margin-top:24px">
                   {checked} names checked."""
    if errors:
        footer += "<br>Could not price: " + ", ".join(errors)
    footer += "</p>"

    return body + footer


def send_email(subject, html):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


# ---------------------------------------------------------------- main

def main():
    today = date.today()
    state = load_state()

    watchlist = fetch_watchlist()
    if not watchlist:
        print("No rows returned from Notion. Check the integration connection.")
        sys.exit(1)

    symbols = {r["symbol"] for r in watchlist} | {r["benchmark"] for r in watchlist}
    closes = fetch_closes(symbols)

    alerts, errors = [], []

    for row in watchlist:
        metrics = compute_metrics(row["symbol"], row["benchmark"], closes)
        if metrics is None:
            errors.append(row["ticker_raw"])
            continue

        try:
            write_prices(row, metrics)
        except Exception as exc:
            print(f"Notion write failed for {row['ticker_raw']}: {exc}")

        signals = evaluate(metrics)
        if signals and not in_cooldown(state, row["ticker_raw"], today):
            alerts.append({**row, **metrics, "signals": signals})

    # Deepest relative decline first, then cap the digest.
    alerts.sort(key=lambda a: a["excess_3m"] if a["excess_3m"] is not None else 0)
    shown = alerts[:MAX_DIGEST_ROWS]

    for alert in shown:
        state[alert["ticker_raw"]] = today.isoformat()
    save_state(prune_state(state, today))

    subject = (
        f"Watchlist — {len(shown)} to look at"
        if shown else "Watchlist — nothing triggered"
    )
    send_email(subject, build_email(shown, len(watchlist), errors))

    print(f"Checked {len(watchlist)}. Triggered {len(alerts)}. Sent {len(shown)}.")


if __name__ == "__main__":
    main()
