# -*- coding: utf-8 -*-
"""
soxl_hl.py  --  Base de dados de candles do SOXL (perp xyz:SOXL) via API da Hyperliquid
=====================================================================================

O QUE FAZ
---------
1. Mantem uma base SQLite local (hl_candles.db) com candles OHLCV de varios
   intervalos, funding horario e snapshots de contexto (OI, mark, oracle, premium).
2. Gera um visualizador HTML autocontido (candles + volume + cobertura da base).

POR QUE RODAR COM FREQUENCIA
----------------------------
A API da Hyperliquid so devolve os ~5000 candles MAIS RECENTES de cada intervalo.
Historico mais antigo que isso nao pode ser recuperado depois:

    1m  -> ~3,5 dias      15m -> ~52 dias
    3m  -> ~10 dias       30m -> ~104 dias
    5m  -> ~17 dias       1h+ -> desde a listagem (ago/2026)

Para acumular historico de 1 minuto sem buracos, rode "update" pelo menos
a cada 3 dias (idealmente 1x por dia). Cada execucao e incremental e idempotente.

USO
---
    python soxl_hl.py                 # update + status + dashboard (padrao; F5 no Spyder)
    python soxl_hl.py update          # so atualiza a base
    python soxl_hl.py status          # cobertura e buracos por intervalo
    python soxl_hl.py verify          # reconcilia 1m agregado contra 5m/15m/1h da API
    python soxl_hl.py export          # CSV por intervalo (com hora de NY e sessao)
    python soxl_hl.py dashboard       # gera soxl_hl_dashboard.html + .json
    python soxl_hl.py update --coins xyz:SOXL xyz:SMH xyz:NVDA

Em codigo (Spyder):
    import soxl_hl
    df = soxl_hl.load_candles("xyz:SOXL", "5m")     # DataFrame pronto para estudos

NOTAS
-----
- Timestamps na base: epoch em milissegundos, UTC (como vem da API).
- Candle diario da Hyperliquid = dia UTC (00:00-24:00 UTC), nao o pregao de NY.
- Sessao (pre / rth / after / overnight / weekend) e derivada do relogio de
  America/New_York. Feriados da NYSE NAO sao tratados (caem como dia normal).
- Minutos sem negocio nao geram candle na API: aparecem como "faltantes" no status.
- Dependencias: Python 3.9+, pandas. Nenhuma chave de API e necessaria.
"""

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Configuracao
# --------------------------------------------------------------------------------------
API_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_COINS = ["xyz:SOXL"]
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}
API_MAX_CANDLES = 5000          # limite de retencao da API por intervalo
FUNDING_PAGE = 500              # limite de linhas por chamada de fundingHistory
NY_TZ = "America/New_York"

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:               # execucao interativa
    BASE_DIR = Path.cwd()
DB_PATH = BASE_DIR / "hl_candles.db"
EXPORT_DIR = BASE_DIR / "export"
DASH_HTML = BASE_DIR / "soxl_hl_dashboard.html"
DASH_JSON = BASE_DIR / "soxl_hl_dashboard.json"
LIB_VERSION = "4.2.3"
LIB_FILE = BASE_DIR / ("lightweight-charts-%s.standalone.js" % LIB_VERSION)
LIB_URLS = [
    "https://unpkg.com/lightweight-charts@%s/dist/lightweight-charts.standalone.production.js" % LIB_VERSION,
    "https://cdn.jsdelivr.net/npm/lightweight-charts@%s/dist/lightweight-charts.standalone.production.js" % LIB_VERSION,
]


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
def api_post(body, retries=5):
    """POST no endpoint /info com retry exponencial (429 / erros de rede)."""
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last_err = err
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError("Falha na API da Hyperliquid apos %d tentativas: %s" % (retries, last_err))


def now_ms():
    return int(time.time() * 1000)


def fmt_utc(ms):
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------------------
# Base de dados
# --------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    coin      TEXT    NOT NULL,
    interval  TEXT    NOT NULL,
    t         INTEGER NOT NULL,   -- abertura do candle, epoch ms UTC
    t_close   INTEGER NOT NULL,   -- fechamento do candle, epoch ms UTC
    o         REAL    NOT NULL,
    h         REAL    NOT NULL,
    l         REAL    NOT NULL,
    c         REAL    NOT NULL,
    v         REAL    NOT NULL,   -- volume em contratos (unidades de SOXL)
    n         INTEGER NOT NULL,   -- numero de negocios
    is_closed INTEGER NOT NULL,   -- 0 = candle ainda em formacao na ultima coleta
    PRIMARY KEY (coin, interval, t)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS funding (
    coin         TEXT    NOT NULL,
    t            INTEGER NOT NULL,   -- epoch ms UTC
    funding_rate REAL    NOT NULL,   -- taxa por hora
    premium      REAL    NOT NULL,
    PRIMARY KEY (coin, t)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ctx_snapshots (
    coin          TEXT    NOT NULL,
    t             INTEGER NOT NULL,  -- momento da coleta, epoch ms UTC
    mark_px       REAL, oracle_px REAL, mid_px REAL,
    funding       REAL, premium   REAL,
    open_interest REAL, day_ntl_vlm REAL, day_base_vlm REAL, prev_day_px REAL,
    PRIMARY KEY (coin, t)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS update_log (
    run_t     INTEGER NOT NULL,
    coin      TEXT    NOT NULL,
    interval  TEXT    NOT NULL,
    fetched   INTEGER NOT NULL,
    new_rows  INTEGER NOT NULL,
    lost_bars INTEGER NOT NULL       -- candles perdidos por atraso na coleta
);
"""


def connect(db_path=DB_PATH):
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)
    return con


def _f(x):
    return None if x is None else float(x)


# --------------------------------------------------------------------------------------
# Update
# --------------------------------------------------------------------------------------
def update_candles(con, coin, interval, run_t):
    """Busca os candles desde o ultimo armazenado e faz upsert. Retorna dict de resumo."""
    step = INTERVAL_MS[interval]
    row = con.execute("SELECT MAX(t), COUNT(*) FROM candles WHERE coin=? AND interval=?",
                      (coin, interval)).fetchone()
    last_t, n_before = row[0], row[1]
    # Re-busca a partir do ultimo candle guardado: ele pode ter sido salvo em formacao.
    start = last_t if last_t is not None else 0
    end = now_ms()
    rows = api_post({"type": "candleSnapshot",
                     "req": {"coin": coin, "interval": interval,
                             "startTime": start, "endTime": end}})
    lost = 0
    if last_t is not None and rows:
        first_t = int(rows[0]["t"])
        if first_t > last_t and len(rows) >= API_MAX_CANDLES:
            # a janela de retencao da API ja passou do ultimo candle guardado
            lost = (first_t - last_t) // step - 1
    payload = [(coin, interval, int(r["t"]), int(r["T"]), float(r["o"]), float(r["h"]),
                float(r["l"]), float(r["c"]), float(r["v"]), int(r["n"]),
                1 if int(r["T"]) < end else 0) for r in rows]
    con.executemany(
        """INSERT INTO candles (coin, interval, t, t_close, o, h, l, c, v, n, is_closed)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(coin, interval, t) DO UPDATE SET
               t_close=excluded.t_close, o=excluded.o, h=excluded.h, l=excluded.l,
               c=excluded.c, v=excluded.v, n=excluded.n, is_closed=excluded.is_closed""",
        payload)
    n_after = con.execute("SELECT COUNT(*) FROM candles WHERE coin=? AND interval=?",
                          (coin, interval)).fetchone()[0]
    con.execute("INSERT INTO update_log VALUES (?,?,?,?,?,?)",
                (run_t, coin, interval, len(rows), n_after - n_before, lost))
    con.commit()
    return {"interval": interval, "fetched": len(rows), "new": n_after - n_before,
            "total": n_after, "lost": lost}


def update_funding(con, coin):
    """Funding horario, paginado (500 por chamada)."""
    last = con.execute("SELECT MAX(t) FROM funding WHERE coin=?", (coin,)).fetchone()[0]
    start = (last + 1) if last is not None else 0
    total = 0
    while True:
        rows = api_post({"type": "fundingHistory", "coin": coin, "startTime": start})
        if not rows:
            break
        con.executemany(
            "INSERT OR REPLACE INTO funding (coin, t, funding_rate, premium) VALUES (?,?,?,?)",
            [(coin, int(r["time"]), float(r["fundingRate"]), float(r["premium"])) for r in rows])
        total += len(rows)
        start = int(rows[-1]["time"]) + 1
        if len(rows) < FUNDING_PAGE:
            break
        time.sleep(0.2)
    con.commit()
    return total


def snapshot_ctx(con, coins, run_t):
    """Guarda mark/oracle/OI/premium do momento (nao existe historico disso na API)."""
    by_dex = {}
    for coin in coins:
        dex = coin.split(":")[0] if ":" in coin else ""
        by_dex.setdefault(dex, []).append(coin)
    for dex, dex_coins in by_dex.items():
        meta, ctxs = api_post({"type": "metaAndAssetCtxs", "dex": dex})
        names = [a["name"] for a in meta["universe"]]
        for coin in dex_coins:
            if coin not in names:
                print("  [aviso] %s nao encontrado no dex '%s'" % (coin, dex))
                continue
            c = ctxs[names.index(coin)]
            con.execute(
                "INSERT OR REPLACE INTO ctx_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (coin, run_t, _f(c.get("markPx")), _f(c.get("oraclePx")), _f(c.get("midPx")),
                 _f(c.get("funding")), _f(c.get("premium")), _f(c.get("openInterest")),
                 _f(c.get("dayNtlVlm")), _f(c.get("dayBaseVlm")), _f(c.get("prevDayPx"))))
    con.commit()


def cmd_update(coins, intervals):
    con = connect()
    run_t = now_ms()
    print("Atualizando base: %s" % DB_PATH)
    for coin in coins:
        print("\n%s" % coin)
        for iv in intervals:
            r = update_candles(con, coin, iv, run_t)
            msg = "  %-4s recebidos %5d | novos %5d | total %7d" % (
                iv, r["fetched"], r["new"], r["total"])
            if r["lost"] > 0:
                msg += "  [ATENCAO] %d candles perdidos (coleta atrasada)" % r["lost"]
            print(msg)
            time.sleep(0.15)
        nf = update_funding(con, coin)
        print("  funding: %d novas linhas" % nf)
    snapshot_ctx(con, coins, run_t)
    con.close()


# --------------------------------------------------------------------------------------
# Leitura / derivados
# --------------------------------------------------------------------------------------
def classify_session(ts_ny):
    """Sessao de mercado a partir de timestamps em America/New_York (Series tz-aware)."""
    minutes = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    dow = ts_ny.dt.dayofweek                       # 0 = segunda
    sess = pd.Series("overnight", index=ts_ny.index)
    sess[(minutes >= 240) & (minutes < 570)] = "pre"      # 04:00-09:30
    sess[(minutes >= 570) & (minutes < 960)] = "rth"      # 09:30-16:00
    sess[(minutes >= 960) & (minutes < 1200)] = "after"   # 16:00-20:00
    weekend = ((dow == 4) & (minutes >= 1200)) | (dow == 5) | ((dow == 6) & (minutes < 1200))
    sess[weekend] = "weekend"
    return sess


def load_candles(coin="xyz:SOXL", interval="5m", only_closed=True, db_path=DB_PATH):
    """DataFrame de candles com hora UTC, hora de NY e sessao. Ponto de entrada dos estudos."""
    con = connect(db_path)
    q = "SELECT t, t_close, o, h, l, c, v, n, is_closed FROM candles WHERE coin=? AND interval=?"
    if only_closed:
        q += " AND is_closed=1"
    df = pd.read_sql_query(q + " ORDER BY t", con, params=(coin, interval))
    con.close()
    df["ts_utc"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["ts_ny"] = df["ts_utc"].dt.tz_convert(NY_TZ)
    df["session"] = classify_session(df["ts_ny"])
    return df


def coverage(con, coin, intervals):
    """Cobertura por intervalo: quantidade, inicio, fim e candles faltantes."""
    out = []
    for iv in intervals:
        ts = [r[0] for r in con.execute(
            "SELECT t FROM candles WHERE coin=? AND interval=? ORDER BY t", (coin, iv))]
        if not ts:
            out.append({"interval": iv, "bars": 0, "first": None, "last": None,
                        "missing": 0, "gaps": 0, "largest_gap_bars": 0, "largest_gap_at": None})
            continue
        step = INTERVAL_MS[iv]
        missing = gaps = largest = 0
        largest_at = None
        for a, b in zip(ts[:-1], ts[1:]):
            k = (b - a) // step - 1
            if k > 0:
                gaps += 1
                missing += k
                if k > largest:
                    largest, largest_at = k, a + step
        out.append({"interval": iv, "bars": len(ts), "first": ts[0], "last": ts[-1],
                    "missing": missing, "gaps": gaps,
                    "largest_gap_bars": largest, "largest_gap_at": largest_at})
    return out


def cmd_status(coins, intervals):
    con = connect()
    for coin in coins:
        print("\nCobertura %s  (horarios em UTC)" % coin)
        print("  %-4s %8s  %-16s  %-16s  %9s  %6s  %s" % (
            "int", "candles", "primeiro", "ultimo", "faltantes", "gaps", "maior gap"))
        for r in coverage(con, coin, intervals):
            big = "-"
            if r["largest_gap_bars"]:
                big = "%d candles em %s" % (r["largest_gap_bars"], fmt_utc(r["largest_gap_at"]))
            print("  %-4s %8d  %-16s  %-16s  %9d  %6d  %s" % (
                r["interval"], r["bars"], fmt_utc(r["first"]), fmt_utc(r["last"]),
                r["missing"], r["gaps"], big))
        nf, f0, f1 = con.execute(
            "SELECT COUNT(*), MIN(t), MAX(t) FROM funding WHERE coin=?", (coin,)).fetchone()
        print("  funding: %d linhas, %s -> %s" % (nf, fmt_utc(f0), fmt_utc(f1)))
    con.close()


def cmd_verify(coins):
    """Reagrega 1m e compara com os candles maiores vindos da API (consistencia da base)."""
    for coin in coins:
        base = load_candles(coin, "1m")
        if base.empty:
            print("%s: sem candles de 1m" % coin)
            continue
        print("\nVerificacao %s: 1m reagregado vs. candle da API" % coin)
        for iv in ["3m", "5m", "15m", "30m", "1h"]:
            step = INTERVAL_MS[iv]
            ref = load_candles(coin, iv)
            b = base.copy()
            b["bucket"] = (b["t"] // step) * step
            agg = b.groupby("bucket").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                                          c=("c", "last"), v=("v", "sum"), n=("n", "sum"),
                                          t_min=("t", "min"))
            # so compara buckets inteiramente cobertos pela janela de 1m
            lo, hi = base["t"].min(), base["t"].max()
            agg = agg[(agg.index >= lo) & (agg.index + step - 60_000 <= hi)]
            m = agg.join(ref.set_index("t")[["o", "h", "l", "c", "v", "n"]],
                         rsuffix="_api", how="inner")
            if m.empty:
                print("  %-4s sem sobreposicao" % iv)
                continue
            px_bad = int(((m["o"] != m["o_api"]) | (m["h"] != m["h_api"]) |
                          (m["l"] != m["l_api"]) | (m["c"] != m["c_api"])).sum())
            n_bad = int((m["n"] != m["n_api"]).sum())
            v_diff = float((m["v"] - m["v_api"]).abs().max())
            print("  %-4s %5d candles comparados | OHLC divergentes: %d | n divergentes: %d | "
                  "max |dif. volume|: %.6f" % (iv, len(m), px_bad, n_bad, v_diff))


def cmd_export(coins, intervals):
    EXPORT_DIR.mkdir(exist_ok=True)
    for coin in coins:
        for iv in intervals:
            df = load_candles(coin, iv, only_closed=False)
            if df.empty:
                continue
            out = df[["t", "ts_utc", "ts_ny", "session", "o", "h", "l", "c", "v", "n", "is_closed"]]
            path = EXPORT_DIR / ("%s_%s.csv" % (coin.replace(":", "_"), iv))
            out.to_csv(path, index=False)
            print("  %s (%d linhas)" % (path, len(out)))


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------
SESSION_CODES = {"rth": 0, "pre": 1, "after": 2, "overnight": 3, "weekend": 4}


def build_dashboard_data(coin, intervals, max_bars):
    con = connect()
    data = {"meta": {"coin": coin,
                     "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                     "tz": NY_TZ, "tz_label": "New York (ET)", "max_bars": max_bars,
                     "session_codes": SESSION_CODES},
            "intervals": {}, "coverage": coverage(con, coin, intervals)}
    for iv in intervals:
        df = load_candles(coin, iv, only_closed=False)
        if df.empty:
            continue
        df = df.tail(max_bars)
        # tempo em epoch UTC; o HTML converte para hora de NY na exibicao (fuso IANA)
        data["intervals"][iv] = {
            "t": (df["t"] // 1000).tolist(),
            "o": df["o"].tolist(), "h": df["h"].tolist(), "l": df["l"].tolist(),
            "c": df["c"].tolist(), "v": df["v"].tolist(), "n": df["n"].tolist(),
            "s": df["session"].map(SESSION_CODES).tolist(),
        }
    ctx = con.execute(
        """SELECT t, mark_px, oracle_px, mid_px, funding, premium, open_interest,
                  day_ntl_vlm, prev_day_px
           FROM ctx_snapshots WHERE coin=? ORDER BY t DESC LIMIT 1""", (coin,)).fetchone()
    if ctx:
        keys = ["t", "mark_px", "oracle_px", "mid_px", "funding", "premium",
                "open_interest", "day_ntl_vlm", "prev_day_px"]
        data["ctx"] = dict(zip(keys, ctx))
    con.close()
    return data


def ensure_chart_lib():
    """Baixa a lightweight-charts uma unica vez e guarda ao lado do script."""
    if LIB_FILE.exists() and LIB_FILE.stat().st_size > 50_000:
        return LIB_FILE.read_text(encoding="utf-8")
    for url in LIB_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                txt = resp.read().decode("utf-8")
            if len(txt) > 50_000:
                LIB_FILE.write_text(txt, encoding="utf-8")
                return txt
        except Exception:
            continue
    return None


def cmd_dashboard(coins, intervals, max_bars):
    coin = coins[0]
    data = build_dashboard_data(coin, intervals, max_bars)
    if not data["intervals"]:
        print("Base vazia para %s: rode 'update' primeiro." % coin)
        return
    payload = json.dumps(data, separators=(",", ":"))
    DASH_JSON.write_text(payload, encoding="utf-8")
    lib = ensure_chart_lib()
    if lib is None:
        lib_tag = '<script src="%s"></script>' % LIB_URLS[0]
        print("  [aviso] nao consegui baixar a lib de graficos; o HTML vai usar o CDN.")
    else:
        lib_tag = "<script>" + lib.replace("</script", "<\\/script") + "</script>"
    html = (HTML_TEMPLATE
            .replace("__LIB__", lib_tag)
            .replace("__DATA__", payload.replace("</", "<\\/")))
    DASH_HTML.write_text(html, encoding="utf-8")
    print("\nDashboard: %s" % DASH_HTML)
    print("JSON:      %s" % DASH_JSON)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOXL candles - Hyperliquid</title>
<style>
  :root {
    --bg:#ffffff; --panel:#f6f7f9; --ink:#14181f; --ink2:#566070; --line:#e2e5ea;
    --up:#0b8f7a; --down:#d6453d; --accent:#2f5fd0;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151a; --panel:#1a1e25; --ink:#e8ebf0; --ink2:#98a2b3; --line:#2a303a;
            --up:#1fa38d; --down:#e2574d; --accent:#7ea2f5; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:20px; }
  h1 { font-size:18px; margin:0; font-weight:600; }
  .sub { color:var(--ink2); font-size:12px; margin-top:2px; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:16px 0; }
  .tile { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .tile .k { color:var(--ink2); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .tile .v { font-size:18px; font-weight:600; margin-top:2px; font-variant-numeric:tabular-nums; }
  .tile .d { color:var(--ink2); font-size:11px; }
  .bar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
  .bar button { background:var(--panel); color:var(--ink); border:1px solid var(--line);
                border-radius:6px; padding:5px 11px; cursor:pointer; font:inherit; }
  .bar button.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
  .bar .sp { flex:1; }
  .bar label { color:var(--ink2); font-size:12px; display:flex; gap:5px; align-items:center; }
  .chartbox { position:relative; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  #chart { height:520px; }
  #vol { height:150px; border-top:1px solid var(--line); }
  .legend { position:absolute; left:12px; top:8px; z-index:5; font-size:12px; pointer-events:none;
            font-variant-numeric:tabular-nums; color:var(--ink2); }
  .legend b { color:var(--ink); font-weight:600; }
  .legend .sw { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:4px; }
  #vlegend { top:auto; bottom:126px; }
  h2 { font-size:14px; margin:22px 0 8px; font-weight:600; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13px; }
  th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--ink2); font-weight:500; font-size:12px; }
  th:first-child, td:first-child { text-align:left; }
  .note { color:var(--ink2); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1 id="title"></h1>
  <div class="sub" id="sub"></div>
  <div class="tiles" id="tiles"></div>
  <div class="bar" id="bar"></div>
  <div class="chartbox">
    <div class="legend" id="legend"></div>
    <div id="chart"></div>
    <div class="legend" id="vlegend"></div>
    <div id="vol"></div>
  </div>
  <div class="note" id="tznote"></div>
  <h2>Cobertura da base</h2>
  <table id="cov"></table>
  <div class="note">Faltantes = candles ausentes entre o primeiro e o ultimo armazenado
  (minutos sem negocio nao geram candle na API; coletas atrasadas tambem aparecem aqui).</div>
</div>
__LIB__
<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  var D = JSON.parse(document.getElementById('payload').textContent);
  var css = getComputedStyle(document.documentElement);
  function v(n) { return css.getPropertyValue(n).trim(); }
  var UP = v('--up'), DOWN = v('--down'), INK2 = v('--ink2'), LINE = v('--line'), BG = v('--bg');
  var SESS = ['RTH', 'Pre-market', 'After-hours', 'Overnight', 'Fim de semana'];
  var order = ['1m','3m','5m','15m','30m','1h','4h','1d'].filter(function (k) { return D.intervals[k]; });
  var cur = order.indexOf('15m') >= 0 ? '15m' : order[0];
  var onlyRth = false;

  function fmt(x, d) { return x == null ? '-' : x.toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d}); }
  function fmtTs(ms) {
    if (ms == null) return '-';
    return new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
  }

  document.getElementById('title').textContent = D.meta.coin + ' - candles Hyperliquid';
  document.getElementById('sub').textContent = 'Base atualizada em ' + D.meta.generated_utc + ' UTC';

  // ---- tiles (tudo vindo do JSON) ----
  var tiles = [];
  var last1 = D.intervals[order[0]];
  var lastPx = last1.c[last1.c.length - 1];
  if (D.ctx) {
    var chg = D.ctx.prev_day_px ? (D.ctx.mark_px / D.ctx.prev_day_px - 1) * 100 : null;
    tiles.push(['Mark', fmt(D.ctx.mark_px, 3), chg == null ? '' : (chg >= 0 ? '+' : '') + fmt(chg, 2) + '% vs. dia anterior']);
    tiles.push(['Oracle', fmt(D.ctx.oracle_px, 3), 'premium ' + fmt(D.ctx.premium * 1e4, 2) + ' bps']);
    tiles.push(['Funding (1h)', fmt(D.ctx.funding * 1e4, 4) + ' bps', fmt(D.ctx.funding * 24 * 365 * 100, 2) + '% a.a. simples']);
    tiles.push(['Open interest', fmt(D.ctx.open_interest, 0), 'USD ' + fmt(D.ctx.open_interest * D.ctx.mark_px / 1e6, 2) + ' mm']);
    tiles.push(['Volume 24h', 'USD ' + fmt(D.ctx.day_ntl_vlm / 1e6, 1) + ' mm', 'snapshot ' + fmtTs(D.ctx.t) + ' UTC']);
  } else {
    tiles.push(['Ultimo', fmt(lastPx, 3), '']);
  }
  document.getElementById('tiles').innerHTML = tiles.map(function (t) {
    return '<div class="tile"><div class="k">' + t[0] + '</div><div class="v">' + t[1] + '</div><div class="d">' + t[2] + '</div></div>';
  }).join('');

  // ---- charts ----
  // eixo de tempo: dados em UTC, exibidos em hora de NY (diario fica em UTC)
  function parts(sec) {
    var tz = cur === '1d' ? 'UTC' : D.meta.tz;
    var p = {};
    new Intl.DateTimeFormat('en-GB', { timeZone: tz, year: 'numeric', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).formatToParts(new Date(sec * 1000))
      .forEach(function (x) { p[x.type] = x.value; });
    return p;
  }
  function tick(sec, type) {
    var p = parts(sec);
    if (type === 0) return p.year;
    if (type === 1) return p.month;
    if (type === 2) return p.day + ' ' + p.month;
    return p.hour + ':' + p.minute;
  }
  function full(sec) {
    var p = parts(sec);
    return p.day + ' ' + p.month + ' ' + p.year + (cur === '1d' ? '' : '  ' + p.hour + ':' + p.minute);
  }
  function mk(el, h, logo) {
    return LightweightCharts.createChart(el, {
      height: h, autoSize: true,
      localization: { timeFormatter: full },
      layout: { background: { color: BG }, textColor: INK2, fontSize: 11, attributionLogo: logo },
      grid: { vertLines: { color: LINE }, horzLines: { color: LINE } },
      rightPriceScale: { borderColor: LINE, minimumWidth: 70 },
      timeScale: { borderColor: LINE, timeVisible: true, secondsVisible: false, tickMarkFormatter: tick },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
    });
  }
  var chart = mk(document.getElementById('chart'), 520, true);
  var vchart = mk(document.getElementById('vol'), 150, false);
  var candles = chart.addCandlestickSeries({
    upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN,
    wickUpColor: UP, wickDownColor: DOWN, priceFormat: { type: 'price', precision: 3, minMove: 0.001 }
  });
  var vols = vchart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceLineVisible: false, lastValueVisible: false });

  var rows = [];        // linhas atualmente desenhadas
  var lock = false;
  var byTime = {};
  function draw() {
    var S = D.intervals[cur];
    rows = []; byTime = {};
    for (var i = 0; i < S.t.length; i++) {
      if (onlyRth && cur !== '1d' && S.s[i] !== 0) continue;
      var r = { time: S.t[i], o: S.o[i], h: S.h[i], l: S.l[i], c: S.c[i], v: S.v[i], n: S.n[i], s: S.s[i] };
      rows.push(r); byTime[r.time] = r;
    }
    lock = true;
    candles.setData(rows.map(function (r) { return { time: r.time, open: r.o, high: r.h, low: r.l, close: r.c }; }));
    vols.setData(rows.map(function (r) { return { time: r.time, value: r.v, color: (r.c >= r.o ? UP : DOWN) + '99' }; }));
    var n = rows.length, show = Math.min(n, 300);
    chart.timeScale().setVisibleLogicalRange({ from: n - show, to: n + 5 });
    vchart.timeScale().setVisibleLogicalRange({ from: n - show, to: n + 5 });
    lock = false;
    legend(rows[n - 1]);
    document.getElementById('tznote').textContent = (cur === '1d'
      ? 'Candle diario = dia UTC (00:00-24:00 UTC).'
      : 'Eixo de tempo em hora de ' + D.meta.tz_label + '.') +
      ' Volume em contratos. ' + n.toLocaleString('en-US') + ' candles carregados neste intervalo.';
    Array.prototype.forEach.call(document.querySelectorAll('#bar button'), function (b) {
      b.className = b.dataset.iv === cur ? 'on' : '';
    });
  }
  function legend(r) {
    if (!r) return;
    var ch = (r.c / r.o - 1) * 100, col = r.c >= r.o ? UP : DOWN;
    document.getElementById('legend').innerHTML =
      '<span class="sw" style="background:' + col + '"></span>' +
      'O <b>' + fmt(r.o, 3) + '</b> &nbsp;H <b>' + fmt(r.h, 3) + '</b> &nbsp;L <b>' + fmt(r.l, 3) +
      '</b> &nbsp;C <b>' + fmt(r.c, 3) + '</b> &nbsp;<b>' + (ch >= 0 ? '+' : '') + fmt(ch, 2) + '%</b>' +
      ' &nbsp;amplitude <b>' + fmt((r.h / r.l - 1) * 100, 2) + '%</b>' +
      (cur === '1d' ? '' : ' &nbsp;' + SESS[r.s]);
    document.getElementById('vlegend').innerHTML =
      'Volume <b>' + fmt(r.v, 2) + '</b> &nbsp;negocios <b>' + fmt(r.n, 0) + '</b>';
  }

  // sincroniza os dois paineis (mesmo eixo de tempo, escalas independentes)
  function link(a, b) {
    a.timeScale().subscribeVisibleLogicalRangeChange(function (rg) {
      if (lock || !rg) return; lock = true; b.timeScale().setVisibleLogicalRange(rg); lock = false;
    });
  }
  link(chart, vchart); link(vchart, chart);
  function hover(src, other, otherSeries, key) {
    src.subscribeCrosshairMove(function (p) {
      if (p && p.time != null && byTime[p.time]) {
        legend(byTime[p.time]);
        other.setCrosshairPosition(byTime[p.time][key], p.time, otherSeries);
      } else {
        other.clearCrosshairPosition();
        legend(rows[rows.length - 1]);
      }
    });
  }
  hover(chart, vchart, vols, 'v'); hover(vchart, chart, candles, 'c');

  // ---- controles ----
  var bar = document.getElementById('bar');
  order.forEach(function (iv) {
    var b = document.createElement('button');
    b.textContent = iv; b.dataset.iv = iv;
    b.onclick = function () { cur = iv; draw(); };
    bar.appendChild(b);
  });
  var sp = document.createElement('span'); sp.className = 'sp'; bar.appendChild(sp);
  var lb = document.createElement('label');
  lb.innerHTML = '<input type="checkbox" id="rth"> So pregao regular (RTH 09:30-16:00 ET)';
  bar.appendChild(lb);
  document.getElementById('rth').onchange = function (e) { onlyRth = e.target.checked; draw(); };

  // ---- cobertura ----
  var head = '<tr><th>Intervalo</th><th>Candles</th><th>Primeiro (UTC)</th><th>Ultimo (UTC)</th><th>Faltantes</th><th>Gaps</th><th>Maior gap</th></tr>';
  document.getElementById('cov').innerHTML = head + D.coverage.map(function (r) {
    return '<tr><td>' + r.interval + '</td><td>' + fmt(r.bars, 0) + '</td><td>' + fmtTs(r.first) +
      '</td><td>' + fmtTs(r.last) + '</td><td>' + fmt(r.missing, 0) + '</td><td>' + fmt(r.gaps, 0) +
      '</td><td>' + (r.largest_gap_bars ? fmt(r.largest_gap_bars, 0) + ' em ' + fmtTs(r.largest_gap_at) : '-') + '</td></tr>';
  }).join('');

  draw();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Base de candles SOXL (Hyperliquid xyz:SOXL)")
    p.add_argument("command", nargs="?", default="all",
                   choices=["all", "update", "status", "verify", "export", "dashboard"])
    p.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    p.add_argument("--intervals", nargs="+", default=INTERVALS, choices=sorted(INTERVAL_MS))
    p.add_argument("--max-bars", type=int, default=20000,
                   help="maximo de candles por intervalo no dashboard")
    a = p.parse_args(argv)

    if a.command in ("all", "update"):
        cmd_update(a.coins, a.intervals)
    if a.command in ("all", "status"):
        cmd_status(a.coins, a.intervals)
    if a.command == "verify":
        cmd_verify(a.coins)
    if a.command == "export":
        cmd_export(a.coins, a.intervals)
    if a.command in ("all", "dashboard"):
        cmd_dashboard(a.coins, a.intervals, a.max_bars)


if __name__ == "__main__":
    main(sys.argv[1:])
