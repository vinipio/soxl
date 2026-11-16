# -*- coding: utf-8 -*-
"""
lab.py -- motor comum de backtest intraday para SOXL (candles de 1 minuto).

Todas as familias de estrategia usam ESTE modulo para carregar dados, simular
saidas e medir resultado. Isso garante a mesma metodologia em todos os estudos:

  * Execucao: sinal calculado com informacao ate o FECHAMENTO do candle i;
    entrada no OPEN do candle i+1 (sem look-ahead).
  * Stop/alvo: verificados dentro do candle via high/low. Se o candle abre alem
    do stop, sai no open. Se stop e alvo caem no mesmo candle, assume STOP.
  * Saida por tempo: no CLOSE do candle indicado.
  * Custo: em bps por lado, aplicado na entrada e na saida.
    Base = 3 bps/lado (6 ida e volta); estresse = 10 bps/lado (20 ida e volta).
  * Retorno por trade = side * (exit/entry - 1) - 2*custo. Sem alavancagem,
    sem composicao: PnL acumulado = soma dos retornos (nocional constante).
  * Amostras: IS (in-sample)  = 2026-05-11 .. 2026-08-31
              OOS (out-of-sample) = 2026-09-01 .. 2026-11-10
    Parametros devem ser escolhidos SO no IS; o OOS e a prova.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "soxl_1m_all.csv"          # 2025-11-11 .. 2026-11-10 (Massive, 1 minuto)
RESULTS = ROOT / os.environ.get("SOXL_RESULTS", "results")
NY = "America/New_York"
# Fim do in-sample. Padrao 2026-08-31 (IS = P0+P1). Use SOXL_IS_END=2026-05-08 para IS = P0 apenas.
IS_END = pd.Timestamp(os.environ.get("SOXL_IS_END", "2026-08-31")).date()
# Tres periodos fixos, sempre reportados em evaluate():
#   P0 = 2025-11-11..2026-05-08 (historico adicionado depois; mercado em alta)
#   P1 = 2026-05-11..2026-08-31 (treino do estudo original)
#   P2 = 2026-09-01..2026-11-10 (teste do estudo original)
P0_END = pd.Timestamp("2026-05-08").date()
P1_END = pd.Timestamp("2026-08-31").date()


def period_of(d):
    return "P0" if d <= P0_END else ("P1" if d <= P1_END else "P2")
COST_BASE = 3.0      # bps por lado
COST_STRESS = 10.0   # bps por lado
RTH_START, RTH_END = 570, 960   # 09:30, 16:00 em minutos do dia (NY)
PRE_START, AH_END = 240, 1200   # 04:00, 20:00

_cache = {}


def load():
    """DataFrame 1m completo (pre + RTH + after). Colunas:
    ts (NY tz-aware), date, m (minuto do dia NY), sess ('pre'|'rth'|'after'),
    open, high, low, close, volume, vwap, n (negocios)."""
    if "df" not in _cache:
        d = pd.read_csv(DATA)
        d["ts"] = pd.to_datetime(d["timestamp_unix_ms"], unit="ms", utc=True).dt.tz_convert(NY)
        d["date"] = d["ts"].dt.date
        d["m"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
        d["sess"] = np.where(d["m"] < RTH_START, "pre", np.where(d["m"] < RTH_END, "rth", "after"))
        d = d.rename(columns={"transactions": "n"})
        d = d[["ts", "date", "m", "sess", "open", "high", "low", "close", "volume", "vwap", "n"]]
        d = d.sort_values("ts").reset_index(drop=True)
        # descarta pregoes curtos/incompletos (ex.: 28/11/2025 e 24/12/2025 fecham as 13:00)
        cnt = d[d["sess"] == "rth"].groupby("date").size()
        d = d[d["date"].isin(set(cnt[cnt == 390].index))].reset_index(drop=True)
        _cache["df"] = d
    return _cache["df"].copy()


def rth():
    """So o pregao regular (09:30-16:00 NY): 390 candles por dia, 128 dias."""
    d = load()
    return d[d["sess"] == "rth"].reset_index(drop=True)


def day_groups(df):
    """Lista de (date, DataFrame do dia com index 0..k-1), em ordem cronologica."""
    return [(dt, g.reset_index(drop=True)) for dt, g in df.groupby("date", sort=True)]


def daily_table():
    """Resumo por dia (RTH + contexto de pre/after), util para filtros de regime.
    Colunas: o,h,l,c (RTH), vol_rth, prev_c, gap (open/prev_c-1), oc (c/o-1),
    pre_h, pre_l, pre_last, pre_vol, ah_last, rng (h/l-1),
    atr14 (media de rng dos 14 dias ANTERIORES; sem look-ahead)."""
    d = load()
    r = d[d["sess"] == "rth"].groupby("date").agg(
        o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
        vol_rth=("volume", "sum"))
    p = d[d["sess"] == "pre"].groupby("date").agg(
        pre_h=("high", "max"), pre_l=("low", "min"), pre_last=("close", "last"),
        pre_vol=("volume", "sum"))
    a = d[d["sess"] == "after"].groupby("date").agg(ah_last=("close", "last"))
    t = r.join(p).join(a)
    t["prev_c"] = t["c"].shift(1)
    t["gap"] = t["o"] / t["prev_c"] - 1
    t["oc"] = t["c"] / t["o"] - 1
    t["rng"] = t["h"] / t["l"] - 1
    t["atr14"] = t["rng"].shift(1).rolling(14, min_periods=5).mean()
    return t


def resample(day_df, minutes):
    """Agrega um dia de 1m em candles de `minutes` (ancorado no 1o candle do dia).
    Retorna DataFrame com i0,i1 (indices 1m do primeiro/ultimo candle do bloco)."""
    k = np.arange(len(day_df)) // minutes
    g = day_df.groupby(k)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"), n=("n", "sum"),
                m=("m", "first"))
    out["i0"] = g.apply(lambda x: x.index[0])
    out["i1"] = g.apply(lambda x: x.index[-1])
    return out.reset_index(drop=True)


def sim_exit(day_df, i_entry, side, stop=None, target=None, i_last=None, trail_pct=None):
    """Simula um trade dentro de UM dia (DataFrame com index 0..k-1).

    i_entry : indice do candle em cujo OPEN se entra (= indice do sinal + 1).
    side    : +1 compra, -1 venda.
    stop / target : niveis de PRECO absolutos (ou None).
    i_last  : ultimo candle permitido; sai no CLOSE dele (default = ultimo do dia).
    trail_pct : trailing stop em fracao (ex. 0.02) a partir do melhor preco.

    Retorna dict(entry_i, entry_px, exit_i, exit_px, reason) ou None se i_entry invalido.
    """
    n = len(day_df)
    if i_last is None or i_last > n - 1:
        i_last = n - 1
    if i_entry is None or i_entry > i_last or i_entry < 0:
        return None
    o = day_df["open"].values
    h = day_df["high"].values
    l = day_df["low"].values
    c = day_df["close"].values
    entry = o[i_entry]
    best = entry
    cur_stop = stop
    for i in range(i_entry, i_last + 1):
        # 1) abertura alem do stop -> sai no open
        if cur_stop is not None and i > i_entry:
            if (side > 0 and o[i] <= cur_stop) or (side < 0 and o[i] >= cur_stop):
                return dict(entry_i=i_entry, entry_px=entry, exit_i=i, exit_px=o[i], reason="stop_gap")
        # 2) stop dentro do candle (prioridade sobre o alvo)
        if cur_stop is not None:
            if (side > 0 and l[i] <= cur_stop) or (side < 0 and h[i] >= cur_stop):
                return dict(entry_i=i_entry, entry_px=entry, exit_i=i, exit_px=cur_stop, reason="stop")
        # 3) alvo
        if target is not None:
            if (side > 0 and h[i] >= target) or (side < 0 and l[i] <= target):
                px = target
                if i > i_entry and ((side > 0 and o[i] >= target) or (side < 0 and o[i] <= target)):
                    px = o[i]
                return dict(entry_i=i_entry, entry_px=entry, exit_i=i, exit_px=px, reason="target")
        # 4) atualiza trailing com o extremo deste candle (vale a partir do proximo)
        if trail_pct is not None:
            if side > 0:
                best = max(best, h[i])
                ts_ = best * (1 - trail_pct)
                cur_stop = ts_ if cur_stop is None else max(cur_stop, ts_)
            else:
                best = min(best, l[i])
                ts_ = best * (1 + trail_pct)
                cur_stop = ts_ if cur_stop is None else min(cur_stop, ts_)
    return dict(entry_i=i_entry, entry_px=entry, exit_i=i_last, exit_px=c[i_last], reason="time")


def make_trade(day_df, res, side, tag=""):
    """Converte a saida de sim_exit em registro de trade padronizado."""
    return dict(date=str(day_df["date"].iloc[0]), side=int(side),
                entry_m=int(day_df["m"].iloc[res["entry_i"]]), exit_m=int(day_df["m"].iloc[res["exit_i"]]),
                entry_px=float(res["entry_px"]), exit_px=float(res["exit_px"]),
                reason=res["reason"], tag=tag)


def trade_from_prices(date, side, entry_m, entry_px, exit_m, exit_px, reason="time", tag=""):
    """Para trades que cruzam sessoes/dias (ex.: fechamento -> abertura seguinte)."""
    return dict(date=str(date), side=int(side), entry_m=int(entry_m), exit_m=int(exit_m),
                entry_px=float(entry_px), exit_px=float(exit_px), reason=reason, tag=tag)


def _all_dates():
    if "dates" not in _cache:
        _cache["dates"] = sorted(load()["date"].unique())
    return _cache["dates"]


def _stats(tr, dates, cost_side):
    """Metricas de um conjunto de trades sobre um conjunto de datas."""
    out = {"n_trades": int(len(tr)), "n_days": int(len(dates))}
    if len(tr) == 0:
        return out
    gross = tr["side"] * (tr["exit_px"] / tr["entry_px"] - 1.0)
    net = gross - 2.0 * cost_side / 1e4
    daily = pd.Series(0.0, index=[str(x) for x in dates])
    daily = daily.add(net.groupby(tr["date"]).sum(), fill_value=0.0).reindex([str(x) for x in dates]).fillna(0.0)
    cum = daily.cumsum()
    dd = float((cum - cum.cummax()).min())
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    tsd = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    pos, neg = float(net[net > 0].sum()), float(-net[net < 0].sum())
    hold = (tr["exit_m"] - tr["entry_m"])
    out.update({
        "days_traded": int(tr["date"].nunique()),
        "win_rate": float((net > 0).mean()),
        "avg_gross_bps": float(gross.mean() * 1e4),
        "avg_net_bps": float(net.mean() * 1e4),
        "median_net_bps": float(net.median() * 1e4),
        "t_stat": float(net.mean() / (tsd / math.sqrt(len(net)))) if tsd and tsd > 0 else float("nan"),
        "total_net_pct": float(net.sum() * 100),
        "avg_daily_net_bps": float(daily.mean() * 1e4),
        "sharpe_daily_ann": float(daily.mean() / sd * math.sqrt(252)) if sd and sd > 0 else float("nan"),
        "max_dd_pct": dd * 100,
        "profit_factor": float(pos / neg) if neg > 0 else float("inf"),
        "avg_win_bps": float(net[net > 0].mean() * 1e4) if (net > 0).any() else 0.0,
        "avg_loss_bps": float(net[net <= 0].mean() * 1e4) if (net <= 0).any() else 0.0,
        "avg_hold_min": float(hold.mean()),
        "pct_long": float((tr["side"] > 0).mean()),
        "net_long_pct": float(net[tr["side"] > 0].sum() * 100),
        "net_short_pct": float(net[tr["side"] < 0].sum() * 100),
    })
    return out


def evaluate(trades, cost_side=COST_BASE):
    """Metricas ALL / IS / OOS para uma lista de trades (dicts de make_trade)."""
    cols = ["date", "side", "entry_m", "exit_m", "entry_px", "exit_px", "reason", "tag"]
    tr = pd.DataFrame(trades, columns=cols)
    dates = _all_dates()
    is_d = [x for x in dates if x <= IS_END]
    oos_d = [x for x in dates if x > IS_END]
    is_set = set(str(x) for x in is_d)
    m_is = tr["date"].isin(is_set)
    out = {"ALL": _stats(tr, dates, cost_side),
           "IS": _stats(tr[m_is], is_d, cost_side),
           "OOS": _stats(tr[~m_is], oos_d, cost_side)}
    per = {str(x): period_of(x) for x in dates}
    tp = tr["date"].map(per)
    for p in ("P0", "P1", "P2"):
        out[p] = _stats(tr[tp == p], [x for x in dates if period_of(x) == p], cost_side)
    return out


def evaluate_is(trades, cost_side=COST_BASE):
    """Use ESTA funcao durante a busca de parametros: so devolve o in-sample."""
    return evaluate(trades, cost_side)["IS"]


def daily_pnl(trades, cost_side=COST_BASE):
    """Serie diaria de PnL liquido (fracao), todas as datas (0 em dia sem trade)."""
    cols = ["date", "side", "entry_m", "exit_m", "entry_px", "exit_px", "reason", "tag"]
    tr = pd.DataFrame(trades, columns=cols)
    idx = [str(x) for x in _all_dates()]
    if len(tr) == 0:
        return pd.Series(0.0, index=idx)
    net = tr["side"] * (tr["exit_px"] / tr["entry_px"] - 1.0) - 2.0 * cost_side / 1e4
    return net.groupby(tr["date"]).sum().reindex(idx).fillna(0.0)


def save_result(family, name, description, params, trades, n_configs_tested,
                selection_rule, live_trigger, notes=""):
    """Grava results/<family>/<name>.json (+ trades em CSV). Chame UMA vez por
    estrategia finalista, depois de escolher os parametros no IS."""
    out_dir = RESULTS / family
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "family": family, "name": name, "description": description, "params": params,
        "n_configs_tested": int(n_configs_tested), "selection_rule": selection_rule,
        "live_trigger": live_trigger, "notes": notes,
        "metrics_base": evaluate(trades, COST_BASE),
        "metrics_stress": evaluate(trades, COST_STRESS),
        "metrics_gross": evaluate(trades, 0.0),
        "daily_pnl_base": {k: float(v) for k, v in daily_pnl(trades, COST_BASE).items()},
    }
    (out_dir / (name + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    pd.DataFrame(trades).to_csv(out_dir / (name + "_trades.csv"), index=False)
    return rec


def fmt(metrics, label=""):
    """Linha-resumo legivel de um bloco de metricas (ex.: evaluate(t)['IS'])."""
    m = metrics
    if m.get("n_trades", 0) == 0:
        return "%s sem trades" % label
    return ("%s n=%d dias=%d win=%.1f%% net=%.1fbps t=%.2f total=%.1f%% sharpe=%.2f dd=%.1f%% pf=%.2f hold=%.0fmin"
            % (label, m["n_trades"], m["days_traded"], m["win_rate"] * 100, m["avg_net_bps"], m["t_stat"],
               m["total_net_pct"], m["sharpe_daily_ann"], m["max_dd_pct"], m["profit_factor"], m["avg_hold_min"]))
