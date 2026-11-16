import os, sys
os.environ['SOXL_RESULTS'] = 'results_v2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import lab
# -*- coding: utf-8 -*-
"""
Familia "gap" -- RODADA 2 (249 pregoes; IS = P0+P1, teste = P2).

Uso (a partir da pasta backtest):
    python3 agents/gap_v2.py --search-only  -> so a busca em P0/P1 (NAO gera nenhum trade em P2)
    python3 agents/gap_v2.py                -> busca + finalistas congelados (grava results_v2/gap/)

Criterio de selecao (so blocos 'P0' e 'P1' de lab.evaluate):
  n>=30 em P0 e em P1; liquido>0 em P0 E em P1 no custo de selecao (base 3 bps/lado para trades 09:30-16:00;
  ESTRESSE 10 bps/lado para qualquer trade com perna fora do pregao regular); ordenar por min(t_P0, t_P1);
  desempate por fracao de vizinhos de grade que tambem passam; preferir compra e venda ambas positivas.
  Ideia 4 (overnight) e so CONTEXTO: contada nas configuracoes testadas, nunca elegivel.

Convencoes de execucao (iguais as da rodada 1, sem look-ahead):
  * Ideias 1, 6, 7 (gap): gap usa o OPEN das 09:30 -> entrada no OPEN das 09:31 (i_entry=1).
  * Ideia 2 (pre-market): sinal ate o close das 09:29 -> entrada no OPEN das 09:30.
  * Ideia 3 (dentro do pre/after) e 4b (after-hours): custo de estresse.
  * Ideia 5 (dia anterior): entrada no OPEN das 09:30.
  * Ideia 6 (NOVA, condicional a regime): 3 variaveis x 3 limiares, so informacao anterior ao trade.
  * Ideia 7 (NOVA): gap que abre FORA vs DENTRO da faixa (max/min) do pregao anterior.
  * atr14 de lab.daily_table() (so dias anteriores). Dias sem atr14 ficam fora de tudo.
"""
import itertools
import numpy as np
import pandas as pd

FAMILY = "gap"
SEARCH_ONLY = "--search-only" in sys.argv

D = lab.load()
T = lab.daily_table()
DAYS = {dt: g for dt, g in lab.day_groups(D)}                       # dia completo (pre+rth+after)
RTH = {dt: g[g["sess"] == "rth"].reset_index(drop=True) for dt, g in DAYS.items()}
PRE = {dt: g[g["sess"] == "pre"].reset_index(drop=True) for dt, g in DAYS.items()}
AFT = {dt: g[g["sess"] == "after"].reset_index(drop=True) for dt, g in DAYS.items()}
ALL_DATES = sorted(DAYS.keys())
IS_DATES = [d for d in ALL_DATES if d <= lab.IS_END]
PREV = {d: (ALL_DATES[i - 1] if i > 0 else None) for i, d in enumerate(ALL_DATES)}


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def ok_day(dt):
    r = T.loc[dt]
    return not (pd.isna(r["atr14"]) or pd.isna(r["prev_c"]))


# ----------------------------------------------------------------------------------------------
# Ideia 1: gap de abertura -- fade vs continuacao
# ----------------------------------------------------------------------------------------------
def gen_gap(dates, action, H, thr_kind, thr, stop_k):
    out = []
    for dt in dates:
        if not ok_day(dt):
            continue
        r = T.loc[dt]; g = RTH[dt]
        gap = g["open"].iloc[0] / r["prev_c"] - 1.0
        size = abs(gap) / r["atr14"] if thr_kind == "atr" else abs(gap)
        if gap == 0 or size < thr:
            continue
        side = -sgn(gap) if action == "fade" else sgn(gap)
        entry = g["open"].iloc[1]
        stop = None if stop_k is None else entry * (1.0 - side * stop_k * r["atr14"])
        res = lab.sim_exit(g, 1, side, stop=stop, i_last=(None if H == "close" else H))
        if res:
            out.append(lab.make_trade(g, res, side, tag="gap_%s" % ("up" if gap > 0 else "dn")))
    return out


GRID1 = [dict(action=a, H=h, thr_kind=k, thr=t, stop_k=s)
         for a in ("fade", "cont")
         for h in (15, 30, 60, 120, "close")
         for (k, t) in (("atr", 0.0), ("atr", 0.15), ("atr", 0.30), ("pct", 0.01), ("pct", 0.02))
         for s in (None, 0.25, 0.50)]


# ----------------------------------------------------------------------------------------------
# Ideia 2: pre-market como preditor do pregao
# ----------------------------------------------------------------------------------------------
def pre_feature(dt, feat):
    p = PRE[dt]
    if len(p) < 60:
        return None
    last = p["close"].iloc[-1]
    if feat == "pre_full":
        return last / p["open"].iloc[0] - 1.0
    if feat in ("pre_60", "pre_120"):
        cut = 570 - (60 if feat == "pre_60" else 120)
        q = p[p["m"] < cut]
        if len(q) == 0:
            return None
        return last / q["close"].iloc[-1] - 1.0
    if feat == "pos":
        hi, lo = p["high"].max(), p["low"].min()
        return None if hi <= lo else (last - lo) / (hi - lo)
    raise ValueError(feat)


def gen_pre(dates, feat, action, thr, H):
    out = []
    for dt in dates:
        if not ok_day(dt):
            continue
        f = pre_feature(dt, feat)
        if f is None:
            continue
        if feat == "pos":
            s = 1 if f >= 1.0 - thr else (-1 if f <= thr else 0)
        else:
            s = sgn(f) if abs(f) / T.loc[dt, "atr14"] >= thr else 0
        if s == 0:
            continue
        side = s if action == "cont" else -s
        g = RTH[dt]
        res = lab.sim_exit(g, 0, side, i_last=(None if H == "close" else H - 1))
        if res:
            out.append(lab.make_trade(g, res, side, tag="%s_%s" % (feat, "hi" if s > 0 else "lo")))
    return out


GRID2 = ([dict(feat=f, action=a, thr=t, H=h)
          for f in ("pre_full", "pre_120", "pre_60") for a in ("cont", "fade")
          for t in (0.0, 0.05, 0.10) for h in (30, 60, "close")] +
         [dict(feat="pos", action=a, thr=t, H=h)
          for a in ("cont", "fade") for t in (0.20, 0.35, 0.50) for h in (30, 60, "close")])


# ----------------------------------------------------------------------------------------------
# Ideia 3: operar dentro do pre-market / after-hours (momentum ou reversao de L min, segura H min)
# ----------------------------------------------------------------------------------------------
def gen_ext(dates, sess, L, H, action, thr):
    out = []
    for dt in dates:
        if not ok_day(dt):
            continue
        g = DAYS[dt]; m = g["m"].values; c = g["close"].values
        atr = T.loc[dt, "atr14"]
        t0, t1 = (lab.PRE_START + L, lab.RTH_START) if sess == "pre" else (lab.RTH_END, lab.AH_END)
        for t in range(t0, t1 - H + 1, H):
            i_sig = np.searchsorted(m, t, side="left") - 1          # ultimo candle com m < t
            i_ref = np.searchsorted(m, t - L, side="left") - 1      # ultimo candle com m < t-L
            i_ent = i_sig + 1
            i_lst = np.searchsorted(m, t + H, side="left") - 1      # ultimo candle com m <= t+H-1
            if i_ref < 0 or i_sig <= i_ref or i_ent >= len(g) or m[i_ent] >= t + H or i_lst < i_ent:
                continue
            if sess == "pre" and m[i_ref] < lab.PRE_START:
                continue
            ret = c[i_sig] / c[i_ref] - 1.0
            if ret == 0 or abs(ret) / atr < thr:
                continue
            side = sgn(ret) if action == "mom" else -sgn(ret)
            res = lab.sim_exit(g, int(i_ent), side, i_last=int(i_lst))
            if res:
                out.append(lab.make_trade(g, res, side, tag="%s_%d" % (sess, t)))
    return out


GRID3 = [dict(sess=s, L=l, H=h, action=a, thr=t)
         for s in ("pre", "after") for l in (30, 60) for h in (30, 60)
         for a in ("mom", "rev") for t in (0.0, 0.03, 0.06)]


# ----------------------------------------------------------------------------------------------
# Ideia 4: overnight (contexto)
# ----------------------------------------------------------------------------------------------
def gen_on(dates, seg, cond, thr, side):
    """seg: c2o (close 15:59 -> open 09:30 seguinte), c2ah (close -> 19:59), ah2pre (19:59 -> 04:00 seguinte),
    pre2o (04:00 -> 09:30 do dia seguinte). cond sobre o dia de ENTRADA (dia d): open -> close das 15:58,
    em multiplos do atr14 de d: 'all', 'dn' (ret <= -thr*atr), 'up' (ret >= thr*atr). date = dia da SAIDA."""
    out = []
    dset = set(dates)
    for d in ALL_DATES:
        if not ok_day(d):
            continue
        g = RTH[d]
        dret = g["close"].iloc[388] / g["open"].iloc[0] - 1.0
        x = dret / T.loc[d, "atr14"]
        if cond == "dn" and not (x <= -thr and dret < 0):
            continue
        if cond == "up" and not (x >= thr and dret > 0):
            continue
        i = ALL_DATES.index(d)
        nxt = ALL_DATES[i + 1] if i + 1 < len(ALL_DATES) else None
        c = g["close"].iloc[-1]
        if seg == "c2ah":
            a = AFT[d]
            if len(a) == 0 or d not in dset:
                continue
            out.append(lab.trade_from_prices(d, side, 960, c, 1199, a["close"].iloc[-1], tag=seg + "_" + cond))
            continue
        if nxt is None or nxt not in dset:
            continue
        if seg == "c2o":
            out.append(lab.trade_from_prices(nxt, side, 960, c, 570 + 1440, RTH[nxt]["open"].iloc[0], tag=seg + "_" + cond))
        elif seg == "ah2pre":
            a, p = AFT[d], PRE[nxt]
            if len(a) and len(p):
                out.append(lab.trade_from_prices(nxt, side, 1199, a["close"].iloc[-1], 240 + 1440, p["open"].iloc[0], tag=seg + "_" + cond))
        elif seg == "pre2o":
            p = PRE[nxt]
            if len(p):
                out.append(lab.trade_from_prices(nxt, side, 240, p["open"].iloc[0], 570, RTH[nxt]["open"].iloc[0], tag=seg + "_" + cond))
    return out


def gen_ahmom(dates, thr):
    """4b: continuacao do dia no after-hours (simetrica): dia de alta (open->15:58 >= thr*atr14) compra no close
    das 15:59; dia de queda (<= -thr*atr14) vende; sai no close das 19:59 do MESMO dia."""
    return gen_on(dates, "c2ah", "up", thr, 1) + gen_on(dates, "c2ah", "dn", thr, -1)


GRID4B = [dict(thr=t) for t in (0.0, 0.25, 0.5)]
GRID4 = [dict(seg=s, cond=c, thr=t, side=sd)
         for s in ("c2o", "c2ah", "ah2pre", "pre2o")
         for (c, t) in (("all", 0.0), ("dn", 0.0), ("dn", 0.25), ("dn", 0.5), ("up", 0.0), ("up", 0.25), ("up", 0.5))
         for sd in (1, -1)]
ON_COST = {"c2o": lab.COST_BASE, "c2ah": lab.COST_STRESS, "ah2pre": lab.COST_STRESS, "pre2o": lab.COST_STRESS}


# ----------------------------------------------------------------------------------------------
# Ideia 5: efeito do dia anterior no intraday
# ----------------------------------------------------------------------------------------------
def gen_prev(dates, meas, action, thr, H):
    out = []
    for dt in dates:
        pv = PREV[dt]
        if pv is None or not ok_day(dt):
            continue
        rp = T.loc[pv]
        if meas == "oc":
            x = rp["oc"]
        else:
            if pd.isna(rp["prev_c"]):
                continue
            x = rp["c"] / rp["prev_c"] - 1.0
        if x == 0 or abs(x) / T.loc[dt, "atr14"] < thr:
            continue
        side = sgn(x) if action == "cont" else -sgn(x)
        g = RTH[dt]
        res = lab.sim_exit(g, 0, side, i_last=(None if H == "close" else H - 1))
        if res:
            out.append(lab.make_trade(g, res, side, tag="prev_%s" % ("up" if x > 0 else "dn")))
    return out


GRID5 = [dict(meas=m, action=a, thr=t, H=h)
         for m in ("oc", "cc") for a in ("cont", "rev")
         for t in (0.0, 0.25, 0.50, 0.75) for h in (60, "close")]


# ----------------------------------------------------------------------------------------------
# Ideia 6 (NOVA): gap condicional a regime. Variavel de regime v (so informacao anterior a entrada):
#   r10   = close(d-1)/close(d-11) - 1          (tendencia dos 10 pregoes anteriores)
#   atr14 = amplitude media dos 14 pregoes anteriores
#   gapsz = |gap do dia| / atr14                 (conhecido as 09:30:00; entrada 09:31)
# orient 'hi_cont': v > thr -> segue o gap; v <= thr -> fade.  'hi_fade': o inverso.
# ----------------------------------------------------------------------------------------------
_C = T["c"]
R10 = (_C.shift(1) / _C.shift(11) - 1.0)


def regime_value(dt, var, gap):
    if var == "r10":
        return R10.loc[dt]
    if var == "atr14":
        return T.loc[dt, "atr14"]
    if var == "gapsz":
        return abs(gap) / T.loc[dt, "atr14"]
    raise ValueError(var)


def gen_gapreg(dates, var, thr, orient, H):
    out = []
    for dt in dates:
        if not ok_day(dt):
            continue
        r = T.loc[dt]; g = RTH[dt]
        gap = g["open"].iloc[0] / r["prev_c"] - 1.0
        v = regime_value(dt, var, gap)
        if gap == 0 or pd.isna(v):
            continue
        hi = v > thr
        cont = hi if orient == "hi_cont" else (not hi)
        side = sgn(gap) if cont else -sgn(gap)
        res = lab.sim_exit(g, 1, side, i_last=(None if H == "close" else H))
        if res:
            out.append(lab.make_trade(g, res, side, tag="%s_%s_%s" % (var, "hi" if hi else "lo", "cont" if cont else "fade")))
    return out


REG_THR = {"r10": (-0.10, 0.0, 0.10), "atr14": (0.08, 0.10, 0.12), "gapsz": (0.15, 0.30, 0.50)}
GRID6 = [dict(var=v, thr=t, orient=o, H=h)
         for v in ("r10", "atr14", "gapsz") for t in REG_THR[v]
         for o in ("hi_cont", "hi_fade") for h in (30, 60, "close")]


# ----------------------------------------------------------------------------------------------
# Ideia 7 (NOVA): gap FORA vs DENTRO da faixa do pregao anterior
# ----------------------------------------------------------------------------------------------
def gen_gaprange(dates, where, action, H):
    out = []
    for dt in dates:
        pv = PREV[dt]
        if pv is None or not ok_day(dt):
            continue
        r = T.loc[dt]; g = RTH[dt]; o = g["open"].iloc[0]
        gap = o / r["prev_c"] - 1.0
        if gap == 0:
            continue
        outside = (o > T.loc[pv, "h"]) or (o < T.loc[pv, "l"])
        if (where == "out") != outside:
            continue
        side = -sgn(gap) if action == "fade" else sgn(gap)
        res = lab.sim_exit(g, 1, side, i_last=(None if H == "close" else H))
        if res:
            out.append(lab.make_trade(g, res, side, tag="gap_%s_%s" % (where, "up" if gap > 0 else "dn")))
    return out


GRID7 = [dict(where=w, action=a, H=h) for w in ("out", "in") for a in ("fade", "cont") for h in (30, 60, "close")]

# nome -> (gerador, grade, custo de selecao, elegivel a finalista)
IDEAS = {
    "1_gap": (gen_gap, GRID1, lambda p: lab.COST_BASE, True),
    "2_pre": (gen_pre, GRID2, lambda p: lab.COST_BASE, True),
    "3_ext": (gen_ext, GRID3, lambda p: lab.COST_STRESS, True),
    "4_on": (gen_on, GRID4, lambda p: ON_COST[p["seg"]], False),      # contexto
    "4b_ahmom": (gen_ahmom, GRID4B, lambda p: lab.COST_STRESS, True),
    "5_prev": (gen_prev, GRID5, lambda p: lab.COST_BASE, True),
    "6_gapreg": (gen_gapreg, GRID6, lambda p: lab.COST_BASE, True),
    "7_gaprange": (gen_gaprange, GRID7, lambda p: lab.COST_BASE, True),
}
N_CONFIGS = sum(len(v[1]) for v in IDEAS.values())
N_REGIME_CONFIGS = len(GRID6)
PER = ("P0", "P1")                      # UNICOS blocos lidos durante a busca
NO_NEIGH = ("action", "side", "cond", "seg", "feat", "meas", "sess", "thr_kind", "orient", "var", "where")


def per_row(trades, cost):
    """Metricas SOMENTE dos blocos P0 e P1 de lab.evaluate: total, so compras, so vendas."""
    ev = lab.evaluate(trades, cost)
    evL = lab.evaluate([x for x in trades if x["side"] == 1], cost)
    evS = lab.evaluate([x for x in trades if x["side"] == -1], cost)
    row = {}
    for p in PER:
        m = ev[p]
        row["n_" + p] = m.get("n_trades", 0)
        row["net_" + p] = m.get("avg_net_bps", np.nan)
        row["t_" + p] = m.get("t_stat", np.nan)
        row["nL_" + p] = evL[p].get("n_trades", 0); row["netL_" + p] = evL[p].get("avg_net_bps", np.nan)
        row["nS_" + p] = evS[p].get("n_trades", 0); row["netS_" + p] = evS[p].get("avg_net_bps", np.nan)
    return row


def add_criteria(df, grid):
    df["pass"] = ((df["n_P0"] >= 30) & (df["n_P1"] >= 30) & (df["net_P0"] > 0) & (df["net_P1"] > 0))
    df["min_t"] = df[["t_P0", "t_P1"]].min(axis=1)
    df["ls_ok"] = ((df["netL_P0"] > 0) & (df["netS_P0"] > 0) & (df["netL_P1"] > 0) & (df["netS_P1"] > 0))
    # vizinhos: mesma ideia, difere em exatamente 1 parametro ordenavel, em posicao adjacente da grade
    keys = list(grid[0].keys())
    order = {k: [] for k in keys}
    for p in grid:
        for k in keys:
            if p[k] not in order[k]:
                order[k].append(p[k])
    nb_n, nb_ok = [], []
    for i, p in enumerate(grid):
        n = ok = 0
        for j, q in enumerate(grid):
            diff = [k for k in keys if p[k] != q[k]]
            if len(diff) != 1 or diff[0] in NO_NEIGH:
                continue
            k = diff[0]
            if abs(order[k].index(p[k]) - order[k].index(q[k])) != 1:
                continue
            n += 1; ok += int(bool(df["pass"].iloc[j]))
        nb_n.append(n); nb_ok.append(ok)
    df["nb_n"] = nb_n; df["nb_pass"] = nb_ok
    df["nb_frac"] = [o / n if n else np.nan for o, n in zip(nb_ok, nb_n)]
    return df


def search():
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 600); pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    (lab.RESULTS / FAMILY).mkdir(parents=True, exist_ok=True)
    for sd, nm in ((-1, "VENDIDO"), (1, "COMPRADO")):
        b = [lab.make_trade(RTH[d], lab.sim_exit(RTH[d], 0, sd), sd) for d in IS_DATES]
        ev = lab.evaluate(b)
        for p in PER:
            print(lab.fmt(ev[p], "BENCH %s %s open->close:" % (p, nm)))
    allrows = []
    for name, (gen, grid, costf, elig) in IDEAS.items():
        rows = []
        for p in grid:
            tr = gen(IS_DATES, **p)                       # so datas <= IS_END: P2 nem e simulado
            r = dict(p); r["cost"] = costf(p); r.update(per_row(tr, costf(p)))
            rows.append(r)
        df = add_criteria(pd.DataFrame(rows), grid)
        df["eligible"] = elig
        df.to_csv(lab.RESULTS / FAMILY / ("search_p0p1_%s.csv" % name), index=False)
        print("\n===== IDEIA %s : %d configs | passam no criterio dos dois regimes: %d%s =====" % (
            name, len(df), int(df["pass"].sum()), "" if elig else "  (CONTEXTO, nao elegivel)"))
        print(df.sort_values("min_t", ascending=False).head(12).to_string(index=False))
        d2 = df.copy(); d2.insert(0, "idea", name)
        d2.insert(1, "cfg", [", ".join("%s=%s" % kv for kv in p.items()) for p in grid])
        allrows.append(d2[["idea", "cfg", "cost", "eligible", "n_P0", "net_P0", "t_P0", "n_P1", "net_P1", "t_P1",
                           "netL_P0", "netS_P0", "netL_P1", "netS_P1", "pass", "min_t", "ls_ok", "nb_pass", "nb_n", "nb_frac"]])
    A = pd.concat(allrows, ignore_index=True)
    A = A.sort_values(["min_t", "nb_frac"], ascending=False).reset_index(drop=True)
    A.to_csv(lab.RESULTS / FAMILY / "search_p0p1_ALL.csv", index=False)
    E = A[A["eligible"]].reset_index(drop=True)
    E["rank"] = np.arange(1, len(E) + 1)
    P = E[E["pass"]]
    print("\n===== RANKING GERAL (elegiveis que PASSAM; ordenado por min(t_P0,t_P1), desempate nb_frac) =====")
    print(P.drop(columns=["eligible", "pass"]).to_string(index=False))
    print("\nTotal de configuracoes testadas: %d (das quais %d condicionais a regime, %d de contexto overnight)" % (
        N_CONFIGS, N_REGIME_CONFIGS, len(GRID4)))
    print("Elegiveis: %d | passam no criterio dos dois regimes: %d | passam com compra e venda positivas nos dois: %d" % (
        len(E), len(P), int(P["ls_ok"].sum())))
    print("Sinais: P0>0 e P1>0: %d | P0>0 so: %d | P1>0 so: %d | nenhum: %d  (elegiveis, sem filtro de n)" % (
        int(((E.net_P0 > 0) & (E.net_P1 > 0)).sum()), int(((E.net_P0 > 0) & ~(E.net_P1 > 0)).sum()),
        int((~(E.net_P0 > 0) & (E.net_P1 > 0)).sum()), int((~(E.net_P0 > 0) & ~(E.net_P1 > 0)).sum())))
    print("corr(t_P0, t_P1) entre configs elegiveis com n>=30 nos dois: %.2f" % (
        E[(E.n_P0 >= 30) & (E.n_P1 >= 30)][["t_P0", "t_P1"]].corr().iloc[0, 1]))
    print("\n--- Finalistas da RODADA 1 na nova busca ---")
    for cfg in ("action=fade, H=60, thr_kind=atr, thr=0.0, stop_k=None", "thr=0.25"):
        r = E[(E["cfg"] == cfg) & (E["idea"].isin(["1_gap", "4b_ahmom"]))]
        print(r.drop(columns=["eligible"]).to_string(index=False))
    return A


def facts(dates, label):
    rows = []
    for dt in dates:
        if not ok_day(dt):
            continue
        r = T.loc[dt]; g = RTH[dt]; o = g["open"].iloc[0]; c = g["close"].values
        p = PRE[dt]; a = AFT[dt]
        rows.append(dict(date=dt, gap=r["gap"], gap_atr=r["gap"] / r["atr14"], atr14=r["atr14"],
                         r30=c[29] / o - 1, r60=c[59] / o - 1, roc=c[-1] / o - 1,
                         pre=(p["close"].iloc[-1] / p["open"].iloc[0] - 1) if len(p) else np.nan,
                         ah=(a["close"].iloc[-1] / c[-1] - 1) if len(a) else np.nan,
                         prev_oc=T.loc[PREV[dt], "oc"] if PREV[dt] else np.nan))
    f = pd.DataFrame(rows)
    up, dn = f[f.gap > 0], f[f.gap < 0]
    print("\n--- FATOS %s (n=%d): bruto, bps ---" % (label, len(f)))
    print("gap medio=%.0f  |gap| mediano=%.0f  |gap|/atr14 mediano=%.2f  atr14 medio=%.1f%%  gaps de alta=%.0f%%" % (
        f.gap.mean() * 1e4, f.gap.abs().median() * 1e4, f.gap_atr.abs().median(), f.atr14.mean() * 100, (f.gap > 0).mean() * 100))
    print("apos gap de ALTA  (n=%d): r30=%.0f r60=%.0f open->close=%.0f" % (len(up), up.r30.mean() * 1e4, up.r60.mean() * 1e4, up.roc.mean() * 1e4))
    print("apos gap de BAIXA (n=%d): r30=%.0f r60=%.0f open->close=%.0f" % (len(dn), dn.r30.mean() * 1e4, dn.r60.mean() * 1e4, dn.roc.mean() * 1e4))
    print("media: pre(04:00->09:29)=%.0f  open->close=%.0f  after(16:00->19:59)=%.0f  soma dos gaps=%.0f%%" % (
        f.pre.mean() * 1e4, f.roc.mean() * 1e4, f.ah.mean() * 1e4, f.gap.sum() * 100))
    print("corr(gap,r60)=%.2f corr(gap,open->close)=%.2f corr(pre,r60)=%.2f corr(prev_oc,open->close)=%.2f" % (
        f.gap.corr(f.r60), f.gap.corr(f.roc), f.pre.corr(f.r60), f.prev_oc.corr(f.roc)))
    return f


# ----------------------------------------------------------------------------------------------
# Finalistas (CONGELADOS com base so em P0/P1; preenchido depois da busca)
# ----------------------------------------------------------------------------------------------
FINALISTS = [
    dict(idea="4b_ahmom", name="afterhours_day_continuation_v2",
         params=dict(thr=0.25),
         description="Continuacao do dia no after-hours: se o retorno open 09:30 -> close 15:58 >= +0,25*atr14 compra "
                     "no close das 15:59; se <= -0,25*atr14 vende; zera no close das 19:59 do mesmo dia.",
         selection_rule="Rodada 2: 1o lugar por min(t_P0,t_P1) entre 371 configs elegiveis (min t = 1,07; P0 t=1,07, "
                        "P1 t=1,30), liquido>0 nos dois regimes no custo de ESTRESSE (10 bps/lado), n>=30 em ambos. "
                        "Vizinhos: thr=0,0 tambem passa (min t 0,45); thr=0,5 nao.",
         live_trigger="As 15:59:00 calcular ret = close do candle 15:58 / abertura das 09:30 - 1 e atr14 = media de "
                      "(maxima/minima - 1) do pregao regular dos 14 dias anteriores. Se ret >= +0,25*atr14: COMPRAR no "
                      "fechamento das 16:00. Se ret <= -0,25*atr14: VENDER. Caso contrario nao operar. Zerar as 20:00 "
                      "(ultimo negocio do after-hours).",
         notes="MENOS PIOR, NAO SIGNIFICATIVO: min t = 1,07 depois de 427 configuracoes testadas nao distingue de ruido. "
               "Ponta VENDIDA NEGATIVA em P0 (-45,6 bps/trade no estresse): nao satisfaz 'compra e venda ambas "
               "positivas'. Perna fora do pregao: criterio e o custo de ESTRESSE -> olhar metrics_stress, nao "
               "metrics_base. Liquidez das 19:59 e baixa."),
    dict(idea="6_gapreg", name="gap_fade_60m_unless_r10_crash",
         params=dict(var="r10", thr=-0.10, orient="hi_fade", H=60),
         description="Gap de abertura condicional a tendencia: r10 = close(d-1)/close(d-11)-1. Se r10 > -10%: FADE do "
                     "gap (gap de alta vende, gap de baixa compra); se r10 <= -10%: SEGUE o gap. Entrada no open das "
                     "09:31, saida por tempo 60 min depois (close do candle das 10:30). Sem stop.",
         selection_rule="Rodada 2: 2o lugar por min(t_P0,t_P1) (min t = 0,72; P0 t=0,92, P1 t=0,72), liquido>0 nos dois "
                        "regimes no custo base, n>=30 em ambos; 2 de 3 vizinhos de grade tambem passam. Regra "
                        "condicional a regime (1 de 54 configs condicionais).",
         live_trigger="Antes da abertura calcular r10 = fechamento de ontem / fechamento de 11 pregoes atras - 1. As "
                      "09:30:00 comparar a abertura com o fechamento das 16:00 de ontem. Se r10 > -10%: abertura acima "
                      "-> VENDER, abaixo -> COMPRAR. Se r10 <= -10%: abertura acima -> COMPRAR, abaixo -> VENDER. "
                      "Executar a mercado as 09:31:00, sem stop. Zerar as 10:31:00. Um trade por dia.",
         notes="MENOS PIOR, NAO SIGNIFICATIVO: min t = 0,72 depois de 427 configuracoes. Ponta COMPRADA NEGATIVA em P1 "
               "(-26,6 bps/trade): resultado de P1 vem todo da ponta vendida, que coincide com a deriva de queda "
               "intraday de P1. Nao satisfaz 'compra e venda ambas positivas'."),
]


def finals():
    for fz in FINALISTS:
        gen = IDEAS[fz["idea"]][0]
        tr = gen(ALL_DATES, **fz["params"])
        rec = lab.save_result(family=FAMILY, name=fz["name"], description=fz["description"], params=fz["params"],
                              trades=tr, n_configs_tested=N_CONFIGS, selection_rule=fz["selection_rule"],
                              live_trigger=fz["live_trigger"], notes=fz["notes"])
        print("\n### %s" % fz["name"])
        for k in ("P0", "P1", "P2"):
            print(lab.fmt(rec["metrics_base"][k], "  base   %s" % k))
            print(lab.fmt(rec["metrics_stress"][k], "  stress %s" % k))
            for lbl, s in (("compra", 1), ("venda", -1)):
                sub = [x for x in tr if x["side"] == s]
                print(lab.fmt(lab.evaluate(sub)[k], "    %s %s (base)" % (lbl, k)))
                print(lab.fmt(lab.evaluate(sub, lab.COST_STRESS)[k], "    %s %s (stress)" % (lbl, k)))


if __name__ == "__main__":
    search()
    facts([d for d in IS_DATES if lab.period_of(d) == "P0"], "P0")
    facts([d for d in IS_DATES if lab.period_of(d) == "P1"], "P1")
    if not SEARCH_ONLY:
        finals()
        facts([d for d in ALL_DATES if d > lab.IS_END], "P2 (so depois de congelar)")
