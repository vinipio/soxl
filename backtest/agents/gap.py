# -*- coding: utf-8 -*-
"""
Familia "gap": gaps de abertura, pre-market, after-hours, overnight e efeito do dia anterior no SOXL (1m).

Uso (a partir de a pasta backtest):
    python3 agents/gap.py            -> refaz a busca no IS, grava results/gap/search_is_*.csv e os finalistas
    python3 agents/gap.py --is-only  -> so a busca no IS (nao toca no OOS)

Convencoes de execucao (sem look-ahead):
  * Ideia 1 (gap): o gap usa o OPEN das 09:30, que so e conhecido as 09:30:00 -> a entrada e no OPEN do candle
    das 09:31 (i_entry=1). Saida no CLOSE do candle de indice H (= 09:30 + H min) ou no fechamento.
  * Ideia 2 (pre-market): sinal com dados ate o close das 09:29 -> entrada no OPEN das 09:30 (i_entry=0).
  * Ideia 3 (dentro do pre/after): sinal ate o close do candle anterior a t, entrada no open do candle t.
    Selecao pelo custo de ESTRESSE (10 bps/lado).
  * Ideia 4 (overnight): contexto; lab.trade_from_prices, date = dia da SAIDA. Condicao do dia usa dados ate o
    close das 15:58; entrada no close das 15:59 (leilao de fechamento).
  * Ideia 5 (dia anterior): sinal do dia anterior -> entrada no OPEN das 09:30.
  * atr14 de lab.daily_table() (so dias anteriores). Dias sem atr14 (5 primeiros) ficam fora de tudo.
"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0]); import lab
import itertools
import numpy as np
import pandas as pd

FAMILY = "gap"
IS_ONLY = "--is-only" in sys.argv

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

IDEAS = {
    "1_gap": (gen_gap, GRID1, lambda p: lab.COST_BASE),
    "2_pre": (gen_pre, GRID2, lambda p: lab.COST_BASE),
    "3_ext": (gen_ext, GRID3, lambda p: lab.COST_STRESS),
    "4_on": (gen_on, GRID4, lambda p: ON_COST[p["seg"]]),
    "4b_ahmom": (gen_ahmom, GRID4B, lambda p: lab.COST_STRESS),
    "5_prev": (gen_prev, GRID5, lambda p: lab.COST_BASE),
}
N_CONFIGS = sum(len(v[1]) for v in IDEAS.values())


def is_row(trades, cost):
    """Metricas IS: total, so compras, so vendas."""
    m = lab.evaluate_is(trades, cost)
    row = dict(n=m.get("n_trades", 0), net=m.get("avg_net_bps", np.nan), t=m.get("t_stat", np.nan),
               sh=m.get("sharpe_daily_ann", np.nan), tot=m.get("total_net_pct", np.nan), win=m.get("win_rate", np.nan))
    for lbl, s in (("L", 1), ("S", -1)):
        ms = lab.evaluate_is([x for x in trades if x["side"] == s], cost)
        row.update({"n" + lbl: ms.get("n_trades", 0), "net" + lbl: ms.get("avg_net_bps", np.nan),
                    "t" + lbl: ms.get("t_stat", np.nan)})
    return row


def search():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    (lab.RESULTS / FAMILY).mkdir(parents=True, exist_ok=True)
    # benchmark de vies direcional (IS)
    for sd, nm in ((-1, "VENDIDO"), (1, "COMPRADO")):
        b = [lab.make_trade(RTH[d], lab.sim_exit(RTH[d], 0, sd), sd) for d in IS_DATES]
        print(lab.fmt(lab.evaluate_is(b), "BENCH IS %s open->close:" % nm))
    tables = {}
    for name, (gen, grid, costf) in IDEAS.items():
        rows = []
        for p in grid:
            tr = gen(IS_DATES, **p)
            r = dict(p); r["cost"] = costf(p); r.update(is_row(tr, costf(p)))
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(lab.RESULTS / FAMILY / ("search_is_%s.csv" % name), index=False)
        tables[name] = df
        print("\n===== IDEIA %s : %d configs (IS, custo de selecao na coluna cost) =====" % (name, len(df)))
        print(df.to_string(index=False))
    print("\nTotal de configuracoes testadas: %d" % N_CONFIGS)
    return tables


def facts(dates, label):
    """Fatos descritivos (media em bps, bruto) por bucket de gap."""
    rows = []
    for dt in dates:
        if not ok_day(dt):
            continue
        r = T.loc[dt]; g = RTH[dt]; o = g["open"].iloc[0]; c = g["close"].values
        p = PRE[dt]; a = AFT[dt]
        rows.append(dict(date=dt, gap=r["gap"], gap_atr=r["gap"] / r["atr14"],
                         r15=c[14] / o - 1, r30=c[29] / o - 1, r60=c[59] / o - 1, r120=c[119] / o - 1, roc=c[-1] / o - 1,
                         pre=(p["close"].iloc[-1] / p["open"].iloc[0] - 1), ah=(a["close"].iloc[-1] / c[-1] - 1),
                         prev_oc=T.loc[PREV[dt], "oc"] if PREV[dt] else np.nan))
    f = pd.DataFrame(rows)
    f["bucket"] = pd.cut(f["gap_atr"], [-9, -0.3, -0.1, 0, 0.1, 0.3, 9])
    print("\n--- FATOS (%s): retorno bruto medio em bps a partir do open 09:30, por gap/atr14 ---" % label)
    agg = f.groupby("bucket", observed=True)[["r15", "r30", "r60", "r120", "roc"]].mean() * 1e4
    agg["n"] = f.groupby("bucket", observed=True).size()
    print(agg.to_string())
    print("gap medio=%.0fbps |gap| mediano=%.0fbps  |gap|/atr14 mediano=%.2f  gaps de alta=%.0f%%" % (
        f["gap"].mean() * 1e4, f["gap"].abs().median() * 1e4, f["gap_atr"].abs().median(), (f["gap"] > 0).mean() * 100))
    print("media bps: pre-market(04:00->09:29)=%.0f  open->close=%.0f  after(16:00->20:00)=%.0f" % (
        f["pre"].mean() * 1e4, f["roc"].mean() * 1e4, f["ah"].mean() * 1e4))
    print("corr(gap, open->close)=%.2f  corr(pre, r60)=%.2f  corr(prev_oc, open->close)=%.2f  corr(gap, r30)=%.2f" % (
        f["gap"].corr(f["roc"]), f["pre"].corr(f["r60"]), f["prev_oc"].corr(f["roc"]), f["gap"].corr(f["r30"])))
    return f


# ----------------------------------------------------------------------------------------------
# Finalistas (parametros CONGELADOS a partir do IS; preenchido depois da busca)
# ----------------------------------------------------------------------------------------------
FINALISTS = [
    dict(idea="1_gap", name="gap_fade_60m",
         params=dict(action="fade", H=60, thr_kind="atr", thr=0.0, stop_k=None),
         description="Fade do gap de abertura: open 09:30 acima do close anterior -> vende; abaixo -> compra. "
                     "Entrada no open das 09:31, saida por tempo 60 min depois (close do candle das 10:30). Sem stop.",
         selection_rule="Maior t-stat IS (custo base) na ideia 1 com n>=30, dentro de regiao estavel: fade em 30 e 60 min "
                        "positivo para todos os limiares de gap (t 0,95-1,86) e continuacao negativa nos mesmos pontos; "
                        "empate de t (1,80) entre 30 e 60 min resolvido pelo maior net/trade. Unica regra com as duas "
                        "pontas positivas no IS apesar da deriva de queda intraday.",
         live_trigger="As 09:30:00 comparar o preco de abertura do pregao regular com o fechamento das 16:00 do pregao "
                      "anterior. Se abertura > fechamento anterior: VENDER a mercado as 09:31:00. Se abertura < fechamento "
                      "anterior: COMPRAR as 09:31:00. Sem filtro de tamanho e sem stop. Zerar as 10:31:00 (60 min). "
                      "Um trade por dia.",
         notes="IS t=1,80: NAO e estatisticamente significativo (73 trades). Ponta vendida (gap de alta) responde pela "
               "maior parte do resultado e coincide com a deriva de queda intraday do IS; ponta comprada positiva mas "
               "fraca (t~0,55). Stops de 0,25/0,5 atr14 pioraram a ponta comprada. Gaps tipicos sao enormes "
               "(|gap| mediano ~5,8% no IS), risco por trade alto."),
    dict(idea="4b_ahmom", name="afterhours_day_continuation",
         params=dict(thr=0.25),
         description="Continuacao do dia no after-hours: se o retorno open 09:30 -> close 15:58 >= +0,25*atr14 compra "
                     "no close das 15:59; se <= -0,25*atr14 vende; zera no close das 19:59 do mesmo dia.",
         selection_rule="Maior t-stat IS no custo de ESTRESSE (10 bps/lado) entre os 3 limiares com n>=30; vizinhos "
                        "(0,0 e 0,5) tambem positivos.",
         live_trigger="As 15:59:00 calcular ret = ultimo preco (close do candle 15:58) / abertura das 09:30 - 1 e "
                      "atr14 = media de (maxima/minima - 1) do pregao regular dos 14 dias anteriores. Se ret >= "
                      "+0,25*atr14: COMPRAR no leilao/fechamento das 16:00. Se ret <= -0,25*atr14: VENDER. Caso "
                      "contrario nao operar. Zerar as 20:00 (ultimo negocio do after-hours).",
         notes="MENOS PIOR fora da ideia de gap -- efeito fraco (IS t=1,16 no estresse, 49 trades). Saida executada "
               "FORA do pregao (after-hours): selecionado pelo custo de ESTRESSE 10 bps/lado via "
               "lab.evaluate_is(cost_side=lab.COST_STRESS); olhar metrics_stress, nao metrics_base. Posicao aberta e "
               "fechada no mesmo dia (nao e overnight). Liquidez das 19:59 e baixa (mediana ~170 negocios/min no after)."),
]

# Contexto (NAO finalistas, nao gravados): overnight. "posicao overnight" -- nao e day trade.
CONTEXT = [
    ("overnight close15:59->open09:30 COMPRADO, todos os dias (posicao overnight)", gen_on, dict(seg="c2o", cond="all", thr=0.0, side=1), lab.COST_BASE),
    ("overnight c2o COMPRADO apos dia de QUEDA (posicao overnight)", gen_on, dict(seg="c2o", cond="dn", thr=0.0, side=1), lab.COST_BASE),
    ("overnight c2o COMPRADO apos dia de ALTA (posicao overnight)", gen_on, dict(seg="c2o", cond="up", thr=0.0, side=1), lab.COST_BASE),
    ("19:59->04:00 COMPRADO (posicao overnight, custo estresse)", gen_on, dict(seg="ah2pre", cond="all", thr=0.0, side=1), lab.COST_STRESS),
    ("pre-market 04:00->09:30 COMPRADO (custo estresse)", gen_on, dict(seg="pre2o", cond="all", thr=0.0, side=1), lab.COST_STRESS),
    ("after 16:00->19:59 COMPRADO (custo estresse)", gen_on, dict(seg="c2ah", cond="all", thr=0.0, side=1), lab.COST_STRESS),
]


def finals():
    for fz in FINALISTS:
        gen = IDEAS[fz["idea"]][0]
        tr = gen(ALL_DATES, **fz["params"])
        rec = lab.save_result(family=FAMILY, name=fz["name"], description=fz["description"], params=fz["params"],
                              trades=tr, n_configs_tested=N_CONFIGS, selection_rule=fz["selection_rule"],
                              live_trigger=fz["live_trigger"], notes=fz["notes"])
        print("\n### %s" % fz["name"])
        for k in ("IS", "OOS"):
            print(lab.fmt(rec["metrics_base"][k], "  base   %s" % k))
            print(lab.fmt(rec["metrics_stress"][k], "  stress %s" % k))
            for lbl, s in (("compra", 1), ("venda", -1)):
                sub = [x for x in tr if x["side"] == s]
                print(lab.fmt(lab.evaluate(sub)[k], "    %s %s (base)" % (lbl, k)))
                print(lab.fmt(lab.evaluate(sub, lab.COST_STRESS)[k], "    %s %s (stress)" % (lbl, k)))
    print("\n### CONTEXTO (nao finalistas; date = dia da saida)")
    for nm, gen, p, cost in CONTEXT:
        ev = lab.evaluate(gen(ALL_DATES, **p), cost)
        print(nm); print(lab.fmt(ev["IS"], "   IS ")); print(lab.fmt(ev["OOS"], "   OOS"))


if __name__ == "__main__":
    search()
    facts(IS_DATES, "IS")
    if not IS_ONLY:
        finals()
        facts([d for d in ALL_DATES if d > lab.IS_END], "OOS (so depois de congelar)")
        for sd, nm in ((-1, "VENDIDO"), (1, "COMPRADO")):
            b = [lab.make_trade(RTH[d], lab.sim_exit(RTH[d], 0, sd), sd) for d in ALL_DATES]
            print(lab.fmt(lab.evaluate(b)["OOS"], "BENCH OOS %s open->close:" % nm))
