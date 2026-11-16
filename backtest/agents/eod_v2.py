import os, sys
os.environ['SOXL_RESULTS'] = 'results_v2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import lab
# -*- coding: utf-8 -*-
"""
Familia "eod" -- RODADA 2 (249 pregoes, IS = P0+P1, teste = P2).

Rodar:  python3 agents/eod_v2.py                   -> busca (so P0/P1) + sazonalidade P0/P1/P2 + grava finalistas
        EOD_STAGE=search python3 agents/eod_v2.py  -> so a busca; NENHUM numero de P2/OOS/ALL e lido ou impresso

Criterio (BRIEF2): n>=30 em P0 e em P1; liquido>0 no custo base em P0 E em P1; ordenar por min(t_P0, t_P1).
A busca recebe SOMENTE dias <= lab.IS_END e le SOMENTE os blocos 'P0' e 'P1' de lab.evaluate.

Convencoes de horario (HHMM, Nova York):
  * "sinal ate T"  = fechamento do candle T-1min (informacao disponivel em T:00).
  * "entrada em T" = OPEN do candle T.
  * "saida em T"   = CLOSE do candle T-1min. Saida 1600 = close do candle 15:59 (ultimo do pregao regular).
"""
import math
import numpy as np
import pandas as pd

STAGE = os.environ.get("EOD_STAGE", "full")
FAMILY = "eod"
OUT = lab.RESULTS / FAMILY
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 280, "display.max_rows", 2000, "display.max_columns", 60)


def mi(hhmm):
    return (hhmm // 100) * 60 + hhmm % 100


def hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


RTH = lab.rth()
DAYS = lab.day_groups(RTH)
for _d, _g in DAYS:
    assert len(_g) == 390 and _g["m"].iloc[0] == 570, _d
DT = lab.daily_table()
# variaveis de regime: so informacao anterior ao trade
DT["ret5"] = DT["prev_c"] / DT["prev_c"].shift(5) - 1.0      # retorno dos 5 pregoes anteriores (fech. a fech.)
DT["absgap"] = DT["gap"].abs()                                # gap do dia (conhecido as 09:30)
FULL = lab.day_groups(lab.load())
IS_DAYS = [(d, g) for d, g in DAYS if d <= lab.IS_END]
IS_FULL = [(d, g) for d, g in FULL if d <= lab.IS_END]
PER = ("P0", "P1")          # unicos blocos lidos durante a busca


# ----------------------------------------------------------------------------- geradores de trades
def _base(d, o, ref):
    if ref == "prev":
        return DT.at[d, "prev_c"]
    if ref == "open":
        return o[0]
    return o[mi(ref) - 570]


def win_trades(days, ref, sig_end, entry, exit_, thr=0.0, thr_kind="abs", mode=1, tag=""):
    """Sinal = retorno de `ref` ('prev' = fechamento anterior, 'open' = abertura do dia,
    ou HHMM = open daquele candle) ate o fechamento do candle sig_end-1min.
    Se |ret| >= limiar: posicao = mode * sinal(ret) (mode=+1 momentum, -1 reversao),
    entrada no OPEN do candle `entry` (>= sig_end), saida no CLOSE do candle exit_-1min."""
    out = []
    i_sig = mi(sig_end) - 571
    i_en = mi(entry) - 570
    i_ex = mi(exit_) - 571
    assert i_en >= i_sig + 1
    for d, g in days:
        o = g["open"].values
        c = g["close"].values
        base = _base(d, o, ref)
        if not np.isfinite(base):
            continue
        r = c[i_sig] / base - 1.0
        t = thr
        if thr_kind == "atr":
            a = DT.at[d, "atr14"]
            if not np.isfinite(a):
                continue
            t = thr * a
        if r == 0 or abs(r) < t:
            continue
        side = int(np.sign(r)) * mode
        res = lab.sim_exit(g, i_en, side, i_last=i_ex)
        if res is not None:
            out.append(lab.make_trade(g, res, side, tag))
    return out


def ah_trades(days_full, ref, exit_, thr=0.0, mode=-1, tag=""):
    """After-hours: sinal = retorno de `ref` ate o CLOSE do candle 15:59; entrada no OPEN do
    candle 16:00 (se existir), saida no CLOSE do ultimo candle com m < exit_."""
    out = []
    for d, g in days_full:
        m = g["m"].values
        o = g["open"].values
        c = g["close"].values
        r_idx = np.where((m >= 570) & (m < 960))[0]
        en = np.where(m == 960)[0]
        ex = np.where((m >= 960) & (m < mi(exit_)))[0]
        if len(en) == 0 or len(ex) == 0:
            continue
        if ref == "prev":
            base = DT.at[d, "prev_c"]
        elif ref == "open":
            base = o[r_idx[0]]
        else:
            base = o[np.where(m == mi(ref))[0][0]]
        if not np.isfinite(base):
            continue
        r = c[r_idx[-1]] / base - 1.0
        if r == 0 or abs(r) < thr:
            continue
        side = int(np.sign(r)) * mode
        res = lab.sim_exit(g, int(en[0]), side, i_last=int(ex[-1]))
        if res is not None:
            out.append(lab.make_trade(g, res, side, tag))
    return out


def x_trades(days_full, ref, sig_end, entry, exit_, thr=0.0, thr_kind="abs", mode=1, tag=""):
    """Como win_trades, mas sobre o dia completo (permite SAIDA no after-hours).
    Sinal = retorno do OPEN do candle `ref` (HHMM) ate o CLOSE do candle sig_end-1min; entrada no OPEN do candle
    `entry`; saida no CLOSE do ultimo candle existente com m < exit_ (exit_ pode ser > 1600)."""
    out = []
    for d, g in days_full:
        m = g["m"].values
        o = g["open"].values
        c = g["close"].values
        i_ref = np.where(m == mi(ref))[0]
        i_sig = np.where(m == mi(sig_end) - 1)[0]
        i_en = np.where(m == mi(entry))[0]
        i_ex = np.where((m >= mi(entry)) & (m < mi(exit_)))[0]
        if len(i_ref) == 0 or len(i_sig) == 0 or len(i_en) == 0 or len(i_ex) == 0:
            continue
        assert i_en[0] >= i_sig[0] + 1
        r = c[i_sig[0]] / o[i_ref[0]] - 1.0
        t = thr
        if thr_kind == "atr":
            a = DT.at[d, "atr14"]
            if not np.isfinite(a):
                continue
            t = thr * a
        if r == 0 or abs(r) < t:
            continue
        side = int(np.sign(r)) * mode
        res = lab.sim_exit(g, int(i_en[0]), side, i_last=int(i_ex[-1]))
        if res is not None:
            out.append(lab.make_trade(g, res, side, tag))
    return out


def agree_trades(days, day_ref, want, ref, sig_end, entry, exit_, thr=0.0, mode=1, tag=""):
    """Momentum do empurrao final (ref->sig_end) filtrado pelo sinal do dia (day_ref='prev'|'open' ate sig_end):
    want='agree' so opera se o empurrao tem o MESMO sinal do retorno do dia; 'disagree' so se tem sinal oposto."""
    out = []
    i_sig = mi(sig_end) - 571
    i_en = mi(entry) - 570
    i_ex = mi(exit_) - 571
    for d, g in days:
        o = g["open"].values
        c = g["close"].values
        bd = _base(d, o, day_ref)
        if not np.isfinite(bd):
            continue
        rd = c[i_sig] / bd - 1.0
        r = c[i_sig] / o[mi(ref) - 570] - 1.0
        if r == 0 or rd == 0 or abs(r) < thr:
            continue
        same = np.sign(r) == np.sign(rd)
        if (want == "agree") != bool(same):
            continue
        side = int(np.sign(r)) * mode
        res = lab.sim_exit(g, i_en, side, i_last=i_ex)
        if res is not None:
            out.append(lab.make_trade(g, res, side, tag))
    return out


GEN = {"win": win_trades, "ah": ah_trades, "x": x_trades, "agree": agree_trades}
FULLKIND = ("ah", "x")

# ----------------------------------------------------------------------------- condicoes de regime (3 variaveis x 3 limiares)
REG_VARS = {"atr14": [0.08, 0.10, 0.12], "ret5": [-0.05, 0.0, 0.05], "absgap": [0.01, 0.02, 0.03]}


def reg_ok(d, reg):
    """reg = None ou (variavel, 'gt'|'le', limiar). Dia sem a variavel (inicio do arquivo) fica fora."""
    if reg is None:
        return True
    v = DT.at[d, reg[0]]
    if not np.isfinite(v):
        return False
    return (v > reg[2]) if reg[1] == "gt" else (v <= reg[2])


def gen_trades(kind, params, reg, days_rth, days_full):
    days = days_full if kind in FULLKIND else days_rth
    if reg is not None:
        days = [(d, g) for d, g in days if reg_ok(d, reg)]
    return GEN[kind](days, **params)


# ----------------------------------------------------------------------------- busca (so P0 e P1)
ROWS = []
NCONF = {}


def _side(tr, per, s, cost):
    """Mesma formula de lab._stats: liquido = side*(exit/entry-1) - 2*custo."""
    x = np.array([t["side"] * (t["exit_px"] / t["entry_px"] - 1.0) - 2.0 * cost / 1e4
                  for t in tr if t["side"] == s and lab.period_of(pd.Timestamp(t["date"]).date()) == per])
    if len(x) == 0:
        return 0, float("nan")
    return len(x), float(x.mean() * 1e4)


def record(idea, label, kind, params, reg=None):
    tr = gen_trades(kind, params, reg, IS_DAYS, IS_FULL)
    NCONF[idea] = NCONF.get(idea, 0) + 1
    ev = lab.evaluate(tr, lab.COST_BASE)
    r = dict(idea=idea, label=label)
    for p in PER:                       # SOMENTE P0 e P1
        m = ev[p]
        n = m.get("n_trades", 0)
        r["n" + p] = n
        if n > 0:
            r["net" + p] = m["avg_net_bps"]
            r["t" + p] = m["t_stat"]
            r["str" + p] = m["avg_gross_bps"] - 2.0 * lab.COST_STRESS      # liquido no custo de estresse
            nl, bl = _side(tr, p, 1, lab.COST_BASE)
            ns, bs = _side(tr, p, -1, lab.COST_BASE)
            r.update({"nL" + p: nl, "L" + p: bl, "nS" + p: ns, "S" + p: bs})
        else:
            r.update({"net" + p: np.nan, "t" + p: np.nan, "str" + p: np.nan})
    r["ok"] = bool(r["nP0"] >= 30 and r["nP1"] >= 30 and r["netP0"] > 0 and r["netP1"] > 0)
    r["mint"] = min(r["tP0"], r["tP1"]) if (r["nP0"] > 1 and r["nP1"] > 1) else np.nan
    r["both_sides"] = bool(r["ok"] and all(r.get(k, np.nan) > 0 for k in ("LP0", "SP0", "LP1", "SP1")))
    r["kind"] = kind
    r["params"] = repr(params)
    r["reg"] = repr(reg)
    ROWS.append(r)
    return r


SHOWCOLS = ["label", "nP0", "netP0", "tP0", "strP0", "LP0", "SP0", "nP1", "netP1", "tP1", "strP1", "LP1", "SP1",
            "mint", "ok", "both_sides"]


def show(idea, title, full=False):
    df = pd.DataFrame([r for r in ROWS if r["idea"] == idea])
    same = ((np.sign(df["netP0"]) == np.sign(df["netP1"])) & (df["nP0"] >= 30) & (df["nP1"] >= 30)).sum()
    print("\n=== %s  (%d configs; passam no criterio P0&P1: %d; ambas as pontas positivas: %d; mesmo sinal: %d) ===" % (
        title, NCONF[idea], int(df["ok"].sum()), int(df["both_sides"].sum()), int(same)))
    if full:
        print(df[SHOWCOLS].round(2).to_string(index=False))
    else:
        print(df.sort_values("mint", ascending=False)[SHOWCOLS].head(6).round(2).to_string(index=False))


def search():
    for a, z in [(930, 1600), (1500, 1600), (1530, 1600), (1550, 1600), (1558, 1600)]:
        b = [lab.make_trade(g, lab.sim_exit(g, mi(a) - 570, 1, i_last=mi(z) - 571), 1, "bench") for d, g in IS_DAYS]
        ev = lab.evaluate(b)
        print("BENCH comprado %d->%d sempre: " % (a, z) + " | ".join(lab.fmt(ev[p], p) for p in PER))

    # ================= grade da rodada 1 (identica; 510 configs) =================
    thr2 = [("abs", 0.0), ("abs", 0.02), ("abs", 0.04), ("abs", 0.06), ("atr", 0.25), ("atr", 0.5)]
    for ref in ["prev", "open"]:
        for T in [1500, 1530, 1545, 1550]:
            for ex in [1600, 1555, 1558]:
                for kind, th in thr2:
                    record("2_mom", "ref=%s T=%d ex=%d thr=%s%.2f" % (ref, T, ex, kind, th), "win",
                           dict(ref=ref, sig_end=T, entry=T, exit_=ex, thr=th, thr_kind=kind, mode=1))
    show("2_mom", "Ideia 2: momentum de fechamento (posicao = sinal do retorno do dia)")
    for ref in [1500, 1530, 1545, 1550]:
        for E in [1555, 1557, 1558]:
            for th in [0.0, 0.0025, 0.005, 0.01]:
                record("3a_fade", "push=%s->%d thr=%.4f fade->1600" % (ref, E, th), "win",
                       dict(ref=ref, sig_end=E, entry=E, exit_=1600, thr=th, mode=-1))
    for E in [1555, 1557, 1558]:
        for th in [0.0, 0.02, 0.04]:
            record("3a_fade", "push=prev->%d thr=%.4f fade->1600" % (E, th), "win",
                   dict(ref="prev", sig_end=E, entry=E, exit_=1600, thr=th, mode=-1))
    show("3a_fade", "Ideia 3a: reversao do empurrao nos ultimos minutos")
    AH = [("prev", [0.0, 0.02, 0.04]), ("open", [0.0, 0.02, 0.04]),
          (1530, [0.0, 0.003, 0.006]), (1550, [0.0, 0.003, 0.006]), (1555, [0.0, 0.002, 0.004])]
    for ref, ths in AH:
        for ex in [1605, 1615, 1630]:
            for th in ths:
                record("3b_ah", "sig=%s->1600 thr=%.4f fade 1600->%d" % (ref, th, ex), "ah",
                       dict(ref=ref, exit_=ex, thr=th, mode=-1))
    show("3b_ah", "Ideia 3b: reversao no after-hours (ver coluna str = custo de estresse)")
    for se in [1100, 1130, 1200]:
        for ex in [1300, 1330, 1400]:
            for th in [0.0, 0.01, 0.02]:
                record("4a_lunch", "manha open->%d thr=%.2f fade->%d" % (se, th, ex), "win",
                       dict(ref="open", sig_end=se, entry=se, exit_=ex, thr=th, mode=-1))
    show("4a_lunch", "Ideia 4a: reversao do almoco (fade da manha)")
    for ref in ["open", "prev"]:
        for ex in [1030, 1100, 1130]:
            for th in [0.0, 0.01, 0.02]:
                record("4b_10h", "ref=%s->1000 thr=%.2f fade 1000->%d" % (ref, th, ex), "win",
                       dict(ref=ref, sig_end=1000, entry=1000, exit_=ex, thr=th, mode=-1))
    show("4b_10h", "Ideia 4b: virada das 10:00 (fade dos primeiros 30 min)")
    for ref in ["prev", "open"]:
        for en in [1500, 1530]:
            for th in [0.0, 0.01, 0.02]:
                record("4c_first_last", "ref=%s->1000 thr=%.2f mom %d->1600" % (ref, th, en), "win",
                       dict(ref=ref, sig_end=1000, entry=en, exit_=1600, thr=th, mode=1))
    for en in [1500, 1530]:
        for th in [0.0, 0.01, 0.02]:
            record("4c_first_last", "ref=1000->%d thr=%.2f mom %d->1600" % (en, th, en), "win",
                   dict(ref=1000, sig_end=en, entry=en, exit_=1600, thr=th, mode=1))
    show("4c_first_last", "Ideia 4c: 1a meia hora / miolo do dia -> ultima (meia) hora, momentum")
    for ref in ["prev", "open", 1000]:
        for T in [1430, 1500]:
            for ex in [1530, 1555, 1600]:
                for th in [0.0, 0.01, 0.02, 0.04]:
                    record("2b_fade", "ref=%s T=%d thr=%.2f fade->%d" % (ref, T, th, ex), "win",
                           dict(ref=ref, sig_end=T, entry=T, exit_=ex, thr=th, mode=-1))
    show("2b_fade", "2b: REVERSAO do movimento do dia na ultima hora")
    for ref in [1500, 1530, 1545, 1550]:
        for E in [1555, 1557, 1558]:
            for th in [0.0, 0.0025, 0.005, 0.01]:
                record("3a_mom", "push=%s->%d thr=%.4f mom->1600" % (ref, E, th), "win",
                       dict(ref=ref, sig_end=E, entry=E, exit_=1600, thr=th, mode=1))
    for E in [1555, 1557, 1558]:
        for th in [0.0, 0.02, 0.04]:
            record("3a_mom", "push=prev->%d thr=%.4f mom->1600" % (E, th), "win",
                   dict(ref="prev", sig_end=E, entry=E, exit_=1600, thr=th, mode=1))
    show("3a_mom", "3a': MOMENTUM do empurrao nos ultimos minutos", full=True)
    for ref, ths in AH:
        for ex in [1605, 1615, 1630]:
            for th in ths:
                record("3b_ah_mom", "sig=%s->1600 thr=%.4f mom 1600->%d" % (ref, th, ex), "ah",
                       dict(ref=ref, exit_=ex, thr=th, mode=1))
    show("3b_ah_mom", "3b': MOMENTUM no after-hours (ver coluna str = custo de estresse)")
    for se in [1100, 1130, 1200]:
        for ex in [1300, 1330, 1400]:
            for th in [0.0, 0.01, 0.02]:
                record("4a_lunch_mom", "manha open->%d thr=%.2f mom->%d" % (se, th, ex), "win",
                       dict(ref="open", sig_end=se, entry=se, exit_=ex, thr=th, mode=1))
    show("4a_lunch_mom", "4a': MOMENTUM da manha atravessando o almoco")

    # ================= ideias novas da rodada 2 (mesma familia: efeito dos ultimos minutos) =================
    # ---- 3c: momentum dos ultimos minutos, grade fina de janela/entrada e limiares maiores (margem p/ custo)
    for ref in [1540, 1550, 1553, 1555]:
        for E in [1556, 1557, 1558, 1559]:
            for kind, th in [("abs", 0.0), ("abs", 0.0025), ("abs", 0.005), ("abs", 0.0075), ("abs", 0.01),
                             ("atr", 0.03), ("atr", 0.06), ("atr", 0.10)]:
                record("3c_lastmin", "push=%d->%d thr=%s%.4f mom->1600" % (ref, E, kind, th), "win",
                       dict(ref=ref, sig_end=E, entry=E, exit_=1600, thr=th, thr_kind=kind, mode=1))
    show("3c_lastmin", "3c (NOVA): momentum dos ultimos minutos, grade fina", full=True)
    # ---- 3d: mesmo sinal, carregando a posicao pelo fechamento para o after-hours
    for E in [1558, 1559]:
        for ex in [1601, 1605, 1615]:
            for th in [0.0, 0.0025, 0.005, 0.01]:
                record("3d_lastmin_ah", "push=1550->%d thr=%.4f mom->%d" % (E, th, ex), "x",
                       dict(ref=1550, sig_end=E, entry=E, exit_=ex, thr=th, mode=1))
    show("3d_lastmin_ah", "3d (NOVA): momentum dos ultimos minutos com saida no after-hours", full=True)
    # ---- 3e: empurrao final a favor / contra o sinal do dia (tese de rebalanceamento: fluxo ~ retorno do dia)
    for dref in ["prev", "open"]:
        for want in ["agree", "disagree"]:
            for th in [0.0025, 0.005]:
                record("3e_agree", "push=1550->1558 thr=%.4f dia(%s)=%s mom->1600" % (th, dref, want), "agree",
                       dict(day_ref=dref, want=want, ref=1550, sig_end=1558, entry=1558, exit_=1600, thr=th, mode=1))
    show("3e_agree", "3e (NOVA): empurrao final x sinal do dia", full=True)
    # ---- 5: regras condicionais a regime (3 variaveis x 3 limiares x acima/abaixo) sobre 4 regras-base fixas
    BASES = [("last2", dict(ref=1550, sig_end=1558, entry=1558, exit_=1600, thr=0.005, thr_kind="abs", mode=1)),
             ("lunch", dict(ref="open", sig_end=1100, entry=1100, exit_=1300, thr=0.0, thr_kind="abs", mode=1)),
             ("phfade", dict(ref=1000, sig_end=1500, entry=1500, exit_=1600, thr=0.0, thr_kind="abs", mode=-1)),
             ("rebal", dict(ref="prev", sig_end=1550, entry=1550, exit_=1600, thr=0.0, thr_kind="abs", mode=1))]
    for bn, bp in BASES:
        for var, ths in REG_VARS.items():
            for th in ths:
                for op in ["gt", "le"]:
                    record("5_regime", "%s | %s %s %.2f" % (bn, var, op, th), "win", bp, reg=(var, op, th))
    show("5_regime", "5 (NOVA): condicional a regime", full=True)

    df = pd.DataFrame(ROWS)
    df.to_csv(OUT / "search_p0p1.csv", index=False)
    n = len(df)
    print("\nTOTAL de configuracoes testadas: %d  %s" % (n, NCONF))
    print("Passam no criterio dos dois regimes (n>=30 e liquido>0 em P0 e P1): %d ; com compra e venda positivas nos dois: %d"
          % (int(df["ok"].sum()), int(df["both_sides"].sum())))
    print("Passam e ainda ficam >0 no custo de estresse em P0 e P1: %d" % int((df["ok"] & (df["strP0"] > 0) & (df["strP1"] > 0)).sum()))
    print("\n=== TOP 45 por min(t_P0, t_P1) entre as que passam ===")
    print(df[df["ok"]].sort_values("mint", ascending=False)[["idea"] + SHOWCOLS].head(45).round(2).to_string(index=False))
    print("\n=== Finalistas da rodada 1 na nova busca (posicao entre TODAS as configs por min t) ===")
    rk = df["mint"].rank(ascending=False, method="min")
    for idea, lab_ in [("3a_mom", "push=1550->1558 thr=0.0050 mom->1600"), ("4a_lunch_mom", "manha open->1100 thr=0.00 mom->1300"),
                       ("2b_fade", "ref=1000 T=1500 thr=0.00 fade->1600")]:
        j = df.index[(df["idea"] == idea) & (df["label"] == lab_)][0]
        print("rank %4d de %d | %s" % (rk[j], n, df.loc[[j], ["idea"] + SHOWCOLS].round(2).to_string(index=False, header=False)))
    return n


# ----------------------------------------------------------------------------- sazonalidade (descricao)
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _st(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean_bps=np.nan, t=np.nan, hit=np.nan)
    sd = x.std(ddof=1) if n > 1 else np.nan
    return dict(n=n, mean_bps=x.mean() * 1e4, t=(x.mean() / (sd / math.sqrt(n))) if sd and sd > 0 else np.nan,
                hit=(x > 0).mean() * 100)


def seasonality(periods=("P0", "P1", "P2")):
    """Retorno por faixa = close do ultimo candle da faixa / close do ultimo candle da faixa anterior - 1
    (1a faixa do dia: a partir do OPEN de 09:30), de modo que as faixas de 30m somam o open->close.
    So entram dias dos `periods` pedidos (na etapa de busca: P0 e P1 apenas)."""
    recs = []
    days = [(d, g) for d, g in DAYS if lab.period_of(d) in periods]
    full = [(d, g) for d, g in FULL if lab.period_of(d) in periods]
    for d, g in days:
        c = g["close"].values
        o0 = g["open"].values[0]
        wd = WD[pd.Timestamp(d).weekday()]
        samp = lab.period_of(d)

        def r(i0, i1):
            base = o0 if i0 == 0 else c[i0 - 1]
            return c[i1] / base - 1.0
        for k in range(13):
            recs.append(("30m", "%s-%s" % (hm(570 + 30 * k), hm(600 + 30 * k)), wd, samp, r(30 * k, 30 * k + 29)))
        for k in range(12):
            recs.append(("5m_first_hour", "%s-%s" % (hm(570 + 5 * k), hm(575 + 5 * k)), wd, samp, r(5 * k, 5 * k + 4)))
            recs.append(("5m_last_hour", "%s-%s" % (hm(900 + 5 * k), hm(905 + 5 * k)), wd, samp,
                         r(330 + 5 * k, 330 + 5 * k + 4)))
        for k in range(5):
            recs.append(("1m_last_5", "%s-%s" % (hm(955 + k), hm(956 + k)), wd, samp, r(385 + k, 385 + k)))
        row = DT.loc[d]
        recs.append(("day", "gap_prevclose->open", wd, samp, row["gap"]))
        recs.append(("day", "open->close", wd, samp, row["oc"]))
        recs.append(("day", "close->close", wd, samp, row["c"] / row["prev_c"] - 1))
        recs.append(("day", "range_h/l-1", wd, samp, row["rng"]))
        # modulo (amplitude) dos ultimos minutos: descreve o tamanho do movimento disponivel
        recs.append(("abs", "|15:50-15:58|", wd, samp, abs(c[387] / g["open"].values[380] - 1.0)))
        recs.append(("abs", "|15:58-16:00|", wd, samp, abs(c[389] / g["open"].values[388] - 1.0)))
    for d, g in full:
        m = g["m"].values
        ix = np.where((m >= 960) & (m < 975))[0]
        if len(ix) == 0:
            continue
        wd = WD[pd.Timestamp(d).weekday()]
        recs.append(("after", "16:00-16:15", wd, lab.period_of(d),
                     g["close"].values[ix[-1]] / g["open"].values[ix[0]] - 1))
    df = pd.DataFrame(recs, columns=["scheme", "bucket", "weekday", "sample", "ret"])
    # fase do mes (3 primeiros / 3 ultimos pregoes de cada mes) -- calculada sobre o calendario completo de pregoes
    dates = [d for d, _ in DAYS]
    ym = pd.Series([(d.year, d.month) for d in dates])
    pos = ym.groupby(ym).cumcount().values
    rev = ym.groupby(ym).cumcount(ascending=False).values
    # mes incompleto nas pontas do arquivo (nov/2025 comeca dia 11, nov/2026 termina dia 10) -> nao rotula
    first_ym, last_ym = ym.iloc[0], ym.iloc[-1]
    phase = {}
    for d, y, p, q in zip(dates, ym, pos, rev):
        if p < 3 and y != first_ym:
            phase[d] = "first3_of_month"
        elif q < 3 and y != last_ym:
            phase[d] = "last3_of_month"
    ext = []
    for d, g in days:
        if d in phase:
            row = DT.loc[d]
            samp = lab.period_of(d)
            c = g["close"].values
            ext.append(("day", "open->close", phase[d], samp, row["oc"]))
            ext.append(("day", "close->close", phase[d], samp, row["c"] / row["prev_c"] - 1))
            ext.append(("day", "gap_prevclose->open", phase[d], samp, row["gap"]))
            ext.append(("30m", "15:30-16:00", phase[d], samp, c[389] / c[359] - 1))
    df = pd.concat([df, pd.DataFrame(ext, columns=df.columns)], ignore_index=True)
    al = df[df["weekday"].isin(WD)].copy()
    al["weekday"] = "ALL"
    df = pd.concat([al, df], ignore_index=True)
    out = []
    for (sc, bk, wd), gg in df.groupby(["scheme", "bucket", "weekday"], sort=False):
        rec = dict(scheme=sc, bucket=bk, weekday=wd)
        for p in periods:
            a = _st(gg.loc[gg["sample"] == p, "ret"])
            rec.update({p + "_n": a["n"], p + "_mean_bps": a["mean_bps"], p + "_t": a["t"], p + "_hit_pct": a["hit"]})
        out.append(rec)
    res = pd.DataFrame(out)
    grp_order = ["ALL"] + WD + ["first3_of_month", "last3_of_month"]
    sch_order = ["30m", "5m_first_hour", "5m_last_hour", "1m_last_5", "after", "abs", "day"]
    res["_g"] = res["weekday"].map({k: i for i, k in enumerate(grp_order)})
    res["_s"] = res["scheme"].map({k: i for i, k in enumerate(sch_order)})
    res["_b"] = np.where(res["scheme"] == "day", "z", res["bucket"])
    res = res.sort_values(["_s", "_g", "_b"], kind="stable").drop(columns=["_g", "_s", "_b"]).reset_index(drop=True)
    return res


# ----------------------------------------------------------------------------- finalistas (congelados apos P0/P1)
# Escolhidos olhando SOMENTE as tabelas P0/P1 impressas por search(). Nao alterar depois de ver P2.
_SEL = ("Rodada 2: so blocos P0 e P1 (custo base 3 bps/lado); n>=30 em P0 e em P1; liquido>0 nos dois; ordenado por "
        "min(t_P0, t_P1); vizinhos da grade com o mesmo sinal; compra e venda positivas nos dois regimes. ")
FINALISTS = [
    dict(kind="win", name="last2min_mom_1553_1558_atr003",
         params=dict(ref=1553, sig_end=1558, entry=1558, exit_=1600, thr=0.03, thr_kind="atr", mode=1),
         description="Momentum dos 2 ultimos minutos com limiar proporcional a volatilidade: se |retorno 15:53->15:58| >= "
                     "0,03 x atr14 (atr14 = media da amplitude high/low-1 dos 14 pregoes anteriores), entra a favor no open "
                     "do candle 15:58 e sai no close do candle 15:59. Sem stop.",
         selection_rule=_SEL + "1o lugar de 742 por min t (P0 t=4,46; P1 t=3,97). Regiao estavel: entrada 15:58 com janela "
                        "15:50/15:53/15:55 e limiar 0,25%/0,5%/0,03xATR tem min t entre 2,6 e 3,6, todas com compra e venda "
                        "positivas em P0 e P1.",
         live_trigger="Antes da abertura: atr14 = media de (maxima/minima - 1) do pregao regular nos 14 pregoes anteriores; "
                      "limiar = 0,03 x atr14. As 15:58:00 NY (fechamento do candle 15:57): r = close(15:57)/open(15:53) - 1. "
                      "Se r >= +limiar COMPRA; se r <= -limiar VENDE, a mercado na abertura do candle 15:58. Saida a mercado "
                      "nos ultimos segundos antes das 16:00:00 (close do candle 15:59; NAO o leilao). Sem stop.",
         notes="Robusto em SINAL nos dois regimes, mas SEM margem para o custo de estresse em P0: liquido de estresse "
               "-3,75 bps (P0) e +16,21 bps (P1). Nenhuma das 128+57 variantes do efeito com n>=30 passa de +0,3 bps em P0 "
               "no estresse. O movimento absoluto medio de 15:58->16:00 e 18,9 bps em P0 (38,9 em P1), abaixo dos 20 bps "
               "de ida e volta do estresse. Carregar para o after-hours (3d) inverte o sinal em P0."),
    dict(kind="win", name="lunch_trend_1130_1300_thr2",
         params=dict(ref="open", sig_end=1130, entry=1130, exit_=1300, thr=0.02, thr_kind="abs", mode=1),
         description="Momentum da manha atravessando o almoco: as 11:30, se |retorno abertura(09:30)->11:30| >= 2%, segue o "
                     "sinal; entra no open do candle 11:30, sai no close do candle 12:59. Sem stop.",
         selection_rule=_SEL + "MENOS PIOR fora do efeito dos ultimos minutos: unica familia com liquido > 0 no custo de "
                        "ESTRESSE em P0 e P1 e com compra e venda positivas. min t = 1,56 (P0 t=1,56; P1 t=1,77). "
                        "Vizinhos com sinal 11:30: 9/9 passam (min t 0,9-1,6); com sinal 11:00 (finalista da rodada 1) "
                        "P0 e ~zero; com sinal 12:00 so 2/9 passam.",
         live_trigger="As 11:30:00 NY (fechamento do candle 11:29): r = close(11:29)/open(09:30) - 1. Se r >= +2% COMPRA; "
                      "se r <= -2% VENDE, a mercado na abertura do candle 11:30. Saida a mercado as 13:00:00 (close do candle "
                      "12:59). Sem stop; no maximo 1 trade por dia.",
         notes="NAO significativo: t~1,6 no pior regime com 742 configuracoes testadas. Salvo como 'menos pior' com margem "
               "de custo; depende do horario do sinal (11:00 nao funciona em P0), ou seja, a regiao estavel e estreita."),
]


def main():
    n_tested = search()
    if STAGE == "search":
        s = seasonality(periods=PER)
        print("\n=== Sazonalidade P0 vs P1 (ALL) ===")
        print(s[s["weekday"] == "ALL"].round(2).to_string(index=False))
        return
    s = seasonality()
    s.to_csv(OUT / "seasonality.csv", index=False)
    print("\n=== Sazonalidade P0 / P1 / P2 (ALL) ===")
    print(s[s["weekday"] == "ALL"].round(2).to_string(index=False))
    a = s[(s["weekday"] == "ALL")]
    for sc in ["30m", "5m_first_hour", "5m_last_hour"]:
        x = a[a["scheme"] == sc]
        for p, q in [("P0", "P1"), ("P0", "P2"), ("P1", "P2")]:
            print("Repeticao %s->%s [%s]: corr das medias = %.2f ; mesmo sinal em %d de %d faixas" % (
                p, q, sc, np.corrcoef(x[p + "_mean_bps"], x[q + "_mean_bps"])[0, 1],
                int((np.sign(x[p + "_mean_bps"]) == np.sign(x[q + "_mean_bps"])).sum()), len(x)))
    for f in FINALISTS:
        tr = gen_trades(f["kind"], f["params"], f.get("reg"), DAYS, FULL)
        rec = lab.save_result(family=FAMILY, name=f["name"], description=f["description"],
                              params=dict(kind=f["kind"], reg=f.get("reg"), **f["params"]),
                              trades=tr, n_configs_tested=n_tested, selection_rule=f["selection_rule"],
                              live_trigger=f["live_trigger"], notes=f["notes"])
        print("\n##### FINALISTA %s" % f["name"])
        for cst, key in [("base", "metrics_base"), ("estresse", "metrics_stress")]:
            for smp in ["P0", "P1", "P2"]:
                print(lab.fmt(rec[key][smp], "%-8s %-3s" % (cst, smp)))
        for smp in ["P0", "P1", "P2"]:
            for sd, nm in [(1, "COMPRA"), (-1, "VENDA")]:
                sub = [t for t in tr if t["side"] == sd]
                mm = lab.evaluate(sub)[smp] if sub else {"n_trades": 0}
                print(lab.fmt(mm, "base %-3s %-6s" % (smp, nm)))


if __name__ == "__main__":
    main()
