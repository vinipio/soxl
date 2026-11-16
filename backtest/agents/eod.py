import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0]); import lab
# -*- coding: utf-8 -*-
"""
Familia "eod": sazonalidade por horario e efeitos de fim de dia no SOXL (1m).

Rodar:  python3 agents/eod.py            -> busca no IS + sazonalidade IS/OOS + grava finalistas
        EOD_STAGE=search python3 agents/eod.py -> so a busca no IS (nada de OOS e tocado)

Convencoes de horario (HHMM, Nova York):
  * "sinal ate T"  = fechamento do candle T-1min (informacao disponivel em T:00).
  * "entrada em T" = OPEN do candle T (i_entry = indice do sinal + 1, ou depois).
  * "saida em T"   = CLOSE do candle T-1min. Saida 1600 = close do candle 15:59 (ultimo do pregao).
A busca de parametros recebe SOMENTE os dias do IS.
"""
import os
import math
import numpy as np
import pandas as pd

STAGE = os.environ.get("EOD_STAGE", "full")
FAMILY = "eod"
OUT = lab.RESULTS / FAMILY
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)


def mi(hhmm):
    return (hhmm // 100) * 60 + hhmm % 100


def hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


RTH = lab.rth()
DAYS = lab.day_groups(RTH)
for _d, _g in DAYS:
    assert len(_g) == 390 and _g["m"].iloc[0] == 570, _d
DT = lab.daily_table()
FULL = lab.day_groups(lab.load())
IS_DAYS = [(d, g) for d, g in DAYS if d <= lab.IS_END]
IS_FULL = [(d, g) for d, g in FULL if d <= lab.IS_END]


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


GEN = {"win": win_trades, "ah": ah_trades}

# ----------------------------------------------------------------------------- busca no IS
ROWS = []
NCONF = {}


def _side_stats(tr, s, cost):
    sub = [t for t in tr if t["side"] == s]
    if not sub:
        return 0, float("nan"), float("nan")
    m = lab.evaluate_is(sub, cost)
    return m["n_trades"], m["avg_net_bps"], m["t_stat"]


def record(idea, label, kind, params, cost=lab.COST_BASE):
    days = IS_FULL if kind == "ah" else IS_DAYS
    tr = GEN[kind](days, **params)
    NCONF[idea] = NCONF.get(idea, 0) + 1
    m = lab.evaluate_is(tr, cost)
    r = dict(idea=idea, label=label, n=m.get("n_trades", 0))
    if r["n"] > 0:
        nl, bl, tl = _side_stats(tr, 1, cost)
        ns, bs, ts = _side_stats(tr, -1, cost)
        r.update(net=m["avg_net_bps"], t=m["t_stat"], win=m["win_rate"] * 100, sharpe=m["sharpe_daily_ann"],
                 tot=m["total_net_pct"], nL=nl, netL=bl, tL=tl, nS=ns, netS=bs, tS=ts)
    ROWS.append(r)
    return r


def show(idea, title):
    df = pd.DataFrame([r for r in ROWS if r["idea"] == idea]).drop(columns=["idea"])
    print("\n=== %s  (IS, %d configs) ===" % (title, NCONF[idea]))
    print(df.round(2).to_string(index=False))


def search():
    # benchmark de vies direcional
    b = [lab.make_trade(g, lab.sim_exit(g, 0, -1), -1, "bench") for d, g in IS_DAYS]
    print("BENCH IS vendido open->close:", lab.fmt(lab.evaluate_is(b), ""))
    for a, z in [(1500, 1600), (1530, 1600), (1550, 1600)]:
        b = [lab.make_trade(g, lab.sim_exit(g, mi(a) - 570, -1, i_last=mi(z) - 571), -1, "bench") for d, g in IS_DAYS]
        print("BENCH IS vendido %d->%d sempre:" % (a, z), lab.fmt(lab.evaluate_is(b), ""))

    # ---- Ideia 2: momentum de fechamento (rebalanceamento)
    thr2 = [("abs", 0.0), ("abs", 0.02), ("abs", 0.04), ("abs", 0.06), ("atr", 0.25), ("atr", 0.5)]
    for ref in ["prev", "open"]:
        for T in [1500, 1530, 1545, 1550]:
            for ex in [1600, 1555, 1558]:
                for kind, th in thr2:
                    record("2_mom", "ref=%s T=%d ex=%d thr=%s%.2f" % (ref, T, ex, kind, th), "win",
                           dict(ref=ref, sig_end=T, entry=T, exit_=ex, thr=th, thr_kind=kind, mode=1))
    show("2_mom", "Ideia 2: momentum de fechamento (posicao = sinal do retorno do dia)")

    # ---- Ideia 3a: reversao nos ultimos minutos (RTH)
    for ref in [1500, 1530, 1545, 1550]:
        for E in [1555, 1557, 1558]:
            for th in [0.0, 0.0025, 0.005, 0.01]:
                record("3a_fade", "push=%s->%d thr=%.4f fade->1600" % (ref, E, th), "win",
                       dict(ref=ref, sig_end=E, entry=E, exit_=1600, thr=th, mode=-1))
    for E in [1555, 1557, 1558]:
        for th in [0.0, 0.02, 0.04]:
            record("3a_fade", "push=prev->%d thr=%.4f fade->1600" % (E, th), "win",
                   dict(ref="prev", sig_end=E, entry=E, exit_=1600, thr=th, mode=-1))
    show("3a_fade", "Ideia 3a: reversao do empurrao nos ultimos minutos (custo base)")

    # ---- Ideia 3b: reversao no after-hours imediato (criterio = custo de estresse)
    for ref, ths in [("prev", [0.0, 0.02, 0.04]), ("open", [0.0, 0.02, 0.04]),
                     (1530, [0.0, 0.003, 0.006]), (1550, [0.0, 0.003, 0.006]), (1555, [0.0, 0.002, 0.004])]:
        for ex in [1605, 1615, 1630]:
            for th in ths:
                record("3b_ah", "sig=%s->1600 thr=%.4f fade 1600->%d" % (ref, th, ex), "ah",
                       dict(ref=ref, exit_=ex, thr=th, mode=-1), cost=lab.COST_STRESS)
    show("3b_ah", "Ideia 3b: reversao no after-hours (CUSTO DE ESTRESSE 10 bps/lado)")

    # ---- Ideia 4a: reversao do almoco
    for se in [1100, 1130, 1200]:
        for ex in [1300, 1330, 1400]:
            for th in [0.0, 0.01, 0.02]:
                record("4a_lunch", "manha open->%d thr=%.2f fade->%d" % (se, th, ex), "win",
                       dict(ref="open", sig_end=se, entry=se, exit_=ex, thr=th, mode=-1))
    show("4a_lunch", "Ideia 4a: reversao do almoco (fade da manha)")

    # ---- Ideia 4b: virada das 10:00
    for ref in ["open", "prev"]:
        for ex in [1030, 1100, 1130]:
            for th in [0.0, 0.01, 0.02]:
                record("4b_10h", "ref=%s->1000 thr=%.2f fade 1000->%d" % (ref, th, ex), "win",
                       dict(ref=ref, sig_end=1000, entry=1000, exit_=ex, thr=th, mode=-1))
    show("4b_10h", "Ideia 4b: virada das 10:00 (fade dos primeiros 30 min)")

    # ---- Ideia 4c: primeira meia hora -> ultima meia hora / power hour
    for ref in ["prev", "open"]:
        for en in [1500, 1530]:
            for th in [0.0, 0.01, 0.02]:
                record("4c_first_last", "ref=%s->1000 thr=%.2f mom %d->1600" % (ref, th, en), "win",
                       dict(ref=ref, sig_end=1000, entry=en, exit_=1600, thr=th, mode=1))
    # miolo do dia (10:00 -> 15:00/15:30) -> ultima hora
    for en in [1500, 1530]:
        for th in [0.0, 0.01, 0.02]:
            record("4c_first_last", "ref=1000->%d thr=%.2f mom %d->1600" % (en, th, en), "win",
                   dict(ref=1000, sig_end=en, entry=en, exit_=1600, thr=th, mode=1))
    show("4c_first_last", "Ideia 4c: 1a meia hora / miolo do dia -> ultima (meia) hora, momentum")

    # ======== Segunda rodada (ainda so IS): o IS mostrou o SINAL OPOSTO ao das hipoteses em 3 lugares.
    # Os espelhos abaixo sao contados como configuracoes adicionais.
    # ---- 2b: reversao do movimento do dia na ultima hora (espelho da ideia 2 com T cedo)
    for ref in ["prev", "open", 1000]:
        for T in [1430, 1500]:
            for ex in [1530, 1555, 1600]:
                for th in [0.0, 0.01, 0.02, 0.04]:
                    record("2b_fade", "ref=%s T=%d thr=%.2f fade->%d" % (ref, T, th, ex), "win",
                           dict(ref=ref, sig_end=T, entry=T, exit_=ex, thr=th, mode=-1))
    show("2b_fade", "2b: REVERSAO do movimento do dia na ultima hora")
    # ---- 3a': momentum nos ultimos minutos (espelho de 3a)
    for ref in [1500, 1530, 1545, 1550]:
        for E in [1555, 1557, 1558]:
            for th in [0.0, 0.0025, 0.005, 0.01]:
                record("3a_mom", "push=%s->%d thr=%.4f mom->1600" % (ref, E, th), "win",
                       dict(ref=ref, sig_end=E, entry=E, exit_=1600, thr=th, mode=1))
    for E in [1555, 1557, 1558]:
        for th in [0.0, 0.02, 0.04]:
            record("3a_mom", "push=prev->%d thr=%.4f mom->1600" % (E, th), "win",
                   dict(ref="prev", sig_end=E, entry=E, exit_=1600, thr=th, mode=1))
    show("3a_mom", "3a': MOMENTUM do empurrao nos ultimos minutos (custo base)")
    # ---- 3b': momentum no after-hours (espelho de 3b), custo de estresse
    for ref, ths in [("prev", [0.0, 0.02, 0.04]), ("open", [0.0, 0.02, 0.04]),
                     (1530, [0.0, 0.003, 0.006]), (1550, [0.0, 0.003, 0.006]), (1555, [0.0, 0.002, 0.004])]:
        for ex in [1605, 1615, 1630]:
            for th in ths:
                record("3b_ah_mom", "sig=%s->1600 thr=%.4f mom 1600->%d" % (ref, th, ex), "ah",
                       dict(ref=ref, exit_=ex, thr=th, mode=1), cost=lab.COST_STRESS)
    show("3b_ah_mom", "3b': MOMENTUM no after-hours (CUSTO DE ESTRESSE 10 bps/lado)")
    # ---- 4a': momentum da manha atravessando o almoco (espelho de 4a)
    for se in [1100, 1130, 1200]:
        for ex in [1300, 1330, 1400]:
            for th in [0.0, 0.01, 0.02]:
                record("4a_lunch_mom", "manha open->%d thr=%.2f mom->%d" % (se, th, ex), "win",
                       dict(ref="open", sig_end=se, entry=se, exit_=ex, thr=th, mode=1))
    show("4a_lunch_mom", "4a': MOMENTUM da manha atravessando o almoco")
    return sum(NCONF.values())


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


def seasonality(with_oos=True):
    """Retorno por faixa = close do ultimo candle da faixa / close do ultimo candle da faixa anterior - 1
    (1a faixa do dia: a partir do OPEN de 09:30), de modo que as faixas de 30m somam o open->close."""
    recs = []
    for d, g in DAYS:
        c = g["close"].values
        o0 = g["open"].values[0]
        wd = WD[pd.Timestamp(d).weekday()]
        samp = "IS" if d <= lab.IS_END else "OOS"

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
    # after-hours imediato 16:00-16:15
    for d, g in FULL:
        m = g["m"].values
        ix = np.where((m >= 960) & (m < 975))[0]
        if len(ix) == 0:
            continue
        wd = WD[pd.Timestamp(d).weekday()]
        samp = "IS" if d <= lab.IS_END else "OOS"
        recs.append(("after", "16:00-16:15", wd, samp, g["close"].values[ix[-1]] / g["open"].values[ix[0]] - 1))
    df = pd.DataFrame(recs, columns=["scheme", "bucket", "weekday", "sample", "ret"])
    # fase do mes (3 primeiros / 3 ultimos pregoes de cada mes) -- so descricao
    dates = [d for d, _ in DAYS]
    ym = pd.Series([(d.year, d.month) for d in dates])
    pos = ym.groupby(ym).cumcount().values
    rev = ym.groupby(ym).cumcount(ascending=False).values
    # mes incompleto nas pontas do arquivo (mai/2026 comeca dia 11, nov/2026 termina dia 10) -> nao rotula
    first_ym, last_ym = ym.iloc[0], ym.iloc[-1]
    phase = {}
    for d, y, p, q in zip(dates, ym, pos, rev):
        if p < 3 and y != first_ym:
            phase[d] = "first3_of_month"
        elif q < 3 and y != last_ym:
            phase[d] = "last3_of_month"
    ext = []
    for d, g in DAYS:
        if d in phase:
            row = DT.loc[d]
            samp = "IS" if d <= lab.IS_END else "OOS"
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
    order = {}
    for (sc, bk, wd), gg in df.groupby(["scheme", "bucket", "weekday"], sort=False):
        a = _st(gg.loc[gg["sample"] == "IS", "ret"])
        rec = dict(scheme=sc, bucket=bk, weekday=wd, is_n=a["n"], is_mean_bps=a["mean_bps"], is_t=a["t"],
                   is_hit_pct=a["hit"])
        if with_oos:
            b = _st(gg.loc[gg["sample"] == "OOS", "ret"])
            rec.update(oos_n=b["n"], oos_mean_bps=b["mean_bps"], oos_t=b["t"], oos_hit_pct=b["hit"])
        out.append(rec)
    res = pd.DataFrame(out)
    grp_order = ["ALL"] + WD + ["first3_of_month", "last3_of_month"]
    sch_order = ["30m", "5m_first_hour", "5m_last_hour", "1m_last_5", "after", "day"]
    res["_g"] = res["weekday"].map({k: i for i, k in enumerate(grp_order)})
    res["_s"] = res["scheme"].map({k: i for i, k in enumerate(sch_order)})
    res["_b"] = np.where(res["scheme"] == "day", "z", res["bucket"])
    res = res.sort_values(["_s", "_g", "_b"], kind="stable").drop(columns=["_g", "_s", "_b"]).reset_index(drop=True)
    return res


# ----------------------------------------------------------------------------- finalistas (congelados apos o IS)
# Escolhidos olhando SOMENTE as tabelas do IS impressas por search(). Nao alterar depois de ver o OOS.
_SEL = ("So IS, custo base 3 bps/lado, n>=30; escolhida a regiao em que TODOS os vizinhos da grade tem o mesmo sinal "
        "e compra e venda sao ambas positivas; dentro da regiao, a configuracao mais simples/central. ")
FINALISTS = [
    dict(kind="win", name="lunch_trend_1100_1300",
         params=dict(ref="open", sig_end=1100, entry=1100, exit_=1300, thr=0.0, thr_kind="abs", mode=1),
         description="Momentum da manha atravessando o almoco: as 11:00 segue o sinal do retorno abertura(09:30)->11:00; "
                     "entra no open do candle 11:00, sai no close do candle 12:59. Sem stop.",
         selection_rule=_SEL + "Espelho da hipotese 'reversao do almoco' (que deu negativa em 27/27 configs no IS); "
                        "momentum positivo em 27/27 (t 1.6-2.3). Escolhida 11:00->13:00 sem limiar (maior Sharpe IS, n=78).",
         live_trigger="As 11:00:00 NY (fechamento do candle 10:59): r = close(10:59)/open(09:30) - 1. Se r>0 COMPRA, se r<0 VENDE, "
                      "a mercado na abertura do candle 11:00. Saida a mercado as 13:00:00 (close do candle 12:59). "
                      "Sem limiar, sem stop; 1 trade por dia.",
         notes="Hipotese original (reversao) rejeitada; finalista e o espelho, descoberto no IS (risco de data-mining: "
               "t~2.2 com 510 configs testadas nao e significativo apos correcao). No IS o lado COMPRA (manha de alta) "
               "e o mais forte, contra a deriva negativa intraday do IS, portanto nao e so vies direcional."),
    dict(kind="win", name="last2min_momentum_1550_1558",
         params=dict(ref=1550, sig_end=1558, entry=1558, exit_=1600, thr=0.005, thr_kind="abs", mode=1),
         description="Momentum dos 2 ultimos minutos: se |retorno 15:50->15:58| >= 0,5%, entra a favor no open do candle "
                     "15:58 e sai no close do candle 15:59 (ultimo negocio antes das 16:00, NAO o leilao).",
         selection_rule=_SEL + "Espelho da hipotese 'reversao no ultimo minuto' (negativa em quase toda a grade). "
                        "Momentum com entrada 15:58 positivo em 12/12 configs com janela intraday (t 1.1-3.9); "
                        "limiar 0,5% escolhido por folga contra custo de estresse (vizinhos 0,25% e 1% tambem t>3).",
         live_trigger="As 15:58:00 NY (fechamento do candle 15:57): r = close(15:57)/open(15:50) - 1. Se r >= +0,5% COMPRA; "
                      "se r <= -0,5% VENDE, a mercado na abertura do candle 15:58. Saida a mercado nos ultimos segundos "
                      "antes das 16:00:00 (close do candle 15:59). Sem stop. Sem sinal -> sem trade.",
         notes="Trade de 2 minutos: ~29 bps liquidos/trade no IS, muito sensivel a execucao (spread/slippage no fechamento). "
               "O sinal e o movimento dos ultimos 10 min, NAO o retorno do dia: seguir o retorno do dia (tese pura de "
               "rebalanceamento) nos ultimos 2 min deu NEGATIVO no IS (t=-2.9)."),
    dict(kind="win", name="power_hour_fade_1000_1500",
         params=dict(ref=1000, sig_end=1500, entry=1500, exit_=1600, thr=0.0, thr_kind="abs", mode=-1),
         description="Reversao na ultima hora: as 15:00 opera CONTRA o sinal do retorno 10:00->15:00; entra no open do candle "
                     "15:00, sai no close do candle 15:59. Sem stop.",
         selection_rule=_SEL + "Espelho da hipotese de momentum de fechamento por rebalanceamento (negativa no IS para T=15:00). "
                        "Fade com T=15:00 positivo em 36/36 configs (ref prev/open/10:00); ref=10:00 sem limiar ate 16:00 "
                        "escolhida por simetria compra/venda e n=78 (t=2.17).",
         live_trigger="As 15:00:00 NY (fechamento do candle 14:59): r = close(14:59)/open(10:00) - 1. Se r>0 VENDE, se r<0 COMPRA, "
                      "a mercado na abertura do candle 15:00. Saida a mercado nos ultimos segundos antes das 16:00:00 "
                      "(close do candle 15:59). Sem limiar, sem stop; 1 trade por dia.",
         notes="Contraria a tese de rebalanceamento do ETF alavancado (que preve momentum). Descoberta no IS como espelho; "
               "t~2.2 com 510 configs testadas nao e significativo apos correcao para multiplos testes. "
               "A reversao no IS concentra-se em 15:00-15:30."),
]


def main():
    n_tested = search()
    print("\nTOTAL de configuracoes testadas no IS: %d  %s" % (n_tested, NCONF))
    if STAGE == "search":
        s = seasonality(with_oos=False)
        print("\n=== Sazonalidade IS (ALL) ===")
        print(s[s["weekday"] == "ALL"].round(2).to_string(index=False))
        return
    s = seasonality(with_oos=True)
    s.to_csv(OUT / "seasonality.csv", index=False)
    print("\n=== Sazonalidade IS vs OOS (ALL) ===")
    print(s[s["weekday"] == "ALL"].round(2).to_string(index=False))
    print("\n=== Dia (por dia da semana / fase do mes) ===")
    print(s[(s["scheme"] == "day") | (s["weekday"].isin(["first3_of_month", "last3_of_month"]))].round(2).to_string(index=False))
    a = s[(s["weekday"] == "ALL")]
    for sc in ["30m", "5m_first_hour", "5m_last_hour"]:
        x = a[a["scheme"] == sc]
        print("Repeticao IS->OOS [%s]: corr das medias = %.2f ; mesmo sinal em %d de %d faixas" % (
            sc, np.corrcoef(x["is_mean_bps"], x["oos_mean_bps"])[0, 1],
            int((np.sign(x["is_mean_bps"]) == np.sign(x["oos_mean_bps"])).sum()), len(x)))
    w = s[(s["scheme"] == "30m") & (s["weekday"].isin(WD))]
    print("Repeticao IS->OOS [30m x dia da semana, 65 celulas]: corr = %.2f ; mesmo sinal em %d de %d" % (
        np.corrcoef(w["is_mean_bps"], w["oos_mean_bps"])[0, 1],
        int((np.sign(w["is_mean_bps"]) == np.sign(w["oos_mean_bps"])).sum()), len(w)))
    for f in FINALISTS:
        days = FULL if f["kind"] == "ah" else DAYS
        tr = GEN[f["kind"]](days, **f["params"])
        rec = lab.save_result(family=FAMILY, name=f["name"], description=f["description"],
                              params=dict(kind=f["kind"], **{k: (v if not isinstance(v, (np.integer,)) else int(v))
                                                            for k, v in f["params"].items()}),
                              trades=tr, n_configs_tested=n_tested, selection_rule=f["selection_rule"],
                              live_trigger=f["live_trigger"], notes=f["notes"])
        print("\n##### FINALISTA %s" % f["name"])
        for cst, key in [("base", "metrics_base"), ("estresse", "metrics_stress")]:
            for smp in ["IS", "OOS"]:
                print(lab.fmt(rec[key][smp], "%-8s %-3s" % (cst, smp)))
        for smp, sel in [("IS", lambda t: t["date"] <= str(lab.IS_END)), ("OOS", lambda t: t["date"] > str(lab.IS_END))]:
            for sd, nm in [(1, "COMPRA"), (-1, "VENDA")]:
                sub = [t for t in tr if t["side"] == sd and sel(t)]
                mm = lab.evaluate(sub)[smp] if sub else {"n_trades": 0}
                print(lab.fmt(mm, "base %-3s %-6s" % (smp, nm)))


if __name__ == "__main__":
    main()
