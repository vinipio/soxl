import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0]); import lab
# -*- coding: utf-8 -*-
"""
Familia "patterns": padroes de candle, volume e volatilidade no SOXL (5m e 15m, pregao regular).

Rodar:  python3 agents/patterns.py          (a partir de a pasta backtest)
  1) varre TODOS os padroes x direcao x modo x horizonte x contexto SOMENTE no in-sample
     -> results/patterns/pattern_scan_is.csv
  2) aplica a regra de selecao (definida so com o IS) e grava os finalistas com lab.save_result
     (so nesse momento o OOS e calculado).

Convencoes
  * bloco de 5m/15m = lab.resample; sinal no fechamento do bloco (indice 1m i1); entrada no OPEN de i1+1.
  * "sig" = direcao textbook do padrao (bull=+1 / bear=-1). modo follow = opera a favor de sig;
    modo fade = opera contra sig. Compra e venda ficam sempre em linhas separadas.
  * horizonte h (min): saida no CLOSE do candle 1m i_entry+h-1; sinais cujo horizonte nao cabe no dia sao descartados.
  * stop (variante): extremo do candle-sinal (minima p/ compra, maxima p/ venda).
"""
import math
from collections import defaultdict

import numpy as np
import pandas as pd

FAMILY = "patterns"
OUT = lab.RESULTS / FAMILY
OUT.mkdir(parents=True, exist_ok=True)
TFS = [5, 15]
HORIZONS = [5, 15, 30, 60]
CONTEXTS = ["all", "with_vwap", "against_vwap", "with_open", "against_open", "t_open", "t_mid", "t_close"]
PATTERNS = ["engulf", "hammer_star", "doji_trend", "inside_break", "run3", "run4", "run5", "wide2x",
            "rvol2", "rvol3", "vol_climax", "nr4_break", "nr7_break", "bb_squeeze"]

DAYS = lab.day_groups(lab.rth())
N_IS_DAYS = sum(1 for dt, _ in DAYS if dt <= lab.IS_END)


# --------------------------------------------------------------------------------------
# 1) Sinais (causais) em todos os dias
# --------------------------------------------------------------------------------------
def build_bars(tf):
    parts = []
    for di, (dt, g) in enumerate(DAYS):
        b = lab.resample(g, tf)
        b["di"] = di
        b["k"] = np.arange(len(b))
        # VWAP da sessao ate o fechamento do bloco (so dados ate i1)
        pv = (g["vwap"] * g["volume"]).cumsum().values
        vv = g["volume"].cumsum().values
        i1 = b["i1"].values.astype(int)
        b["svwap"] = pv[i1] / np.where(vv[i1] > 0, vv[i1], np.nan)
        b["day_open"] = g["open"].iloc[0]
        b["run_hi"] = b["high"].cummax()
        b["run_lo"] = b["low"].cummin()
        parts.append(b)
    B = pd.concat(parts, ignore_index=True)
    return B


def build_signals(tf):
    """Devolve dict pattern -> array sig (+1/-1/0) alinhado a B, e B."""
    B = build_bars(tf)
    o, h, l, c, v, k = (B[x].values for x in ["open", "high", "low", "close", "volume", "k"])
    n = len(B)
    rng = h - l
    body = np.abs(c - o)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    sgn = np.sign(c - o)
    S = pd.Series
    avg20 = S(rng).shift(1).rolling(20).mean().values           # 20 blocos ANTERIORES (pode cruzar dia)
    sh = lambda a, j: S(a).shift(j).values
    sig = {}

    # engolfo: corpo atual engolfa o corpo anterior de cor oposta
    ok = k >= 1
    bull = ok & (sh(c, 1) < sh(o, 1)) & (c > o) & (o <= sh(c, 1)) & (c >= sh(o, 1)) & (body > sh(body, 1))
    bear = ok & (sh(c, 1) > sh(o, 1)) & (c < o) & (o >= sh(c, 1)) & (c <= sh(o, 1)) & (body > sh(body, 1))
    sig["engulf"] = bull.astype(int) - bear.astype(int)

    # tendencia previa de 3 blocos (mesmo dia): close[i-1] vs close[i-4]
    okt = k >= 4
    up_tr = okt & (sh(c, 1) > sh(c, 4))
    dn_tr = okt & (sh(c, 1) < sh(c, 4))
    big_enough = rng >= 0.5 * avg20

    # martelo (apos queda) / estrela cadente (apos alta): pavio >= 2x corpo e >= 2x pavio oposto
    ham = dn_tr & big_enough & (rng > 0) & (lw >= 2 * body) & (lw >= 2 * uw) & (lw > 0)
    star = up_tr & big_enough & (rng > 0) & (uw >= 2 * body) & (uw >= 2 * lw) & (uw > 0)
    sig["hammer_star"] = ham.astype(int) - star.astype(int)

    # doji (corpo <= 10% da amplitude) apos tendencia -> sinal de reversao
    doji = big_enough & (rng > 0) & (body <= 0.1 * rng)
    sig["doji_trend"] = (doji & dn_tr).astype(int) - (doji & up_tr).astype(int)

    # inside bar (i-1 dentro de i-2) seguido de fechamento alem do extremo do inside
    ins = (k >= 2) & (sh(h, 1) <= sh(h, 2)) & (sh(l, 1) >= sh(l, 2))
    sig["inside_break"] = (ins & (c > sh(h, 1))).astype(int) - (ins & (c < sh(l, 1))).astype(int)

    # N candles consecutivos na mesma direcao (dispara quando a sequencia atinge exatamente N, no mesmo dia)
    run = np.zeros(n, dtype=int)
    for i in range(n):
        if sgn[i] == 0:
            run[i] = 0
        elif k[i] > 0 and sgn[i - 1] == sgn[i]:
            run[i] = run[i - 1] + 1
        else:
            run[i] = 1
    for N in (3, 4, 5):
        sig["run%d" % N] = np.where(run == N, sgn, 0).astype(int)

    # candle de amplitude larga: >= 2x a media dos 20 blocos anteriores
    sig["wide2x"] = np.where((rng >= 2 * avg20) & (sgn != 0), sgn, 0).astype(int)

    # volume relativo: vs media do MESMO horario nos 14 pregoes ANTERIORES (min 10)
    piv = B.pivot(index="di", columns="k", values="volume")
    base = piv.rolling(14, min_periods=10).mean().shift(1)
    rvol = v / base.stack(future_stack=True).reindex(list(zip(B["di"], B["k"]))).values
    B["rvol"] = rvol
    sig["rvol2"] = np.where((rvol >= 2) & (sgn != 0), sgn, 0).astype(int)
    sig["rvol3"] = np.where((rvol >= 3) & (sgn != 0), sgn, 0).astype(int)

    # climax de volume no extremo do dia: rvol >= 2 e o bloco faz nova maxima/minima do dia (apos 30 min)
    late = k >= (30 // tf)
    nh = late & (rvol >= 2) & (h >= B["run_hi"].values) & ~(l <= B["run_lo"].values)
    nl = late & (rvol >= 2) & (l <= B["run_lo"].values) & ~(h >= B["run_hi"].values)
    sig["vol_climax"] = nh.astype(int) - nl.astype(int)

    # NR4 / NR7: bloco i-1 e o de menor amplitude dos N ultimos (mesmo dia); i fecha alem do extremo dele
    for N in (4, 7):
        rmin = S(rng).shift(1).rolling(N).min().values
        nr = (k >= N) & (sh(rng, 1) <= rmin)
        sig["nr%d_break" % N] = (nr & (c > sh(h, 1))).astype(int) - (nr & (c < sh(l, 1))).astype(int)

    # Bollinger squeeze: largura BB(20,2) do bloco i-1 <= quantil 20% das 100 larguras anteriores; i fecha fora da banda
    mid = S(c).rolling(20).mean()
    sd = S(c).rolling(20).std(ddof=0)
    width = (4 * sd / mid)
    q20 = width.shift(1).rolling(100).quantile(0.2)
    sq = (width <= q20).shift(1).fillna(False).values & (k >= 1)
    up_b = (mid + 2 * sd).shift(1).values
    lo_b = (mid - 2 * sd).shift(1).values
    sig["bb_squeeze"] = (sq & (c > up_b)).astype(int) - (sq & (c < lo_b)).astype(int)

    for p in sig:
        sig[p] = np.nan_to_num(sig[p]).astype(int)
    return B, sig


# --------------------------------------------------------------------------------------
# 2) Simulacao
# --------------------------------------------------------------------------------------
def simulate(g, i_entry, side, i_last, stop):
    """lab.sim_exit; se o open de entrada ja esta alem do stop, sai no proprio open (perde so o custo)."""
    if stop is not None:
        e = g["open"].values[i_entry]
        if (side > 0 and e <= stop) or (side < 0 and e >= stop):
            return dict(entry_i=i_entry, entry_px=e, exit_i=i_entry, exit_px=e, reason="stop_at_entry")
    return lab.sim_exit(g, i_entry, side, stop=stop, i_last=i_last)


def ctx_flags(row, s):
    """Contextos do sinal s (+1/-1) no bloco `row` (tudo conhecido no fechamento do bloco)."""
    c = row["close"]
    em = row["m_entry"]
    f = {"all": True,
         "with_vwap": (c > row["svwap"]) if s > 0 else (c < row["svwap"]),
         "against_vwap": (c < row["svwap"]) if s > 0 else (c > row["svwap"]),
         "with_open": (c > row["day_open"]) if s > 0 else (c < row["day_open"]),
         "against_open": (c < row["day_open"]) if s > 0 else (c > row["day_open"]),
         "t_open": em < 630, "t_mid": 630 <= em < 870, "t_close": em >= 870}
    return f


def scan_is():
    rows = []
    drift_rows = []
    for tf in TFS:
        B, sig = build_signals(tf)
        B["m_entry"] = B["m"] + tf
        isB = np.array([DAYS[di][0] <= lab.IS_END for di in B["di"].values])
        recs = B.to_dict("records")

        # deriva incondicional no IS (todos os blocos), por horizonte e faixa horaria -- referencia de vies
        dr = defaultdict(list)
        for idx in np.where(isB)[0]:
            r = recs[idx]
            g = DAYS[r["di"]][1]
            ie = int(r["i1"]) + 1
            for hmin in HORIZONS:
                il = ie + hmin - 1
                if il > len(g) - 1:
                    continue
                ret = g["close"].values[il] / g["open"].values[ie] - 1
                bucket = "t_open" if r["m_entry"] < 630 else ("t_mid" if r["m_entry"] < 870 else "t_close")
                dr[(hmin, "all")].append(ret)
                dr[(hmin, bucket)].append(ret)
        drift = {kk: float(np.mean(vv) * 1e4) for kk, vv in dr.items()}
        for (hmin, b), val in sorted(drift.items()):
            drift_rows.append(dict(tf=tf, horizon=hmin, bucket=b, n=len(dr[(hmin, b)]), drift_long_gross_bps=val,
                                   t=float(np.mean(dr[(hmin, b)]) / (np.std(dr[(hmin, b)], ddof=1) / math.sqrt(len(dr[(hmin, b)]))))))

        for p in PATTERNS:
            s_arr = sig[p]
            # trades[(sigdir, mode, h, stop)] = lista de (trade, flags)
            T = defaultdict(list)
            for idx in np.where((s_arr != 0) & isB)[0]:
                r = recs[idx]
                s = int(s_arr[idx])
                g = DAYS[r["di"]][1]
                ie = int(r["i1"]) + 1
                fl = ctx_flags(r, s)
                for hmin in HORIZONS:
                    il = ie + hmin - 1
                    if il > len(g) - 1:
                        continue
                    res = lab.sim_exit(g, ie, s, i_last=il)
                    t_fo = lab.make_trade(g, res, s, tag=p)
                    T[(s, "follow", hmin, 0)].append((t_fo, fl))
                    t_fa = dict(t_fo); t_fa["side"] = -s        # mesmo trade com o lado invertido
                    T[(s, "fade", hmin, 0)].append((t_fa, fl))
                    # variantes com stop no extremo do candle-sinal
                    st_fo = r["low"] if s > 0 else r["high"]
                    st_fa = r["high"] if s > 0 else r["low"]
                    T[(s, "follow", hmin, 1)].append((lab.make_trade(g, simulate(g, ie, s, il, st_fo), s, tag=p), fl))
                    T[(s, "fade", hmin, 1)].append((lab.make_trade(g, simulate(g, ie, -s, il, st_fa), -s, tag=p), fl))
            for s in (1, -1):
                for mode in ("follow", "fade"):
                    for hmin in HORIZONS:
                        for stop in (0, 1):
                            for ctx in (CONTEXTS if stop == 0 else ["all"]):
                                tl = [t for t, fl in T.get((s, mode, hmin, stop), []) if fl[ctx]]
                                side = s if mode == "follow" else -s
                                row = dict(tf=tf, pattern=p, sig="bull" if s > 0 else "bear", mode=mode,
                                           side="long" if side > 0 else "short", horizon=hmin, stop=stop, context=ctx,
                                           n=len(tl))
                                if len(tl) >= 2:
                                    m = lab.evaluate_is(tl)
                                    db = drift.get((hmin, ctx if ctx.startswith("t_") else "all"), 0.0)
                                    row.update(days=m["days_traded"], win=m["win_rate"], gross_bps=m["avg_gross_bps"],
                                               net_bps=m["avg_net_bps"], t=m["t_stat"],
                                               t_day=m["sharpe_daily_ann"] * math.sqrt(N_IS_DAYS / 252.0),
                                               sharpe=m["sharpe_daily_ann"], total_pct=m["total_net_pct"],
                                               excess_gross_bps=m["avg_gross_bps"] - side * db)
                                rows.append(row)
    scan = pd.DataFrame(rows)
    scan.to_csv(OUT / "pattern_scan_is.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUT / "drift_is.csv", index=False)
    return scan, pd.DataFrame(drift_rows)


# --------------------------------------------------------------------------------------
# 3) Simetria compra x venda (mesmos testes, pareados) e finalistas
# --------------------------------------------------------------------------------------
def pair_table(scan):
    key = ["tf", "pattern", "mode", "horizon", "stop", "context"]
    cols = ["n", "net_bps", "t", "t_day", "excess_gross_bps"]
    L = scan[scan["side"] == "long"].set_index(key)[cols]
    S = scan[scan["side"] == "short"].set_index(key)[cols]
    j = L.join(S, lsuffix="_L", rsuffix="_S").dropna()
    j = j[(j["n_L"] >= 30) & (j["n_S"] >= 30)].copy()
    sd = lambda n, m, t: m / t * np.sqrt(n)
    j["n"] = j["n_L"] + j["n_S"]
    j["net_bps"] = (j["n_L"] * j["net_bps_L"] + j["n_S"] * j["net_bps_S"]) / j["n"]
    j["t_pool"] = (j["n_L"] * j["net_bps_L"] + j["n_S"] * j["net_bps_S"]) / np.sqrt(
        j["n_L"] * sd(j["n_L"], j["net_bps_L"], j["t_L"]) ** 2 + j["n_S"] * sd(j["n_S"], j["net_bps_S"], j["t_S"]) ** 2)
    j = j.sort_values("t_pool", ascending=False)
    j.to_csv(OUT / "pattern_pairs_is.csv")
    return j


def finalist_trades(tf, pattern, mode, hmin, ctx, stop=0):
    """Trades em TODOS os dias, uma posicao por vez (sinal ignorado enquanto ha posicao aberta)."""
    B, sig = build_signals(tf)
    B["m_entry"] = B["m"] + tf
    recs = B.to_dict("records")
    s_arr = sig[pattern]
    trades = []
    busy = {}                                   # di -> ultimo indice 1m ocupado
    for idx in np.where(s_arr != 0)[0]:
        r = recs[idx]
        s = int(s_arr[idx])
        if not ctx_flags(r, s)[ctx]:
            continue
        g = DAYS[r["di"]][1]
        ie = int(r["i1"]) + 1
        il = ie + hmin - 1
        if il > len(g) - 1 or ie <= busy.get(r["di"], -1):
            continue
        side = s if mode == "follow" else -s
        st = None
        if stop:
            st = (r["low"] if side > 0 else r["high"]) if mode == "follow" else (r["high"] if side < 0 else r["low"])
        res = simulate(g, ie, side, il, st)
        busy[r["di"]] = res["exit_i"]
        trades.append(lab.make_trade(g, res, side, tag="%s_%s_%dm" % (pattern, mode, tf)))
    return trades


# Finalistas CONGELADOS com base apenas no scan do IS (ver SELECTION). Nenhum passou no criterio duro
# (t >= 3 no IS apos ~4000 testes); estes sao os "menos piores": efeito simetrico (compra E venda > 0 no IS),
# vizinhos de horizonte tambem positivos, melhor t pareado.
FINALISTS = [
    dict(name="bbsqueeze_false_break_fade_5m_mid_60",
         tf=5, pattern="bb_squeeze", mode="fade", h=60, ctx="t_mid", stop=0,
         description="Fade do rompimento de Bollinger squeeze em 5m (falso rompimento), entradas 10:30-14:30, saida em 60 min.",
         live="Candles de 5m (ancorados 09:30). BB(20,2) nos fechamentos de 5m (janela continua, cruza dias). "
              "Squeeze = largura (4*sd/media) do candle anterior <= quantil 20% das 100 larguras anteriores a ele. "
              "Se o candle atual FECHA acima da banda superior do candle anterior -> VENDER no open do minuto seguinte; "
              "se fecha abaixo da banda inferior -> COMPRAR. So entradas entre 10:30 e 14:29 NY, sem posicao aberta. "
              "Saida: a mercado 60 minutos depois (close do 60o candle de 1m). Sem stop."),
    dict(name="run3_fade_5m_close_30",
         tf=5, pattern="run3", mode="fade", h=30, ctx="t_close", stop=0,
         description="Apos 14:30, fade de 3 candles de 5m consecutivos na mesma direcao; saida em 30 min.",
         live="Candles de 5m. Quando o 3o candle consecutivo de mesma cor fecha (sequencia exatamente = 3, no mesmo dia) "
              "e a entrada cai entre 14:30 e 15:30 NY: 3 verdes -> VENDER, 3 vermelhos -> COMPRAR no open do minuto seguinte. "
              "Sem posicao aberta. Saida a mercado 30 minutos depois. Sem stop."),
    dict(name="run5_fade_5m_60",
         tf=5, pattern="run5", mode="fade", h=60, ctx="all", stop=0,
         description="Exaustao: fade de 5 candles de 5m consecutivos na mesma direcao, qualquer horario; saida em 60 min.",
         live="Candles de 5m. Quando o 5o candle consecutivo de mesma cor fecha (sequencia exatamente = 5, no mesmo dia), "
              "ate 15:00 NY: 5 verdes -> VENDER, 5 vermelhos -> COMPRAR no open do minuto seguinte. Sem posicao aberta. "
              "Saida a mercado 60 minutos depois. Sem stop."),
]
SELECTION = ("Scan IS de 4032 combinacoes (14 padroes x bull/bear x follow/fade x 2 timeframes x 4 horizontes x "
             "[8 contextos sem stop + 1 com stop]); 2580 com n>=30. Criterio duro: t>=3 no IS -> apenas 1 celula (esperado "
             "por acaso ~3,5), logo NADA passa em testes multiplos. Finalistas = menos piores: exige compra E venda com "
             "net>0 no IS (n>=30 cada), horizontes vizinhos tambem positivos, maior t pareado (2,3-2,4).")


def side_split(trades):
    out = {}
    for nm, sd_ in (("long", 1), ("short", -1)):
        m = lab.evaluate([t for t in trades if t["side"] == sd_])
        out[nm] = {k: (m[k].get("n_trades", 0), round(m[k].get("avg_net_bps", float("nan")), 1),
                       round(m[k].get("t_stat", float("nan")), 2)) for k in ("IS", "OOS")}
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    scan, drift = scan_is()
    v = scan[scan["n"] >= 30]
    n_cfg = len(scan)
    print("configs testadas (linhas do scan):", n_cfg, "| com n>=30:", len(v))
    for thr in (2, 2.5, 3):
        print("  t>=%.1f: %d celulas | t_day>=%.1f: %d" % (thr, (v["t"] >= thr).sum(), thr, (v["t_day"] >= thr).sum()))
    print(drift.round(2).to_string())
    pairs = pair_table(scan)
    print("pares compra/venda validos:", len(pairs), "| ambos net>0:", int(((pairs.net_bps_L > 0) & (pairs.net_bps_S > 0)).sum()),
          "| t_pool>=3:", int((pairs.t_pool >= 3).sum()))
    for f in FINALISTS:
        tr = finalist_trades(f["tf"], f["pattern"], f["mode"], f["h"], f["ctx"], f["stop"])
        sp = side_split(tr)
        rec = lab.save_result(
            family=FAMILY, name=f["name"], description=f["description"],
            params={k: f[k] for k in ("tf", "pattern", "mode", "h", "ctx", "stop")},
            trades=tr, n_configs_tested=n_cfg, selection_rule=SELECTION, live_trigger=f["live"],
            notes="MENOS PIOR, nao significativo: nenhum padrao passou t>=3 no IS apos ~4000 testes. "
                  "Trades sem sobreposicao (1 posicao por vez). (n, net bps, t) por lado: %s" % sp)
        print("\n==", f["name"])
        for blk in ("IS", "OOS"):
            print(lab.fmt(rec["metrics_base"][blk], blk + " base  "))
        print(lab.fmt(rec["metrics_stress"]["OOS"], "OOS stress"))
        print("lado (n, net bps, t):", sp)
