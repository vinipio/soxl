import os, sys
os.environ['SOXL_RESULTS'] = 'results_v2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import lab
# -*- coding: utf-8 -*-
"""
Familia "orb" -- RODADA 2 (249 pregoes; IS = P0+P1; teste = P2).

Ideias (as 4 da rodada 1, mesma grade, + 3 novas da mesma familia):
  1. ORB classico (faixa 5/15/30/60 min; sinal 1m ou 5m; stop oposto/meio/2%; saida close/1R/2R/3R/trailing 1R)
     1b. com reentrada na direcao oposta.
  2. Filtros sobre o ORB classico (gap a favor/contra, faixa estreita/larga, volume alto/baixo, horario limite).
  3. Rompimento da maxima/minima do pre-market durante o pregao regular.
  4. Fade do falso rompimento da faixa de abertura.
  5. (nova) Fade do falso rompimento da maxima/minima do PRE-MARKET.
  6. (nova) Regra condicional a regime, como FILTRO: 4 regras-base x 3 variaveis x 3 limiares x 2 lados do limiar.
  7. (nova) Regra condicional a regime, como CHAVE: acima do limiar segue o rompimento / abaixo faz o fade (e o inverso).

Variaveis de regime (todas conhecidas ANTES do trade):
  ret10 = fechamento de ontem / fechamento de 11 pregoes atras - 1      limiares -10%, 0, +10%
  atr14 = lab.daily_table()['atr14'] (amplitude media dos 14 pregoes anteriores)   limiares 8%, 10%, 12%
  wrel  = (largura da faixa de hoje / atr14) / mediana desse quociente nos dias anteriores   limiares 0,75, 1,00, 1,25

CRITERIO DE SELECAO (BRIEF2) -- usa SOMENTE os blocos 'P0' e 'P1' de lab.evaluate:
  - n >= 30 em P0 e n >= 30 em P1;
  - liquido medio > 0 no custo base em P0 E em P1;
  - ordena por min(t_P0, t_P1); empate (diferenca <= 0,25 de t para o lider) decidido por
    (a) compra e venda ambas positivas, (b) media do min(t) dos vizinhos (configs que diferem em 1 parametro);
  - F1 = vencedor; F2 = vencedor entre as demais configs estruturalmente distintas (outra ideia ou outro N).
Os blocos 'P2', 'OOS' e 'ALL' so sao lidos depois de congelar os finalistas (funcao final()).

Rodar:  python3 agents/orb_v2.py              (busca + finalistas + P2 + grava results_v2/orb/)
        ORB_STAGE=search python3 agents/orb_v2.py   (so a busca em P0/P1; nao calcula nem imprime P2)
"""
import itertools
import numpy as np
import pandas as pd

FAMILY = "orb"
LAST_SIGNAL_I = 387          # ultimo candle de sinal (15:57) -> entrada 15:58, saida no close de 15:59
PCT_STOP = 0.02              # stop percentual fixo
MIN_TRADES = 30              # por periodo (P0 e P1)
TIE_T = 0.25                 # tolerancia de empate em min(t)

# ----------------------------------------------------------------------------- dados
DT = lab.daily_table()
DAYS = []
for dt, g in lab.day_groups(lab.rth()):
    assert len(g) == 390, (dt, len(g))
    DAYS.append(dict(date=dt, key=str(dt), df=g, o=g["open"].values, h=g["high"].values, l=g["low"].values,
                     c=g["close"].values, v=g["volume"].values, m=g["m"].values))
KEYS = [d["key"] for d in DAYS]
PERIOD = {d["key"]: lab.period_of(d["date"]) for d in DAYS}
DT_IDX = [str(x) for x in DT.index]


def build_context():
    """Variaveis de filtro por dia, sempre comparadas com dias ANTERIORES (sem look-ahead)."""
    ctx = {}
    atr = pd.Series(DT["atr14"].values, index=DT_IDX).reindex(KEYS)
    for N in (5, 15, 30, 60):
        width, vol = [], []
        for d in DAYS:
            hi, lo = d["h"][:N].max(), d["l"][:N].min()
            width.append(hi / lo - 1.0)
            vol.append(float(d["v"][:N].sum()))
        width = pd.Series(width, index=KEYS)
        vol = pd.Series(vol, index=KEYS)
        w_rel = width / atr                                              # largura da faixa / atr14 (dias anteriores)
        w_med_prev = w_rel.shift(1).expanding(min_periods=10).median()   # mediana so de dias anteriores
        v_rel = vol / vol.shift(1).rolling(20, min_periods=5).mean()     # volume da faixa / media dos 20 dias anteriores
        ctx[N] = dict(w_rel=w_rel, w_med_prev=w_med_prev, v_rel=v_rel, wrel=w_rel / w_med_prev)
    gap = pd.Series(DT["gap"].values, index=DT_IDX).reindex(KEYS)
    ret10 = pd.Series((DT["c"].shift(1) / DT["c"].shift(11) - 1.0).values, index=DT_IDX).reindex(KEYS)
    return ctx, gap, atr, ret10


CTX, GAP, ATR, RET10 = build_context()
REGIME_THR = {"ret10": (-0.10, 0.0, 0.10), "atr14": (0.08, 0.10, 0.12), "wrel": (0.75, 1.00, 1.25)}


def regime_value(var, key, N):
    if var == "ret10":
        return RET10[key]
    if var == "atr14":
        return ATR[key]
    if var == "wrel":
        return CTX[N]["wrel"][key]
    raise ValueError(var)


# ----------------------------------------------------------------------------- blocos
def first_break(d, hi, lo, start, sig, i_max=LAST_SIGNAL_I, want=None):
    """Primeiro candle i>=start cujo FECHAMENTO esta fora de [lo, hi]. sig=1 -> candles de 1m;
    sig=5 -> so fechamentos de candles de 5m (ancorados em 09:30). want=+1/-1 restringe a direcao."""
    c = d["c"]
    for i in range(start, min(i_max, LAST_SIGNAL_I) + 1):
        if sig == 5 and (i + 1) % 5 != 0:
            continue
        if c[i] > hi and want in (None, 1):
            return i, 1
        if c[i] < lo and want in (None, -1):
            return i, -1
    return None, 0


def bracket_trade(d, i_sig, side, hi, lo, stop_mode, exit_mode, tag):
    """Entra no OPEN de i_sig+1; niveis de stop/alvo a partir do preco de entrada efetivo."""
    i_e = i_sig + 1
    entry = d["o"][i_e]
    if stop_mode == "opp":
        stop = lo if side > 0 else hi
    elif stop_mode == "mid":
        stop = (hi + lo) / 2.0
    elif stop_mode == "pct":
        stop = entry * (1 - side * PCT_STOP)
    else:
        raise ValueError(stop_mode)
    if side * (entry - stop) <= 0:      # abriu alem do stop: nao entra
        return None, None
    R = abs(entry - stop)
    target, trail = None, None
    if exit_mode in ("1R", "2R", "3R"):
        target = entry + side * float(exit_mode[0]) * R
    elif exit_mode == "trail":
        trail = R / entry               # trailing de 1R
    res = lab.sim_exit(d["df"], i_e, side, stop=stop, target=target, trail_pct=trail)
    if res is None:
        return None, None
    return lab.make_trade(d["df"], res, side, tag=tag), res


def gen_orb(N, sig, stop_mode, exit_mode, reentry=False, i_max=LAST_SIGNAL_I):
    """Ideia 1: ORB classico. Faixa = N primeiros minutos do pregao regular."""
    out = []
    for d in DAYS:
        hi, lo = d["h"][:N].max(), d["l"][:N].min()
        i, side = first_break(d, hi, lo, N, sig, i_max)
        if i is None:
            continue
        t, res = bracket_trade(d, i, side, hi, lo, stop_mode, exit_mode, "orb")
        if t is None:
            continue
        out.append(t)
        if reentry and res["reason"].startswith("stop"):
            j, s2 = first_break(d, hi, lo, res["exit_i"], sig, i_max, want=-side)
            if j is not None:
                t2, _ = bracket_trade(d, j, s2, hi, lo, stop_mode, exit_mode, "orb_re")
                if t2 is not None:
                    out.append(t2)
    return out


def pm_levels(d):
    row = DT.loc[d["date"]]
    hi, lo = row["pre_h"], row["pre_l"]
    if not (np.isfinite(hi) and np.isfinite(lo)):
        return None, None
    return float(hi), float(lo)


def gen_premarket(sig, stop_mode, exit_mode, i_max):
    """Ideia 3: rompimento da maxima/minima do pre-market (04:00-09:29) durante o pregao regular."""
    out = []
    for d in DAYS:
        hi, lo = pm_levels(d)
        if hi is None:
            continue
        i, side = first_break(d, hi, lo, 0, sig, i_max)
        if i is None:
            continue
        t, _ = bracket_trade(d, i, side, hi, lo, stop_mode, exit_mode, "pm")
        if t is not None:
            out.append(t)
    return out


def _fade_day(d, hi, lo, start, wait, target_mode, stop_mode, tag):
    """Falso rompimento de [lo, hi]: 1o fechamento de 1m fora (direcao b); se em ate `wait` candles um fechamento
    voltar para dentro, entra CONTRA o rompimento no open seguinte."""
    i, b = first_break(d, hi, lo, start, 1)
    if i is None:
        return None
    c = d["c"]
    j_end = LAST_SIGNAL_I if wait is None else min(LAST_SIGNAL_I, i + wait)
    j = None
    for k in range(i + 1, j_end + 1):
        if lo <= c[k] <= hi:
            j = k
            break
    if j is None:
        return None
    side = -b
    ext = d["h"][i:j + 1].max() if b > 0 else d["l"][i:j + 1].min()
    stop = ext if stop_mode == "ext" else None
    mid = (hi + lo) / 2.0
    target = mid if target_mode == "mid" else ((lo if b > 0 else hi) if target_mode == "opp" else None)
    if target is not None and side * (target - c[j]) <= 0:   # alvo ja ultrapassado no fechamento do sinal
        return None
    res = lab.sim_exit(d["df"], j + 1, side, stop=stop, target=target)
    if res is None:
        return None
    if stop is not None and side * (res["entry_px"] - stop) <= 0:
        return None
    return lab.make_trade(d["df"], res, side, tag=tag)


def gen_fade(N, wait, target_mode, stop_mode):
    """Ideia 4: falso rompimento da faixa de abertura de N minutos."""
    out = []
    for d in DAYS:
        hi, lo = d["h"][:N].max(), d["l"][:N].min()
        t = _fade_day(d, hi, lo, N, wait, target_mode, stop_mode, "fade")
        if t is not None:
            out.append(t)
    return out


def gen_pm_fade(wait, target_mode, stop_mode):
    """Ideia 5 (nova): falso rompimento da maxima/minima do pre-market no pregao regular."""
    out = []
    for d in DAYS:
        hi, lo = pm_levels(d)
        if hi is None:
            continue
        t = _fade_day(d, hi, lo, 0, wait, target_mode, stop_mode, "pm_fade")
        if t is not None:
            out.append(t)
    return out


def apply_filter(trades, N, filt):
    """Ideia 2: filtros sobre os trades do ORB classico (decididos com informacao ate o candle de sinal)."""
    out = []
    for t in trades:
        k = t["date"]
        if filt == "gap_favor":
            g = GAP[k]
            ok = np.isfinite(g) and g != 0 and np.sign(g) == t["side"]
        elif filt == "gap_contra":
            g = GAP[k]
            ok = np.isfinite(g) and g != 0 and np.sign(g) == -t["side"]
        elif filt in ("estreita", "larga"):
            w, med = CTX[N]["w_rel"][k], CTX[N]["w_med_prev"][k]
            ok = np.isfinite(w) and np.isfinite(med) and ((w < med) if filt == "estreita" else (w >= med))
        elif filt in ("vol_alto", "vol_baixo"):
            v = CTX[N]["v_rel"][k]
            ok = np.isfinite(v) and ((v > 1.0) if filt == "vol_alto" else (v <= 1.0))
        elif filt.startswith("ate_"):
            hh, mm = filt[4:].split(":")
            ok = t["entry_m"] <= int(hh) * 60 + int(mm)      # entrada (candle seguinte ao sinal) ate o horario
        else:
            raise ValueError(filt)
        if ok:
            out.append(t)
    return out


def regime_filter(trades, N, var, thr, lado):
    """Ideia 6: mantem o trade so se a variavel de regime do dia estiver acima ('>') ou abaixo/igual ('<=') do limiar."""
    out = []
    for t in trades:
        x = regime_value(var, t["date"], N)
        if not np.isfinite(x):
            continue
        if (x > thr) if lado == ">" else (x <= thr):
            out.append(t)
    return out


def regime_switch(follow, fade, N, var, thr, modo):
    """Ideia 7: modo 'segue_acima' -> var>thr usa o trade de rompimento do dia, senao o de fade;
    modo 'fade_acima' -> o inverso. Dias sem a variavel ficam de fora."""
    out = []
    for hi_set, lo_set in ((follow, fade),) if modo == "segue_acima" else ((fade, follow),):
        for t in hi_set:
            x = regime_value(var, t["date"], N)
            if np.isfinite(x) and x > thr:
                out.append(t)
        for t in lo_set:
            x = regime_value(var, t["date"], N)
            if np.isfinite(x) and x <= thr:
                out.append(t)
    out.sort(key=lambda t: (t["date"], t["entry_m"]))
    return out


# ----------------------------------------------------------------------------- metricas (so P0 e P1)
def p01(trades):
    """Devolve SOMENTE os blocos P0 e P1 (in-sample) de lab.evaluate."""
    ev = lab.evaluate(trades)
    return ev["P0"], ev["P1"]


def side_split(trades):
    return [t for t in trades if t["side"] > 0], [t for t in trades if t["side"] < 0]


def g(m, k):
    return m.get(k, float("nan")) if m.get("n_trades", 0) > 0 else float("nan")


def make_row(idea, params, trades):
    a, b = p01(trades)
    L, S = side_split(trades)
    La, Lb = p01(L)
    Sa, Sb = p01(S)
    n0, n1 = a.get("n_trades", 0), b.get("n_trades", 0)
    t0, t1 = g(a, "t_stat"), g(b, "t_stat")
    mint = min(t0, t1) if (np.isfinite(t0) and np.isfinite(t1)) else float("nan")
    totL = np.nansum([g(La, "total_net_pct"), g(Lb, "total_net_pct")])
    totS = np.nansum([g(Sa, "total_net_pct"), g(Sb, "total_net_pct")])
    nL = La.get("n_trades", 0) + Lb.get("n_trades", 0)
    nS = Sa.get("n_trades", 0) + Sb.get("n_trades", 0)
    return dict(idea=idea, params=params, trades=trades,
                n0=n0, net0=g(a, "avg_net_bps"), t0=t0, n1=n1, net1=g(b, "avg_net_bps"), t1=t1, mint=mint,
                nL=nL, totL=float(totL), nS=nS, totS=float(totS),
                L0=g(La, "avg_net_bps"), L1=g(Lb, "avg_net_bps"), S0=g(Sa, "avg_net_bps"), S1=g(Sb, "avg_net_bps"),
                nL0=La.get("n_trades", 0), nL1=Lb.get("n_trades", 0), nS0=Sa.get("n_trades", 0), nS1=Sb.get("n_trades", 0),
                sym=bool(nL >= 10 and nS >= 10 and totL > 0 and totS > 0))


def enough(r):
    return r["n0"] >= MIN_TRADES and r["n1"] >= MIN_TRADES


def passes(r):
    return enough(r) and r["net0"] > 0 and r["net1"] > 0


def pstr(p):
    return ",".join("%s=%s" % (k, v) for k, v in p.items())


def line(r):
    return ("%-17s %-58s P0 n=%3d net=%7.1f t=%5.2f | P1 n=%3d net=%7.1f t=%5.2f | min t=%5.2f | "
            "C %3d tot=%7.1f%% V %3d tot=%7.1f%% %s" % (
                r["idea"], pstr(r["params"]), r["n0"], r["net0"], r["t0"], r["n1"], r["net1"], r["t1"], r["mint"],
                r["nL"], r["totL"], r["nS"], r["totS"], "SIM" if r["sym"] else ""))


def show(rows, title, top=8):
    ok = [r for r in rows if enough(r)]
    ps = [r for r in ok if passes(r)]
    print("\n=== %s  (%d configs) | n>=%d em P0 e P1: %d | liquido>0 nos dois: %d | P0>0: %d | P1>0: %d | "
          "mediana net P0=%.1f P1=%.1f bps ===" % (
              title, len(rows), MIN_TRADES, len(ok), len(ps), sum(r["net0"] > 0 for r in ok), sum(r["net1"] > 0 for r in ok),
              float(np.median([r["net0"] for r in ok])) if ok else float("nan"),
              float(np.median([r["net1"] for r in ok])) if ok else float("nan")))
    for r in sorted(ok, key=lambda r: -r["mint"])[:top]:
        print("  " + line(r))


def neighbor_mint(r, rows):
    """Media do min(t_P0,t_P1) dos vizinhos (mesma ideia, diferem em exatamente 1 parametro, n>=30 nos dois)."""
    ts = []
    for q in rows:
        if q is r or q["idea"] != r["idea"] or not enough(q):
            continue
        if set(q["params"]) != set(r["params"]):
            continue
        if sum(q["params"][k] != r["params"][k] for k in r["params"]) == 1:
            ts.append(q["mint"])
    return float(np.mean(ts)) if ts else float("nan")


# ----------------------------------------------------------------------------- busca (P0/P1)
def search():
    rows = []
    bench = [lab.trade_from_prices(d["key"], -1, d["m"][0], d["o"][0], d["m"][-1], d["c"][-1], tag="bench") for d in DAYS]
    a, b = p01(bench)
    print(lab.fmt(a, "BENCH vendido open->close P0"))
    print(lab.fmt(b, "BENCH vendido open->close P1"))

    # ideia 1 -- 4*2*3*5 = 120 + 8 com reentrada
    for N, sig, st, ex in itertools.product((5, 15, 30, 60), (1, 5), ("opp", "mid", "pct"),
                                            ("close", "1R", "2R", "3R", "trail")):
        rows.append(make_row("1_orb", dict(N=N, sig=sig, stop=st, exit=ex), gen_orb(N, sig, st, ex)))
    for N, sig in itertools.product((5, 15, 30, 60), (1, 5)):
        rows.append(make_row("1b_orb_reentrada", dict(N=N, sig=sig, stop="opp", exit="close"),
                             gen_orb(N, sig, "opp", "close", reentry=True)))
    # ideia 2 -- filtros sobre ORB (sinal 1m, stop oposto, saida no close): 4*10 = 40
    base_orb = {}
    for N in (5, 15, 30, 60):
        base_orb[N] = gen_orb(N, 1, "opp", "close")
        for f in ("gap_favor", "gap_contra", "estreita", "larga", "vol_alto", "vol_baixo",
                  "ate_10:30", "ate_11:00", "ate_12:00", "ate_14:00"):
            rows.append(make_row("2_filtro", dict(N=N, filtro=f), apply_filter(base_orb[N], N, f)))
    # ideia 3 -- pre-market: 2*3*3*2 = 36
    for sig, st, ex, cut in itertools.product((1, 5), ("opp", "mid", "pct"), ("close", "2R", "trail"), ("fim", "11:00")):
        i_max = LAST_SIGNAL_I if cut == "fim" else (11 * 60 - lab.RTH_START - 1)
        rows.append(make_row("3_premarket", dict(sig=sig, stop=st, exit=ex, ate=cut), gen_premarket(sig, st, ex, i_max)))
    # ideia 4 -- fade: 4*3*3*2 = 72
    base_fade = {}
    for N, wait, tg, st in itertools.product((5, 15, 30, 60), (10, 30, None), ("mid", "opp", "close"), ("ext", "none")):
        tr = gen_fade(N, wait, tg, st)
        base_fade[(N, wait, tg, st)] = tr
        rows.append(make_row("4_fade", dict(N=N, espera=wait, alvo=tg, stop=st), tr))
    # ideia 5 (nova) -- fade do falso rompimento do pre-market: 3*3*2 = 18
    for wait, tg, st in itertools.product((10, 30, None), ("mid", "opp", "close"), ("ext", "none")):
        rows.append(make_row("5_pm_fade", dict(espera=wait, alvo=tg, stop=st), gen_pm_fade(wait, tg, st)))
    # ideia 6 (nova) -- filtro de regime: 4 bases * 3 variaveis * 3 limiares * 2 lados = 72
    bases = {"orb15": (15, base_orb[15]), "orb60": (60, base_orb[60]),
             "fade15": (15, base_fade[(15, 30, "mid", "none")]), "fade30": (30, base_fade[(30, 30, "mid", "none")])}
    for bname, (N, tr) in bases.items():
        for var, thrs in REGIME_THR.items():
            for thr, lado in itertools.product(thrs, (">", "<=")):
                rows.append(make_row("6_regime_filtro", dict(base=bname, var=var, lim=thr, lado=lado),
                                     regime_filter(tr, N, var, thr, lado)))
    # ideia 7 (nova) -- chave de regime entre rompimento e fade: 2 N * 3 variaveis * 3 limiares * 2 modos = 36
    for N in (15, 30):
        for var, thrs in REGIME_THR.items():
            for thr, modo in itertools.product(thrs, ("segue_acima", "fade_acima")):
                rows.append(make_row("7_regime_chave", dict(N=N, var=var, lim=thr, modo=modo),
                                     regime_switch(base_orb[N], base_fade[(N, 30, "mid", "none")], N, var, thr, modo)))
    return rows, bench


IDEAS = ("1_orb", "1b_orb_reentrada", "2_filtro", "3_premarket", "4_fade", "5_pm_fade", "6_regime_filtro", "7_regime_chave")
R1_FINALISTS = [("4_fade", dict(N=15, espera=30, alvo="mid", stop="none")),
                ("4_fade", dict(N=30, espera=10, alvo="opp", stop="ext")),
                ("2_filtro", dict(N=60, filtro="gap_favor"))]


def distinct(r, f):
    """Estruturalmente distinta de f: outra ideia, ou outra faixa (N / base) na mesma ideia."""
    if r["idea"] != f["idea"]:
        return True
    for k in ("N", "base"):
        if k in f["params"]:
            return r["params"].get(k) != f["params"].get(k)
    return False


def pick_one(cands):
    """Lider por min(t); empate (<= TIE_T do lider) decidido por simetria compra/venda e depois pelos vizinhos."""
    if not cands:
        return None
    top = max(r["mint"] for r in cands)
    tied = [r for r in cands if r["mint"] >= top - TIE_T]
    tied.sort(key=lambda r: (-int(r["sym"]), -(r["nt"] if np.isfinite(r["nt"]) else -9.0), -r["mint"]))
    return tied[0]


def pick_finalists(cand):
    fin = []
    f1 = pick_one(cand)
    if f1 is None:
        return fin
    fin.append(f1)
    f2 = pick_one([r for r in cand if r is not f1 and distinct(r, f1)])
    if f2 is not None:
        fin.append(f2)
    return fin


SELECTION_RULE = ("Rodada 2. Somente blocos P0 e P1 de lab.evaluate, custo base 3 bps/lado: n>=30 em P0 e em P1; liquido medio >0 "
                  "em P0 E em P1; ordenacao por min(t_P0, t_P1); empate (ate 0,25 de t do lider) decidido por compra e venda ambas "
                  "positivas (soma P0+P1, >=10 trades cada) e depois pela media do min(t) dos vizinhos (1 parametro diferente). "
                  "F1 = vencedor; F2 = vencedor entre configs estruturalmente distintas (outra ideia ou outra faixa). "
                  "P2 so foi calculado depois de congelar.")


def main():
    rows, bench = search()
    for idea in IDEAS:
        show([r for r in rows if r["idea"] == idea], idea)
    n_cfg = len(rows)
    ok = [r for r in rows if enough(r)]
    cand = [r for r in ok if passes(r)]
    for r in ok:
        r["nt"] = neighbor_mint(r, rows)
    cand.sort(key=lambda r: -r["mint"])
    print("\nTOTAL de configuracoes testadas: %d | com n>=%d em P0 e P1: %d | PASSAM no criterio dos dois regimes: %d "
          "(dessas, compra e venda ambas positivas: %d)" % (n_cfg, MIN_TRADES, len(ok), len(cand), sum(r["sym"] for r in cand)))
    print("Sinais (configs com n suficiente): P0>0 e P1>0: %d | P0>0 e P1<=0: %d | P0<=0 e P1>0: %d | ambos <=0: %d" % (
        sum(r["net0"] > 0 and r["net1"] > 0 for r in ok), sum(r["net0"] > 0 and r["net1"] <= 0 for r in ok),
        sum(r["net0"] <= 0 and r["net1"] > 0 for r in ok), sum(r["net0"] <= 0 and r["net1"] <= 0 for r in ok)))
    print("\n=== CANDIDATOS (passam nos dois regimes), por min(t) ===")
    for r in cand[:25]:
        print("  " + line(r) + " t_viz=%5.2f" % r["nt"])
    # onde ficam os finalistas da rodada 1
    ranked = sorted(ok, key=lambda r: -r["mint"])
    print("\n=== FINALISTAS DA RODADA 1 na nova busca (posicao por min(t) entre %d configs com n suficiente) ===" % len(ranked))
    for idea, p in R1_FINALISTS:
        hit = [r for r in rows if r["idea"] == idea and r["params"] == p]
        if not hit:
            print("  %s %s: nao encontrado" % (idea, pstr(p)))
            continue
        r = hit[0]
        pos = (ranked.index(r) + 1) if r in ranked else None
        print("  pos=%s passa=%s | %s" % (pos, passes(r), line(r)))
    return rows, cand, n_cfg, bench, ok


def hm(N):
    return "%02d:%02d" % divmod(lab.RTH_START + N, 60)


VAR_TXT = {"ret10": "retorno dos 10 pregoes anteriores (fechamento de ontem / fechamento de 11 pregoes atras - 1)",
           "atr14": "atr14 (amplitude media max/min-1 dos 14 pregoes anteriores)",
           "wrel": "largura relativa da faixa ((max/min-1 da faixa)/atr14, dividida pela mediana desse quociente nos dias anteriores)"}


def describe_base(idea, p):
    """(nome, descricao, gatilho) das ideias 1-5."""
    if idea == "4_fade":
        N = p["N"]
        alvo = {"mid": "o meio da faixa", "opp": "o lado oposto da faixa", "close": "nenhum (sai no fechamento)"}[p["alvo"]]
        stop = "extremo (max/min) atingido durante o rompimento" if p["stop"] == "ext" else "sem stop de preco"
        esp = "qualquer momento ate 15:57" if p["espera"] is None else "ate %d candles de 1m depois do rompimento" % p["espera"]
        name = "fade_or%d_esp%s_%s_%s" % (N, "livre" if p["espera"] is None else p["espera"], p["alvo"], p["stop"])
        desc = ("Fade de falso rompimento da faixa de abertura de %d min: 1o fechamento de 1m fora da faixa e depois "
                "fechamento de volta para dentro (%s) -> entra contra o rompimento. Alvo: %s. Stop: %s. Saida final 15:59."
                % (N, esp, alvo, stop))
        live = ("1) Marque maxima H e minima L de 09:30 ate %s (candles de 1m, pregao regular). 2) Espere o PRIMEIRO fechamento "
                "de 1m acima de H ou abaixo de L (so vale o primeiro do dia) e anote o extremo atingido desde entao. "
                "3) Se, %s, um candle de 1m FECHAR de volta dentro de [L, H], entre no open do candle seguinte CONTRA o "
                "rompimento (rompeu para cima -> vende; para baixo -> compra). Nao entra se esse fechamento ja passou do alvo. "
                "4) Alvo: %s. Stop: %s. 5) Sem alvo/stop, zera no fechamento de 15:59. Maximo 1 trade por dia."
                % (hm(N), esp, alvo, stop))
    elif idea == "5_pm_fade":
        alvo = {"mid": "o meio da faixa do pre-market", "opp": "o lado oposto da faixa do pre-market",
                "close": "nenhum (sai no fechamento)"}[p["alvo"]]
        stop = "extremo (max/min) atingido durante o rompimento" if p["stop"] == "ext" else "sem stop de preco"
        esp = "qualquer momento ate 15:57" if p["espera"] is None else "ate %d candles de 1m depois do rompimento" % p["espera"]
        name = "pmfade_esp%s_%s_%s" % ("livre" if p["espera"] is None else p["espera"], p["alvo"], p["stop"])
        desc = ("Fade de falso rompimento da maxima/minima do pre-market (04:00-09:29) no pregao regular: 1o fechamento de 1m fora "
                "da faixa e depois fechamento de volta para dentro (%s) -> entra contra o rompimento. Alvo: %s. Stop: %s. "
                "Saida final 15:59." % (esp, alvo, stop))
        live = ("1) As 09:29 anote maxima PH e minima PL do pre-market (04:00-09:29). 2) A partir de 09:30 espere o PRIMEIRO "
                "fechamento de 1m acima de PH ou abaixo de PL e anote o extremo atingido desde entao. 3) Se, %s, um candle de 1m "
                "FECHAR de volta dentro de [PL, PH], entre no open do candle seguinte CONTRA o rompimento. Nao entra se esse "
                "fechamento ja passou do alvo. 4) Alvo: %s. Stop: %s. 5) Sem alvo/stop, zera 15:59. Maximo 1 trade por dia."
                % (esp, alvo, stop))
    elif idea == "3_premarket":
        stop = {"opp": "lado oposto da faixa do pre-market", "mid": "meio da faixa do pre-market",
                "pct": "%.0f%% do preco de entrada" % (PCT_STOP * 100)}[p["stop"]]
        ex = {"close": "fechamento 15:59", "2R": "alvo de 2R (2x a distancia ate o stop) ou 15:59",
              "trail": "trailing de 1R ou 15:59"}[p["exit"]]
        lim = "15:57" if p["ate"] == "fim" else "10:59"
        name = "premkt_sig%d_%s_%s_%s" % (p["sig"], p["stop"], p["exit"], p["ate"].replace(":", ""))
        desc = ("Rompimento da maxima/minima do pre-market (04:00-09:29) no pregao regular: 1o fechamento de %dm fora da faixa "
                "(sinal ate %s) -> entra a favor. Stop: %s. Saida: %s." % (p["sig"], lim, stop, ex))
        live = ("1) As 09:29 anote maxima PH e minima PL do pre-market (04:00-09:29). 2) A partir de 09:30, no PRIMEIRO fechamento de "
                "candle de %dm acima de PH (compra) ou abaixo de PL (venda), desde que ocorra ate %s, entre no open do candle de 1m "
                "seguinte. 3) Stop: %s. 4) Saida: %s. Maximo 1 trade por dia." % (p["sig"], lim, stop, ex))
    else:
        N = p["N"]
        f = p.get("filtro", "")
        sig, st, ex = p.get("sig", 1), p.get("stop", "opp"), p.get("exit", "close")
        re_ = idea == "1b_orb_reentrada"
        name = ("orb%d_%s" % (N, f.replace(":", ""))) if f else ("orb%d_sig%s_%s_%s%s" % (N, sig, st, ex, "_re" if re_ else ""))
        fdesc = {"gap_favor": "so se o rompimento for na MESMA direcao do gap de abertura (open vs fechamento regular anterior)",
                 "gap_contra": "so se o rompimento for CONTRA o gap de abertura",
                 "estreita": "so se largura da faixa/atr14 < mediana dos dias anteriores",
                 "larga": "so se largura da faixa/atr14 >= mediana dos dias anteriores",
                 "vol_alto": "so se volume da faixa > media dos 20 dias anteriores no mesmo intervalo",
                 "vol_baixo": "so se volume da faixa <= media dos 20 dias anteriores no mesmo intervalo"}.get(
                     f, ("so se a entrada ocorrer %s" % f.replace("_", " ")) if f else "sem filtro")
        stop = {"opp": "lado oposto da faixa", "mid": "meio da faixa", "pct": "%.0f%% do preco de entrada" % (PCT_STOP * 100)}[st]
        exd = {"close": "fechamento 15:59", "1R": "alvo de 1R ou 15:59", "2R": "alvo de 2R ou 15:59", "3R": "alvo de 3R ou 15:59",
               "trail": "trailing de 1R ou 15:59"}[ex]
        red = " Se stopar, reentra no 1o fechamento fora do lado oposto (mesmas regras)." if re_ else ""
        desc = ("ORB %d min, sinal no 1o fechamento de %dm fora da faixa, entrada a favor, stop: %s, saida: %s; filtro: %s.%s"
                % (N, sig, stop, exd, fdesc, red))
        live = ("1) Marque maxima H e minima L de 09:30 ate %s. 2) No PRIMEIRO fechamento de candle de %dm acima de H (compra) ou "
                "abaixo de L (venda), verifique o filtro: %s. 3) Se passar, entre no open do candle de 1m seguinte. 4) Stop: %s "
                "(R = distancia entrada-stop). 5) Saida: %s.%s Maximo 1 trade por dia%s."
                % (hm(N), sig, fdesc, stop, exd, red, " (2 com a reentrada)" if re_ else ""))
    return name, desc, live


def describe(r):
    p, idea = r["params"], r["idea"]
    if idea == "6_regime_filtro":
        bidea, bp = {"orb15": ("1_orb", dict(N=15, sig=1, stop="opp", exit="close")),
                     "orb60": ("1_orb", dict(N=60, sig=1, stop="opp", exit="close")),
                     "fade15": ("4_fade", dict(N=15, espera=30, alvo="mid", stop="none")),
                     "fade30": ("4_fade", dict(N=30, espera=30, alvo="mid", stop="none"))}[p["base"]]
        bn, bd, bl = describe_base(bidea, bp)
        cond = "%s %s %s" % (VAR_TXT[p["var"]], p["lado"], p["lim"])
        name = "%s_se_%s_%s_%s" % (bn, p["var"], "acima" if p["lado"] == ">" else "ate", str(p["lim"]).replace("-", "m").replace(".", "p"))
        return name, "CONDICIONAL A REGIME: so opera se %s. Regra-base: %s" % (cond, bd), \
            "0) Antes de operar (dado conhecido antes do trade), confira: %s. Se nao, nao opera no dia. %s" % (cond, bl)
    if idea == "7_regime_chave":
        N = p["N"]
        _, fd, fl = describe_base("1_orb", dict(N=N, sig=1, stop="opp", exit="close"))
        _, gd, gl = describe_base("4_fade", dict(N=N, espera=30, alvo="mid", stop="none"))
        hi, lo = ("ROMPIMENTO", "FADE") if p["modo"] == "segue_acima" else ("FADE", "ROMPIMENTO")
        name = "chave_or%d_%s_%s_%s" % (N, p["var"], str(p["lim"]).replace("-", "m").replace(".", "p"), p["modo"])
        desc = ("CHAVE DE REGIME na faixa de %d min: se %s > %s opera %s; caso contrario opera %s. ROMPIMENTO = %s FADE = %s"
                % (N, VAR_TXT[p["var"]], p["lim"], hi, lo, fd, gd))
        live = ("0) Calcule %s. Se > %s, o dia e de %s; senao, de %s. ROMPIMENTO: %s FADE: %s"
                % (VAR_TXT[p["var"]], p["lim"], hi, lo, fl, gl))
        return name, desc, live
    return describe_base(idea, p)


# ----------------------------------------------------------------------------- depois de congelar
def facts():
    """Fatos descritivos por periodo (calculados depois de congelar os finalistas)."""
    print("\n=== FATOS por periodo ===")
    for N in (15, 30, 60):
        rec = []
        for d in DAYS:
            hi, lo = d["h"][:N].max(), d["l"][:N].min()
            i, side = first_break(d, hi, lo, N, 1)
            if i is None:
                rec.append(dict(k=d["key"], brk=0))
                continue
            e = d["o"][i + 1]
            back30 = any(lo <= x <= hi for x in d["c"][i + 1:i + 31])
            back = any(lo <= x <= hi for x in d["c"][i + 1:])
            opp = (d["l"][i + 1:].min() <= lo) if side > 0 else (d["h"][i + 1:].max() >= hi)
            rec.append(dict(k=d["key"], brk=1, side=side, ret=side * (d["c"][-1] / e - 1), back=float(back),
                            back30=float(back30), opp=float(opp), w=hi / lo - 1,
                            same=float(np.sign(d["c"][-1] - d["o"][0]) == side)))
        f = pd.DataFrame(rec)
        f["per"] = f["k"].map(PERIOD)
        for per in ("P0", "P1", "P2"):
            sub = f[f["per"] == per]
            b = sub[sub["brk"] == 1]
            up, dn = b[b["side"] > 0], b[b["side"] < 0]
            print("OR%-2d %s dias=%d romp=%d (cima %d) | largura med=%.2f%% | volta p/ dentro em 30min=%.0f%% no dia=%.0f%% | "
                  "toca lado oposto=%.0f%% | dia fecha na dir.=%.0f%% | ret bruto ate o close: media=%.0fbps (cima %.0f, baixo %.0f) "
                  "mediana=%.0fbps" % (
                      N, per, len(sub), len(b), len(up), b["w"].median() * 100, b["back30"].mean() * 100, b["back"].mean() * 100,
                      b["opp"].mean() * 100, b["same"].mean() * 100, b["ret"].mean() * 1e4, up["ret"].mean() * 1e4,
                      dn["ret"].mean() * 1e4, b["ret"].median() * 1e4))


def final(rows, cand, n_cfg, bench, ok):
    finalists = pick_finalists(cand)
    extra = ""
    if not finalists:
        r = max(ok, key=lambda r: r["mint"])
        finalists = [r]
        extra = "NADA satisfez o criterio dos dois regimes (liquido>0 em P0 e em P1 com n>=30); este e apenas o 'menos pior' (maior min t). "
    print("\n=== FINALISTAS (congelados so com P0/P1) ===")
    for r in finalists:
        print("  " + line(r) + " t_viz=%5.2f" % r["nt"])
    print("\n################ A PARTIR DAQUI: P2 (nada e reajustado) ################")
    eb = lab.evaluate(bench)
    for per in ("P0", "P1", "P2"):
        print(lab.fmt(eb[per], "BENCH vendido open->close %s" % per))
    for k, r in enumerate(finalists, 1):
        name, desc, live = describe(r)
        trades = r["trades"]
        ev, evs = lab.evaluate(trades), lab.evaluate(trades, lab.COST_STRESS)
        L, S = side_split(trades)
        eL, eS = lab.evaluate(L), lab.evaluate(S)
        sidetxt = " | ".join("%s compra n=%d net=%.1fbps, venda n=%d net=%.1fbps" % (
            per, eL[per].get("n_trades", 0), g(eL[per], "avg_net_bps"), eS[per].get("n_trades", 0), g(eS[per], "avg_net_bps"))
            for per in ("P0", "P1", "P2"))
        notes = extra + ("Ideia %s. min(t_P0,t_P1)=%.2f (t_P0=%.2f, t_P1=%.2f), media do min(t) dos vizinhos=%.2f, simetria "
                         "compra/venda no IS=%s. %s. P2 base: n=%d net=%.1fbps t=%.2f; P2 estresse: net=%.1fbps t=%.2f. "
                         "Benchmark vendido open->close Sharpe: P0 %.2f, P1 %.2f, P2 %.2f. %d configs testadas, %d passaram no "
                         "criterio dos dois regimes: com esse numero de testes a evidencia e fraca."
                         % (r["idea"], r["mint"], r["t0"], r["t1"], r["nt"], "sim" if r["sym"] else "nao", sidetxt,
                            ev["P2"].get("n_trades", 0), g(ev["P2"], "avg_net_bps"), g(ev["P2"], "t_stat"),
                            g(evs["P2"], "avg_net_bps"), g(evs["P2"], "t_stat"),
                            eb["P0"]["sharpe_daily_ann"], eb["P1"]["sharpe_daily_ann"], eb["P2"]["sharpe_daily_ann"],
                            n_cfg, len(cand)))
        lab.save_result(family=FAMILY, name=name, description=desc, params=dict(r["params"], idea=r["idea"]),
                        trades=trades, n_configs_tested=n_cfg, selection_rule=SELECTION_RULE,
                        live_trigger=live, notes=notes)
        print("\n--- F%d %s ---" % (k, name))
        for per in ("P0", "P1", "P2"):
            print(lab.fmt(ev[per], "  %s base    " % per))
            print(lab.fmt(evs[per], "  %s estresse" % per))
            print(lab.fmt(eL[per], "  %s COMPRA  " % per))
            print(lab.fmt(eS[per], "  %s VENDA   " % per))
        print("  motivos de saida:", pd.Series([t["reason"] for t in trades]).value_counts().to_dict())
        print("  descricao:", desc)
        print("  gatilho:", live)
    # finalistas da rodada 1, agora com P2 (apenas informativo)
    print("\n=== FINALISTAS DA RODADA 1 (informativo, P0/P1/P2) ===")
    for idea, p in R1_FINALISTS:
        for r in rows:
            if r["idea"] == idea and r["params"] == p:
                ev = lab.evaluate(r["trades"])
                print("  %s %s" % (idea, pstr(p)))
                for per in ("P0", "P1", "P2"):
                    print(lab.fmt(ev[per], "     %s" % per))
    # variaveis de regime por periodo
    print("\n=== VARIAVEIS DE REGIME por periodo (media / mediana) ===")
    per = pd.Series(PERIOD)
    for nm, s in (("ret10", RET10), ("atr14", ATR), ("gap", GAP), ("largura OR15", CTX[15]["w_rel"] * ATR),
                  ("largura OR30", CTX[30]["w_rel"] * ATR)):
        print("  %-13s " % nm + " | ".join("%s %.2f%% / %.2f%%" % (q, s[per == q].mean() * 100, s[per == q].median() * 100)
                                            for q in ("P0", "P1", "P2")))
    facts()


if __name__ == "__main__":
    rows, cand, n_cfg, bench, ok = main()
    if os.environ.get("ORB_STAGE") == "search":
        sys.exit(0)
    final(rows, cand, n_cfg, bench, ok)
