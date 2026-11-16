import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0]); import lab
# -*- coding: utf-8 -*-
"""
Familia "orb" -- rompimento de faixa de abertura no SOXL (1 minuto).

Ideias:
  1. ORB classico (faixa 5/15/30/60 min; sinal 1m ou 5m; stop oposto/meio/2%; saida close/1R/2R/3R/trailing 1R;
     variante com reentrada na direcao oposta).
  2. Filtros sobre o ORB classico (gap a favor/contra, faixa estreita/larga vs atr14, volume da faixa alto/baixo,
     horario limite do rompimento).
  3. Rompimento da maxima/minima do pre-market durante o pregao regular.
  4. Fade do falso rompimento (fecha fora e volta a fechar dentro da faixa).

Selecao SOMENTE no in-sample (lab.evaluate_is). Rodar: python3 agents/orb.py   (a partir de a pasta backtest)
ORB_STAGE=search python3 agents/orb.py  -> so a busca no IS (nao calcula nem imprime OOS).
"""
import os
import itertools
import numpy as np
import pandas as pd

FAMILY = "orb"
LAST_SIGNAL_I = 387          # ultimo candle de sinal (15:57) -> entrada 15:58, saida no close de 15:59
PCT_STOP = 0.02              # stop percentual fixo
MIN_TRADES_IS = 30

# ----------------------------------------------------------------------------- dados
DT = lab.daily_table()
DAYS = []
for dt, g in lab.day_groups(lab.rth()):
    assert len(g) == 390, (dt, len(g))
    DAYS.append(dict(date=dt, key=str(dt), df=g, o=g["open"].values, h=g["high"].values, l=g["low"].values,
                     c=g["close"].values, v=g["volume"].values, m=g["m"].values))
KEYS = [d["key"] for d in DAYS]
IS_KEYS = set(k for d, k in zip(DAYS, KEYS) if d["date"] <= lab.IS_END)


def build_context():
    """Variaveis de filtro por dia, sempre comparadas com dias ANTERIORES (sem look-ahead)."""
    ctx = {}
    for N in (5, 15, 30, 60):
        width, vol = [], []
        for d in DAYS:
            hi, lo = d["h"][:N].max(), d["l"][:N].min()
            width.append(hi / lo - 1.0)
            vol.append(float(d["v"][:N].sum()))
        width = pd.Series(width, index=KEYS)
        vol = pd.Series(vol, index=KEYS)
        atr = pd.Series(DT["atr14"].values, index=[str(x) for x in DT.index]).reindex(KEYS)
        w_rel = width / atr                                              # largura da faixa / atr14 (dias anteriores)
        w_med_prev = w_rel.shift(1).expanding(min_periods=10).median()   # mediana so de dias anteriores
        v_rel = vol / vol.shift(1).rolling(20, min_periods=5).mean()     # volume da faixa / media dos 20 dias anteriores
        ctx[N] = dict(w_rel=w_rel, w_med_prev=w_med_prev, v_rel=v_rel)
    gap = pd.Series(DT["gap"].values, index=[str(x) for x in DT.index]).reindex(KEYS)
    return ctx, gap


CTX, GAP = build_context()


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


def gen_premarket(sig, stop_mode, exit_mode, i_max):
    """Ideia 3: rompimento da maxima/minima do pre-market (04:00-09:29) durante o pregao regular."""
    out = []
    for d in DAYS:
        row = DT.loc[d["date"]]
        hi, lo = row["pre_h"], row["pre_l"]
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        i, side = first_break(d, hi, lo, 0, sig, i_max)
        if i is None:
            continue
        t, _ = bracket_trade(d, i, side, hi, lo, stop_mode, exit_mode, "pm")
        if t is not None:
            out.append(t)
    return out


def gen_fade(N, wait, target_mode, stop_mode):
    """Ideia 4: falso rompimento. 1o fechamento de 1m fora da faixa (direcao b); se em ate `wait` candles
    um fechamento voltar para dentro da faixa, entra CONTRA o rompimento. Stop = extremo da excursao (ou sem stop);
    alvo = meio da faixa, lado oposto da faixa, ou fechamento."""
    out = []
    for d in DAYS:
        hi, lo = d["h"][:N].max(), d["l"][:N].min()
        i, b = first_break(d, hi, lo, N, 1)
        if i is None:
            continue
        c = d["c"]
        j_end = LAST_SIGNAL_I if wait is None else min(LAST_SIGNAL_I, i + wait)
        j = None
        for k in range(i + 1, j_end + 1):
            if lo <= c[k] <= hi:
                j = k
                break
        if j is None:
            continue
        side = -b
        ext = d["h"][i:j + 1].max() if b > 0 else d["l"][i:j + 1].min()
        stop = ext if stop_mode == "ext" else None
        mid = (hi + lo) / 2.0
        target = mid if target_mode == "mid" else ((lo if b > 0 else hi) if target_mode == "opp" else None)
        if target is not None and side * (target - c[j]) <= 0:   # alvo ja ultrapassado no fechamento do sinal
            continue
        res = lab.sim_exit(d["df"], j + 1, side, stop=stop, target=target)
        if res is None:
            continue
        if stop is not None and side * (res["entry_px"] - stop) <= 0:
            continue
        out.append(lab.make_trade(d["df"], res, side, tag="fade"))
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


# ----------------------------------------------------------------------------- metricas
def side_split(trades):
    return [t for t in trades if t["side"] > 0], [t for t in trades if t["side"] < 0]


def is_row(idea, params, trades):
    m = lab.evaluate_is(trades)
    L, S = side_split(trades)
    mL, mS = lab.evaluate_is(L), lab.evaluate_is(S)
    return dict(idea=idea, params=params, trades=trades,
                n=m.get("n_trades", 0), net=m.get("avg_net_bps", np.nan), t=m.get("t_stat", np.nan),
                sharpe=m.get("sharpe_daily_ann", np.nan), total=m.get("total_net_pct", np.nan),
                nL=mL.get("n_trades", 0), netL=mL.get("avg_net_bps", np.nan), tL=mL.get("t_stat", np.nan),
                nS=mS.get("n_trades", 0), netS=mS.get("avg_net_bps", np.nan), tS=mS.get("t_stat", np.nan))


def pstr(p):
    return ",".join("%s=%s" % (k, v) for k, v in p.items())


def show(rows, title, top=12):
    print("\n=== %s  (%d configs) ===" % (title, len(rows)))
    ok = [r for r in rows if r["n"] >= MIN_TRADES_IS]
    print("configs com n>=%d: %d | net>0: %d | compra>0 e venda>0: %d | mediana net=%.1fbps" % (
        MIN_TRADES_IS, len(ok), sum(r["net"] > 0 for r in ok),
        sum((r["netL"] > 0) and (r["netS"] > 0) for r in ok),
        float(np.median([r["net"] for r in ok])) if ok else float("nan")))
    for r in sorted(ok, key=lambda r: -r["t"])[:top]:
        print("  %-52s n=%3d net=%7.1f t=%5.2f sh=%5.2f | C n=%3d net=%7.1f t=%5.2f | V n=%3d net=%7.1f t=%5.2f" % (
            pstr(r["params"]), r["n"], r["net"], r["t"], r["sharpe"],
            r["nL"], r["netL"], r["tL"], r["nS"], r["netS"], r["tS"]))


def neighbor_t(r, rows):
    """t medio dos vizinhos (mesma ideia, diferem em exatamente 1 parametro, n>=30)."""
    ts = []
    for q in rows:
        if q is r or q["idea"] != r["idea"] or q["n"] < MIN_TRADES_IS:
            continue
        if set(q["params"]) != set(r["params"]):
            continue
        if sum(q["params"][k] != r["params"][k] for k in r["params"]) == 1:
            ts.append(q["t"])
    return float(np.mean(ts)) if ts else float("nan")


# ----------------------------------------------------------------------------- busca no IS
def search():
    rows = []
    # benchmark: vendido do open ao close
    bench = [lab.trade_from_prices(d["key"], -1, d["m"][0], d["o"][0], d["m"][-1], d["c"][-1], tag="bench") for d in DAYS]
    print("BENCH vendido open->close  " + lab.fmt(lab.evaluate_is(bench), "IS"))

    # ideia 1 -- 4*2*3*5 = 120 + 8 com reentrada
    for N, sig, st, ex in itertools.product((5, 15, 30, 60), (1, 5), ("opp", "mid", "pct"),
                                            ("close", "1R", "2R", "3R", "trail")):
        rows.append(is_row("1_orb", dict(N=N, sig=sig, stop=st, exit=ex), gen_orb(N, sig, st, ex)))
    for N, sig in itertools.product((5, 15, 30, 60), (1, 5)):
        rows.append(is_row("1b_orb_reentrada", dict(N=N, sig=sig, stop="opp", exit="close"),
                           gen_orb(N, sig, "opp", "close", reentry=True)))
    # ideia 2 -- filtros sobre ORB (sinal 1m, stop oposto, saida no close): 4*10 = 40
    for N in (5, 15, 30, 60):
        base = gen_orb(N, 1, "opp", "close")
        for f in ("gap_favor", "gap_contra", "estreita", "larga", "vol_alto", "vol_baixo",
                  "ate_10:30", "ate_11:00", "ate_12:00", "ate_14:00"):
            rows.append(is_row("2_filtro", dict(N=N, filtro=f), apply_filter(base, N, f)))
    # ideia 3 -- pre-market: 2*3*3*2 = 36
    for sig, st, ex, cut in itertools.product((1, 5), ("opp", "mid", "pct"), ("close", "2R", "trail"), ("fim", "11:00")):
        i_max = LAST_SIGNAL_I if cut == "fim" else (11 * 60 - lab.RTH_START - 1)
        rows.append(is_row("3_premarket", dict(sig=sig, stop=st, exit=ex, ate=cut), gen_premarket(sig, st, ex, i_max)))
    # ideia 4 -- fade: 4*3*3*2 = 72
    for N, wait, tg, st in itertools.product((5, 15, 30, 60), (10, 30, None), ("mid", "opp", "close"), ("ext", "none")):
        rows.append(is_row("4_fade", dict(N=N, espera=wait, alvo=tg, stop=st), gen_fade(N, wait, tg, st)))
    return rows, bench


def facts():
    """Fatos descritivos (calculados depois de congelar os finalistas; separados por amostra)."""
    print("\n=== FATOS (IS | OOS) ===")
    for N in (5, 15, 30, 60):
        rec = []
        for d in DAYS:
            hi, lo = d["h"][:N].max(), d["l"][:N].min()
            i, side = first_break(d, hi, lo, N, 1)
            if i is None:
                rec.append(dict(k=d["key"], brk=0))
                continue
            e = d["o"][i + 1]
            ret_close = side * (d["c"][-1] / e - 1)
            # voltou a fechar dentro da faixa depois do rompimento?
            back = any(lo <= x <= hi for x in d["c"][i + 1:])
            # tocou o lado oposto?
            opp = (d["l"][i + 1:].min() <= lo) if side > 0 else (d["h"][i + 1:].max() >= hi)
            # extremo do dia formado dentro da faixa?
            rec.append(dict(k=d["key"], brk=1, side=side, m=int(d["m"][i]), ret=ret_close, back=back, opp=opp,
                            w=hi / lo - 1, same=np.sign(d["c"][-1] - d["o"][0]) == side,
                            hod_in=d["h"].max() <= hi, lod_in=d["l"].min() >= lo))
        f = pd.DataFrame(rec)
        for lbl, sub in (("IS ", f[f["k"].isin(IS_KEYS)]), ("OOS", f[~f["k"].isin(IS_KEYS)])):
            b = sub[sub["brk"] == 1]
            b = b.assign(**{k_: b[k_].astype(float) for k_ in ("back", "opp", "same", "hod_in", "lod_in")})
            print("OR%-2d %s dias=%d romp=%d (cima %d) | largura med=%.2f%% | min. mediano do romp=%s | "
                  "volta p/ dentro=%.0f%% | toca lado oposto=%.0f%% | dia fecha na dir. do romp=%.0f%% | "
                  "ret bruto ate o close: media=%.0fbps mediana=%.0fbps | max do dia na faixa=%.0f%% min na faixa=%.0f%%" % (
                      N, lbl, len(sub), len(b), int((b["side"] > 0).sum()), b["w"].median() * 100,
                      "%02d:%02d" % divmod(int(b["m"].median()), 60), b["back"].mean() * 100, b["opp"].mean() * 100,
                      b["same"].mean() * 100, b["ret"].mean() * 1e4, b["ret"].median() * 1e4,
                      b["hod_in"].mean() * 100, b["lod_in"].mean() * 100))


def report(name, trades):
    ev = lab.evaluate(trades)
    evs = lab.evaluate(trades, lab.COST_STRESS)
    L, S = side_split(trades)
    eL, eS = lab.evaluate(L), lab.evaluate(S)
    print("\n--- %s ---" % name)
    for blk in ("IS", "OOS"):
        print(lab.fmt(ev[blk], "  %-3s base   " % blk))
        print(lab.fmt(evs[blk], "  %-3s estresse" % blk))
        print(lab.fmt(eL[blk], "  %-3s COMPRA " % blk))
        print(lab.fmt(eS[blk], "  %-3s VENDA  " % blk))


def main():
    rows, bench = search()
    n_cfg = len(rows)
    for idea in ("1_orb", "1b_orb_reentrada", "2_filtro", "3_premarket", "4_fade"):
        show([r for r in rows if r["idea"] == idea], idea)
    print("\nTOTAL de configuracoes testadas: %d" % n_cfg)

    # ---- regra de selecao (so IS): n>=30, liquido>0 no custo base, COMPRA e VENDA ambas >0 com >=10 trades cada,
    #      score = media entre o t da config e o t medio dos vizinhos (estabilidade); 1 finalista por ideia; top 3.
    cand = []
    for r in rows:
        if r["n"] >= MIN_TRADES_IS and r["net"] > 0 and r["nL"] >= 10 and r["nS"] >= 10 and r["netL"] > 0 and r["netS"] > 0:
            nt = neighbor_t(r, rows)
            r["nt"] = nt
            r["score"] = 0.5 * r["t"] + 0.5 * (nt if np.isfinite(nt) else 0.0)
            cand.append(r)
    cand.sort(key=lambda r: -r["score"])
    print("\n=== CANDIDATOS SIMETRICOS no IS (compra>0 e venda>0): %d ===" % len(cand))
    for r in cand[:20]:
        print("  %-18s %-48s n=%3d net=%6.1f t=%5.2f t_viz=%5.2f score=%5.2f | C %3d %6.1f | V %3d %6.1f" % (
            r["idea"], pstr(r["params"]), r["n"], r["net"], r["t"], r["nt"], r["score"],
            r["nL"], r["netL"], r["nS"], r["netS"]))
    return rows, cand, n_cfg, bench


def regen(r):
    """Regera os trades (todas as datas) de uma linha da busca."""
    return r["trades"]


def describe(r):
    p = r["params"]
    hm = lambda N: "%02d:%02d" % divmod(lab.RTH_START + N, 60)
    if r["idea"] == "4_fade":
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
    elif r["idea"] == "3_premarket":
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
        name = "orb%d_%s" % (N, f.replace(":", "")) if f else "orb%d_sig%s_%s_%s" % (N, p.get("sig"), p.get("stop"), p.get("exit"))
        fdesc = {"gap_favor": "so se o rompimento for na MESMA direcao do gap de abertura (open vs fechamento regular anterior)",
                 "gap_contra": "so se o rompimento for CONTRA o gap de abertura",
                 "estreita": "so se largura da faixa/atr14 < mediana dos dias anteriores",
                 "larga": "so se largura da faixa/atr14 >= mediana dos dias anteriores",
                 "vol_alto": "so se volume da faixa > media dos 20 dias anteriores no mesmo intervalo",
                 "vol_baixo": "so se volume da faixa <= media dos 20 dias anteriores no mesmo intervalo"}.get(
                     f, "so se a entrada ocorrer %s" % f.replace("_", " ") if f else "sem filtro")
        desc = ("ORB %d min, sinal no 1o fechamento de 1m fora da faixa, entrada a favor, stop no lado oposto da faixa, "
                "saida 15:59; filtro: %s." % (N, fdesc))
        live = ("1) Marque maxima H e minima L de 09:30 ate %s. 2) No PRIMEIRO fechamento de 1m acima de H (compra) ou abaixo de L "
                "(venda), verifique o filtro: %s. 3) Se passar, entre no open do candle seguinte. 4) Stop no lado oposto da faixa. "
                "5) Zera no fechamento de 15:59. Maximo 1 trade por dia." % (hm(N), fdesc))
    return name, desc, live


def pick_finalists(cand):
    """Regra congelada ANTES de olhar o OOS (usa apenas metricas do IS):
       F1 = maior score entre os candidatos simetricos;
       F2 = maior score da mesma ideia de F1 com N diferente e COM stop de preco (variante estruturalmente distinta);
       F3 = maior score entre candidatos simetricos de OUTRA ideia."""
    fin = []
    if not cand:
        return fin
    f1 = cand[0]
    fin.append(f1)
    for r in cand[1:]:
        if r["idea"] == f1["idea"] and r["params"].get("N") != f1["params"].get("N") and r["params"].get("stop") != "none":
            fin.append(r)
            break
    for r in cand[1:]:
        if r["idea"] != f1["idea"]:
            fin.append(r)
            break
    return fin


SELECTION_RULE = ("Somente IS, custo base 3 bps/lado: n>=30; liquido medio >0; COMPRA e VENDA ambas com liquido medio >0 e >=10 "
                  "trades cada (simetria, para nao capturar so a deriva vendida intraday do IS); score = 0,5*t + 0,5*t medio dos "
                  "vizinhos (configs da mesma ideia que diferem em 1 parametro). F1 = maior score; F2 = maior score da mesma ideia "
                  "com N diferente e com stop de preco; F3 = maior score de outra ideia.")

if __name__ == "__main__":
    rows, cand, n_cfg, bench = main()
    if os.environ.get("ORB_STAGE") == "search":
        sys.exit(0)
    finalists = pick_finalists(cand)
    print("\n=== FINALISTAS (congelados pela regra do IS) ===")
    for r in finalists:
        print("  %s  %s  score=%.2f" % (r["idea"], pstr(r["params"]), r["score"]))
    print("\n################ A PARTIR DAQUI: OOS (nada e reajustado) ################")
    print(lab.fmt(lab.evaluate(bench)["OOS"], "BENCH vendido open->close OOS"))
    if not finalists:
        # nada simetrico sobreviveu: salva o "menos pior" (maior t no IS com n>=30)
        ok = [r for r in rows if r["n"] >= MIN_TRADES_IS]
        r = max(ok, key=lambda r: r["t"])
        r["score"], r["nt"] = float("nan"), float("nan")
        finalists = [r]
        extra = "NADA sobreviveu ao criterio de simetria no IS; este e apenas o 'menos pior'. "
    else:
        extra = ""
    for k, r in enumerate(finalists, 1):
        name, desc, live = describe(r)
        trades = regen(r)
        L, S = side_split(trades)
        eL, eS = lab.evaluate(L), lab.evaluate(S)
        notes = extra + ("Ideia %s. Score IS=%.2f (t=%.2f, t medio vizinhos=%.2f). "
                         "IS compra: n=%d net=%.1fbps | IS venda: n=%d net=%.1fbps. "
                         "OOS compra: n=%d net=%.1fbps total=%.1f%% | OOS venda: n=%d net=%.1fbps total=%.1f%%. "
                         "Benchmark vendido open->close: IS Sharpe %.2f, OOS Sharpe %.2f. Amostra pequena (t<~2): evidencia fraca."
                         % (r["idea"], r["score"], r["t"], r["nt"], r["nL"], r["netL"], r["nS"], r["netS"],
                            eL["OOS"].get("n_trades", 0), eL["OOS"].get("avg_net_bps", float("nan")),
                            eL["OOS"].get("total_net_pct", float("nan")),
                            eS["OOS"].get("n_trades", 0), eS["OOS"].get("avg_net_bps", float("nan")),
                            eS["OOS"].get("total_net_pct", float("nan")),
                            lab.evaluate(bench)["IS"]["sharpe_daily_ann"], lab.evaluate(bench)["OOS"]["sharpe_daily_ann"]))
        lab.save_result(family=FAMILY, name=name, description=desc, params=dict(r["params"], idea=r["idea"]),
                        trades=trades, n_configs_tested=n_cfg, selection_rule=SELECTION_RULE,
                        live_trigger=live, notes=notes)
        report("F%d %s" % (k, name), trades)
        ex = pd.Series([t["reason"] for t in trades]).value_counts().to_dict()
        print("  motivos de saida (todas as datas):", ex)
        print("  gatilho:", live)
    facts()
