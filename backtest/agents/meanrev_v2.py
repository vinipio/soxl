import os, sys
os.environ['SOXL_RESULTS'] = 'results_v2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import lab
# -*- coding: utf-8 -*-
"""
Familia "meanrev" -- RODADA 2 (249 pregoes; IS = P0+P1 = 199 pregoes; teste = P2).

Rodar:  python3 agents/meanrev_v2.py
  1) refaz a busca SOMENTE com os dias do IS (P0 e P1), lendo apenas os blocos 'P0' e 'P1' de lab.evaluate;
  2) grava a grade em results_v2/meanrev/is_grid_v2.csv (+ grade condicional a regime em regime_grid_v2.csv);
  3) avalia os FINALISTAS congelados (lista FINALISTS) em P0/P1/P2 e grava via lab.save_result.

Criterio (BRIEF2): n>=30 em P0 e em P1; liquido>0 (custo base) em P0 E em P1; ordenar por min(t_P0, t_P1);
desempate por estabilidade dos vizinhos; preferir compra e venda ambas positivas.

Convencoes iguais as da rodada 1: sinal no FECHAMENTO do candle i, entrada no OPEN de i+1; uma posicao por vez;
saida por tempo de H minutos = CLOSE do candle i+H; indicadores so com candles do proprio dia ate i.
Variaveis de regime (ideia 7) usam so informacao anterior ao trade (dias anteriores ou gap de abertura).
"""
import math
import itertools
import numpy as np
import pandas as pd

FAMILY = "meanrev"
COST = lab.COST_BASE
WINDOWS = {"all": (575, 950), "mid": (660, 870)}   # minuto do SINAL: 09:35-15:50 / 11:00-14:30


# ----------------------------------------------------------------------------- dados
class _Col:
    """Coluna leve (numpy) compativel com o que lab.sim_exit/lab.make_trade usam (.values / .iloc[i])."""
    def __init__(self, a):
        self.values = a
        self.iloc = a


class Day:
    """Dia de pregao com as colunas originais + indicadores pre-calculados (sem look-ahead)."""
    def __init__(self, date, g, atr):
        self.date = date
        self.cols = {k: _Col(g[k].values) for k in ["date", "m", "open", "high", "low", "close", "volume", "vwap"]}
        self.n = len(g)
        self.atr = atr
        o = g["open"].values.astype(float); h = g["high"].values.astype(float)
        l = g["low"].values.astype(float); c = g["close"].values.astype(float)
        v = g["volume"].values.astype(float); vw = g["vwap"].values.astype(float)
        self.o, self.h, self.l, self.c, self.m = o, h, l, c, g["m"].values
        cv = np.cumsum(v)
        cv_safe = np.where(cv > 0, cv, np.nan)
        self.svwap = np.cumsum(vw * v) / cv_safe                       # VWAP acumulada da sessao
        var = np.cumsum(v * vw * vw) / cv_safe - self.svwap ** 2
        sd = np.sqrt(np.clip(var, 0, None))                            # banda de VWAP (desvio ponderado por volume)
        z = np.full(self.n, np.nan)
        ok = (sd > 0) & (np.arange(self.n) >= 15)
        z[ok] = (c[ok] - self.svwap[ok]) / sd[ok]
        self.z_band = z
        self.z_atr = (c / self.svwap - 1.0) / atr if (atr is not None and atr == atr and atr > 0) else np.full(self.n, np.nan)
        r1 = np.empty(self.n); r1[0] = c[0] / o[0] - 1.0; r1[1:] = c[1:] / c[:-1] - 1.0
        self.r1 = r1
        self.sig = pd.Series(r1).rolling(60, min_periods=20).std().values   # vol recente de 1m
        # osciladores 1m e 5m
        self.c5 = c[4::5]
        self.i5 = np.arange(4, self.n, 5)[: len(self.c5)]
        self.rsi = {(1, n): _rsi(c, n) for n in (2, 5, 14)}
        self.rsi.update({(5, n): _rsi(self.c5, n) for n in (2, 5, 14)})
        self.bb = {1: _bb(c), 5: _bb(self.c5)}
        # extremos do dia ate o candle anterior
        ph = np.maximum.accumulate(h); pl = np.minimum.accumulate(l)
        self.ph_prev = np.concatenate([[np.nan], ph[:-1]])
        self.pl_prev = np.concatenate([[np.nan], pl[:-1]])
        ah = np.zeros(self.n, dtype=int); al = np.zeros(self.n, dtype=int)   # indice do extremo vigente ate i-1
        ih, il = 0, 0
        for i in range(1, self.n):
            ah[i], al[i] = ih, il
            if h[i] > h[ih]: ih = i
            if l[i] < l[il]: il = i
        self.ih_prev, self.il_prev = ah, al

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        return self.cols[k]


def _rsi(c, n):
    """RSI de Wilder; out[i] usa fechamentos ate i."""
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    up = np.clip(d, 0, None); dn = np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    out[n] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    for j in range(n, len(d)):
        au = (au * (n - 1) + up[j]) / n
        ad = (ad * (n - 1) + dn[j]) / n
        out[j + 1] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return out


def _bb(c, n=20):
    s = pd.Series(c)
    return s.rolling(n).mean().values, s.rolling(n).std(ddof=0).values


def build_days():
    dt = lab.daily_table()
    # variaveis de regime, todas conhecidas ANTES do primeiro trade do dia
    reg = pd.DataFrame({"ret5": dt["prev_c"] / dt["c"].shift(6) - 1.0,     # retorno dos 5 pregoes anteriores (fech/fech)
                        "atr14": dt["atr14"],                               # amplitude media dos 14 pregoes anteriores
                        "gap": dt["gap"]})                                  # gap de abertura do dia
    out = []
    for date, g in lab.day_groups(lab.rth()):
        atr = dt.loc[date, "atr14"] if date in dt.index else np.nan
        D = Day(date, g, float(atr) if atr == atr else np.nan)
        D.reg = {k: float(reg.loc[date, k]) for k in reg.columns}
        out.append(D)
    return out


# ----------------------------------------------------------------------------- geradores de sinal
# cada gerador devolve, para um dia, lista de (i_sinal, side, target, stop, hold_min) em ordem de i

def gen_vwap(D, norm, thr, exitmode, stop_mult):
    """Ideia 1: fade do desvio vs VWAP da sessao (cruzamento do limiar)."""
    z = D.z_band if norm == "band" else D.z_atr
    hold = {"vwap60": 60, "t15": 15, "t30": 30, "t60": 60}[exitmode]
    out = []
    for i in range(1, D.n):
        a, b = z[i - 1], z[i]
        if not (a == a and b == b):
            continue
        if b > thr and a <= thr: side = -1
        elif b < -thr and a >= -thr: side = 1
        else: continue
        dist = D.c[i] - D.svwap[i]
        tg = D.svwap[i] if exitmode == "vwap60" else None
        st = D.c[i] + stop_mult * dist if stop_mult else None
        out.append((i, side, tg, st, hold))
    return out


def gen_rsi(D, tf, n, lo, hold, flip=1):
    """Ideia 2a: RSI cruza para extremo (lo / 100-lo) -> fade."""
    r = D.rsi[(tf, n)]
    out = []
    for j in range(1, len(r)):
        a, b = r[j - 1], r[j]
        if not (a == a and b == b):
            continue
        if b < lo and a >= lo: side = 1
        elif b > 100 - lo and a <= 100 - lo: side = -1
        else: continue
        i = j if tf == 1 else int(D.i5[j])
        out.append((i, side * flip, None, None, hold))
    return out


def gen_bb(D, tf, k, exitmode):
    """Ideia 2b: fechamento cruza para fora da Banda de Bollinger(20,k) -> fade; saida na media ou por tempo."""
    cc = D.c if tf == 1 else D.c5
    ma, sd = D.bb[tf]
    hold = {"mean": 30 if tf == 1 else 60, "t15": 15, "t30": 30}[exitmode]
    up, dn = ma + k * sd, ma - k * sd
    out = []
    for j in range(1, len(cc)):
        if not (sd[j] == sd[j] and sd[j - 1] == sd[j - 1]) or sd[j] <= 0:
            continue
        if cc[j] > up[j] and cc[j - 1] <= up[j - 1]: side = -1
        elif cc[j] < dn[j] and cc[j - 1] >= dn[j - 1]: side = 1
        else: continue
        i = j if tf == 1 else int(D.i5[j])
        out.append((i, side, ma[j] if exitmode == "mean" else None, None, hold))
    return out


def gen_stretch(D, k, X, hold, flip=1):
    """Ideia 3: movimento de k minutos > X * sigma1m * sqrt(k) (sigma medido ANTES do movimento).
    flip=+1 fade; flip=-1 continuacao."""
    out = []
    rk = math.sqrt(k)
    for i in range(k, D.n):
        s = D.sig[i - k]
        if not (s == s) or s <= 0:
            continue
        mv = D.c[i] / D.c[i - k] - 1.0
        if abs(mv) > X * s * rk:
            out.append((i, (-1 if mv > 0 else 1) * flip, None, None, hold))
    return out


def gen_extreme(D, minwait, kind, age, hold, use_stop, flip=1):
    """Ideia 5: candle faz nova maxima/minima do dia.
    kind='reject': fura o extremo anterior e FECHA de volta dentro; kind='touch': qualquer furo.
    age: o extremo anterior precisa ter pelo menos `age` minutos. flip=-1 = rompimento (continuacao)."""
    out = []
    for i in range(max(minwait, 1), D.n):
        hi = D.h[i] > D.ph_prev[i]; lo = D.l[i] < D.pl_prev[i]
        if hi and lo:
            continue
        if hi and (i - D.ih_prev[i]) >= age and (kind == "touch" or D.c[i] < D.ph_prev[i]):
            out.append((i, -1 * flip, None, D.h[i] * 1.0005 if use_stop else None, hold))
        elif lo and (i - D.il_prev[i]) >= age and (kind == "touch" or D.c[i] > D.pl_prev[i]):
            out.append((i, 1 * flip, None, D.l[i] * 0.9995 if use_stop else None, hold))
    return out


def gen_dayext(D, thr, hold, flip=1):
    """Ideia 6 (nova): esticada do dia. x = (close/open do dia - 1) / atr14; cruzamento de +-thr -> fade."""
    out = []
    if not (D.atr == D.atr) or D.atr <= 0:
        return out
    x = (D.c / D.o[0] - 1.0) / D.atr
    for i in range(1, D.n):
        if x[i] > thr and x[i - 1] <= thr: side = -1
        elif x[i] < -thr and x[i - 1] >= -thr: side = 1
        else: continue
        out.append((i, side * flip, None, None, hold))
    return out


GENS = {"dayext": gen_dayext, "vwap": gen_vwap, "rsi": gen_rsi, "bb": gen_bb, "stretch": gen_stretch, "extreme": gen_extreme}


# ----------------------------------------------------------------------------- motor
def run(days, gen, params, window, tag="", regime=None):
    """regime = (var, thr, modo) ou None. modo 'hi': opera so se var > thr; 'lo': so se var <= thr (dia sem var = nao opera)."""
    w0, w1 = WINDOWS[window]
    f = GENS[gen]
    trades = []
    for D in days:
        if regime is not None:
            x = D.reg.get(regime[0], np.nan)
            if not (x == x) or (regime[2] == "hi") != (x > regime[1]):
                continue
        last_exit = -1
        for (i, side, tg, st, hold) in f(D, **params):
            if i < last_exit or D.m[i] < w0 or D.m[i] >= w1 or i + 1 > D.n - 1:
                continue
            res = lab.sim_exit(D, i + 1, side, stop=st, target=tg, i_last=min(i + hold, D.n - 1))
            if res is None:
                continue
            trades.append(lab.make_trade(D, res, side, tag=tag))
            last_exit = res["exit_i"]
    return trades


def side_stats(trades, cost=COST, dates=None):
    """n, bps liquido medio e t por lado (mesma formula de lab._stats)."""
    out = {}
    for nm, sd in (("L", 1), ("S", -1)):
        x = np.array([t["side"] * (t["exit_px"] / t["entry_px"] - 1.0) - 2 * cost / 1e4
                      for t in trades if t["side"] == sd and (dates is None or t["date"] in dates)])
        n = len(x)
        out["n" + nm] = n
        out["bps" + nm] = float(x.mean() * 1e4) if n else float("nan")
        out["t" + nm] = float(x.mean() / (x.std(ddof=1) / math.sqrt(n))) if n > 2 and x.std(ddof=1) > 0 else float("nan")
    return out


def grid():
    """Lista (ideia, gen, params, window). Contagem total = n_configs_tested."""
    G = []
    for norm, thrs in (("band", (1.5, 2.0, 2.5, 3.0)), ("atr", (0.10, 0.15, 0.20, 0.30))):
        for thr, ex, sm, w in itertools.product(thrs, ("vwap60", "t15", "t30", "t60"), (0.0, 1.0), ("all", "mid")):
            G.append(("1_vwap", "vwap", dict(norm=norm, thr=thr, exitmode=ex, stop_mult=sm), w))
    for tf in (1, 5):
        for (n, los), w in itertools.product(((2, (5, 10)), (5, (15, 20)), (14, (25, 30))), ("all", "mid")):
            for lo, hold in itertools.product(los, (5, 15, 30) if tf == 1 else (15, 30, 60)):
                G.append(("2_rsi", "rsi", dict(tf=tf, n=n, lo=lo, hold=hold), w))
        for k, ex, w in itertools.product((2.0, 2.5, 3.0), ("mean", "t15", "t30"), ("all", "mid")):
            G.append(("2_bb", "bb", dict(tf=tf, k=k, exitmode=ex), w))
    for k, X, hold, w, flip in itertools.product((1, 3, 5), (2.0, 3.0, 4.0), (5, 15, 30), ("all", "mid"), (1, -1)):
        G.append(("3_stretch_fade" if flip == 1 else "3_stretch_cont", "stretch", dict(k=k, X=X, hold=hold, flip=flip), w))
    for mw, kind, age, hold, us in itertools.product((30, 60), ("reject", "touch"), (0, 15), (15, 30, 60), (False, True)):
        G.append(("5_extreme_fade", "extreme", dict(minwait=mw, kind=kind, age=age, hold=hold, use_stop=us, flip=1), "all"))
        if not us:
            G.append(("5_extreme_cont", "extreme", dict(minwait=mw, kind=kind, age=age, hold=hold, use_stop=us, flip=-1), "all"))
    for thr, hold, w, flip in itertools.product((0.5, 0.75, 1.0), (30, 60, 390), ("all", "mid"), (1, -1)):
        G.append(("6_dayext_fade" if flip == 1 else "6_dayext_cont", "dayext", dict(thr=thr, hold=hold, flip=flip), w))
    return G



REGIMES = [(v, t, m) for v, ts in (("ret5", (-0.05, 0.0, 0.05)), ("atr14", (0.08, 0.10, 0.12)), ("gap", (-0.01, 0.0, 0.01)))
           for t in ts for m in ("hi", "lo")]          # 3 variaveis x 3 limiares x 2 lados = 18 filtros
FADE_IDEAS = ("1_vwap", "2_rsi", "2_bb", "3_stretch_fade", "5_extreme_fade", "6_dayext_fade")
P0S = P1S = None


def _period_sets():
    global P0S, P1S
    if P0S is None:
        ds = lab._all_dates()
        P0S = set(str(x) for x in ds if lab.period_of(x) == "P0")
        P1S = set(str(x) for x in ds if lab.period_of(x) == "P1")
    return P0S, P1S


def row_of(tr):
    """Metricas de UMA configuracao usando SOMENTE os blocos 'P0' e 'P1' de lab.evaluate."""
    p0s, p1s = _period_sets()
    ev = lab.evaluate(tr)
    r = {}
    for p, ds in (("P0", p0s), ("P1", p1s)):
        m = ev[p]
        r["n_" + p] = m.get("n_trades", 0)
        r["net_" + p] = m.get("avg_net_bps", np.nan)
        r["gross_" + p] = m.get("avg_gross_bps", np.nan)
        r["t_" + p] = m.get("t_stat", np.nan)
        ss = side_stats(tr, dates=ds)
        for k, v in ss.items():
            r[k + "_" + p] = v
    r["tmin"] = min(r["t_P0"], r["t_P1"]) if (r["t_P0"] == r["t_P0"] and r["t_P1"] == r["t_P1"]) else np.nan
    r["ok_n"] = bool(r["n_P0"] >= 30 and r["n_P1"] >= 30)
    r["pass"] = bool(r["ok_n"] and r["net_P0"] > 0 and r["net_P1"] > 0)
    r["both_sides"] = bool(all(r[k] > 0 for k in ("bpsL_P0", "bpsS_P0", "bpsL_P1", "bpsS_P1")))
    return r


def search_is(days_is):
    rows = []
    for idea, gen, params, w in grid():
        tr = run(days_is, gen, params, w)
        rows.append(dict(idea=idea, gen=gen, window=w, params=str(params), _p=params, **row_of(tr)))
    g = pd.DataFrame(rows)
    # estabilidade: vizinhos = mesma ideia, diferem em exatamente 1 parametro (janela conta como parametro)
    nb_mean, nb_pass, nb_n = [], [], []
    recs = g.to_dict("records")
    for a in recs:
        vs = []
        for b in recs:
            if b["idea"] != a["idea"] or b is a:
                continue
            diff = sum(a["_p"][k] != b["_p"][k] for k in a["_p"]) + (a["window"] != b["window"])
            if diff == 1 and b["ok_n"]:
                vs.append(b)
        nb_n.append(len(vs))
        nb_mean.append(float(np.mean([v["tmin"] for v in vs])) if vs else np.nan)
        nb_pass.append(float(np.mean([v["pass"] for v in vs])) if vs else np.nan)
    g["nb_n"], g["nb_tmin"], g["nb_pass"] = nb_n, nb_mean, nb_pass
    return g


def search_regime(days_is, g):
    """Ideia 7: regra condicional a regime. Base = melhor config (maior tmin, n>=30 nos dois) de cada ideia de fade."""
    rows = []
    for idea in FADE_IDEAS:
        s = g[(g["idea"] == idea) & g["ok_n"]].sort_values("tmin", ascending=False)
        if len(s) == 0:
            continue
        b = s.iloc[0]
        for rg in REGIMES:
            tr = run(days_is, b["gen"], b["_p"], b["window"], regime=rg)
            rows.append(dict(idea=idea, gen=b["gen"], window=b["window"], params=str(b["_p"]), regime=str(rg), **row_of(tr)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- finalistas (congelados apos a busca P0+P1)
# Congelados em 2026-11-11 olhando SOMENTE P0 e P1 (log: logs/meanrev_v2_search.log). Nao reajustar.
# RESULTADO NEGATIVO: de 560 configuracoes, 32 tem liquido>0 nos dois regimes e o melhor min(t_P0,t_P1) e 0,81
# (nada perto de significancia). Os dois abaixo sao os "menos piores": (1) maior min-t da busca inteira;
# (2) maior min-t entre as configuracoes com compra E venda positivas em P0 e em P1.
FINALISTS = [
    dict(name="vwapatr0.10_fade_mid_h60_gapup", gen="vwap", window="mid",
         params=dict(norm="atr", thr=0.10, exitmode="t60", stop_mult=0.0), regime=("gap", 0.0, "hi"),
         description="So em dias de gap de abertura positivo: fade quando (close/VWAP da sessao - 1)/atr14 cruza +-0.10, "
                     "sinal entre 11:00 e 14:30, saida por tempo em 60 min, sem stop.",
         live_trigger="Pre-condicao do dia: open 09:30 > fechamento RTH anterior (gap > 0). atr14 = media de (maxima/minima - 1) do pregao "
                      "regular nos 14 pregoes anteriores. Entre 11:00 e 14:29 NY, sem posicao aberta: x = (close 1m / VWAP acumulada da sessao - 1) / atr14. "
                      "Se x cruza para acima de +0.10 -> VENDER no open do candle seguinte; se cruza para abaixo de -0.10 -> COMPRAR. "
                      "Sair no fechamento do 60o candle apos o sinal. Sem stop/alvo. Uma posicao por vez.",
         notes="MENOS PIOR, NAO SIGNIFICATIVO (t_P0=0.81, t_P1=0.95 apos 560 testes). Sem o filtro de gap a base perde em P1; "
               "o lado da VENDA e negativo em P0 (-11 bps). Filtros vizinhos gap>-1% e gap>+1% tambem positivos nos dois periodos."),
    dict(name="stretch1m_fade_h30_gapdown", gen="stretch", window="all",
         params=dict(k=1, X=2.0, hold=30, flip=1), regime=("gap", -0.01, "lo"),
         description="So em dias de gap de abertura <= -1%: fade de candle de 1m com |retorno| > 2x desvio-padrao dos 60 retornos "
                     "de 1m anteriores, sinal entre 09:35 e 15:50, saida por tempo em 30 min, sem stop.",
         live_trigger="Pre-condicao do dia: open 09:30 / fechamento RTH anterior - 1 <= -1%. Entre 09:35 e 15:49 NY, sem posicao aberta: "
                      "no fechamento de um candle de 1m, r = close/close anterior - 1 e s = desvio-padrao amostral dos 60 retornos de 1m "
                      "imediatamente anteriores (minimo 20, so do dia; na pratica sinais a partir de ~09:51). Se r > +2*s -> VENDER no open do "
                      "candle seguinte; se r < -2*s -> COMPRAR. Sair no fechamento do 30o candle apos o sinal (ou 16:00). Sem stop/alvo.",
         notes="MENOS PIOR, NAO SIGNIFICATIVO (t_P0=1.18, t_P1=0.51 apos 560 testes). Compra e venda positivas em P0 e P1. "
               "Sem filtro a mesma regra da +0.1 bps em P0 (t=0.02); vizinho gap<=0 tambem positivo nos dois periodos."),
]
R1_FINALISTS = [("stretch1m_fade_mid_h30", "3_stretch_fade", "mid", dict(k=1, X=2.0, hold=30, flip=1)),
                ("bb5m_k2.5_fade_to_mean", "2_bb", "all", dict(tf=5, k=2.5, exitmode="mean")),
                ("vwapband2.5_fade_h60", "1_vwap", "all", dict(norm="band", thr=2.5, exitmode="t60", stop_mult=0.0))]
SELECTION_RULE = ("Blocos P0 e P1 de lab.evaluate (ambos in-sample), custo base 3 bps/lado: n>=30 em cada; liquido>0 em P0 E em P1; "
                  "ordenado por min(t_P0,t_P1); desempate por vizinhos de parametros; preferencia por compra e venda positivas. "
                  "Congelado antes de qualquer leitura de P2/OOS/ALL.")


def facts(days_set, label):
    """Fatos descritivos por periodo: autocorrelacao lag-1 de retornos de 5m e 1m, vol de 1m."""
    a5, b5, a1, b1 = [], [], [], []
    for D in days_set:
        c5 = np.concatenate([[D.o[0]], D.c5]); r = c5[1:] / c5[:-1] - 1
        a5.extend(r[:-1]); b5.extend(r[1:])
        a1.extend(D.r1[:-1]); b1.extend(D.r1[1:])
    print("%s: corr lag-1 5m=%.4f (n=%d) | corr lag-1 1m=%.4f (n=%d) | vol 1m=%.1f bps"
          % (label, np.corrcoef(a5, b5)[0, 1], len(a5), np.corrcoef(a1, b1)[0, 1], len(a1), np.std(a1) * 1e4))


def report_finalists(days, n_cfg, n_pass):
    ds = lab._all_dates()
    sets = {p: set(str(x) for x in ds if lab.period_of(x) == p) for p in ("P0", "P1", "P2")}
    print("\n================ FINALISTAS (congelados) ================")
    for f in FINALISTS:
        tr = run(days, f["gen"], f["params"], f["window"], tag=f["name"], regime=f.get("regime"))
        rec = lab.save_result(family=FAMILY, name=f["name"], description=f["description"],
                              params=dict(f["params"], window=f["window"], gen=f["gen"], regime=f.get("regime")), trades=tr,
                              n_configs_tested=n_cfg, selection_rule=SELECTION_RULE,
                              live_trigger=f["live_trigger"],
                              notes=f["notes"] + " | configs testadas=%d, aprovadas no criterio dos dois regimes=%d" % (n_cfg, n_pass))
        print("\n### %s" % f["name"])
        for blk in ("P0", "P1", "P2"):
            print(lab.fmt(rec["metrics_base"][blk], "  %s base  " % blk))
            print(lab.fmt(rec["metrics_stress"][blk], "  %s stress" % blk))
            ss = side_stats(tr, dates=sets[blk])
            print("    %s base por lado: COMPRA n=%d net=%.1fbps t=%.2f | VENDA n=%d net=%.1fbps t=%.2f"
                  % (blk, ss["nL"], ss["bpsL"], ss["tL"], ss["nS"], ss["bpsS"], ss["tS"]))


def main():
    days = build_days()
    days_is = [D for D in days if D.date <= lab.IS_END]          # P0 + P1; P2 nao entra na busca
    print("dias: total=%d IS=%d (P0=%d, P1=%d)" % (len(days), len(days_is),
          sum(lab.period_of(D.date) == "P0" for D in days_is), sum(lab.period_of(D.date) == "P1" for D in days_is)))
    g = search_is(days_is)
    rg = search_regime(days_is, g)
    out = lab.RESULTS / FAMILY
    out.mkdir(parents=True, exist_ok=True)
    g.drop(columns=["_p"]).to_csv(out / "is_grid_v2.csv", index=False)
    rg.to_csv(out / "regime_grid_v2.csv", index=False)
    n_cfg = len(g) + len(rg)
    n_pass = int(g["pass"].sum()) + int(rg["pass"].sum())
    pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 95); pd.set_option("display.max_rows", 500)
    print("\nconfiguracoes testadas: %d (grade %d + regime %d) | aprovadas (n>=30 e liquido>0 em P0 e P1): %d (grade %d + regime %d)"
          % (n_cfg, len(g), len(rg), n_pass, int(g["pass"].sum()), int(rg["pass"].sum())))
    print("aprovadas com compra e venda positivas nos dois periodos: grade %d + regime %d"
          % (int((g["pass"] & g["both_sides"]).sum()), int((rg["pass"] & rg["both_sides"]).sum())))
    cols = ["window", "params", "n_P0", "net_P0", "t_P0", "n_P1", "net_P1", "t_P1", "tmin", "nb_n", "nb_tmin", "nb_pass",
            "bpsL_P0", "bpsS_P0", "bpsL_P1", "bpsS_P1"]
    for idea, s in g.groupby("idea"):
        v = s[s["ok_n"]]
        print("\n=== %s: %d cfgs (%d com n>=30 nos dois) | aprovadas: %d | mediana net P0=%.1f P1=%.1f | mediana gross P0=%.1f P1=%.1f | corr(net_P0,net_P1)=%.2f"
              % (idea, len(s), len(v), int(v["pass"].sum()), v["net_P0"].median(), v["net_P1"].median(),
                 v["gross_P0"].median(), v["gross_P1"].median(), v["net_P0"].corr(v["net_P1"]) if len(v) > 2 else np.nan))
        print(v.sort_values("tmin", ascending=False)[cols].head(5).round(2).to_string(index=False))
    print("\n=== TOP 15 GERAL por min(t_P0,t_P1) entre aprovadas ===")
    print(g[g["pass"]].sort_values("tmin", ascending=False)[["idea"] + cols].head(15).round(2).to_string(index=False))
    print("\n=== 7_regime: aprovadas por tmin ===")
    rc = ["idea", "window", "params", "regime", "n_P0", "net_P0", "t_P0", "n_P1", "net_P1", "t_P1", "tmin",
          "bpsL_P0", "bpsS_P0", "bpsL_P1", "bpsS_P1"]
    print(rg[rg["pass"]].sort_values("tmin", ascending=False)[rc].head(12).round(2).to_string(index=False))
    print("\n=== finalistas da rodada 1 na nova busca (rank por tmin entre as %d com n>=30 nos dois) ===" % int(g["ok_n"].sum()))
    rk = g[g["ok_n"]]["tmin"].rank(ascending=False)
    for nm, idea, w, p in R1_FINALISTS:
        s = g[(g["idea"] == idea) & (g["window"] == w) & (g["params"] == str(p))]
        for ix, r in s.iterrows():
            print("%-26s n_P0=%d net_P0=%.1f t_P0=%.2f | n_P1=%d net_P1=%.1f t_P1=%.2f | tmin=%.2f rank=%s aprovado=%s"
                  % (nm, r["n_P0"], r["net_P0"], r["t_P0"], r["n_P1"], r["net_P1"], r["t_P1"], r["tmin"],
                     ("%d" % rk[ix]) if ix in rk.index else "n<30", r["pass"]))
    print("\n--- fatos por periodo (so IS) ---")
    facts([D for D in days_is if lab.period_of(D.date) == "P0"], "P0")
    facts([D for D in days_is if lab.period_of(D.date) == "P1"], "P1")
    if FINALISTS:
        # ---- a partir daqui: finalistas congelados; P2 apenas reportado ----
        report_finalists(days, n_cfg, n_pass)
    else:
        print("\n(FINALISTS vazio: etapa de busca; P2 nao foi calculado)")


if __name__ == "__main__":
    main()
