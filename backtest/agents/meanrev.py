import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0]); import lab
# -*- coding: utf-8 -*-
"""
Familia "meanrev" -- reversao a media intraday no SOXL (1m, pregao regular).

Rodar:  python3 agents/meanrev.py   (a partir de a pasta backtest)
  1) refaz a busca de parametros SOMENTE no in-sample (lab.evaluate_is);
  2) grava a grade IS em results/meanrev/is_grid.csv;
  3) avalia os FINALISTAS congelados (lista FINALISTS abaixo) em IS+OOS e grava via lab.save_result.

Convencoes (iguais para todas as ideias):
  * sinal no FECHAMENTO do candle i, entrada no OPEN do candle i+1 (lab.sim_exit);
  * uma posicao por vez (novo sinal so e aceito se i >= candle de saida do trade anterior);
  * saida por tempo de H minutos = CLOSE do candle i+H; alvo/stop sao niveis fixos conhecidos no sinal;
  * todos os indicadores sao calculados so com candles do proprio dia ate i (atr14 = dias anteriores).
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
    out = []
    for date, g in lab.day_groups(lab.rth()):
        atr = dt.loc[date, "atr14"] if date in dt.index else np.nan
        out.append(Day(date, g, float(atr) if atr == atr else np.nan))
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


GENS = {"vwap": gen_vwap, "rsi": gen_rsi, "bb": gen_bb, "stretch": gen_stretch, "extreme": gen_extreme}


# ----------------------------------------------------------------------------- motor
def run(days, gen, params, window, tag=""):
    w0, w1 = WINDOWS[window]
    f = GENS[gen]
    trades = []
    for D in days:
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
    return G


def search_is(days_is):
    rows = []
    for idea, gen, params, w in grid():
        tr = run(days_is, gen, params, w)
        m = lab.evaluate_is(tr)
        ss = side_stats(tr)
        rows.append(dict(idea=idea, gen=gen, window=w, params=str(params), n=m.get("n_trades", 0),
                         gross=m.get("avg_gross_bps", np.nan), net=m.get("avg_net_bps", np.nan),
                         t=m.get("t_stat", np.nan), sharpe=m.get("sharpe_daily_ann", np.nan),
                         total=m.get("total_net_pct", np.nan), win=m.get("win_rate", np.nan),
                         hold=m.get("avg_hold_min", np.nan), **ss))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- finalistas (congelados apos a busca IS)
# Regra de selecao (so IS, custo base): maior t-stat com n>=30 em cada ideia, exigindo vizinhos de parametros
# tambem positivos e os dois lados (compra/venda) positivos; um finalista por ideia, top-3 por t.
# Congelados ANTES de qualquer leitura do OOS. Nao reajustar.
FINALISTS = [
    dict(name="stretch1m_fade_mid_h30", gen="stretch", window="mid",
         params=dict(k=1, X=2.0, hold=30, flip=1),
         description="Fade de candle de 1m com |retorno| > 2x desvio-padrao dos 60 retornos de 1m anteriores, "
                     "sinal entre 11:00 e 14:30, saida por tempo em 30 min, sem stop.",
         live_trigger="Entre 11:00 e 14:29 NY, sem posicao aberta: no fechamento de um candle de 1m, calcule r = close/close anterior - 1 "
                      "e s = desvio-padrao amostral dos 60 retornos de 1m imediatamente anteriores (minimo 20, so do dia). "
                      "Se r > +2*s -> VENDER no open do candle seguinte; se r < -2*s -> COMPRAR no open do candle seguinte. "
                      "Sair no fechamento do 30o candle apos o sinal (ou 16:00). Sem stop/alvo. Uma posicao por vez.",
         notes="IS: pico relativamente isolado (X=4 e holds de 5/15 min sao piores); so a faixa 11:00-14:30 tem t alto."),
    dict(name="bb5m_k2.5_fade_to_mean", gen="bb", window="all",
         params=dict(tf=5, k=2.5, exitmode="mean"),
         description="Fade de fechamento de 5m fora da Banda de Bollinger(20, 2.5) em candles de 5m do dia; "
                     "alvo = media de 20 no momento do sinal; saida por tempo em 60 min; sem stop.",
         live_trigger="Candles de 5m ancorados em 09:30 (precisa de 20 candles: sinais a partir de ~11:05) ate 15:50 NY, sem posicao aberta: "
                      "calcule SMA20 e desvio-padrao populacional de 20 fechamentos de 5m do dia. Se o fechamento cruza para ACIMA de "
                      "SMA20+2.5*sd (o anterior estava dentro) -> VENDER no open do minuto seguinte; se cruza para ABAIXO de SMA20-2.5*sd -> COMPRAR. "
                      "Alvo fixo = SMA20 do momento do sinal; se nao bater, sair no fechamento do 60o minuto apos o sinal (ou 16:00). Sem stop.",
         notes="IS: vizinhos (k=2.0/3.0, saidas t15/t30, janela mid) todos positivos; poucos trades (54)."),
    dict(name="vwapband2.5_fade_h60", gen="vwap", window="all",
         params=dict(norm="band", thr=2.5, exitmode="t60", stop_mult=0.0),
         description="Fade quando o fechamento de 1m cruza 2.5 desvios (banda de VWAP ponderada por volume) em relacao "
                     "a VWAP acumulada da sessao; saida por tempo em 60 min; sem stop.",
         live_trigger="Das 09:45 as 15:50 NY, sem posicao aberta: z = (close - VWAP da sessao) / desvio da banda de VWAP "
                      "(raiz de [soma(vol*vwap_candle^2)/soma(vol) - VWAP^2], acumulado desde 09:30). Se z cruza para acima de +2.5 -> VENDER "
                      "no open do candle seguinte; se cruza para abaixo de -2.5 -> COMPRAR. Sair no fechamento do 60o candle apos o sinal (ou 16:00). Sem stop.",
         notes="IS: so 36 trades; limiares menores (1.5-2.0) perdem dinheiro (fade de desvio moderado NAO funciona)."),
]
EXTRA_CONFIGS = 9   # quebra por faixa horaria (3 faixas x 3 configs de referencia) -- ideia 4


def band_of(m):
    return "abertura<11:00" if m < 660 else ("meio 11:00-14:30" if m < 870 else "fim>=14:30")


def hour_breakdown(days_set, label):
    """Ideia 4: mesmo sinal na janela 'all', resultado por faixa horaria do sinal (custo base)."""
    refs = [("stretch k1 X2 h30", "stretch", dict(k=1, X=2.0, hold=30, flip=1)),
            ("bb5m k2.0 media", "bb", dict(tf=5, k=2.0, exitmode="mean")),
            ("rsi5m(5) 15/85 h15", "rsi", dict(tf=5, n=5, lo=15, hold=15))]
    print("\n--- faixa horaria (%s), net bps/trade no custo base ---" % label)
    for nm, gen, params in refs:
        tr = run(days_set, gen, params, "all")
        parts = []
        for b in ("abertura<11:00", "meio 11:00-14:30", "fim>=14:30"):
            x = np.array([t["side"] * (t["exit_px"] / t["entry_px"] - 1) - 2 * COST / 1e4 for t in tr if band_of(t["entry_m"] - 1) == b])
            tt = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 else float("nan")
            parts.append("%s: n=%d net=%.1f t=%.2f" % (b, len(x), x.mean() * 1e4 if len(x) else float("nan"), tt))
        print("%-20s | %s" % (nm, " | ".join(parts)))


def facts(days_set, label):
    """Fatos descritivos: autocorrelacao de retornos de 5m e de 1m por faixa horaria; reversao apos candle extremo."""
    rows = {b: [[], []] for b in ("abertura<11:00", "meio 11:00-14:30", "fim>=14:30")}
    r1 = {b: [[], []] for b in rows}
    for D in days_set:
        c5 = np.concatenate([[D.o[0]], D.c5]); r = c5[1:] / c5[:-1] - 1
        for j in range(len(r) - 1):
            b = band_of(int(D.m[D.i5[j]]))
            rows[b][0].append(r[j]); rows[b][1].append(r[j + 1])
        for i in range(D.n - 1):
            b = band_of(int(D.m[i]))
            r1[b][0].append(D.r1[i]); r1[b][1].append(D.r1[i + 1])
    print("\n--- autocorrelacao lag-1 (%s) ---" % label)
    for b in rows:
        print("%-18s corr 5m=%.3f (n=%d) | corr 1m=%.3f (n=%d) | vol 1m=%.1f bps" % (
            b, np.corrcoef(rows[b][0], rows[b][1])[0, 1], len(rows[b][0]),
            np.corrcoef(r1[b][0], r1[b][1])[0, 1], len(r1[b][0]), np.std(r1[b][0]) * 1e4))


def report_finalists(days, n_cfg):
    dates = lab._all_dates()
    is_set = set(str(x) for x in dates if x <= lab.IS_END)
    oos_set = set(str(x) for x in dates if x > lab.IS_END)
    dt = lab.daily_table()
    bench = (-dt["oc"]); bench.index = [str(x) for x in bench.index]      # vendido open->close (bruto)
    print("\n================ FINALISTAS (congelados) ================")
    for f in FINALISTS:
        tr = run(days, f["gen"], f["params"], f["window"], tag=f["name"])
        rec = lab.save_result(family=FAMILY, name=f["name"], description=f["description"],
                              params=dict(f["params"], window=f["window"], gen=f["gen"]), trades=tr,
                              n_configs_tested=n_cfg,
                              selection_rule="So IS, custo base 3 bps/lado: maior t-stat com n>=30 por ideia, vizinhos de parametros "
                                             "positivos e compra/venda ambos positivos; top-3 entre ideias. Congelado antes do OOS.",
                              live_trigger=f["live_trigger"], notes=f["notes"])
        print("\n### %s" % f["name"])
        for blk in ("IS", "OOS"):
            print(lab.fmt(rec["metrics_base"][blk], "  %s base  " % blk))
            print(lab.fmt(rec["metrics_stress"][blk], "  %s stress" % blk))
            ds = is_set if blk == "IS" else oos_set
            for cst, nm in ((COST, "base"), (lab.COST_STRESS, "stress")):
                ss = side_stats(tr, cost=cst, dates=ds)
                print("    %s %s por lado: COMPRA n=%d net=%.1fbps t=%.2f | VENDA n=%d net=%.1fbps t=%.2f"
                      % (blk, nm, ss["nL"], ss["bpsL"], ss["tL"], ss["nS"], ss["bpsS"], ss["tS"]))
            dp = pd.Series(rec["daily_pnl_base"])
            idx = [d for d in dp.index if d in ds]
            print("    %s corr(PnL diario, vendido open->close) = %.2f" % (blk, np.corrcoef(dp[idx], bench[idx])[0, 1]))
    for blk, ds in (("IS", is_set), ("OOS", oos_set)):
        b = bench[[d for d in bench.index if d in ds]]
        print("benchmark vendido open->close %s: soma=%.1f%% sharpe=%.2f" % (blk, b.sum() * 100, b.mean() / b.std(ddof=1) * math.sqrt(252)))


def main():
    days = build_days()
    days_is = [D for D in days if D.date <= lab.IS_END]
    days_oos = [D for D in days if D.date > lab.IS_END]
    print("dias: total=%d IS=%d" % (len(days), len(days_is)))
    g = search_is(days_is)
    (lab.RESULTS / FAMILY).mkdir(parents=True, exist_ok=True)
    g.to_csv(lab.RESULTS / FAMILY / "is_grid.csv", index=False)
    n_cfg = len(g) + EXTRA_CONFIGS
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 500)
    print("\nconfiguracoes testadas (IS): %d (grade %d + %d quebras por horario)" % (n_cfg, len(g), EXTRA_CONFIGS))
    for idea, s in g.groupby("idea"):
        v = s[s["n"] >= 30]
        print("\n=== %s: %d cfgs (%d com n>=30) | net>0: %d | mediana net=%.1f bps | mediana gross=%.1f bps | melhor t=%.2f"
              % (idea, len(s), len(v), int((v["net"] > 0).sum()), v["net"].median(), v["gross"].median(), v["t"].max()))
        cols = ["window", "params", "n", "gross", "net", "t", "sharpe", "nL", "bpsL", "tL", "nS", "bpsS", "tS", "hold"]
        print(v.sort_values("t", ascending=False)[cols].head(8).round(2).to_string(index=False))
    hour_breakdown(days_is, "IS")
    facts(days_is, "IS")
    # ---- a partir daqui: finalistas congelados; OOS apenas reportado ----
    report_finalists(days, n_cfg)
    hour_breakdown(days_oos, "OOS, apenas informativo")
    facts(days_oos, "OOS, apenas informativo")


if __name__ == "__main__":
    main()
