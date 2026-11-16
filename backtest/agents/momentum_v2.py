import os, sys
os.environ['SOXL_RESULTS'] = 'results_v2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import lab
# -*- coding: utf-8 -*-
"""
Familia "momentum" -- RODADA 2 (1 ano de dados; IS = P0+P1, teste = P2).

Rodar:  python3 agents/momentum_v2.py
  1) refaz TODA a busca usando SOMENTE os dias in-sample (P0 e P1); os dias de P2 nem entram no loop;
     criterio: n>=30 em P0 e em P1, liquido>0 (custo base) em P0 E em P1, ordenar por min(t_P0, t_P1),
     desempate por estabilidade nos vizinhos da grade, preferindo compra e venda positivas;
  2) roda os finalistas congelados (FINALISTS) na amostra inteira e grava results_v2/momentum/*.json.

Execucao: sinal no FECHAMENTO do candle i, entrada/saida por sinal no OPEN do candle i+1;
saida por tempo no CLOSE do ultimo candle do dia (15:59). Tudo day trade.
"""
import math
import numpy as np
import pandas as pd

FAMILY = "momentum"
LOOKBACK = 14          # dias anteriores para perfil de ruido / volume tipico
NB = 390

# ----------------------------------------------------------------------------- dados
def build_days():
    daily = lab.daily_table()
    # retorno fech->fech dos N pregoes ANTERIORES (termina no fechamento de ontem; sem look-ahead)
    RETN = {N: daily["c"].shift(1) / daily["c"].shift(N + 1) - 1.0 for N in (3, 5, 10)}
    groups = lab.day_groups(lab.rth())
    days = []
    for dt, g in groups:
        assert len(g) == NB, (dt, len(g))
        o = g["open"].values.astype(float); h = g["high"].values.astype(float)
        l = g["low"].values.astype(float); c = g["close"].values.astype(float)
        v = g["volume"].values.astype(float)
        vw = np.where(np.isfinite(g["vwap"].values) & (v > 0), g["vwap"].values, c)
        cv = np.cumsum(v); pv = np.cumsum(vw * v)
        sv = np.where(cv > 0, pv / np.where(cv > 0, cv, 1.0), c)       # VWAP da sessao (acumulada)
        row = daily.loc[dt]
        days.append(dict(date=dt, df=g, o=o, h=h, l=l, c=c, v=v, sv=sv, cv=cv,
                         prev_c=float(row["prev_c"]) if pd.notna(row["prev_c"]) else None,
                         atr14=float(row["atr14"]) if pd.notna(row["atr14"]) else None,
                         absmove=np.abs(c / o[0] - 1.0),
                         rng_sofar=np.maximum.accumulate(h) / np.minimum.accumulate(l) - 1.0,
                         gap=float(row["gap"]) if pd.notna(row["gap"]) else None,
                         ret={N: (float(RETN[N].loc[dt]) if pd.notna(RETN[N].loc[dt]) else None) for N in (3, 5, 10)},
                         per=lab.period_of(dt), is_=dt <= lab.IS_END))
    # perfis com dias ANTERIORES apenas
    for k, d in enumerate(days):
        if k >= LOOKBACK:
            prev = days[k - LOOKBACK:k]
            d["sig"] = np.mean([p["absmove"] for p in prev], axis=0)
            d["rvol"] = d["cv"] / np.mean([p["cv"] for p in prev], axis=0)
        else:
            d["sig"] = None; d["rvol"] = None
        d["rngfrac"] = d["rng_sofar"] / d["atr14"] if d["atr14"] else None
        # candles de 5m
        d["H5"] = d["h"].reshape(-1, 5).max(1); d["L5"] = d["l"].reshape(-1, 5).min(1); d["C5"] = d["c"].reshape(-1, 5)[:, -1]
    return days


def tfp(d, side, ei, epx, xi, xpx, reason, tag=""):
    return lab.trade_from_prices(d["date"], side, lab.RTH_START + ei, epx, lab.RTH_START + xi, xpx, reason, tag)


def path_trades(d, changes, tag=""):
    """changes = [(i_sinal, nova_posicao)]; executa no OPEN de i+1; zera no CLOSE do ultimo candle."""
    o, c = d["o"], d["c"]; pos = 0; ei = None; out = []
    for i, new in changes:
        if new == pos:
            continue
        x = i + 1
        if pos != 0:
            out.append(tfp(d, pos, ei, o[ei], x, o[x], "signal", tag))
        pos = new; ei = x
    if pos != 0:
        out.append(tfp(d, pos, ei, o[ei], NB - 1, c[NB - 1], "time", tag))
    return out


def allowed(d, i, flt):
    """Filtro de regime avaliado no fechamento do candle i (so informacao ate i)."""
    if flt is None:
        return True
    kind, x = flt
    if kind == "rng":
        return d["rngfrac"] is not None and d["rngfrac"][i] >= x
    if kind == "rvol":
        return d["rvol"] is not None and d["rvol"][i] >= x
    raise ValueError(kind)

# ----------------------------------------------------------------------------- estrategias
def s_ghlz(d, ctx, W=30, ref="prevc", hold="rest", flt=None, sides=(1, -1)):
    """Sinal = retorno ate o fechamento do candle W-1 (desde fech. anterior ou abertura)."""
    if ref == "prevc" and d["prev_c"] is None:
        return []
    c = d["c"]
    if ref == "h12":                       # retorno 15:00-15:30 (penultima meia hora)
        r = c[359] / c[329] - 1.0; isig = 359
    else:
        base = d["prev_c"] if ref == "prevc" else d["o"][0]
        r = c[W - 1] / base - 1.0; isig = W - 1
    if r == 0:
        return []
    side = 1 if r > 0 else -1
    if side not in sides or not allowed(d, isig, flt):
        return []
    ie = {"rest": isig + 1, "last30": 360, "last60": 330}[hold]
    ie = max(ie, isig + 1)
    res = lab.sim_exit(d["df"], ie, side)
    return [lab.make_trade(d["df"], res, side, "ghlz")]


def s_noise(d, ctx, k=1.0, step=30, base="gap", exitm="vwap", flt=None):
    """Bandas de ruido: UB/LB = base*(1 +/- k*sig[t]), sig = media de |close_t/open-1| dos 14 dias anteriores."""
    if d["sig"] is None or d["prev_c"] is None:
        return []
    c, sv, sig = d["c"], d["sv"], d["sig"]
    O = d["o"][0]
    hb = max(O, d["prev_c"]) if base == "gap" else O
    lb = min(O, d["prev_c"]) if base == "gap" else O
    pos = 0; ch = []
    for i in range(step - 1, NB - 1, step):
        px = c[i]; ub = hb * (1 + k * sig[i]); lo = lb * (1 - k * sig[i])
        if exitm == "vwap":
            up_lvl = max(ub, sv[i]); dn_lvl = min(lo, sv[i])
        else:
            up_lvl = ub; dn_lvl = lo
        new = pos
        if exitm != "eod":
            if pos == 1 and px < up_lvl: new = 0
            if pos == -1 and px > dn_lvl: new = 0
        want = 1 if px > up_lvl else (-1 if px < dn_lvl else 0)
        if want != 0 and want != pos:
            new = want if allowed(d, i, flt) else (0 if exitm != "eod" else pos)
        if new != pos:
            ch.append((i, new)); pos = new
    return path_trades(d, ch, "noise")


def s_vwap(d, ctx, step=5, buf=0.0, start=0, flt=None):
    """Comprado acima da VWAP da sessao*(1+buf), vendido abaixo de VWAP*(1-buf); entre as duas mantem."""
    c, sv = d["c"], d["sv"]; pos = 0; ch = []
    for i in range(step - 1, NB - 1, step):
        if i < start:
            continue
        want = 1 if c[i] > sv[i] * (1 + buf) else (-1 if c[i] < sv[i] * (1 - buf) else pos)
        if want != pos:
            new = want if allowed(d, i, flt) else 0
            if new != pos:
                ch.append((i, new)); pos = new
    return path_trades(d, ch, "vwap")


def ma_states(days, fast, slow, kind):
    """Estado sign(MA rapida - MA lenta) em candles de 5m, serie continua (dias anteriores entram na media)."""
    c5 = pd.Series(np.concatenate([d["C5"] for d in days]))
    if kind == "sma":
        f = c5.rolling(fast).mean(); s = c5.rolling(slow).mean()
    else:
        f = c5.ewm(span=fast, adjust=False).mean(); s = c5.ewm(span=slow, adjust=False).mean()
    st = np.sign((f - s).fillna(0.0).values).reshape(len(days), 78)
    return {d["date"]: st[k] for k, d in enumerate(days)}, (st[:-1, -1] if len(days) > 1 else None)


def s_ma(d, ctx, fast=9, slow=21, ma="ema", mode="cross", flt=None):
    key = (fast, slow, ma)
    if key not in ctx["ma"]:
        ctx["ma"][key] = ma_states(ctx["all_days"], fast, slow, ma)[0]
    st = ctx["ma"][key][d["date"]]
    k = ctx["idx"][d["date"]]
    prev = ctx["ma"][key][ctx["all_days"][k - 1]["date"]][-1] if k > 0 else 0
    pos = 0; ch = []
    for j in range(0, 77):                 # barra 5m j fecha no candle 1m 5j+4; entrada 5j+5 <= 389
        s = st[j]
        if s == 0:
            prev = s; continue
        if mode == "state":
            want = int(s)
        else:
            want = int(s) if s != prev else pos
        prev = s
        if want != pos:
            new = want if allowed(d, 5 * j + 4, flt) else 0
            if new != pos:
                ch.append((5 * j + 4, new)); pos = new
    return path_trades(d, ch, "ma")


def s_donch(d, ctx, N=12, trail=None, flt=None):
    """Rompimento do max/min dos N candles de 5m anteriores (do proprio dia); saida: trailing % ou fechamento."""
    H5, L5, C5 = d["H5"], d["L5"], d["C5"]; out = []; j = N
    while j <= 76:
        hh = H5[j - N:j].max(); ll = L5[j - N:j].min()
        side = 1 if C5[j] > hh else (-1 if C5[j] < ll else 0)
        if side and allowed(d, 5 * j + 4, flt):
            res = lab.sim_exit(d["df"], 5 * j + 5, side, trail_pct=trail)
            out.append(lab.make_trade(d["df"], res, side, "donch"))
            j = max(j + 1, res["exit_i"] // 5)
        else:
            j += 1
    return out


def s_bench(d, ctx, side=-1):
    res = lab.sim_exit(d["df"], 0, side)
    return [lab.make_trade(d["df"], res, side, "bench")]


KINDS = dict(ghlz=s_ghlz, noise=s_noise, vwap=s_vwap, ma=s_ma, donch=s_donch, bench=s_bench)


def reg_ok(d, reg):
    """Regime do DIA (so informacao anterior a abertura / a abertura): devolve None (dia fora) ou lados permitidos."""
    if reg is None:
        return (1, -1)
    var, x = reg
    if var == "atr":                       # atr14 (media da amplitude dos 14 pregoes anteriores) >= x
        return (1, -1) if (d["atr14"] is not None and d["atr14"] >= x) else None
    if var == "gapabs":                    # |gap de abertura| >= x
        return (1, -1) if (d["gap"] is not None and abs(d["gap"]) >= x) else None
    if var == "trend":                     # so trades a favor do retorno dos x pregoes anteriores
        r = d["ret"][x]
        if r is None or r == 0:
            return None
        return (1,) if r > 0 else (-1,)
    raise ValueError(var)


def run(spec, days, ctx):
    spec = dict(spec); fn = KINDS[spec.pop("kind")]; reg = spec.pop("reg", None)
    out = []
    for d in days:
        sides = reg_ok(d, reg)
        if sides is None:
            continue
        tr = fn(d, ctx, **spec)
        out.extend(tr if len(sides) == 2 else [t for t in tr if t["side"] in sides])
    return out


# ----------------------------------------------------------------------------- metricas por periodo
def pstats(tr, per):
    """n, bps liquido/trade, t, e bps de compra / venda no periodo `per` (custo base)."""
    m = lab.evaluate(tr)[per]
    n = m.get("n_trades", 0)
    if n == 0:
        return dict(n=0, net=float("nan"), t=float("nan"), tot=0.0, nL=0, L=float("nan"), nS=0, S=float("nan"))
    nl = int(round(n * m["pct_long"])); ns = n - nl
    return dict(n=n, net=m["avg_net_bps"], t=m["t_stat"], tot=m["total_net_pct"],
                nL=nl, L=m["net_long_pct"] * 100 / nl if nl else float("nan"),
                nS=ns, S=m["net_short_pct"] * 100 / ns if ns else float("nan"))


def row(block, spec, tr):
    r = dict(block=block, spec=repr({k: v for k, v in spec.items() if k != "kind"}))
    for p in ("P0", "P1"):                 # SOMENTE P0 e P1 (in-sample)
        s = pstats(tr, p)
        r.update({p + "_n": s["n"], p + "_bps": round(s["net"], 1), p + "_t": round(s["t"], 2),
                  p + "_L": round(s["L"], 1), p + "_S": round(s["S"], 1)})
    ok_n = r["P0_n"] >= 30 and r["P1_n"] >= 30
    r["mint"] = round(min(r["P0_t"], r["P1_t"]), 2) if ok_n else float("nan")
    r["pass"] = bool(ok_n and r["P0_bps"] > 0 and r["P1_bps"] > 0)
    r["both_sides"] = bool(all((r[k] > 0) for k in ("P0_L", "P0_S", "P1_L", "P1_S")))
    return r


def neighbours(specs):
    """Vizinhos na grade: diferem em exatamente 1 parametro; se numerico, valor adjacente na grade."""
    keys = sorted({k for s in specs for k in s})
    vals = {k: sorted({s.get(k) for s in specs}, key=lambda x: (str(type(x)), x if x is not None else 0)) for k in keys}
    nb = []
    for a in specs:
        cur = []
        for j, b in enumerate(specs):
            diff = [k for k in keys if a.get(k) != b.get(k)]
            if len(diff) != 1:
                continue
            k = diff[0]; va, vb = a.get(k), b.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                vv = [x for x in vals[k] if isinstance(x, (int, float))]
                if abs(vv.index(va) - vv.index(vb)) != 1:
                    continue
            cur.append(j)
        nb.append(cur)
    return nb


def search(block, specs, days_is, ctx, with_nb=True):
    rows = [row(block, sp, run(sp, days_is, ctx)) for sp in specs]
    if with_nb:
        nb = neighbours(specs)
        for r, js in zip(rows, nb):
            mt = [rows[j]["mint"] for j in js if not math.isnan(rows[j]["mint"])]
            r["nb_n"] = len(js)
            r["nb_pass"] = round(float(np.mean([rows[j]["pass"] for j in js])), 2) if js else float("nan")
            r["nb_mint"] = round(float(np.mean(mt)), 2) if mt else float("nan")
    df = pd.DataFrame(rows)
    print("\n===== IS (P0|P1): %s  (%d configuracoes; %d passam) -- top 10 por min(t_P0,t_P1) =====" % (
        block, len(specs), int(df["pass"].sum())))
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 70):
        print(df.sort_values("mint", ascending=False).head(10).drop(columns=["block"]).to_string(index=False))
    return df


def best_spec(df, specs):
    ok = df[df["mint"].notna()]
    return None if len(ok) == 0 else specs[int(ok["mint"].idxmax())]

# ----------------------------------------------------------------------------- fatos
def facts(days, label):
    dd = [d for d in days if d["prev_c"] is not None]
    print("\n===== FATOS (%s, %d dias) =====" % (label, len(dd)))
    edges = list(range(0, 391, 30))
    R = np.array([[d["c"][b - 1] / (d["o"][a] if a == 0 else d["c"][a - 1]) - 1 for a, b in zip(edges[:-1], edges[1:])]
                  for d in dd]) * 1e4
    first_o = R[:, 0]; last = R[:, 12]
    r2h = np.array([d["c"][119] / d["o"][0] - 1 for d in dd]) * 1e4
    rest2h = np.array([d["c"][389] / d["o"][120] - 1 for d in dd]) * 1e4
    rest = np.array([d["c"][389] / d["o"][30] - 1 for d in dd]) * 1e4
    oc = np.array([d["c"][389] / d["o"][0] - 1 for d in dd]) * 1e4
    gap = np.array([d["o"][0] / d["prev_c"] - 1 for d in dd]) * 1e4
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print("  open->close medio=%.1f bps (soma %.1f%%, %%pos=%.1f) | gap medio=%.1f bps" % (oc.mean(), oc.sum() / 100, (oc > 0).mean() * 100, gap.mean()))
    print("  corr(1a meia hora, resto)=%.3f | corr(1as 2h, resto apos 11:30)=%.3f | corr(1a meia hora, ultima meia hora)=%.3f" % (
        cc(first_o, rest), cc(r2h, rest2h), cc(first_o, last)))
    print("  %% dias mesmo sinal 1a meia hora x resto=%.1f%% | 1as 2h x resto=%.1f%% | corr(gap, open->close)=%.3f" % (
        (np.sign(first_o) == np.sign(rest)).mean() * 100, (np.sign(r2h) == np.sign(rest2h)).mean() * 100, cc(gap, oc)))
    up = r2h > 0
    print("  resto do dia apos 11:30: quando 2h>0 (n=%d) media=%.1f bps | quando 2h<0 (n=%d) media=%.1f bps" % (
        up.sum(), rest2h[up].mean(), (~up).sum(), rest2h[~up].mean()))
    # eficiencia de tendencia: |open->close| / amplitude do dia
    eff = np.array([abs(d["c"][389] / d["o"][0] - 1) / (d["h"].max() / d["l"].min() - 1) for d in dd])
    # numero de cruzamentos da VWAP (fechamentos de 5m)
    xs = np.array([int((np.diff(np.sign(d["c"][4::5] - d["sv"][4::5])) != 0).sum()) for d in dd])
    print("  eficiencia |open->close|/amplitude media=%.3f | cruzamentos da VWAP por dia (5m)=%.1f | amplitude media=%.2f%%" % (
        eff.mean(), xs.mean(), np.mean([d["h"].max() / d["l"].min() - 1 for d in dd]) * 100))

# ----------------------------------------------------------------------------- grades
FILTERS = [("rng", 0.3), ("rng", 0.5), ("rng", 0.7), ("rvol", 1.0), ("rvol", 1.2), ("rvol", 1.5)]
# regime condicional (item 3 do BRIEF2): 3 variaveis x 3 limiares, todas conhecidas antes do trade
REGIMES = [("atr", 0.08), ("atr", 0.10), ("atr", 0.12),
           ("gapabs", 0.01), ("gapabs", 0.02), ("gapabs", 0.03),
           ("trend", 3), ("trend", 5), ("trend", 10)]


def grids():
    g = {}
    g["1_ghlz"] = [dict(kind="ghlz", W=W, ref=ref, hold=hold) for W in (15, 30, 60, 120)
                   for ref in ("prevc", "open") for hold in ("rest", "last60", "last30")] + \
                  [dict(kind="ghlz", ref="h12", hold="last30")]
    g["2_noise"] = [dict(kind="noise", k=k, step=st, base=b, exitm=e)
                    for b in ("gap", "open") for e in ("band", "vwap", "eod")
                    for st in (1, 5, 15, 30) for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    g["3a_vwap"] = [dict(kind="vwap", step=st, buf=b, start=s) for s in (0, 29)
                    for st in (5, 15, 30) for b in (0.0, 0.003, 0.006, 0.012)]
    g["3b_ma"] = [dict(kind="ma", fast=f, slow=s, ma=kd, mode=m) for (f, s) in ((5, 20), (9, 21), (10, 30), (20, 50))
                  for kd in ("sma", "ema") for m in ("state", "cross")]
    g["3c_donch"] = [dict(kind="donch", N=N, trail=tr) for N in (6, 12, 18, 24, 36)
                     for tr in (None, 0.02, 0.03, 0.05, 0.08)]
    return g


# Finalistas da rodada 1 (para verificar onde caem na nova busca)
R1 = [("mom_2h_resto_do_dia_rng30", dict(kind="ghlz", W=120, ref="open", hold="rest", flt=("rng", 0.3))),
      ("noise_band_open_k1_15m_eod", dict(kind="noise", k=1.0, step=15, base="open", exitm="eod")),
      ("ma5x20_5m_rvol1", dict(kind="ma", fast=5, slow=20, ma="sma", mode="cross", flt=("rvol", 1.0)))]

# Finalistas CONGELADOS depois de olhar apenas P0 e P1 (ver selection_rule).
FINALISTS = [
    dict(name="mom_2h_resto_gap1pct",
         spec=dict(kind="ghlz", W=120, ref="open", hold="rest", reg=("gapabs", 0.01)),
         description="Momentum intraday (Gao et al.) condicional ao gap: so em dias com |gap de abertura| >= 1% (open 09:30 vs fechamento RTH anterior). "
                     "Sinal = retorno da abertura ate o fechamento do candle 11:29; entra na direcao do sinal no open de 11:30 e sai no fechamento (15:59).",
         selection_rule="P0 e P1 apenas (P2 nao calculado antes de congelar). Criterio: n>=30 e liquido>0 nos dois, ordenar por min(t_P0,t_P1). "
                        "min-t=1,05 (2o lugar, empatado; 1o foi a mesma base com filtro 'trend 10d', min-t=1,12, descartado por instabilidade: "
                        "vizinhos trend 5d e 3d perdem em P1). Aqui os vizinhos |gap|>=2% (0,93) e >=3% (0,81) tambem passam e compra/venda sao positivas nos dois periodos.",
         live_trigger="Na abertura (09:30 NY): gap = open/fechamento RTH anterior - 1; se |gap| < 1%, nao opera no dia. As 11:29:59: r = close(11:29)/open(09:30) - 1. "
                      "Compra no open de 11:30 se r>0, vende se r<0. Sem stop. Zera no fechamento de 15:59.",
         notes="NAO SIGNIFICATIVO: melhor min-t da familia = 1,12 em 309 configuracoes (nenhuma com min-t>=1,5). Passa no criterio formal dos dois regimes, "
               "mas a evidencia e fraca; tratar como hipotese, nao como padrao confirmado."),
    dict(name="noise_band_open_k125_15m_eod",
         spec=dict(kind="noise", k=1.25, step=15, base="open", exitm="eod"),
         description="Bandas de ruido (Zarattini): banda = abertura x (1 +/- 1,25 x media de |close_t/open-1| no mesmo minuto nos 14 pregoes anteriores). "
                     "Checagem a cada 15 min: fecha acima da banda superior -> compra; abaixo da inferior -> vende; inverte se cruzar a banda oposta; zera no fechamento.",
         selection_rule="P0 e P1 apenas. min-t=0,93 (6o no ranking global; 2a ideia distinta da familia). Preferido ao pico k=1,25/30min/base gap (min-t=1,05, so 41/32 trades, "
                        "metade dos vizinhos falha) por estabilidade: 5 de 7 vizinhos passam, e a regiao base=abertura + saida no fechamento e positiva nos dois periodos "
                        "em quase toda a grade; compra e venda positivas em P0 e P1.",
         live_trigger="A cada 15 min (09:44:59, 09:59:59, ... 15:44:59 NY): sigma_t = media, nos 14 pregoes anteriores, de |close daquele minuto/open do dia - 1|. "
                      "Se close > open_hoje x (1+1,25 x sigma_t): fica comprado (open do minuto seguinte); se close < open_hoje x (1-1,25 x sigma_t): fica vendido. "
                      "Mantem ate o sinal oposto ou 15:59.",
         notes="NAO SIGNIFICATIVO: min-t=0,93 em 309 configuracoes. Vizinhos k=1,0 e k=1,5 no mesmo passo tem min-t de apenas 0,17 e 0,24 (positivos, mas ~zero)."),
]


def main():
    days = build_days()
    ctx = dict(ma={}, all_days=days, idx={d["date"]: k for k, d in enumerate(days)})
    days_is = [d for d in days if d["is_"]]
    print("dias: total=%d IS=%d (P0=%d P1=%d) P2=%d" % (len(days), len(days_is), sum(d["per"] == "P0" for d in days),
                                                       sum(d["per"] == "P1" for d in days), len(days) - len(days_is)))
    for p in ("P0", "P1"):
        facts([d for d in days_is if d["per"] == p], p)
    for s in (-1, 1):
        tr = run(dict(kind="bench", side=s), days_is, ctx)
        print("BENCH open->close side=%+d: " % s + " | ".join("%s n=%d %.1f bps t=%.2f" % (p, x["n"], x["net"], x["t"])
              for p, x in ((p, pstats(tr, p)) for p in ("P0", "P1"))))

    G = grids(); bases = {}; dfs = []; seen = set()
    for name, specs in G.items():
        df = search(name, specs, days_is, ctx); dfs.append(df)
        bases[name] = best_spec(df, specs); seen.update(repr(s) for s in specs)
    print("\nbases (maior min-t de cada bloco, n>=30 nos dois periodos) para filtros/regimes:", bases)
    fl = [dict(b, flt=f) for b in bases.values() if b is not None for f in FILTERS]
    dfs.append(search("4_filtros_intradia", fl, days_is, ctx, with_nb=False)); seen.update(repr(s) for s in fl)
    rg = [dict(b, reg=r) for b in bases.values() if b is not None for r in REGIMES]
    dfs.append(search("5_regime_condicional", rg, days_is, ctx, with_nb=False)); seen.update(repr(s) for s in rg)
    r1x = [sp for _, sp in R1 if repr(sp) not in seen]
    if r1x:
        dfs.append(search("6_rodada1_fora_da_grade", r1x, days_is, ctx, with_nb=False)); seen.update(repr(s) for s in r1x)
    allr = pd.concat(dfs, ignore_index=True)
    n_total = len(allr)
    # ranking global: passa no criterio -> min-t -> estabilidade dos vizinhos
    allr["rank"] = allr["mint"].rank(ascending=False, method="min")
    passed = allr[allr["pass"]].sort_values(["mint", "nb_mint"], ascending=False)
    os.makedirs("logs", exist_ok=True)
    allr.sort_values("mint", ascending=False).to_csv("logs/momentum_v2_search.csv", index=False)
    print("\nTOTAL de configuracoes testadas (P0+P1): %d | passam no criterio dos dois regimes: %d | passam com compra e venda >0 nos dois: %d" % (
        n_total, len(passed), int((passed["both_sides"]).sum())))
    print("distribuicao de min(t): max=%.2f  p90=%.2f  mediana=%.2f | configs com min-t>=1: %d, >=1.5: %d, >=2: %d" % (
        allr["mint"].max(), allr["mint"].quantile(0.9), allr["mint"].median(),
        int((allr["mint"] >= 1).sum()), int((allr["mint"] >= 1.5).sum()), int((allr["mint"] >= 2).sum())))
    with pd.option_context("display.width", 280, "display.max_columns", 40, "display.max_colwidth", 80):
        print("\n===== RANKING GLOBAL: os que PASSAM (top 25) =====")
        print(passed.head(25).to_string(index=False))
        print("\n===== Finalistas da rodada 1 na nova busca =====")
        for nm, sp in R1:
            key = repr({k: v for k, v in sp.items() if k != "kind"})
            hit = allr[allr["spec"] == key]
            print(nm); print(hit.to_string(index=False))

    if not FINALISTS:
        print("\nFINALISTS vazio -- busca IS apenas; nada gravado; P2 nao foi calculado.")
        return
    print("\n================ FINALISTAS (amostra inteira; congelados com P0+P1) ================")
    for f in FINALISTS:
        tr = run(f["spec"], days, ctx)
        rec = lab.save_result(family=FAMILY, name=f["name"], description=f["description"], params=f["spec"],
                              trades=tr, n_configs_tested=n_total, selection_rule=f["selection_rule"],
                              live_trigger=f["live_trigger"], notes=f.get("notes", ""))
        print("\n## %s" % f["name"])
        for p in ("P0", "P1", "P2"):
            s = pstats(tr, p); ms = rec["metrics_stress"][p]
            print("  %s: n=%d  %.1f bps/trade  t=%.2f  total=%.1f%% | compra n=%d %.1f bps | venda n=%d %.1f bps | estresse: %.1f bps t=%.2f" % (
                p, s["n"], s["net"], s["t"], s["tot"], s["nL"], s["L"], s["nS"], s["S"],
                ms.get("avg_net_bps", float("nan")), ms.get("t_stat", float("nan"))))
            print(lab.fmt(rec["metrics_base"][p], "      "))
    for s in (-1, 1):
        tr = run(dict(kind="bench", side=s), days, ctx)
        print("BENCH open->close side=%+d: " % s + " | ".join("%s n=%d %.1f bps t=%.2f" % (p, x["n"], x["net"], x["t"])
              for p, x in ((p, pstats(tr, p)) for p in ("P0", "P1", "P2"))))
    facts([d for d in days if d["per"] == "P2"], "P2 (apenas informativo, apos congelar)")


if __name__ == "__main__":
    main()
