import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(sys.path[0]); import lab
# -*- coding: utf-8 -*-
"""
Familia "momentum" -- momentum / seguimento de tendencia intraday no SOXL (1m).

Rodar a partir de a pasta backtest:  python3 agents/momentum.py
  1) refaz TODA a busca usando SOMENTE os dias in-sample (os dias OOS nem entram no loop de busca);
  2) roda os finalistas congelados (FINALISTS) na amostra inteira e grava results/momentum/*.json.

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
                         is_=dt <= lab.IS_END))
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


def run(spec, days, ctx):
    spec = dict(spec); fn = KINDS[spec.pop("kind")]
    out = []
    for d in days:
        out.extend(fn(d, ctx, **spec))
    return out


def side_bps(m, which):
    n = m.get("n_trades", 0)
    if n == 0:
        return (0, float("nan"), 0.0)
    nl = int(round(n * m["pct_long"])); ns = n - nl
    if which == "L":
        return (nl, m["net_long_pct"] * 100 / nl if nl else float("nan"), m["net_long_pct"])
    return (ns, m["net_short_pct"] * 100 / ns if ns else float("nan"), m["net_short_pct"])


def row(spec, m):
    r = dict(spec)
    if m.get("n_trades", 0) == 0:
        r.update(n=0); return r
    nl, lb, lt = side_bps(m, "L"); ns, sb, stt = side_bps(m, "S")
    r.update(n=m["n_trades"], net=round(m["avg_net_bps"], 1), t=round(m["t_stat"], 2),
             sh=round(m["sharpe_daily_ann"], 2), tot=round(m["total_net_pct"], 1), dd=round(m["max_dd_pct"], 1),
             hmin=round(m["avg_hold_min"]), nL=nl, L_bps=round(lb, 1), L_tot=round(lt, 1),
             nS=ns, S_bps=round(sb, 1), S_tot=round(stt, 1))
    return r


def search(name, specs, days_is, ctx):
    rows = []
    for sp in specs:
        rows.append(row({k: v for k, v in sp.items() if k != "kind"}, lab.evaluate_is(run(sp, days_is, ctx))))
    df = pd.DataFrame(rows)
    print("\n===== IS: %s  (%d configuracoes) =====" % (name, len(specs)))
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_columns", 40):
        print(df.to_string(index=False))
    return df, len(specs)


def best_spec(df, specs):
    ok = df[(df["n"] >= 30)]
    if len(ok) == 0:
        return None
    return specs[int(ok["sh"].idxmax())]

# ----------------------------------------------------------------------------- fatos
def facts(days, label):
    print("\n===== FATOS (%s, %d dias) =====" % (label, len(days)))
    dd = [d for d in days if d["prev_c"] is not None]
    edges = list(range(0, 391, 30))
    R = np.array([[d["c"][b - 1] / (d["o"][a] if a == 0 else d["c"][a - 1]) - 1 for a, b in zip(edges[:-1], edges[1:])]
                  for d in dd]) * 1e4
    for k in range(13):
        x = R[:, k]
        print("  %02d:%02d-%02d:%02d media=%7.1f bps  t=%5.2f  %%pos=%4.1f  |media|=%6.1f" % (
            (570 + edges[k]) // 60, (570 + edges[k]) % 60, (570 + edges[k + 1]) // 60, (570 + edges[k + 1]) % 60,
            x.mean(), x.mean() / (x.std(ddof=1) / math.sqrt(len(x))), (x > 0).mean() * 100, np.abs(x).mean()))
    gap = np.array([d["o"][0] / d["prev_c"] - 1 for d in dd]) * 1e4
    first_pc = np.array([d["c"][29] / d["prev_c"] - 1 for d in dd]) * 1e4
    first_o = R[:, 0]; last = R[:, 12]; h12 = R[:, 11]
    rest = np.array([d["c"][389] / d["o"][30] - 1 for d in dd]) * 1e4
    oc = np.array([d["c"][389] / d["o"][0] - 1 for d in dd]) * 1e4
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print("  gap medio=%.1f bps | open->close medio=%.1f bps (soma %.1f%%)" % (gap.mean(), oc.mean(), oc.sum() / 100))
    print("  corr(r 1a meia hora desde fech.ant, ultima meia hora)=%.3f | desde abertura=%.3f" % (cc(first_pc, last), cc(first_o, last)))
    print("  corr(r 1a meia hora desde fech.ant, resto do dia)=%.3f | desde abertura=%.3f" % (cc(first_pc, rest), cc(first_o, rest)))
    print("  corr(gap, open->close)=%.3f | corr(15:00-15:30, 15:30-16:00)=%.3f" % (cc(gap, oc), cc(h12, last)))
    print("  %% dias com mesmo sinal 1a meia hora(abertura) e resto do dia: %.1f%%" % ((np.sign(first_o) == np.sign(rest)).mean() * 100))
    am = np.mean([d["absmove"] for d in dd], axis=0) * 1e4
    print("  |mov. desde abertura| medio (bps) 10:00=%.0f 11:00=%.0f 12:00=%.0f 14:00=%.0f 15:30=%.0f 16:00=%.0f" % (
        am[29], am[89], am[149], am[269], am[359], am[389]))

# ----------------------------------------------------------------------------- grades
FILTERS = [("rng", 0.3), ("rng", 0.5), ("rng", 0.7), ("rvol", 1.0), ("rvol", 1.2), ("rvol", 1.5)]


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


# Finalistas CONGELADOS depois de olhar apenas o IS (ver justificativa em selection_rule).
FINALISTS = [
    dict(name="mom_2h_resto_do_dia_rng30",
         spec=dict(kind="ghlz", W=120, ref="open", hold="rest", flt=("rng", 0.3)),
         description="Momentum intraday (Gao et al.): sinal = retorno da abertura 09:30 ate o fechamento do candle 11:29; "
                     "se a amplitude max/min do dia ate 11:29 >= 0,3 x atr14, entra na direcao do sinal no open de 11:30 e sai no fechamento (15:59).",
         selection_rule="IS apenas. Melhor bloco GHLZ por Sharpe com n>=30 (W=120, ref=abertura, resto do dia: t=1,04) e o unico com compra e venda "
                        "positivas; filtro de amplitude 0,3 e 0,5 dao resultado quase igual (t=1,46). Vizinhos em W (60) e ref (fech. anterior) NAO confirmam -> pico isolado.",
         live_trigger="As 11:29:59 NY: r = close(11:29)/open(09:30)-1 e amp = max(high)/min(low)-1 desde 09:30. Se amp >= 0,3 x (media da amplitude high/low dos 14 pregoes anteriores): "
                      "compra no open de 11:30 se r>0, vende se r<0. Sem stop. Zera no fechamento de 15:59.",
         notes="Evidencia fraca no IS (t~1,5, 264 configuracoes testadas na familia -> nao significativo apos multiplos testes). Efeito simetrico no IS."),
    dict(name="noise_band_open_k1_15m_eod",
         spec=dict(kind="noise", k=1.0, step=15, base="open", exitm="eod"),
         description="Bandas de ruido (Zarattini): banda = abertura x (1 +/- 1,0 x media de |close_t/open-1| no mesmo minuto nos 14 pregoes anteriores). "
                     "Checagem a cada 15 min: fecha acima da banda superior -> compra; abaixo da inferior -> vende; inverte se cruzar a banda oposta; zera no fechamento.",
         selection_rule="IS apenas. Unica regiao das bandas de ruido consistentemente positiva: base=abertura, saida so no fechamento (k 0,5-1,5, checagens 1-30 min quase todas >0). "
                        "Escolhido o centro da regiao (k=1,0, 15 min; vizinhos Sharpe 1,1-1,4) em vez do pico isolado (k=0,75, 5 min, Sharpe 3,0).",
         live_trigger="A cada 15 min (09:44:59, 09:59:59, ... 15:44:59 NY): sigma_t = media, nos 14 pregoes anteriores, de |close daquele minuto/open do dia - 1|. "
                      "Se close > open_hoje x (1+sigma_t): fica comprado (open do minuto seguinte); se close < open_hoje x (1-sigma_t): fica vendido. Mantem ate o sinal oposto ou 15:59.",
         notes="ATENCAO vies direcional: no IS todo o lucro vem da venda (compra ~ -16 bps/trade); short open->close nos mesmos dias deu Sharpe 2,4. "
               "Provavelmente captura a deriva de queda intraday do IS, nao momentum. Saidas por banda/VWAP (versao original) foram negativas."),
    dict(name="ma5x20_5m_rvol1",
         spec=dict(kind="ma", fast=5, slow=20, ma="sma", mode="cross", flt=("rvol", 1.0)),
         description="Cruzamento de medias simples 5/20 em candles de 5m (serie continua), somente quando o volume acumulado do dia ate o momento >= 1,0 x a media "
                     "dos 14 pregoes anteriores no mesmo horario. Segue o cruzamento (compra/venda), inverte no cruzamento oposto, zera no fechamento.",
         selection_rule="IS apenas. Entre os filtros de regime, o unico caso positivo e simetrico (compra +23, venda +39 bps/trade) com vizinho tambem positivo (rvol>=1,2: +29 bps). "
                        "Sem filtro a regra e ~zero (5,7 bps, t=0,4).",
         live_trigger="No fechamento de cada candle de 5m (09:34:59 ... 15:54:59 NY): SMA5 e SMA20 dos closes de 5m (incluindo candles RTH do dia anterior). Quando o sinal de (SMA5-SMA20) muda "
                      "E volume RTH acumulado hoje >= media do acumulado no mesmo minuto nos 14 pregoes anteriores: entra na direcao do cruzamento no open seguinte. "
                      "Cruzamento oposto com volume relativo <1: apenas zera. Zera as 15:59.",
         notes="Evidencia fraca no IS (t~1,0). Menos pior da ideia 3/4; marcado como nao significativo."),
]


def main():
    days = build_days()
    ctx = dict(ma={}, all_days=days, idx={d["date"]: k for k, d in enumerate(days)})
    days_is = [d for d in days if d["is_"]]
    print("dias: total=%d IS=%d OOS=%d" % (len(days), len(days_is), len(days) - len(days_is)))
    facts(days_is, "IN-SAMPLE")
    for s in (-1, 1):
        print(lab.fmt(lab.evaluate_is(run(dict(kind="bench", side=s), days_is, ctx)), "BENCH IS open->close side=%+d:" % s))
    # benchmark restrito aos dias com perfil de 14 dias (mesmo universo das bandas de ruido)
    d14 = [d for d in days_is if d["sig"] is not None]
    print(lab.fmt(lab.evaluate_is(run(dict(kind="bench", side=-1), d14, ctx)), "BENCH IS short (so dias com perfil 14d, n=%d):" % len(d14)))

    n_total = 0; G = grids(); bases = {}
    for name, specs in G.items():
        df, n = search(name, specs, days_is, ctx); n_total += n
        bases[name] = best_spec(df, specs)
    # ideia 4: filtros de regime sobre a melhor configuracao IS (Sharpe, n>=30) de cada bloco
    fl = []
    for name, b in bases.items():
        if b is None:
            continue
        for f in FILTERS:
            sp = dict(b); sp["flt"] = f; fl.append(sp)
    print("\nbases para filtros:", bases)
    rows = []
    for sp in fl:
        r = row({k: v for k, v in sp.items()}, lab.evaluate_is(run(sp, days_is, ctx))); rows.append(r)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n===== IS: 4_filtros (%d configuracoes) =====" % len(fl))
        print(pd.DataFrame(rows).to_string(index=False))
    n_total += len(fl)
    print("\nTOTAL de configuracoes testadas no IS: %d" % n_total)

    if not FINALISTS:
        print("\nFINALISTS vazio -- busca IS apenas; nada gravado.")
        return
    print("\n================ FINALISTAS (amostra inteira; parametros congelados no IS) ================")
    for f in FINALISTS:
        tr = run(f["spec"], days, ctx)
        rec = lab.save_result(family=FAMILY, name=f["name"], description=f["description"], params=f["spec"],
                              trades=tr, n_configs_tested=n_total, selection_rule=f["selection_rule"],
                              live_trigger=f["live_trigger"], notes=f.get("notes", ""))
        print("\n## %s" % f["name"])
        for blk in ("IS", "OOS"):
            print(lab.fmt(rec["metrics_base"][blk], "  base   %s:" % blk))
            m = rec["metrics_base"][blk]
            nl, lb, lt = side_bps(m, "L"); ns, sb, stt = side_bps(m, "S")
            print("      compra: n=%d %.1f bps/trade total=%.1f%% | venda: n=%d %.1f bps/trade total=%.1f%%" % (nl, lb, lt, ns, sb, stt))
            for sd in (1, -1):
                sub = [t for t in tr if t["side"] == sd]
                if sub:
                    print(lab.fmt(lab.evaluate(sub)[blk], "      so %s:" % ("compra" if sd > 0 else "venda ")))
        print(lab.fmt(rec["metrics_stress"]["IS"], "  stress IS :"))
        print(lab.fmt(rec["metrics_stress"]["OOS"], "  stress OOS:"))
    days_oos = [d for d in days if not d["is_"]]
    for s in (-1, 1):
        print(lab.fmt(lab.evaluate(run(dict(kind="bench", side=s), days, ctx))["OOS"], "BENCH OOS open->close side=%+d:" % s))
    facts(days_oos, "OUT-OF-SAMPLE (apenas informativo, apos congelar)")


if __name__ == "__main__":
    main()
