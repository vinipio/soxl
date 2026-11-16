import os, sys
os.environ['SOXL_RESULTS'] = 'results_v2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import lab
# -*- coding: utf-8 -*-
"""
Familia "patterns" -- RODADA 2 (249 pregoes; IS = P0 + P1; P2 = teste).

Rodar:  python3 agents/patterns_v2.py            (refaz o scan P0/P1, a selecao e regrava os finalistas)
        python3 agents/patterns_v2.py scan       (so o scan e a selecao; NAO calcula P2)

Mudancas em relacao a patterns.py
  * timeframes 1, 5, 15 e 30 minutos (antes 5 e 15);
  * 5 padroes novos de niveis do dia anterior (maxima, minima e fechamento do pregao regular de ontem):
    rompimento, falso rompimento no mesmo candle (pavio) e rompimento que falha em ate 3 candles;
  * scan vetorizado, feito SOMENTE com trades de P0 e P1 (P2 nem entra nas contas do scan);
  * selecao pelos dois regimes: n>=30, liquido>0 em P0 E em P1, ordenado por min(t_P0, t_P1), exigindo
    horizontes vizinhos tambem positivos nos dois regimes e compra e venda ambas positivas.

Convencoes (iguais as da rodada 1)
  * bloco de tf minutos ancorado em 09:30; sinal no fechamento do bloco (indice 1m i1); entrada no OPEN de i1+1.
  * "sig" = direcao textbook do padrao (bull=+1 / bear=-1). follow = opera a favor de sig; fade = contra.
  * horizonte h (min): saida no CLOSE do candle 1m i_entry+h-1; sinais cujo horizonte nao cabe no dia sao descartados.
  * stop (variante): extremo do candle-sinal. Mesma semantica de lab.sim_exit (conferida em check_stop_engine).
  * custo base 3 bps/lado; t = media/(dp/raiz(n)) por trade, igual a lab._stats.
"""
import math

import numpy as np
import pandas as pd

FAMILY = "patterns"
OUT = lab.RESULTS / FAMILY
OUT.mkdir(parents=True, exist_ok=True)
TFS = [1, 5, 15, 30]
HORIZONS = [5, 15, 30, 60]
CONTEXTS = ["all", "with_vwap", "against_vwap", "with_open", "against_open", "t_open", "t_mid", "t_close"]
PATTERNS_R1 = ["engulf", "hammer_star", "doji_trend", "inside_break", "run3", "run4", "run5", "wide2x",
               "rvol2", "rvol3", "vol_climax", "nr4_break", "nr7_break", "bb_squeeze"]
PATTERNS_PD = ["pdhl_break", "pdhl_reject", "pdhl_fail", "pdc_cross", "pdc_reject"]
PATTERNS = PATTERNS_R1 + PATTERNS_PD
COST = 2.0 * lab.COST_BASE / 1e4
NBAR = 390

DAYS = lab.day_groups(lab.rth())
ND = len(DAYS)
DATES = [dt for dt, _ in DAYS]
PER = np.array([{"P0": 0, "P1": 1, "P2": 2}[lab.period_of(dt)] for dt in DATES])
_R = pd.concat([g for _, g in DAYS], ignore_index=True)
M1 = {c: _R[c].values.astype(float).reshape(ND, NBAR) for c in ["open", "high", "low", "close", "volume", "vwap"]}
# niveis do pregao regular ANTERIOR (dia anterior presente na base)
PDH = np.r_[np.nan, M1["high"].max(axis=1)[:-1]]
PDL = np.r_[np.nan, M1["low"].min(axis=1)[:-1]]
PDC = np.r_[np.nan, M1["close"][:-1, -1]]


# --------------------------------------------------------------------------------------
# 1) Candles e sinais (causais) em todos os dias
# --------------------------------------------------------------------------------------
def build_bars(tf):
    nb = NBAR // tf
    r = lambda a: a.reshape(ND, nb, tf)
    pv = (M1["vwap"] * M1["volume"]).cumsum(axis=1)
    vv = M1["volume"].cumsum(axis=1)
    i1 = (np.arange(nb) + 1) * tf - 1
    B = pd.DataFrame({
        "di": np.repeat(np.arange(ND), nb), "k": np.tile(np.arange(nb), ND),
        "open": r(M1["open"])[:, :, 0].ravel(), "high": r(M1["high"]).max(axis=2).ravel(),
        "low": r(M1["low"]).min(axis=2).ravel(), "close": r(M1["close"])[:, :, -1].ravel(),
        "volume": r(M1["volume"]).sum(axis=2).ravel(),
        "i1": np.tile(i1, ND), "m": np.tile(lab.RTH_START + np.arange(nb) * tf, ND),
        "svwap": (pv[:, i1] / np.where(vv[:, i1] > 0, vv[:, i1], np.nan)).ravel(),
        "day_open": np.repeat(M1["open"][:, 0], nb)})
    B["run_hi"] = B.groupby("di")["high"].cummax()
    B["run_lo"] = B.groupby("di")["low"].cummin()
    B["m_entry"] = B["m"] + tf
    return B


def build_signals(tf):
    """Devolve B e dict pattern -> array sig (+1/-1/0) alinhado a B."""
    B = build_bars(tf)
    nb = NBAR // tf
    o, h, l, c, v, k, di = (B[x].values for x in ["open", "high", "low", "close", "volume", "k", "di"])
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

    ok = k >= 1
    bull = ok & (sh(c, 1) < sh(o, 1)) & (c > o) & (o <= sh(c, 1)) & (c >= sh(o, 1)) & (body > sh(body, 1))
    bear = ok & (sh(c, 1) > sh(o, 1)) & (c < o) & (o >= sh(c, 1)) & (c <= sh(o, 1)) & (body > sh(body, 1))
    sig["engulf"] = bull.astype(int) - bear.astype(int)

    okt = k >= 4
    up_tr = okt & (sh(c, 1) > sh(c, 4))
    dn_tr = okt & (sh(c, 1) < sh(c, 4))
    big_enough = rng >= 0.5 * avg20

    ham = dn_tr & big_enough & (rng > 0) & (lw >= 2 * body) & (lw >= 2 * uw) & (lw > 0)
    star = up_tr & big_enough & (rng > 0) & (uw >= 2 * body) & (uw >= 2 * lw) & (uw > 0)
    sig["hammer_star"] = ham.astype(int) - star.astype(int)

    doji = big_enough & (rng > 0) & (body <= 0.1 * rng)
    sig["doji_trend"] = (doji & dn_tr).astype(int) - (doji & up_tr).astype(int)

    ins = (k >= 2) & (sh(h, 1) <= sh(h, 2)) & (sh(l, 1) >= sh(l, 2))
    sig["inside_break"] = (ins & (c > sh(h, 1))).astype(int) - (ins & (c < sh(l, 1))).astype(int)

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

    sig["wide2x"] = np.where((rng >= 2 * avg20) & (sgn != 0), sgn, 0).astype(int)

    # volume relativo: vs media do MESMO horario nos 14 pregoes ANTERIORES (min 10)
    base = pd.DataFrame(v.reshape(ND, nb)).rolling(14, min_periods=10).mean().shift(1).values.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = v / base
    B["rvol"] = rvol
    sig["rvol2"] = np.where((rvol >= 2) & (sgn != 0), sgn, 0).astype(int)
    sig["rvol3"] = np.where((rvol >= 3) & (sgn != 0), sgn, 0).astype(int)

    late = k >= max(1, 30 // tf)
    nh = late & (rvol >= 2) & (h >= B["run_hi"].values) & ~(l <= B["run_lo"].values)
    nl = late & (rvol >= 2) & (l <= B["run_lo"].values) & ~(h >= B["run_hi"].values)
    sig["vol_climax"] = nh.astype(int) - nl.astype(int)

    for N in (4, 7):
        rmin = S(rng).shift(1).rolling(N).min().values
        nr = (k >= N) & (sh(rng, 1) <= rmin)
        sig["nr%d_break" % N] = (nr & (c > sh(h, 1))).astype(int) - (nr & (c < sh(l, 1))).astype(int)

    mid = S(c).rolling(20).mean()
    sd = S(c).rolling(20).std(ddof=0)
    width = (4 * sd / mid)
    q20 = width.shift(1).rolling(100).quantile(0.2)
    sq = (width <= q20).shift(1).fillna(False).values & (k >= 1)
    up_b = (mid + 2 * sd).shift(1).values
    lo_b = (mid - 2 * sd).shift(1).values
    sig["bb_squeeze"] = (sq & (c > up_b)).astype(int) - (sq & (c < lo_b)).astype(int)

    # ---------------- niveis do dia anterior (novos na rodada 2) ----------------
    ph, pl, pc = PDH[di], PDL[di], PDC[di]
    ref = np.where(k == 0, o, sh(c, 1))          # referencia "antes do candle": close anterior (ou o open, no 1o candle)
    with np.errstate(invalid="ignore"):
        up_h = (ref <= ph) & (c > ph)            # fecha acima da maxima de ontem vindo de baixo
        dn_l = (ref >= pl) & (c < pl)            # fecha abaixo da minima de ontem vindo de cima
        sig["pdhl_break"] = up_h.astype(int) - dn_l.astype(int)
        # falso rompimento no mesmo candle: fura o nivel e fecha de volta (vinha do lado de dentro)
        rej_h = (ref <= ph) & (h > ph) & (c < ph)
        rej_l = (ref >= pl) & (l < pl) & (c > pl)
        sig["pdhl_reject"] = rej_l.astype(int) - rej_h.astype(int)
        # rompimento que falha: fechou alem do nivel em um dos 3 candles anteriores (mesmo dia) e agora fecha de volta
        rec_up = np.zeros(n, bool); rec_dn = np.zeros(n, bool)
        for j in (1, 2, 3):
            rec_up |= (k >= j) & np.nan_to_num(sh(up_h.astype(float), j)).astype(bool)
            rec_dn |= (k >= j) & np.nan_to_num(sh(dn_l.astype(float), j)).astype(bool)
        fail_h = rec_up & (sh(c, 1) > ph) & (c < ph)
        fail_l = rec_dn & (sh(c, 1) < pl) & (c > pl)
        sig["pdhl_fail"] = fail_l.astype(int) - fail_h.astype(int)
        # fechamento de ontem: cruzamento e rejeicao
        sig["pdc_cross"] = ((ref <= pc) & (c > pc)).astype(int) - ((ref >= pc) & (c < pc)).astype(int)
        rej_cu = (ref < pc) & (h > pc) & (c < pc)      # testa por baixo e falha -> bear
        rej_cd = (ref > pc) & (l < pc) & (c > pc)      # testa por cima e segura -> bull
        sig["pdc_reject"] = rej_cd.astype(int) - rej_cu.astype(int)

    for p in sig:
        sig[p] = np.nan_to_num(sig[p]).astype(int)
    return B, sig


def ctx_masks(B, s):
    c, em = B["close"].values, B["m_entry"].values
    vw, do = B["svwap"].values, B["day_open"].values
    return {"all": np.ones(len(B), bool),
            "with_vwap": (c > vw) if s > 0 else (c < vw),
            "against_vwap": (c < vw) if s > 0 else (c > vw),
            "with_open": (c > do) if s > 0 else (c < do),
            "against_open": (c < do) if s > 0 else (c > do),
            "t_open": em < 630, "t_mid": (em >= 630) & (em < 870), "t_close": em >= 870}


# --------------------------------------------------------------------------------------
# 2) Retornos vetorizados (identicos a lab.sim_exit)
# --------------------------------------------------------------------------------------
def gross_time(di, ie, hmin):
    """Retorno bruto COMPRADO: open[ie] -> close[ie+h-1]."""
    return M1["close"][di, ie + hmin - 1] / M1["open"][di, ie] - 1.0


def gross_stop(di, ie, hmin, side, stop):
    """Retorno bruto do lado `side` com stop fixo; mesma ordem de eventos de lab.sim_exit
    (+ regra da rodada 1: se o open de entrada ja esta alem do stop, sai no proprio open)."""
    W = ie[:, None] + np.arange(hmin)[None, :]
    D = di[:, None]
    o, hi, lo = M1["open"][D, W], M1["high"][D, W], M1["low"][D, W]
    entry = o[:, 0]
    st = stop[:, None]
    if side > 0:
        gap = o <= st; hit = lo <= st
    else:
        gap = o >= st; hit = hi >= st
    gap[:, 0] = False
    ev = gap | hit
    any_ev = ev.any(axis=1)
    j = ev.argmax(axis=1)
    rows = np.arange(len(ie))
    px_ev = np.where(gap[rows, j], o[rows, j], stop)
    exit_px = np.where(any_ev, px_ev, M1["close"][di, ie + hmin - 1])
    at_entry = (entry <= stop) if side > 0 else (entry >= stop)
    exit_px = np.where(at_entry, entry, exit_px)
    return side * (exit_px / entry - 1.0)


def simulate(g, i_entry, side, i_last, stop):
    if stop is not None:
        e = g["open"].values[i_entry]
        if (side > 0 and e <= stop) or (side < 0 and e >= stop):
            return dict(entry_i=i_entry, entry_px=e, exit_i=i_entry, exit_px=e, reason="stop_at_entry")
    return lab.sim_exit(g, i_entry, side, stop=stop, i_last=i_last)


def check_stop_engine(n_check=400, seed=7):
    """Confere gross_stop contra lab.sim_exit em amostra aleatoria (so dias de P0/P1)."""
    rs = np.random.RandomState(seed)
    is_days = np.where(PER < 2)[0]
    di = rs.choice(is_days, n_check); ie = rs.randint(1, 320, n_check); hh = 30
    worst = 0.0
    for side in (1, -1):
        base = M1["open"][di, ie]
        stop = base * (1 - side * rs.uniform(-0.002, 0.01, n_check))
        vec = gross_stop(di, ie, hh, side, stop)
        for a in range(n_check):
            g = DAYS[di[a]][1]
            res = simulate(g, int(ie[a]), side, int(ie[a]) + hh - 1, float(stop[a]))
            worst = max(worst, abs(side * (res["exit_px"] / res["entry_px"] - 1) - vec[a]))
    return worst


def st(net, d):
    """n, bruto bps, liquido bps, t por trade, t por dia (soma diaria dos trades; dias com trade)."""
    n = len(net)
    if n < 2:
        return n, np.nan, np.nan, np.nan, np.nan
    mu = net.mean(); sd = net.std(ddof=1)
    t = mu / (sd / math.sqrt(n)) if sd > 0 else np.nan
    u, inv = np.unique(d, return_inverse=True)
    ds = np.bincount(inv, weights=net)
    td = ds.mean() / (ds.std(ddof=1) / math.sqrt(len(ds))) if len(ds) > 2 and ds.std(ddof=1) > 0 else np.nan
    return n, (mu + COST) * 1e4, mu * 1e4, t, td


# --------------------------------------------------------------------------------------
# 3) Scan P0/P1
# --------------------------------------------------------------------------------------
def scan():
    rows = []
    facts = []
    for tf in TFS:
        B, sig = build_signals(tf)
        di_all = B["di"].values; ie_all = B["i1"].values + 1
        per_all = PER[di_all]
        lo_all, hi_all = B["low"].values, B["high"].values
        # fato de regime: autocorrelacao do retorno do bloco com o bloco seguinte (mesmo dia), por periodo
        r_b = B["close"].values / B["open"].values - 1
        nxt = np.r_[r_b[1:], np.nan]; same = np.r_[di_all[1:] == di_all[:-1], False]
        for P in (0, 1):
            mk = same & (per_all == P)
            facts.append(dict(tf=tf, period="P%d" % P, autocorr_lag1=float(np.corrcoef(r_b[mk], nxt[mk])[0, 1]),
                              mean_abs_bar_bps=float(np.abs(r_b[per_all == P]).mean() * 1e4)))
        for p in PATTERNS:
            s_arr = sig[p]
            store = {}                      # (mode, h, stop) -> {s: (net, per, di, masks)}
            for s in (1, -1):
                idx0 = np.where((s_arr == s) & (per_all < 2))[0]
                cm = ctx_masks(B, s)
                for hmin in HORIZONS:
                    idx = idx0[ie_all[idx0] + hmin - 1 <= NBAR - 1]
                    d_, e_ = di_all[idx], ie_all[idx]
                    masks = {cx: cm[cx][idx] for cx in CONTEXTS}
                    pr = per_all[idx]
                    if len(idx) == 0:
                        z = np.array([])
                        for mode in ("follow", "fade"):
                            for stp in (0, 1):
                                store.setdefault((mode, hmin, stp), {})[s] = (z, pr, d_, masks)
                        continue
                    gl = gross_time(d_, e_, hmin)
                    store.setdefault(("follow", hmin, 0), {})[s] = (s * gl - COST, pr, d_, masks)
                    store.setdefault(("fade", hmin, 0), {})[s] = (-s * gl - COST, pr, d_, masks)
                    st_fo = lo_all[idx] if s > 0 else hi_all[idx]
                    st_fa = hi_all[idx] if s > 0 else lo_all[idx]
                    store.setdefault(("follow", hmin, 1), {})[s] = (gross_stop(d_, e_, hmin, s, st_fo) - COST, pr, d_, masks)
                    store.setdefault(("fade", hmin, 1), {})[s] = (gross_stop(d_, e_, hmin, -s, st_fa) - COST, pr, d_, masks)
            for (mode, hmin, stp), bys in store.items():
                for cx in (CONTEXTS if stp == 0 else ["all"]):
                    parts = {}
                    for s in (1, -1):
                        net, pr, d_, masks = bys[s]
                        mk = masks[cx] if len(net) else np.zeros(0, bool)
                        parts[s] = (net[mk], pr[mk], d_[mk])
                    both = tuple(np.concatenate([parts[1][j], parts[-1][j]]) for j in range(3))
                    for label, (net, pr, d_) in (("bull", parts[1]), ("bear", parts[-1]), ("both", both)):
                        if label == "both":
                            side = "both"
                        else:
                            sg = 1 if label == "bull" else -1
                            side = "long" if (sg if mode == "follow" else -sg) > 0 else "short"
                        row = dict(tf=tf, pattern=p, sig=label, mode=mode, side=side, horizon=hmin, stop=stp, context=cx)
                        for P in (0, 1):
                            n, gb, nb_, t, td = st(net[pr == P], d_[pr == P])
                            row.update({"n_P%d" % P: n, "gross_bps_P%d" % P: gb, "bps_P%d" % P: nb_,
                                        "t_P%d" % P: t, "tday_P%d" % P: td})
                        rows.append(row)
    sc = pd.DataFrame(rows)
    sc["min_t"] = sc[["t_P0", "t_P1"]].min(axis=1)
    sc["same_sign"] = np.sign(sc["gross_bps_P0"]) == np.sign(sc["gross_bps_P1"])
    sc.to_csv(OUT / "pattern_scan_p0p1.csv", index=False)
    pd.DataFrame(facts).to_csv(OUT / "regime_facts_p0p1.csv", index=False)
    return sc, pd.DataFrame(facts)


# --------------------------------------------------------------------------------------
# 4) Selecao pelos dois regimes
# --------------------------------------------------------------------------------------
KEY = ["tf", "pattern", "mode", "horizon", "stop", "context"]


def select(sc):
    b = sc[sc["side"] == "both"].set_index(KEY)
    L = sc[sc["side"] == "long"].set_index(KEY)
    Sh = sc[sc["side"] == "short"].set_index(KEY)
    b = b.join(L[["n_P0", "bps_P0", "t_P0", "n_P1", "bps_P1", "t_P1"]].add_suffix("_L")).join(
        Sh[["n_P0", "bps_P0", "t_P0", "n_P1", "bps_P1", "t_P1"]].add_suffix("_S"))
    b["valid"] = (b["n_P0"] >= 30) & (b["n_P1"] >= 30)
    b["pass2"] = b["valid"] & (b["bps_P0"] > 0) & (b["bps_P1"] > 0)
    b["sides_ok"] = ((b["bps_P0_L"] > 0) & (b["bps_P1_L"] > 0) & (b["bps_P0_S"] > 0) & (b["bps_P1_S"] > 0)
                     & (b[["n_P0_L", "n_P1_L", "n_P0_S", "n_P1_S"]].min(axis=1) >= 15))
    # vizinhos: horizontes adjacentes (mesmo resto) e timeframes adjacentes (mesmo resto)
    pas = b["pass2"].to_dict()
    def nb(ix, pos, grid):
        v = list(ix); a = grid.index(v[pos]); res = []
        for q in (a - 1, a + 1):
            if 0 <= q < len(grid):
                w = list(v); w[pos] = grid[q]
                res.append(bool(pas.get(tuple(w), False)))
        return res
    hn = [nb(ix, 3, HORIZONS) for ix in b.index]
    tn = [nb(ix, 0, TFS) for ix in b.index]
    b["h_nb_ok"] = [all(x) for x in hn]
    b["h_nb_frac"] = [float(np.mean(x)) for x in hn]
    b["tf_nb_frac"] = [float(np.mean(x)) for x in tn]
    b["elig"] = b["pass2"] & b["h_nb_ok"] & b["sides_ok"]
    b = b.sort_values(["min_t", "tf_nb_frac"], ascending=False)
    b.to_csv(OUT / "pattern_select_p0p1.csv")
    return b


def finalist_trades(tf, pattern, mode, hmin, ctx, stop=0):
    """Trades em TODOS os dias, uma posicao por vez (sinal ignorado enquanto ha posicao aberta), via lab.sim_exit."""
    B, sig = build_signals(tf)
    s_arr = sig[pattern]
    cm = {1: ctx_masks(B, 1)[ctx], -1: ctx_masks(B, -1)[ctx]}
    dis, i1s, lows, highs = B["di"].values, B["i1"].values, B["low"].values, B["high"].values
    trades = []
    busy = {}
    for idx in np.where(s_arr != 0)[0]:
        s = int(s_arr[idx])
        if not cm[s][idx]:
            continue
        d = int(dis[idx]); g = DAYS[d][1]
        ie = int(i1s[idx]) + 1
        il = ie + hmin - 1
        if il > len(g) - 1 or ie <= busy.get(d, -1):
            continue
        side = s if mode == "follow" else -s
        stp = None
        if stop:
            stp = float(lows[idx] if side > 0 else highs[idx])
        res = simulate(g, ie, side, il, stp)
        busy[d] = res["exit_i"]
        trades.append(lab.make_trade(g, res, side, tag="%s_%s_%dm" % (pattern, mode, tf)))
    return trades


def is_blocks(trades):
    """So P0 e P1 (in-sample) -- total e por lado. Nao expoe P2/OOS/ALL."""
    out = {}
    for nm, tl in (("both", trades), ("long", [t for t in trades if t["side"] > 0]),
                   ("short", [t for t in trades if t["side"] < 0])):
        m = lab.evaluate(tl)
        out[nm] = {P: (m[P].get("n_trades", 0), round(m[P].get("avg_net_bps", float("nan")), 1),
                       round(m[P].get("t_stat", float("nan")), 2)) for P in ("P0", "P1")}
    return out


# Finalistas CONGELADOS olhando apenas P0 e P1 (ver SELECTION). Preenchido depois do scan.
FINALISTS = [
    dict(name="pdhl_break_follow_1m_mid_60",
         tf=1, pattern="pdhl_break", mode="follow", h=60, ctx="t_mid", stop=0,
         description="Rompimento da maxima/minima do pregao anterior em candle de 1m (fechamento alem do nivel vindo do lado "
                     "de dentro), a favor do rompimento, entradas 10:30-14:29, saida em 60 min.",
         live="Niveis fixos antes da abertura: maxima e minima do pregao regular (09:30-16:00 NY) de ontem. Candles de 1m. "
              "Se um candle FECHA acima da maxima de ontem e o fechamento anterior estava em/abaixo dela -> COMPRAR no open do "
              "minuto seguinte; se FECHA abaixo da minima de ontem e o fechamento anterior estava em/acima dela -> VENDER. "
              "So entradas entre 10:30 e 14:29 NY e sem posicao aberta (um trade por vez). Saida a mercado 60 minutos depois "
              "(close do 60o candle de 1m). Sem stop.",
         notes="NAO SIGNIFICATIVO (menos pior). No scan (trades sobrepostos) P0 t=1,95 / P1 t=2,20; com 1 posicao por vez cai "
               "para t=0,56 / 1,06. Passa no criterio formal (n>=30, liquido>0 em P0 e P1, compra e venda positivas nos dois, "
               "horizontes vizinhos positivos), mas 9 regras com min(t)>=1,5 em 3944 validas e MENOS que o esperado por acaso (~18)."),
    dict(name="run4_fade_5m_open_30",
         tf=5, pattern="run4", mode="fade", h=30, ctx="t_open", stop=0,
         description="Na 1a hora, fade de 4 candles de 5m consecutivos na mesma direcao; saida em 30 min.",
         live="Candles de 5m ancorados em 09:30. Quando o 4o candle consecutivo de mesma cor fecha (sequencia exatamente = 4, "
              "contada dentro do dia) e a entrada cai antes de 10:30 NY (na pratica 09:50-10:29): 4 verdes -> VENDER, "
              "4 vermelhos -> COMPRAR no open do minuto seguinte. Sem posicao aberta. Saida a mercado 30 minutos depois. Sem stop.",
         notes="NAO SIGNIFICATIVO (menos pior). P0 t=0,79 / P1 t=2,32 com 1 posicao por vez. Fragil: o mesmo sinal em 1m e "
               "negativo nos dois regimes e o horizonte de 60 min sem sobreposicao e negativo em P0 (-13 bps)."),
]
SELECTION = ("Scan so com P0 e P1: 5472 regras (19 padroes x follow/fade x 4 timeframes [1,5,15,30m] x 4 horizontes x "
             "[8 contextos sem stop + 1 com stop]; cada regra = compra+venda), 3944 com n>=30 nos dois periodos; mais 6 "
             "checagens de candidatos na versao sem sobreposicao (so P0/P1). Criterio: n>=30 e liquido>0 em P0 E P1 (373 "
             "regras), horizontes vizinhos tambem positivos nos dois (126), compra e venda positivas nos dois (20). Ordenado "
             "por min(t_P0,t_P1). Finalista 1 = 1o elegivel. O 2o elegivel (pdhl_fail fade 1m) e a mesma aposta do 1o "
             "(continuacao do rompimento, mesmas janelas) e foi pulado; finalista 2 = 3o elegivel. Nenhum e significativo "
             "apos testes multiplos: min(t)>=2 em 1 regra contra ~2 esperadas por acaso.")
R1 = [(5, "bb_squeeze", "fade", 60, 0, "t_mid"), (5, "run3", "fade", 30, 0, "t_close"), (5, "run5", "fade", 60, 0, "all")]

if __name__ == "__main__":
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60)
    print("dias: P0=%d P1=%d P2=%d | erro max motor de stop vs lab.sim_exit: %.2e"
          % ((PER == 0).sum(), (PER == 1).sum(), (PER == 2).sum(), check_stop_engine()))
    sc, facts = scan()
    b = select(sc)
    n_cfg = len(b)
    print("regras (compra+venda juntas):", n_cfg, "| linhas no csv:", len(sc), "| validas n>=30 nos 2:", int(b.valid.sum()))
    print("passam nos dois regimes (net>0 P0 e P1):", int(b.pass2.sum()),
          "| + vizinhos de horizonte:", int((b.pass2 & b.h_nb_ok).sum()),
          "| + compra e venda positivas:", int(b.elig.sum()))
    v = b[b.valid]
    for thr in (1.0, 1.5, 2.0, 2.5):
        print("  min(t)>=%.1f: %d | espelho max(t)<=-%.1f: %d | acaso (indep.) ~%.1f"
              % (thr, (v.min_t >= thr).sum(), thr, (v[["t_P0", "t_P1"]].max(axis=1) <= -thr).sum(),
                 len(v) * (0.5 * math.erfc(thr / math.sqrt(2))) ** 2))
    print("corr(bps bruto P0, bps bruto P1) entre regras validas: %.3f | mesmo sinal: %.1f%%"
          % (np.corrcoef(v.gross_bps_P0, v.gross_bps_P1)[0, 1], 100 * v.same_sign.mean()))
    vf = v.reset_index()
    vf = vf[(vf["mode"] == "fade") & (vf["stop"] == 0) & (vf["context"] == "all")]
    print("fade (sem stop, ctx all) bruto medio bps por tf  P0 / P1:")
    print(vf.groupby("tf")[["gross_bps_P0", "gross_bps_P1"]].mean().round(2).to_string())
    print(facts.round(4).to_string())
    cols = ["n_P0", "bps_P0", "t_P0", "tday_P0", "n_P1", "bps_P1", "t_P1", "tday_P1", "min_t", "bps_P0_L", "bps_P1_L",
            "bps_P0_S", "bps_P1_S", "h_nb_frac", "tf_nb_frac", "sides_ok", "elig"]
    print("\nTOP 30 passam nos dois regimes (por min t):")
    print(b[b.pass2][cols].head(30).round(2).to_string())
    print("\nTOP 30 elegiveis (pass2 + vizinhos de horizonte + lados):")
    print(b[b.elig][cols].head(30).round(2).to_string())
    rk = b[b.valid].copy(); rk["rank"] = np.arange(1, len(rk) + 1)
    print("\nFinalistas da rodada 1 na nova busca (rank por min t entre %d validas):" % len(rk))
    print(rk.loc[[x for x in R1 if x in rk.index], cols + ["rank"]].round(2).to_string())
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        sys.exit(0)
    for f in FINALISTS:
        tr = finalist_trades(f["tf"], f["pattern"], f["mode"], f["h"], f["ctx"], f["stop"])
        if len(sys.argv) > 1 and sys.argv[1] == "check":
            print("\n==", f["name"], "(1 posicao por vez) P0/P1:", is_blocks(tr))
            continue
        rec = lab.save_result(
            family=FAMILY, name=f["name"], description=f["description"],
            params={k: f[k] for k in ("tf", "pattern", "mode", "h", "ctx", "stop")},
            trades=tr, n_configs_tested=n_cfg, selection_rule=SELECTION, live_trigger=f["live"], notes=f["notes"])
        print("\n==", f["name"])
        for blk in ("P0", "P1", "P2"):
            print(lab.fmt(rec["metrics_base"][blk], blk + " base  "))
        print(lab.fmt(rec["metrics_stress"]["P2"], "P2 stress"))
        for nm, sd_ in (("compra", 1), ("venda", -1)):
            m = lab.evaluate([t for t in tr if t["side"] == sd_])
            print(" ", nm, {P: (m[P].get("n_trades", 0), round(m[P].get("avg_net_bps", float("nan")), 1),
                                round(m[P].get("t_stat", float("nan")), 2)) for P in ("P0", "P1", "P2")})
