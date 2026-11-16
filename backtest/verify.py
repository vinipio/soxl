# -*- coding: utf-8 -*-
"""verify.py -- reimplementacao independente dos finalistas principais + robustez (pos-hoc)."""
import sys, json, itertools
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(sys.path[0])
import lab, numpy as np, pandas as pd

DAYS = lab.day_groups(lab.rth())
DT = lab.daily_table()

def fade(N, wait, target_mode, stop_mode, cat_stop=None, last_sig=385):
    out = []
    for dt, g in DAYS:
        h, l, c = g.high.values, g.low.values, g.close.values
        hi, lo = h[:N].max(), l[:N].min()
        i = b = None
        for k in range(N, last_sig + 1):
            if c[k] > hi: i, b = k, 1; break
            if c[k] < lo: i, b = k, -1; break
        if i is None: continue
        j = None
        for k in range(i + 1, min(last_sig, i + wait) + 1):
            if lo <= c[k] <= hi: j = k; break
        if j is None: continue
        side = -b
        ext = h[i:j+1].max() if b > 0 else l[i:j+1].min()
        mid = (hi + lo) / 2
        target = mid if target_mode == 'mid' else (lo if b > 0 else hi)
        if side * (target - c[j]) <= 0: continue
        stop = ext if stop_mode == 'ext' else None
        if cat_stop is not None:
            stop = g.open.values[j+1] * (1 - side * cat_stop)
        res = lab.sim_exit(g, j + 1, side, stop=stop, target=target)
        if res is None: continue
        if stop_mode == 'ext' and side * (res['entry_px'] - stop) <= 0: continue
        out.append(lab.make_trade(g, res, side, tag='fade'))
    return out

def mom2h():
    out = []
    for dt, g in DAYS:
        a = DT.loc[dt, 'atr14']
        if not np.isfinite(a): continue
        rng = g.high.values[:120].max() / g.low.values[:120].min() - 1
        if rng < 0.3 * a: continue
        r = g.close.values[119] / g.open.values[0] - 1
        if r == 0: continue
        side = 1 if r > 0 else -1
        out.append(lab.make_trade(g, lab.sim_exit(g, 120, side), side))
    return out

def cmp(name, fam, tr):
    ref = pd.read_csv('results/%s/%s_trades.csv' % (fam, name))
    mine = pd.DataFrame(tr)
    same = len(ref) == len(mine) and np.allclose(ref.entry_px.values, mine.entry_px.values) and np.allclose(ref.exit_px.values, mine.exit_px.values) and (ref.side.values == mine.side.values).all()
    print('%-34s agente n=%d | reimplementado n=%d | identico=%s' % (name, len(ref), len(mine), same))
    return mine

if __name__ == '__main__':
    t1 = cmp('fade_or15_esp30_mid_none', 'orb', fade(15, 30, 'mid', 'none'))
    t2 = cmp('fade_or30_esp10_opp_ext', 'orb', fade(30, 10, 'opp', 'ext'))
    t3 = cmp('mom_2h_resto_do_dia_rng30', 'momentum', mom2h())
    print('\n== fade_or15: detalhe ==')
    t1['net'] = t1.side * (t1.exit_px / t1.entry_px - 1) - 6e-4
    t1['oos'] = t1.date > str(lab.IS_END); t1['mes'] = t1.date.str[:7]
    print(t1.groupby('mes').agg(n=('net','size'), win=('net', lambda x: (x>0).mean()), soma_pct=('net', lambda x: x.sum()*100), pior_pct=('net', lambda x: x.min()*100)).round(2))
    print(t1.groupby(['oos','reason']).agg(n=('net','size'), media_bps=('net', lambda x: x.mean()*1e4), pior_bps=('net', lambda x: x.min()*1e4)).round(0))
    print('hold mediano (min):', (t1.exit_m - t1.entry_m).median(), '| entrada mediana (min do dia):', t1.entry_m.median())
    print('5 piores:', (t1.nsmallest(5, 'net')[['date','side','net']].assign(net=lambda x: (x.net*100).round(2))).values.tolist())
    print('\n== grade completa do fade (pos-hoc, so para robustez): OOS liquido base ==')
    rows = []
    for N, w, tm, sm in itertools.product([5, 15, 30, 60], [5, 10, 30], ['mid', 'opp'], ['none', 'ext']):
        e = lab.evaluate(fade(N, w, tm, sm))
        if e['IS'].get('n_trades', 0) < 20 or e['OOS'].get('n_trades', 0) < 15: continue
        rows.append(dict(N=N, wait=w, alvo=tm, stop=sm, n_is=e['IS']['n_trades'], is_bps=e['IS']['avg_net_bps'], is_t=e['IS']['t_stat'],
                         n_oos=e['OOS']['n_trades'], oos_bps=e['OOS']['avg_net_bps'], oos_t=e['OOS']['t_stat']))
    G = pd.DataFrame(rows); print(G.round(1).to_string(index=False))
    print('positivas IS: %d/%d | positivas OOS: %d/%d | positivas nas duas: %d' % ((G.is_bps>0).sum(), len(G), (G.oos_bps>0).sum(), len(G), ((G.is_bps>0)&(G.oos_bps>0)).sum()))
    G.to_csv('results/orb/fade_grid_robustez.csv', index=False)
    print('\n== fade_or15 com stop de catastrofe ==')
    for cs in [None, 0.02, 0.03, 0.05, 0.08]:
        e = lab.evaluate(fade(15, 30, 'mid', 'none', cat_stop=cs))
        print('stop', cs, '|', lab.fmt(e['IS'], 'IS'), '|', lab.fmt(e['OOS'], 'OOS'))
    print('\n== sensibilidade a custo (fade_or15) ==')
    tr = fade(15, 30, 'mid', 'none')
    for cst in [0, 3, 10, 20, 30]:
        e = lab.evaluate(tr, cst); print('custo %2d bps/lado | IS %.1f bps t=%.2f | OOS %.1f bps t=%.2f' % (cst, e['IS']['avg_net_bps'], e['IS']['t_stat'], e['OOS']['avg_net_bps'], e['OOS']['t_stat']))
