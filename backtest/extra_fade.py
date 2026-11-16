# -*- coding: utf-8 -*-
"""extra_fade.py -- variacoes do falso rompimento de 15 min: limite de tempo e reentradas no mesmo dia."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(sys.path[0])
import lab, numpy as np, pandas as pd
DAYS = lab.day_groups(lab.rth()); LAST = 387

def fade(N=15, wait=30, tmax=None, max_trades=1, strict=True):
    """strict=True: o 1o trade do dia segue a regra original (so o PRIMEIRO rompimento do dia vale);
    reentradas (max_trades>1) so acontecem depois que o 1o trade existiu e foi encerrado."""
    out = []
    for dt, g in DAYS:
        h, l, c = g.high.values, g.low.values, g.close.values
        hi, lo = h[:N].max(), l[:N].min(); mid = (hi + lo) / 2
        k, n = N, 0
        while k <= LAST and n < max_trades:
            # proximo fechamento fora da faixa
            i = b = None
            for q in range(k, LAST + 1):
                if c[q] > hi: i, b = q, 1; break
                if c[q] < lo: i, b = q, -1; break
            if i is None: break
            j = None
            for q in range(i + 1, min(LAST, i + wait) + 1):
                if lo <= c[q] <= hi: j = q; break
            if j is None:
                if strict and n == 0: break
                # rompimento se manteve: espera voltar para dentro antes de procurar outro
                q = i + wait + 1
                while q <= LAST and not (lo <= c[q] <= hi): q += 1
                k = q + 1; continue
            side = -b
            if side * (mid - c[j]) <= 0:
                if strict and n == 0: break
                k = j + 1; continue
            i_last = None if tmax is None else min(j + 1 + tmax - 1, 389)
            res = lab.sim_exit(g, j + 1, side, target=mid, i_last=i_last)
            if res is None: break
            out.append(lab.make_trade(g, res, side, tag='t%d' % (n + 1))); n += 1
            k = res['exit_i'] + 1
    return out

def line(lbl, tr):
    e = lab.evaluate(tr); a, b = e['IS'], e['OOS']
    t = pd.DataFrame(tr); net = t.side * (t.exit_px / t.entry_px - 1) - 6e-4
    print('%-34s| IS n=%3d %6.1fbps t=%5.2f win=%2.0f%% dd=%6.1f%% | OOS n=%3d %6.1fbps t=%5.2f win=%2.0f%% dd=%5.1f%% | pior %.1f%% | dur.med %3.0f min | soma IS %5.1f%% OOS %5.1f%%' % (
        lbl, a['n_trades'], a['avg_net_bps'], a['t_stat'], a['win_rate']*100, a['max_dd_pct'], b['n_trades'], b['avg_net_bps'], b['t_stat'], b['win_rate']*100, b['max_dd_pct'],
        net.min()*100, (t.exit_m - t.entry_m).median(), a['total_net_pct'], b['total_net_pct']))
    return t.assign(net=net)

print('== duracao dos trades da regra original =='); t0 = line('original (limite 15:59)', fade())
d = t0.exit_m - t0.entry_m
print('quantis da duracao (min): 25%%=%d 50%%=%d 75%%=%d 90%%=%d | saiu no alvo: %d de %d | alvo em ate 30 min: %d, ate 60: %d, ate 120: %d' % (
    d.quantile(.25), d.median(), d.quantile(.75), d.quantile(.9), (t0.reason=='target').sum(), len(t0),
    ((t0.reason=='target')&(d<=30)).sum(), ((t0.reason=='target')&(d<=60)).sum(), ((t0.reason=='target')&(d<=120)).sum()))
print('\n== limite de tempo (1 trade por dia) ==')
for T in [15, 30, 60, 90, 120, 180, 240]: line('sai em ate %d min' % T, fade(tmax=T))
print('\n== reentradas no mesmo dia (limite 15:59) ==')
for K in [2, 3, 5]: tk = line('ate %d trades por dia' % K, fade(max_trades=K))
print('\n== por ordem do trade no dia (ate 5 por dia, limite 15:59) ==')
tk['oos'] = tk.date > str(lab.IS_END)
print(tk.groupby(['tag','oos']).agg(n=('net','size'), bps=('net', lambda x: x.mean()*1e4), win=('net', lambda x: (x>0).mean()*100)).round(1))
print('\n== reentradas + limite de tempo ==')
for K, T in [(3, 60), (3, 120), (5, 60), (5, 120)]: line('ate %d por dia, sai em %d min' % (K, T), fade(tmax=T, max_trades=K))
