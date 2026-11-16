# -*- coding: utf-8 -*-
"""pnl_usd.py -- PnL em dolares de cada finalista com nocional fixo por trade (padrao USD 50.000)."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(sys.path[0])
import lab, pandas as pd
N = float(sys.argv[1]) if len(sys.argv) > 1 else 50000.0
rows = []
for f in sorted(glob.glob('results/*/*.json')):
    r = json.load(open(f)); t = pd.read_csv(f.replace('.json', '_trades.csv'))
    g = t.side * (t.exit_px / t.entry_px - 1) * N
    cost = 2 * lab.COST_BASE / 1e4 * N; cost_s = 2 * lab.COST_STRESS / 1e4 * N
    net = g - cost; oos = t.date > str(lab.IS_END)
    daily = net.groupby(t.date).sum().reindex([str(d) for d in sorted(lab.load().date.unique())]).fillna(0.0)
    cum = daily.cumsum()
    rows.append(dict(estrategia=r['name'], trades=len(t), bruto=g.sum(), custo=cost * len(t), liquido=net.sum(),
                     liq_treino=net[~oos].sum(), liq_teste=net[oos].sum(), liq_estresse=(g - cost_s).sum(),
                     pior_trade=net.min(), melhor_trade=net.max(), pior_dia=daily.min(), max_queda=(cum - cum.cummax()).min()))
D = pd.DataFrame(rows).sort_values('liq_teste', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda x: '%.0f' % x)
print('Nocional por trade: USD %.0f | custo base USD %.0f por trade | estresse USD %.0f por trade' % (N, 2*lab.COST_BASE/1e4*N, 2*lab.COST_STRESS/1e4*N))
print(D.to_string(index=False))
D.to_csv('results/pnl_usd_%d.csv' % N, index=False)
