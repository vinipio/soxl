# -*- coding: utf-8 -*-
"""periods.py -- tabela P0/P1/P2 dos finalistas de uma pasta de resultados (regras congeladas)."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(sys.path[0])
import pandas as pd
folder = sys.argv[1] if len(sys.argv) > 1 else 'results_frozen'
N = 50000.0
rows = []
for f in sorted(glob.glob(folder + '/*/*.json')):
    r = json.load(open(f)); b, s = r['metrics_base'], r['metrics_stress']
    row = dict(familia=r['family'], estrategia=r['name'])
    for p in ('P0', 'P1', 'P2'):
        m = b[p]
        row.update({p + '_n': m.get('n_trades', 0), p + '_bps': m.get('avg_net_bps'), p + '_t': m.get('t_stat'), p + '_win': (m.get('win_rate') or 0) * 100,
                    p + '_usd': (m.get('total_net_pct') or 0) / 100 * N, p + '_dd_usd': (m.get('max_dd_pct') or 0) / 100 * N})
    row['P0_bps_estresse'] = s['P0'].get('avg_net_bps')
    row['P0_compra_usd'] = b['P0'].get('net_long_pct', 0) / 100 * N; row['P0_venda_usd'] = b['P0'].get('net_short_pct', 0) / 100 * N
    row['ALL_t'] = b['ALL'].get('t_stat'); row['ALL_usd'] = (b['ALL'].get('total_net_pct') or 0) / 100 * N
    rows.append(row)
D = pd.DataFrame(rows).sort_values('P0_t', ascending=False)
pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40); pd.set_option('display.float_format', lambda x: '%.1f' % x)
print(D[['estrategia', 'P0_n', 'P0_bps', 'P0_t', 'P0_win', 'P0_usd', 'P0_dd_usd', 'P0_bps_estresse', 'P0_compra_usd', 'P0_venda_usd', 'P1_n', 'P1_bps', 'P1_t', 'P2_n', 'P2_bps', 'P2_t', 'ALL_t', 'ALL_usd']].to_string(index=False))
D.to_csv(folder + '/periodos.csv', index=False)
