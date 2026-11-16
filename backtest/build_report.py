# -*- coding: utf-8 -*-
"""build_report.py -- consolida results/*/*.json em soxl_estrategias.json e soxl_estrategias.html.
Todos os numeros do relatorio saem dos arquivos de resultado (nada digitado a mao)."""
import sys, json, glob
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(sys.path[0])
import lab, numpy as np, pandas as pd

def verdict(b, s):
    i, o, so = b['IS'], b['OOS'], s['OOS']
    if o.get('n_trades', 0) == 0: return 'descartada'
    if i['avg_net_bps'] > 0 and o['avg_net_bps'] > 0 and so['avg_net_bps'] > 0 and o['t_stat'] >= 2: return 'candidata'
    if i['avg_net_bps'] > 0 and o['avg_net_bps'] > 0 and so['avg_net_bps'] > 0 and o['t_stat'] >= 1: return 'hipotese'
    return 'descartada'

FAM = {'orb': 'Rompimento de abertura', 'momentum': 'Momentum intraday', 'meanrev': 'Reversão à média',
       'gap': 'Gaps e fora do pregão', 'eod': 'Fim de dia e sazonalidade', 'patterns': 'Padrões de candle e volume'}
recs, fam_cfg = [], {}
for f in sorted(glob.glob('results/*/*.json')):
    r = json.load(open(f)); b, s = r['metrics_base'], r['metrics_stress']
    fam_cfg[r['family']] = max(fam_cfg.get(r['family'], 0), r['n_configs_tested'])
    tr = pd.read_csv(f.replace('.json', '_trades.csv'))
    net = tr.side * (tr.exit_px / tr.entry_px - 1) - 2 * lab.COST_BASE / 1e4
    oos = tr.date > str(lab.IS_END)
    def side_bps(mask): return float(net[mask].mean() * 1e4) if mask.any() else None
    recs.append(dict(
        family=r['family'], family_label=FAM[r['family']], name=r['name'], description=r['description'],
        live_trigger=r['live_trigger'], verdict=verdict(b, s),
        IS={k: b['IS'].get(k) for k in ['n_trades', 'win_rate', 'avg_net_bps', 't_stat', 'sharpe_daily_ann', 'max_dd_pct', 'total_net_pct']},
        OOS={k: b['OOS'].get(k) for k in ['n_trades', 'win_rate', 'avg_net_bps', 't_stat', 'sharpe_daily_ann', 'max_dd_pct', 'total_net_pct']},
        ALL={k: b['ALL'].get(k) for k in ['n_trades', 'win_rate', 'avg_net_bps', 't_stat', 'max_dd_pct', 'total_net_pct', 'avg_hold_min', 'avg_win_bps', 'avg_loss_bps']},
        oos_stress_bps=s['OOS'].get('avg_net_bps'),
        long_is=side_bps((tr.side > 0) & ~oos), short_is=side_bps((tr.side < 0) & ~oos),
        long_oos=side_bps((tr.side > 0) & oos), short_oos=side_bps((tr.side < 0) & oos),
        worst_trade_pct=float(net.min() * 100),
        daily=r['daily_pnl_base']))
order = {'candidata': 0, 'hipotese': 1, 'descartada': 2}
recs.sort(key=lambda x: (order[x['verdict']], -(x['OOS']['t_stat'] or -9)))

dt = lab.daily_table()
dates = [str(d) for d in dt.index]
bench = {'oc_is_pct': float(dt.oc[dt.index <= lab.IS_END].sum() * 100), 'oc_oos_pct': float(dt.oc[dt.index > lab.IS_END].sum() * 100),
         'rng_is_pct': float(dt.rng[dt.index <= lab.IS_END].mean() * 100), 'rng_oos_pct': float(dt.rng[dt.index > lab.IS_END].mean() * 100)}
top = [r for r in recs if r['verdict'] != 'descartada']
D = pd.DataFrame({r['name']: pd.Series(r['daily']) for r in top}).reindex(dates).fillna(0.0)
corr = D.corr().round(2).values.tolist() if len(top) > 1 else []
curves = [{'name': r['name'], 'cum': (D[r['name']].cumsum() * 100).round(4).tolist()} for r in top]
for r in recs: del r['daily']
out = dict(meta=dict(symbol='SOXL', first=dates[0], last=dates[-1], n_days=len(dates), is_end=str(lab.IS_END),
                     n_is=int((dt.index <= lab.IS_END).sum()), n_oos=int((dt.index > lab.IS_END).sum()),
                     cost_base=lab.COST_BASE, cost_stress=lab.COST_STRESS,
                     n_configs=int(sum(fam_cfg.values())), n_finalists=len(recs), fam_cfg={FAM[k]: v for k, v in fam_cfg.items()}),
           bench=bench, strategies=recs, dates=dates, curves=curves, corr=corr, corr_names=[r['name'] for r in top])
payload = json.dumps(out, separators=(',', ':'), ensure_ascii=False)
open('soxl_estrategias.json', 'w', encoding='utf-8').write(payload)
html = open('report_template.html', encoding='utf-8').read().replace('__DATA__', payload.replace('</', '<\\/'))
open('soxl_estrategias.html', 'w', encoding='utf-8').write(html)
for r in recs: print('%-11s %-38s IS %6.1f t=%5.2f | OOS %6.1f t=%5.2f | ALL t=%5.2f' % (r['verdict'], r['name'], r['IS']['avg_net_bps'], r['IS']['t_stat'], r['OOS']['avg_net_bps'], r['OOS']['t_stat'], r['ALL']['t_stat']))
print('configs', out['meta']['n_configs'], fam_cfg); print(out['corr_names'], corr); print(bench)
