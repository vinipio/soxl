# -*- coding: utf-8 -*-
"""take_profit.py -- efeito de um alvo fixo de ganho por trade (em % do preco de entrada) nas estrategias do monitor.

Usa o MESMO motor do monitor (soxl_live.py, na pasta acima) sobre data/soxl_1m_all.csv.
O alvo fixo e somado as regras existentes: se a estrategia ja tem alvo, vale o mais proximo.
Stops e saidas por horario nao mudam. Depois de bater o alvo, a estrategia segue a propria regra
(as de 1 trade por dia encerram o dia; banda de ruido, VWAP e esticada podem entrar de novo).
Custo: 3 bps por lado. Nocional fixo por trade (padrao USD 50.000).

Uso: python take_profit.py [nocional]
"""
import os, sys, csv, math
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), '/home/claude'):
    if os.path.exists(os.path.join(p, 'soxl_live.py')):
        sys.path.insert(0, p); break
import soxl_live as L

N = float(sys.argv[1]) if len(sys.argv) > 1 else 50000.0
COST = 2 * 3.0 / 1e4
STOPS = [None, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
P0_END, P1_END = '2026-05-08', '2026-08-31'
HARD = [None]
_orig = L.Position.__init__

def _init(self, side, entry_i, entry_px, stop=None, target=None, i_last=L.NB - 1):
    x = HARD[0]
    if x is not None:
        tp = entry_px * (1 + side * x)
        target = tp if target is None else (min(target, tp) if side > 0 else max(target, tp))
    _orig(self, side, entry_i, entry_px, stop, target, i_last)
L.Position.__init__ = _init

def load_days(path):
    days = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ny = L.to_ny(datetime.fromtimestamp(int(r['timestamp_unix_ms']) / 1000, tz=timezone.utc))
            m = ny.hour * 60 + ny.minute
            if L.RTH_OPEN_MIN <= m < L.RTH_OPEN_MIN + L.NB:
                days.setdefault(ny.date(), {})[m - L.RTH_OPEN_MIN] = (float(r['open']), float(r['high']), float(r['low']), float(r['close']),
                                                                    float(r['volume']), float(r['vwap']) if r['vwap'] not in ('', 'nan') else float('nan'))
    return {d: v for d, v in days.items() if len(v) == L.NB}

def run(days, x):
    HARD[0] = x
    hist, out, meta = [], {}, {}
    for d in sorted(days):
        day = days[d]
        sess = L.Session(d, L.context_from_days(hist))
        for i in range(L.NB):
            sess.close_bar(i, *day[i])
        for st in sess.strats:
            meta[st.key] = (st.label, st.verdict)
            for t in st.trades:
                out.setdefault(st.key, []).append((str(d), t.pnl() - COST, t.reason))
        hist.append(dict(date=d, o0=day[0][0], marks={i: day[i][3] for i in range(14, L.NB - 1, 15)}, c_last=day[L.NB - 1][3],
                         rng=max(b[1] for b in day.values()) / min(b[2] for b in day.values()) - 1.0))
    return out, meta, [str(d) for d in sorted(days)]

def stats(tr, dates):
    r = [t[1] for t in tr]; n = len(r)
    mean = sum(r) / n; sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1))
    daily = {d: 0.0 for d in dates}
    for d, x, _ in tr: daily[d] += x
    cum = peak = dd = 0.0
    for d in dates:
        cum += daily[d]; peak = max(peak, cum); dd = min(dd, cum - peak)
    per = lambda a, b: sum(x for d, x, _ in tr if a < d <= b) * N
    return dict(n=n, bps=mean * 1e4, t=mean / (sd / math.sqrt(n)), usd=sum(r) * N, pior=min(r) * N, dd=dd * N,
                stopped=sum(1 for t in tr if t[2] == 'target') / n * 100, win=sum(1 for x in r if x > 0) / n * 100,
                p0=per('', P0_END), p1=per(P0_END, P1_END), p2=per(P1_END, '9999'))

if __name__ == '__main__':
    days = load_days(os.path.join(HERE, 'data', 'soxl_1m_all.csv'))
    rows = []
    for x in STOPS:
        out, meta, dates = run(days, x)
        for k, tr in out.items():
            s = stats(tr, dates); s.update(estrategia=k, nome=meta[k][0], grupo=meta[k][1], stop=('sem' if x is None else '%g%%' % (x * 100)))
            rows.append(s)
    keys = list(dict.fromkeys(r['estrategia'] for r in rows))
    print('Nocional USD %.0f por trade, custo 3 bps por lado, %d pregoes' % (N, len(days)))
    for k in keys:
        rs = [r for r in rows if r['estrategia'] == k]
        print('\n%s  [%s]' % (rs[0]['nome'], rs[0]['grupo']))
        print('  %-5s %6s %8s %6s %7s %9s %9s %9s %8s | %8s %8s %8s' % ('alvo', 'trades', 'bps/trade', 't', 'acerto', 'PnL USD', 'pior trade', 'max queda', 'no alvo', 'P0', 'P1', 'P2'))
        for r in rs:
            print('  %-5s %6d %8.1f %6.2f %6.0f%% %9.0f %9.0f %9.0f %7.0f%% | %8.0f %8.0f %8.0f' % (
                r['stop'], r['n'], r['bps'], r['t'], r['win'], r['usd'], r['pior'], r['dd'], r['stopped'], r['p0'], r['p1'], r['p2']))
    os.makedirs(os.path.join(HERE, 'results_v2'), exist_ok=True)
    with open(os.path.join(HERE, 'results_v2', 'take_profit_%d.csv' % N), 'w', newline='', encoding='utf-8') as f:
        cols = ['estrategia', 'nome', 'grupo', 'stop', 'n', 'bps', 't', 'win', 'usd', 'pior', 'dd', 'stopped', 'p0', 'p1', 'p2']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in cols})
