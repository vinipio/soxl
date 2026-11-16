# -*- coding: utf-8 -*-
"""seasonality.py -- quando as estrategias do monitor ganham: dia da semana, mes, parte do mes,
tipo de dia, e comportamento em conjunto. Sem stop/alvo adicionais (regras originais).

Usa o motor do monitor (soxl_live.py, na pasta acima) sobre data/soxl_1m_all.csv.
PnL em USD com nocional fixo por trade (padrao 50.000) e custo de 3 bps por lado.
Uso: python seasonality.py [nocional]
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), '/home/claude'):
    if os.path.exists(os.path.join(p, 'soxl_live.py')):
        sys.path.insert(0, p); break
sys.path.insert(0, HERE)
import soxl_live as L
import pandas as pd, numpy as np
import hard_stop as H

N = float(sys.argv[1]) if len(sys.argv) > 1 else 50000.0
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40); pd.set_option('display.float_format', lambda x: '%.0f' % x)
days = H.load_days(os.path.join(HERE, 'data', 'soxl_1m_all.csv'))
out, meta, dates = H.run(days, None)
keys = list(out.keys())
D = pd.DataFrame(0.0, index=dates, columns=keys)
for k, tr in out.items():
    for d, x, _ in tr: D.loc[d, k] += x * N
A = [k for k in keys if meta[k][1] == 'robusta']
D['grupoA'] = D[A].sum(axis=1); D['todas'] = D[keys].sum(axis=1)
idx = pd.to_datetime(D.index)
# caracteristicas do dia
info = pd.DataFrame(index=D.index)
ds = sorted(days)
o = np.array([days[d][0][0] for d in ds]); c = np.array([days[d][L.NB - 1][3] for d in ds])
hi = np.array([max(b[1] for b in days[d].values()) for d in ds]); lo = np.array([min(b[2] for b in days[d].values()) for d in ds])
info['oc'] = (c / o - 1) * 100; info['amp'] = (hi / lo - 1) * 100
info['gap'] = np.r_[np.nan, (o[1:] / c[:-1] - 1) * 100]
info['amp_ontem'] = info['amp'].shift(1); info['ret_ontem'] = pd.Series((c / np.r_[np.nan, c[:-1]] - 1) * 100, index=D.index).shift(1)
info['efic'] = info['oc'].abs() / info['amp']           # 1 = dia de tendencia pura; 0 = vai e volta
info['periodo'] = ['P0' if d <= H.P0_END else ('P1' if d <= H.P1_END else 'P2') for d in D.index]

def tab(group, cols, name):
    g = D.groupby(group)
    t = g[cols].sum()
    t.insert(0, 'dias', g.size())
    print('\n== %s: PnL total em USD ==' % name); print(t.to_string())
    return t

def tstat(x):
    x = np.asarray(x, float); return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else float('nan')

show = A + ['grupoA', 'todas']
names = {k: meta[k][0][:22] for k in keys}
print('Estrategias:', {k: meta[k] for k in keys})
print('Grupo A =', A)

# 1) dia da semana
wd = pd.Series(idx.dayofweek, index=D.index).map({0: '1-seg', 1: '2-ter', 2: '3-qua', 3: '4-qui', 4: '5-sex'})
tab(wd, keys + ['grupoA', 'todas'], 'Dia da semana')
print('\nDia da semana: media diaria USD e t (grupo A / todas), e por periodo (grupo A)')
for w in sorted(wd.unique()):
    m = (wd == w).values
    per = ' '.join('%s %7.0f' % (p, D.loc[m & (info.periodo == p).values, 'grupoA'].sum()) for p in ('P0', 'P1', 'P2'))
    print('  %s n=%d | grupoA media %6.0f t=%5.2f %%dias+ %2.0f%% | todas media %6.0f t=%5.2f | %s | amp media %.1f%% efic %.2f' % (
        w, m.sum(), D.loc[m, 'grupoA'].mean(), tstat(D.loc[m, 'grupoA']), (D.loc[m, 'grupoA'] > 0).mean() * 100,
        D.loc[m, 'todas'].mean(), tstat(D.loc[m, 'todas']), per, info.loc[m, 'amp'].mean(), info.loc[m, 'efic'].mean()))

# 2) mes
mo = pd.Series(idx.strftime('%Y-%m'), index=D.index)
tab(mo, keys + ['grupoA', 'todas'], 'Mes')
# 3) parte do mes
dom = pd.Series(np.where(idx.day <= 10, '1-dias 01-10', np.where(idx.day <= 20, '2-dias 11-20', '3-dias 21-31')), index=D.index)
tab(dom, show, 'Parte do mes')
for w in sorted(dom.unique()):
    m = (dom == w).values
    print('  %s n=%d grupoA media %6.0f t=%5.2f | por periodo: %s' % (w, m.sum(), D.loc[m, 'grupoA'].mean(), tstat(D.loc[m, 'grupoA']),
          ' '.join('%s %7.0f' % (p, D.loc[m & (info.periodo == p).values, 'grupoA'].sum()) for p in ('P0', 'P1', 'P2'))))

# 4) em conjunto
print('\n== Correlacao do PnL diario ==')
C = D[keys].corr().round(2); C.index = [k[:11] for k in C.index]; C.columns = [k[:6] for k in C.columns]
pd.set_option('display.float_format', lambda x: '%.2f' % x); print(C.to_string()); pd.set_option('display.float_format', lambda x: '%.0f' % x)
act = (D[keys] != 0).sum(axis=1); pos = (D[keys] > 0).sum(axis=1)
print('\nDias por numero de estrategias no lucro (das 10): PnL medio de TODAS')
print(D.groupby(pos)['todas'].agg(['size', 'mean', 'sum']).to_string())
print('\nMelhores e piores 10 dias (todas): data, PnL, oc%%, amp%%, gap%%')
for lbl, sub in (('melhores', D['todas'].nlargest(10)), ('piores', D['todas'].nsmallest(10))):
    print(' ', lbl, [(d[5:], int(v), round(info.oc[d], 1), round(info.amp[d], 1), None if pd.isna(info.gap[d]) else round(info.gap[d], 1)) for d, v in sub.items()])
print('Concentracao: os 10 melhores dias somam %.0f de %.0f (%.0f%%); 20 melhores %.0f%%' % (D['todas'].nlargest(10).sum(), D['todas'].sum(), D['todas'].nlargest(10).sum() / D['todas'].sum() * 100, D['todas'].nlargest(20).sum() / D['todas'].sum() * 100))

# 5) tipo de dia
def bucket(col, edges, labels, title, ex_ante):
    b = pd.cut(info[col].abs() if col in ('gap', 'ret_ontem', 'oc') else info[col], edges, labels=labels)
    g = D.groupby(b, observed=True)
    t = g[show + ['fade_or15', 'vwap_band', 'stretch_1m']].mean(); t.insert(0, 'dias', g.size())
    print('\n== %s (%s): PnL MEDIO por dia em USD ==' % (title, 'conhecido ANTES do trade' if ex_ante else 'so conhecido DEPOIS'))
    print(t.to_string())
    for lab in labels:
        m = (b == lab).values
        if m.sum() > 5:
            print('   %s: grupoA t=%5.2f | por periodo media: %s' % (lab, tstat(D.loc[m, 'grupoA']), ' '.join('%s %6.0f (n=%d)' % (p, D.loc[m & (info.periodo == p).values, 'grupoA'].mean() if (m & (info.periodo == p).values).sum() else float('nan'), (m & (info.periodo == p).values).sum()) for p in ('P0', 'P1', 'P2'))))
bucket('gap', [-1, 1, 3, 6, 99], ['|gap|<1%', '1-3%', '3-6%', '>6%'], 'Tamanho do gap de abertura', True)
bucket('amp_ontem', [0, 7, 10, 14, 99], ['amp ontem <7%', '7-10%', '10-14%', '>14%'], 'Amplitude do pregao anterior', True)
bucket('ret_ontem', [-1, 2, 5, 99], ['|ret ontem|<2%', '2-5%', '>5%'], 'Retorno do pregao anterior (fech/fech)', True)
bucket('amp', [0, 7, 10, 14, 99], ['amp <7%', '7-10%', '10-14%', '>14%'], 'Amplitude do proprio dia', False)
bucket('efic', [-1, 0.25, 0.5, 0.75, 2], ['vai-e-volta <0.25', '0.25-0.5', '0.5-0.75', 'tendencia >0.75'], 'Eficiencia do dia = |abertura->fechamento| / amplitude', False)
D.join(info).to_csv(os.path.join(HERE, 'results_v2', 'pnl_diario_%d.csv' % N))
