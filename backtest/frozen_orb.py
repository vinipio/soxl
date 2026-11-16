# -*- coding: utf-8 -*-
"""frozen_orb.py -- reavalia os 3 finalistas ORB originais (regras congeladas) em todo o historico."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(sys.path[0])
import lab
sys.path.insert(0, 'agents'); import orb
specs = {'fade_or15_esp30_mid_none': lambda: orb.gen_fade(15, 30, 'mid', 'none'),
         'fade_or30_esp10_opp_ext': lambda: orb.gen_fade(30, 10, 'opp', 'ext'),
         'orb60_gap_favor': lambda: orb.apply_filter(orb.gen_orb(60, 1, 'opp', 'close'), 60, 'gap_favor')}
for name, f in specs.items():
    old = json.load(open('results_v1/orb/%s.json' % name))
    tr = f()
    lab.save_result('orb', name, old['description'], old['params'], tr, old['n_configs_tested'], old['selection_rule'], old['live_trigger'], old['notes'])
    e = lab.evaluate(tr); print(name); [print('  ', lab.fmt(e[p], p)) for p in ('P0', 'P1', 'P2')]
