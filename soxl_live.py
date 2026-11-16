# -*- coding: utf-8 -*-
"""
soxl_live.py -- Monitor em tempo real das estrategias intraday do SOXL
========================================================================

Feed: perp xyz:SOXL da Hyperliquid (candles de 1 minuto, API publica, sem chave).
Referencia de horario: pregao regular de Nova York, 09:30-16:00 (o perp negocia 24h,
mas os padroes foram medidos no pregao do ETF).

ESTRATEGIAS ACOMPANHADAS (mesmas regras do backtest, candle a candle)
  Grupo A -- mesmo sinal nos 3 periodos do estudo de 1 ano (rodada 2; evidencia ainda fraca):
    lunch_trend   se 09:30-11:30 andou >= 2%, segue de 11:30 a 13:00
    mom_2h_gap    com |gap de abertura| >= 1%, segue a direcao de 09:30-11:30 ate 15:59
    noise_125     banda de ruido 1.25x em torno da abertura, checada a cada 15 min; ate 15:59
    last2min      se 15:53-15:58 andou >= 0.03 x amplitude media de 14 dias, segue ate 16:00
  Grupo B -- versoes da rodada 1 das mesmas ideias:
    mom_2h        segue 09:30-11:30 ate 15:59 (filtro de amplitude)
    noise_band    banda de ruido 1.0x
  Grupo C -- dependentes de regime (ganharam de mai a nov/2026, perderam de nov/2025 a mai/2026):
    fade_or15, fade_or30 (falso rompimento da faixa de abertura), vwap_band, stretch_1m

USO
  python soxl_live.py                         # ao vivo: abre o painel em http://localhost:8765
  python soxl_live.py replay 2026-11-10       # repassa um dia recente (ate ~3 dias atras) acelerado
  python soxl_live.py replay 2026-11-10 --speed 30
  python soxl_live.py validate --csv backtest\\data\\soxl_1m_all.csv --trades-dir backtest
                                              # roda o motor sobre o CSV da Massive e compara trade a trade com o backtest
  Opcoes: --port 8765  --coin xyz:SOXL  --no-browser  --poll 4

VERSAO WEB (GitHub Pages)
  python build_web.py                         # regera index.html a partir do painel (PAGE) deste arquivo
  index.html + soxl_web_worker.js carregam ESTE arquivo no navegador (Pyodide) e chamam live_step();
  a unica troca e o transporte HTTP (http_post). Regras, estados e painel sao os mesmos.

SAIDAS
  Painel local (grafico de 1m com niveis, estado de cada estrategia, alertas com som).
  soxl_live_log.csv : todo sinal, entrada e saida ao vivo, para comparar com o backtest depois.

REGRAS DE EXECUCAO (identicas ao backtest)
  Sinal no FECHAMENTO do candle de 1m; entrada na ABERTURA do candle seguinte.
  Stop/alvo conferidos dentro do candle; saida por tempo no fechamento do candle indicado.

DIFERENCAS EM RELACAO AO BACKTEST (dados do ETF)
  - Precos e volumes sao do perp, nao do ETF. A VWAP da sessao usa (max+min+fech)/3 de cada
    candle, porque a Hyperliquid nao informa o preco medio do candle.
  - O contexto historico (banda de ruido e amplitude media de 14 dias) vem dos candles de
    15 minutos do perp, em dias uteis. Feriados e pregoes curtos da NYSE NAO sao tratados:
    nesses dias ignore os sinais.
  - Ao vivo, alvo/stop sao registrados no instante em que o preco toca (ordem real);
    no backtest, se os dois caem no mesmo candle vale o stop.

INDICADOR "DIA ATIVO / DIA CALMO" (so informativo; nao altera nenhum sinal)
  Mostra a amplitude (maxima/minima - 1) do pregao regular anterior. No estudo de 1 ano, as
  estrategias de tendencia so ganharam depois de um pregao com amplitude >= 10%.

Dependencias: somente a biblioteca padrao do Python 3.8+.
"""
import argparse
import csv
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone, date as date_cls
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

API_URL = "https://api.hyperliquid.xyz/info"
RTH_OPEN_MIN = 570            # 09:30 em minutos do dia (NY)
NB = 390                      # candles de 1m no pregao regular
LAST_SIGNAL_I = 387           # ultimo candle que pode gerar sinal (15:57)
LOOKBACK = 14                 # pregoes anteriores para banda de ruido e amplitude media
ACTIVE_RNG = 0.10             # indicador "dia ativo": amplitude do pregao anterior >= 10% (so informativo)
ATR_MIN = 5                   # minimo de pregoes para a amplitude media

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
LOG_PATH = BASE_DIR / "soxl_live_log.csv"
LIB_VERSION = "4.2.3"
LIB_FILE = BASE_DIR / ("lightweight-charts-%s.standalone.js" % LIB_VERSION)
LIB_URLS = [
    "https://unpkg.com/lightweight-charts@%s/dist/lightweight-charts.standalone.production.js" % LIB_VERSION,
    "https://cdn.jsdelivr.net/npm/lightweight-charts@%s/dist/lightweight-charts.standalone.production.js" % LIB_VERSION,
]


# ======================================================================================
# Horario de Nova York (regra de horario de verao dos EUA, sem depender de tzdata)
# ======================================================================================
def _nth_sunday(year, month, n):
    d = date_cls(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(days=7 * (n - 1))


def ny_offset_hours(utc_dt):
    """-4 no horario de verao dos EUA (2o domingo de marco 07:00 UTC ate 1o domingo de novembro 06:00 UTC), senao -5."""
    y = utc_dt.year
    start = datetime(y, 3, _nth_sunday(y, 3, 2).day, 7, 0, tzinfo=timezone.utc)
    end = datetime(y, 11, _nth_sunday(y, 11, 1).day, 6, 0, tzinfo=timezone.utc)
    return -4 if start <= utc_dt < end else -5


def to_ny(utc_dt):
    return (utc_dt + timedelta(hours=ny_offset_hours(utc_dt))).replace(tzinfo=None)


def rth_open_utc_ms(ny_date):
    """Epoch ms (UTC) das 09:30 de Nova York na data dada."""
    for off in (-4, -5):
        cand = datetime(ny_date.year, ny_date.month, ny_date.day, 9, 30, tzinfo=timezone.utc) - timedelta(hours=off)
        if ny_offset_hours(cand) == off:
            return int(cand.timestamp() * 1000)
    raise RuntimeError("data invalida")


def hhmm(i):
    """Indice do candle (0 = 09:30) -> 'HH:MM' de NY."""
    m = RTH_OPEN_MIN + i
    return "%02d:%02d" % (m // 60, m % 60)


# ======================================================================================
# Motor: posicao + estrategias (usado igual no validate, no replay e ao vivo)
# ======================================================================================
class Position:
    """Mesma logica de saida do backtest (lab.sim_exit)."""

    def __init__(self, side, entry_i, entry_px, stop=None, target=None, i_last=NB - 1):
        self.side, self.entry_i, self.entry_px = side, entry_i, entry_px
        self.stop, self.target, self.i_last = stop, target, min(i_last, NB - 1)
        self.exit_i = self.exit_px = self.reason = None

    @property
    def open(self):
        return self.exit_i is None

    def _close(self, i, px, reason):
        self.exit_i, self.exit_px, self.reason = i, px, reason

    def on_open(self, i, o):
        """Abertura do candle i (i > entrada): abriu alem do stop -> sai na abertura."""
        if not self.open or i <= self.entry_i or self.stop is None:
            return
        if (self.side > 0 and o <= self.stop) or (self.side < 0 and o >= self.stop):
            self._close(i, o, "stop_gap")

    def check(self, i, o, h, l):
        """Stop (prioridade) e alvo dentro do candle i, com a maxima/minima vistas ate agora."""
        if not self.open:
            return
        s = self.side
        if self.stop is not None and ((s > 0 and l <= self.stop) or (s < 0 and h >= self.stop)):
            self._close(i, self.stop, "stop")
            return
        if self.target is not None and ((s > 0 and h >= self.target) or (s < 0 and l <= self.target)):
            px = self.target
            if i > self.entry_i and ((s > 0 and o >= self.target) or (s < 0 and o <= self.target)):
                px = o
            self._close(i, px, "target")

    def on_close(self, i, o, h, l, c):
        self.check(i, o, h, l)
        if self.open and i >= self.i_last:
            self._close(i, c, "time")

    def pnl(self, px=None):
        px = self.exit_px if px is None else px
        return self.side * (px / self.entry_px - 1.0)


class Strategy:
    key = label = verdict = rule = ""

    def __init__(self, ctx):
        self.ctx = ctx
        self.pos = None
        self.pending = None        # dict(side, stop, target, i_last, check_stop) para a abertura seguinte
        self.trades = []           # posicoes encerradas hoje
        self.events = []           # (i, tipo, texto, side, px)
        self.stage = "aguardando"
        self.status = ""
        self.levels = []           # [(nome, preco)]
        self.done = False

    # ---- utilidades
    def ev(self, i, kind, text, side=0, px=None):
        self.events.append(dict(i=i, kind=kind, text=text, side=side, px=px))

    def enter(self, i_sig, side, stop=None, target=None, i_last=NB - 1, check_stop=False):
        self.pending = dict(side=side, stop=stop, target=target, i_last=i_last, check_stop=check_stop)
        self.stage = "sinal"
        self.ev(i_sig, "sinal", "%s na abertura de %s" % ("COMPRAR" if side > 0 else "VENDER", hhmm(i_sig + 1)), side)

    def _settle(self):
        if self.pos is not None and not self.pos.open:
            p = self.pos
            txt = {"target": "alvo atingido", "stop": "stop atingido", "stop_gap": "stop (abriu alem do nivel)",
                   "time": "saida por horario", "signal": "saida por sinal contrario"}[p.reason]
            self.ev(p.exit_i, "saida", "%s: %+.2f%% (bruto)" % (txt, p.pnl() * 100), -p.side, p.exit_px)
            self.trades.append(p)
            self.pos = None
            self.after_exit(p)

    def after_exit(self, p):
        self.stage = "encerrada"

    # ---- ciclo por candle
    def open_bar(self, i, o):
        if self.pos is not None:
            self.pos.on_open(i, o)
            self._settle()
        if self.pending is not None:
            pe, self.pending = self.pending, None
            if self.pos is not None:                       # inversao: fecha a anterior na mesma abertura
                self.pos._close(i, o, "signal")
                self._settle()
            if pe["check_stop"] and pe["stop"] is not None and pe["side"] * (o - pe["stop"]) <= 0:
                self.stage, self.done = "sem trade", True
                self.status = "Abriu alem do stop; entrada cancelada."
                self.ev(i, "info", "entrada cancelada: abertura alem do stop")
            else:
                self.pos = Position(pe["side"], i, o, pe["stop"], pe["target"], pe["i_last"])
                self.stage = "posicionada"
                self.ev(i, "entrada", "%s a %.3f" % ("comprado" if pe["side"] > 0 else "vendido", o), pe["side"], o)

    def peek(self, i, o, h, l):
        """Candle i ainda em formacao: so confere toque em stop/alvo."""
        if self.pos is not None:
            self.pos.check(i, o, h, l)
            self._settle()

    def close_bar(self, i, B):
        if self.pos is not None:
            self.pos.on_close(i, B.o[i], B.h[i], B.l[i], B.c[i])
            self._settle()
        self.on_bar(i, B)

    def on_bar(self, i, B):
        raise NotImplementedError

    def describe(self, last):
        if self.pos is not None:
            p = self.pos
            s = "%s desde %s a %.3f" % ("COMPRADO" if p.side > 0 else "VENDIDO", hhmm(p.entry_i), p.entry_px)
            if p.target is not None:
                s += " | alvo %.3f" % p.target
            if p.stop is not None:
                s += " | stop %.3f" % p.stop
            s += " | saida limite %s" % hhmm(p.i_last)
            if last:
                s += " | agora %+.2f%%" % (p.pnl(last) * 100)
            return s
        return self.status


class Bars:
    """Arrays do dia (indice 0 = 09:30). vw = preco medio do candle (vwap real ou tipico)."""

    def __init__(self):
        self.o, self.h, self.l, self.c, self.v, self.vw = [], [], [], [], [], []

    def add(self, o, h, l, c, v, vw=None):
        self.o.append(o); self.h.append(h); self.l.append(l); self.c.append(c); self.v.append(v)
        self.vw.append(vw if (vw is not None and vw == vw and v > 0) else (c if vw is not None else (h + l + c) / 3.0))


class FadeOR(Strategy):
    """Falso rompimento da faixa de abertura."""

    def __init__(self, ctx, n, wait, target_mode, stop_mode, key, label, verdict):
        super().__init__(ctx)
        self.n, self.wait, self.target_mode, self.stop_mode = n, wait, target_mode, stop_mode
        self.key, self.label, self.verdict = key, label, verdict
        self.rule = ("Faixa de 09:30 a %s. Primeiro fechamento de 1m fora; se em ate %d min um candle fechar de volta dentro, "
                     "entra contra. Alvo: %s. Stop: %s. Saida limite 15:59."
                     % (hhmm(n), wait, "meio da faixa" if target_mode == "mid" else "lado oposto da faixa",
                        "extremo do rompimento" if stop_mode == "ext" else "sem stop"))
        self.hi = self.lo = self.i_break = self.b = self.ext = None
        self.status = "Formando a faixa ate %s." % hhmm(n)
        self.stage = "formando faixa"

    def after_exit(self, p):
        self.stage, self.done = "encerrada", True
        self.status = "Trade do dia encerrado: %+.2f%% (bruto)." % (p.pnl() * 100)

    def on_bar(self, i, B):
        if self.done or self.pos is not None or self.pending is not None:
            return
        if i < self.n - 1:
            return
        if i == self.n - 1:
            self.hi, self.lo = max(B.h[:self.n]), min(B.l[:self.n])
            self.levels = [("max. faixa", self.hi), ("min. faixa", self.lo)]
            if self.target_mode == "mid":
                self.levels.append(("meio", (self.hi + self.lo) / 2.0))
            self.stage = "aguardando rompimento"
            self.status = "Faixa %.3f - %.3f (largura %.2f%%). Aguardando fechamento de 1m fora." % (
                self.lo, self.hi, (self.hi / self.lo - 1) * 100)
            self.ev(i, "info", "faixa definida: %.3f - %.3f" % (self.lo, self.hi))
            return
        c = B.c[i]
        if self.i_break is None:
            if i > LAST_SIGNAL_I:
                self.stage, self.done, self.status = "sem trade", True, "Sem rompimento hoje."
                return
            if c > self.hi or c < self.lo:
                self.i_break, self.b = i, (1 if c > self.hi else -1)
                self.ext = B.h[i] if self.b > 0 else B.l[i]
                self.stage = "aguardando retorno"
                self.status = "Rompeu para %s as %s. Se fechar de volta dentro da faixa ate %s, %s." % (
                    "CIMA" if self.b > 0 else "BAIXO", hhmm(i), hhmm(min(LAST_SIGNAL_I, i + self.wait)),
                    "VENDER" if self.b > 0 else "COMPRAR")
                self.ev(i, "alerta", "rompeu a faixa para %s (%.3f)" % ("cima" if self.b > 0 else "baixo", c))
            return
        # aguardando retorno
        self.ext = max(self.ext, B.h[i]) if self.b > 0 else min(self.ext, B.l[i])
        if self.lo <= c <= self.hi:
            side = -self.b
            mid = (self.hi + self.lo) / 2.0
            target = mid if self.target_mode == "mid" else (self.lo if self.b > 0 else self.hi)
            if side * (target - c) <= 0:
                self.stage, self.done, self.status = "sem trade", True, "Voltou para a faixa ja alem do alvo; sem trade."
                self.ev(i, "info", "retorno ja alem do alvo; sem trade")
                return
            stop = self.ext if self.stop_mode == "ext" else None
            self.status = "Voltou para dentro da faixa as %s." % hhmm(i)
            self.enter(i, side, stop=stop, target=target, check_stop=(stop is not None))
            return
        if i >= min(LAST_SIGNAL_I, self.i_break + self.wait):
            self.stage, self.done = "sem trade", True
            self.status = "Rompimento se manteve por %d min; sem trade hoje." % self.wait


class NoiseBand(Strategy):
    def __init__(self, ctx, k=1.0, key="noise_band", label="Banda de ruido 1.0x (15 min)", verdict="anterior"):
        super().__init__(ctx)
        self.k, self.key, self.label, self.verdict = k, key, label, verdict
        self.rule = ("A cada 15 min: fechamento acima de abertura x (1 + %sx ruido) -> comprado; abaixo de abertura x (1 - %sx ruido) -> vendido. "
                     "Ruido = media, nos 14 pregoes anteriores, de |fechamento naquele minuto / abertura - 1|. Inverte na banda oposta; zera 15:59."
                     % ("%.2f" % k, "%.2f" % k))
        self.sig = ctx.get("noise_sigma")      # dict {i: sigma}
        if not self.sig:
            self.done, self.stage, self.status = True, "indisponivel", "Sem 14 pregoes de historico para a banda."
        else:
            self.status = "Primeira checagem as %s." % hhmm(15)

    def after_exit(self, p):
        if p.reason == "time":
            self.stage, self.status = "encerrada", "Zerado no fechamento: %+.2f%% (bruto) no ultimo trade." % (p.pnl() * 100)

    def next_check(self, i):
        nxt = ((i + 1) // 15 + 1) * 15 - 1
        return nxt if nxt < NB - 1 else None

    def on_bar(self, i, B):
        if self.done:
            return
        nxt = self.next_check(i)
        if nxt is not None and nxt in self.sig:
            O = B.o[0]
            ub, lb = O * (1 + self.k * self.sig[nxt]), O * (1 - self.k * self.sig[nxt])
            self.levels = [("banda sup. (%s)" % hhmm(nxt + 1), ub), ("banda inf. (%s)" % hhmm(nxt + 1), lb)]
        if (i + 1) % 15 != 0 or i >= NB - 1 or i not in self.sig:
            return
        O, px = B.o[0], B.c[i]
        ub, lb = O * (1 + self.k * self.sig[i]), O * (1 - self.k * self.sig[i])
        cur = self.pos.side if self.pos is not None else 0
        want = 1 if px > ub else (-1 if px < lb else 0)
        if want != 0 and want != cur:
            self.status = "Fechou %s da banda as %s." % ("acima" if want > 0 else "abaixo", hhmm(i + 1))
            self.enter(i, want)
        elif cur == 0:
            self.stage = "aguardando"
            self.status = "Dentro da banda as %s (%.3f entre %.3f e %.3f)." % (hhmm(i + 1), px, lb, ub)


class Mom2h(Strategy):
    key, label, verdict = "mom_2h", "Momentum de 2 horas (filtro de amplitude)", "anterior"
    rule = ("As 11:30: se a amplitude max/min desde 09:30 for >= 0.3 x amplitude media dos 14 pregoes anteriores, "
            "segue o sinal do retorno 09:30->11:30 ate 15:59. Sem stop.")

    def __init__(self, ctx):
        super().__init__(ctx)
        self.atr = ctx.get("atr14")
        if not self.atr:
            self.done, self.stage, self.status = True, "indisponivel", "Sem historico de amplitude diaria."
        else:
            self.status = "Decisao as 11:30 (amplitude minima %.2f%%)." % (0.3 * self.atr * 100)

    def after_exit(self, p):
        self.stage, self.done = "encerrada", True
        self.status = "Trade do dia encerrado: %+.2f%% (bruto)." % (p.pnl() * 100)

    def on_bar(self, i, B):
        if self.done:
            return
        if i < 119:
            rng = max(B.h) / min(B.l) - 1.0
            r = B.c[i] / B.o[0] - 1.0
            self.status = "Ate agora: retorno %+.2f%%, amplitude %.2f%% (minimo %.2f%%). Decisao as 11:30." % (
                r * 100, rng * 100, 0.3 * self.atr * 100)
            return
        if i > 119:
            return
        rng = max(B.h[:120]) / min(B.l[:120]) - 1.0
        r = B.c[119] / B.o[0] - 1.0
        if rng / self.atr < 0.3 or r == 0:
            self.stage, self.done = "sem trade", True
            self.status = "Amplitude %.2f%% abaixo do minimo %.2f%%; sem trade hoje." % (rng * 100, 0.3 * self.atr * 100)
            self.ev(i, "info", "filtro de amplitude nao atendido; sem trade")
            return
        self.status = "Retorno 09:30->11:30 de %+.2f%%, amplitude %.2f%%." % (r * 100, rng * 100)
        self.enter(i, 1 if r > 0 else -1)


class Mom2hGap(Strategy):
    key, label, verdict = "mom_2h_gap", "Momentum de 2 horas com gap >= 1%", "robusta"
    rule = ("So em dia com |abertura 09:30 / fechamento 16:00 anterior - 1| >= 1%. As 11:30 segue o sinal do retorno "
            "09:30->11:30 ate 15:59. Sem stop.")
    GAP = 0.01

    def __init__(self, ctx):
        super().__init__(ctx)
        self.prev_c = ctx.get("prev_close")
        self.gap = None
        if not self.prev_c:
            self.done, self.stage, self.status = True, "indisponivel", "Sem fechamento do pregao anterior."
        else:
            self.status = "Aguardando a abertura para medir o gap."

    def after_exit(self, p):
        self.stage, self.done = "encerrada", True
        self.status = "Trade do dia encerrado: %+.2f%% (bruto)." % (p.pnl() * 100)

    def on_bar(self, i, B):
        if self.done:
            return
        if self.gap is None:
            self.gap = B.o[0] / self.prev_c - 1.0
            if abs(self.gap) < self.GAP:
                self.stage, self.done = "sem trade", True
                self.status = "Gap de %+.2f%% abaixo de 1%%; sem trade hoje." % (self.gap * 100)
                return
        if i < 119:
            self.status = "Gap de %+.2f%%. Ate agora 09:30->agora: %+.2f%%. Decisao as 11:30." % (
                self.gap * 100, (B.c[i] / B.o[0] - 1.0) * 100)
            return
        if i > 119:
            return
        r = B.c[119] / B.o[0] - 1.0
        if r == 0:
            self.stage, self.done, self.status = "sem trade", True, "Retorno nulo as 11:30; sem trade."
            return
        self.status = "Gap de %+.2f%%; retorno 09:30->11:30 de %+.2f%%." % (self.gap * 100, r * 100)
        self.enter(i, 1 if r > 0 else -1)


class LunchTrend(Strategy):
    key, label, verdict = "lunch_trend", "Manha -> almoco (11:30 a 13:00)", "robusta"
    rule = "As 11:30: se o retorno 09:30->11:30 for >= +2% compra, <= -2% vende. Sai as 13:00. Sem stop."
    THR, I_EXIT = 0.02, 209

    def __init__(self, ctx):
        super().__init__(ctx)
        self.status = "Decisao as 11:30 (movimento minimo de 2% desde a abertura)."

    def after_exit(self, p):
        self.stage, self.done = "encerrada", True
        self.status = "Trade do dia encerrado: %+.2f%% (bruto)." % (p.pnl() * 100)

    def on_bar(self, i, B):
        if self.done or i > 119:
            return
        r = B.c[i] / B.o[0] - 1.0
        if i < 119:
            self.status = "09:30->agora: %+.2f%% (precisa de +/-2.00%% as 11:30)." % (r * 100)
            return
        if r == 0 or abs(r) < self.THR:
            self.stage, self.done = "sem trade", True
            self.status = "Retorno 09:30->11:30 de %+.2f%%, abaixo de 2%%; sem trade hoje." % (r * 100)
            return
        self.status = "Retorno 09:30->11:30 de %+.2f%%." % (r * 100)
        self.enter(i, 1 if r > 0 else -1, i_last=self.I_EXIT)


class Last2Min(Strategy):
    key, label, verdict = "last2min", "Ultimos minutos (15:58 a 16:00)", "robusta"
    rule = ("As 15:58: se |fechamento de 15:57 / abertura de 15:53 - 1| >= 0.03 x amplitude media dos 14 pregoes anteriores, "
            "segue a direcao ate o fechamento de 15:59. Margem pequena: so compensa com custo baixo.")
    K, I_REF, I_SIG = 0.03, 383, 387

    def __init__(self, ctx):
        super().__init__(ctx)
        self.atr = ctx.get("atr14")
        if not self.atr:
            self.done, self.stage, self.status = True, "indisponivel", "Sem historico de amplitude diaria."
        else:
            self.status = "Decisao as 15:58 (movimento minimo de %.2f%% desde 15:53)." % (self.K * self.atr * 100)

    def after_exit(self, p):
        self.stage, self.done = "encerrada", True
        self.status = "Trade do dia encerrado: %+.2f%% (bruto)." % (p.pnl() * 100)

    def on_bar(self, i, B):
        if self.done or i < self.I_REF or i > self.I_SIG:
            return
        r = B.c[i] / B.o[self.I_REF] - 1.0
        thr = self.K * self.atr
        if i < self.I_SIG:
            self.status = "15:53->agora: %+.2f%% (precisa de +/-%.2f%% as 15:58)." % (r * 100, thr * 100)
            return
        if r == 0 or abs(r) < thr:
            self.stage, self.done = "sem trade", True
            self.status = "Movimento 15:53->15:58 de %+.2f%%, abaixo de %.2f%%; sem trade hoje." % (r * 100, thr * 100)
            return
        self.status = "Movimento 15:53->15:58 de %+.2f%%." % (r * 100)
        self.enter(i, 1 if r > 0 else -1)


class VwapBand(Strategy):
    key, label, verdict = "vwap_band", "Fade de 2.5 desvios da VWAP", "regime"
    rule = ("Das 09:46 as 15:50: quando z = (fechamento - VWAP da sessao) / desvio ponderado por volume cruza +2.5 -> vender; "
            "cruza -2.5 -> comprar. Sai no fechamento do 60o candle. Sem stop. Uma posicao por vez.")
    THR = 2.5

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cv = self.pv = self.pv2 = 0.0
        self.z_prev = float("nan")
        self.status = "Calculando VWAP (primeiro sinal possivel as 09:46)."

    def after_exit(self, p):
        self.stage = "aguardando"

    def on_bar(self, i, B):
        v, vw = B.v[i], B.vw[i]
        self.cv += v; self.pv += v * vw; self.pv2 += v * vw * vw
        z = float("nan")
        if self.cv > 0:
            vwap = self.pv / self.cv
            sd = math.sqrt(max(self.pv2 / self.cv - vwap * vwap, 0.0))
            if sd > 0 and i >= 15:
                z = (B.c[i] - vwap) / sd
                self.levels = [("VWAP", vwap), ("VWAP +2.5 dp", vwap + self.THR * sd), ("VWAP -2.5 dp", vwap - self.THR * sd)]
        a, self.z_prev = self.z_prev, z
        if self.pos is None and self.pending is None:
            self.stage = "aguardando"
            self.status = ("z = %+.2f (sinal em +/-2.5)." % z) if z == z else self.status
        if not (a == a and z == z):
            return
        if z > self.THR and a <= self.THR:
            side = -1
        elif z < -self.THR and a >= -self.THR:
            side = 1
        else:
            return
        m = RTH_OPEN_MIN + i
        if self.pos is not None or self.pending is not None or m < 575 or m >= 950 or i + 1 > NB - 1:
            return
        self.status = "z cruzou %+.2f as %s." % (z, hhmm(i))
        self.enter(i, side, i_last=min(i + 60, NB - 1))


class StretchFade(Strategy):
    key, label, verdict = "stretch_1m", "Fade de esticada de 1 min (meio do dia)", "regime"
    rule = ("Das 11:00 as 14:30: se o retorno de um candle de 1m passar de 2x o desvio-padrao dos 60 retornos de 1m anteriores "
            "(minimo 20), entra contra na abertura seguinte. Sai no fechamento do 30o candle. Sem stop. Uma posicao por vez.")
    X, HOLD, WIN, MINP = 2.0, 30, 60, 20

    def __init__(self, ctx):
        super().__init__(ctx)
        self.r1 = []
        self.status = "Janela de sinais: 11:00 a 14:30."

    def after_exit(self, p):
        self.stage = "aguardando"

    def on_bar(self, i, B):
        r = (B.c[i] / B.o[i] - 1.0) if i == 0 else (B.c[i] / B.c[i - 1] - 1.0)
        w = self.r1[-self.WIN:]
        self.r1.append(r)
        if len(w) < self.MINP:
            return
        mean = sum(w) / len(w)
        sd = math.sqrt(sum((x - mean) ** 2 for x in w) / (len(w) - 1))
        m = RTH_OPEN_MIN + i
        in_window = 660 <= m < 870
        free = self.pos is None and self.pending is None
        if free:
            self.stage = "aguardando"
            if m < 660:
                self.status = "Janela de sinais comeca as 11:00."
            elif m >= 870:
                self.stage, self.status = "encerrada", "Janela de sinais encerrada as 14:30."
            else:
                self.status = "Ultimo candle %+.2f%%; limite +/-%.2f%% (2 desvios)." % (r * 100, self.X * sd * 100)
        if sd <= 0 or i == 0 or abs(r) <= self.X * sd:
            return
        if not free or not in_window or i + 1 > NB - 1:
            return
        self.status = "Candle de %s andou %+.2f%% (limite %.2f%%)." % (hhmm(i), r * 100, self.X * sd * 100)
        self.enter(i, -1 if r > 0 else 1, i_last=min(i + self.HOLD, NB - 1))


def build_strategies(ctx):
    return [
        LunchTrend(ctx), Mom2hGap(ctx),
        NoiseBand(ctx, 1.25, "noise_125", "Banda de ruido 1.25x (15 min)", "robusta"),
        Last2Min(ctx),
        Mom2h(ctx), NoiseBand(ctx),
        FadeOR(ctx, 15, 30, "mid", "none", "fade_or15", "Falso rompimento da faixa de 15 min", "regime"),
        FadeOR(ctx, 30, 10, "opp", "ext", "fade_or30", "Falso rompimento da faixa de 30 min", "regime"),
        VwapBand(ctx), StretchFade(ctx),
    ]


class Session:
    """Um pregao: recebe candles em ordem e mantem as estrategias."""

    def __init__(self, ny_date, ctx):
        self.date, self.ctx = ny_date, ctx
        self.B = Bars()
        self.strats = build_strategies(ctx)
        self.opened = -1      # ultimo indice com open_bar chamado
        self.closed = -1      # ultimo indice com close_bar chamado

    def open_bar(self, i, o):
        if i != self.opened + 1:
            raise RuntimeError("candles fora de ordem")
        self.opened = i
        for s in self.strats:
            s.open_bar(i, o)

    def peek(self, i, o, h, l):
        for s in self.strats:
            s.peek(i, o, h, l)

    def close_bar(self, i, o, h, l, c, v, vw=None):
        if i != self.closed + 1:
            raise RuntimeError("fechamento fora de ordem")
        if self.opened < i:
            self.open_bar(i, o)
        self.B.add(o, h, l, c, v, vw)
        self.closed = i
        for s in self.strats:
            s.close_bar(i, self.B)


# ======================================================================================
# Contexto historico (banda de ruido e amplitude media)
# ======================================================================================
def context_from_days(prev_days):
    """prev_days: lista cronologica de dicts(o0, marks={i: close}, rng) dos pregoes ANTERIORES."""
    ctx = {}
    if prev_days and prev_days[-1].get("c_last"):
        ctx["prev_close"] = prev_days[-1]["c_last"]
    if prev_days:
        # amplitude (maxima/minima - 1) do pregao regular anterior: so informativo, nao altera sinais
        ctx["prev_rng"] = prev_days[-1]["rng"]
        ctx["prev_date"] = str(prev_days[-1]["date"])
    rngs = [d["rng"] for d in prev_days[-LOOKBACK:]]
    if len(rngs) >= ATR_MIN:
        ctx["atr14"] = sum(rngs) / len(rngs)
    if len(prev_days) >= LOOKBACK:
        last = prev_days[-LOOKBACK:]
        sig = {}
        for i in range(14, NB - 1, 15):
            vals = [abs(d["marks"][i] / d["o0"] - 1.0) for d in last if i in d["marks"]]
            if len(vals) == LOOKBACK:
                sig[i] = sum(vals) / LOOKBACK
        ctx["noise_sigma"] = sig
    return ctx


# ======================================================================================
# Hyperliquid
# ======================================================================================
CLOCK_OFFSET = [0.0]     # relogio do servidor - relogio local (s)


def http_post(url, data):
    """Transporte HTTP: envia `data` (bytes, JSON) e devolve (cabecalho Date, texto da resposta).
    A versao web (navegador) troca SO esta funcao; todo o resto do motor e o mesmo codigo."""
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.headers.get("Date"), resp.read().decode("utf-8")


def api_post(body, retries=4):
    data = json.dumps(body).encode("utf-8")
    err = None
    for k in range(retries):
        try:
            hdr, txt = http_post(API_URL, data)
            if hdr:
                try:
                    srv = datetime.strptime(hdr, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
                    off = srv.timestamp() + 0.5 - time.time()
                    if abs(off) > 2.0:
                        CLOCK_OFFSET[0] = off
                except ValueError:
                    pass
            return json.loads(txt)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
            time.sleep(1.0 * (2 ** k))
    raise RuntimeError("Hyperliquid indisponivel: %s" % err)


def now_utc():
    return datetime.fromtimestamp(time.time() + CLOCK_OFFSET[0], tz=timezone.utc)


def hl_candles(coin, interval, start_ms, end_ms):
    return api_post({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
                                                       "startTime": int(start_ms), "endTime": int(end_ms)}})


def hl_context(coin, ny_date):
    """Contexto do dia a partir dos candles de 15m do perp (dias uteis anteriores a ny_date)."""
    t_open = rth_open_utc_ms(ny_date)
    rows = hl_candles(coin, "15m", t_open - 40 * 86400000, t_open - 1)
    by_day = {}
    for r in rows:
        t = int(r["t"])
        ny = to_ny(datetime.fromtimestamp(t / 1000, tz=timezone.utc))
        if ny.weekday() >= 5:
            continue
        m = ny.hour * 60 + ny.minute
        if m < RTH_OPEN_MIN or m >= RTH_OPEN_MIN + NB:
            continue
        by_day.setdefault(ny.date(), []).append((m, float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"])))
    days = []
    for d in sorted(by_day):
        bars = sorted(by_day[d])
        if len(bars) < 24 or bars[0][0] != RTH_OPEN_MIN:      # dia incompleto (26 candles de 15m no pregao)
            continue
        marks = {(m - RTH_OPEN_MIN) + 14: c for (m, o, h, l, c) in bars}
        days.append(dict(date=d, o0=bars[0][1], marks=marks, c_last=marks.get(NB - 1),
                         rng=max(b[2] for b in bars) / min(b[3] for b in bars) - 1.0))
    ctx = context_from_days(days)
    ctx["hist_days"] = len(days)
    return ctx


def hl_day_bars(coin, ny_date, end_ms=None):
    """Candles de 1m do pregao (dict indice -> (o,h,l,c,v)) a partir da Hyperliquid."""
    t0 = rth_open_utc_ms(ny_date)
    t1 = t0 + NB * 60000 - 1
    if end_ms is not None:
        t1 = min(t1, end_ms)
    out = {}
    for r in hl_candles(coin, "1m", t0, t1):
        i = (int(r["t"]) - t0) // 60000
        if 0 <= i < NB:
            out[i] = (float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"]))
    return out


# ======================================================================================
# Estado compartilhado com o painel
# ======================================================================================
class Monitor:
    def __init__(self, coin):
        self.coin = coin
        self.lock = threading.Lock()
        self.session = None
        self.raw = {}              # indice -> (o,h,l,c,v) mais recente (inclui candle em formacao)
        self.mode = "ao vivo"
        self.phase = "iniciando"
        self.message = ""
        self.sim_now = None        # relogio simulado no replay
        self.feed = []             # eventos do dia (mais recente primeiro no painel)
        self.seen = 0
        self.hist_cut = -1         # eventos com i <= hist_cut foram reconstruidos (sem alerta sonoro)
        self.logged = self._load_log_keys()
        self.error = ""

    # ---- log em CSV
    def _load_log_keys(self):
        keys = set()
        if LOG_PATH.exists():
            with open(LOG_PATH, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    keys.add((r.get("data"), r.get("estrategia"), r.get("evento"), r.get("hora_ny")))
        return keys

    def _log(self, row):
        key = (row["data"], row["estrategia"], row["evento"], row["hora_ny"])
        if key in self.logged or self.mode != "ao vivo":
            return
        self.logged.add(key)
        new = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["data", "hora_ny", "estrategia", "evento", "lado", "preco", "texto", "reconstruido"])
            if new:
                w.writeheader()
            w.writerow(row)

    def collect_events(self):
        """Move eventos novos das estrategias para o feed (e para o CSV)."""
        s = self.session
        new = []
        for st in s.strats:
            while st.events:
                e = st.events.pop(0)
                e.update(strategy=st.key, label=st.label, verdict=st.verdict, hist=(e["i"] <= self.hist_cut))
                new.append(e)
        new.sort(key=lambda e: e["i"])
        for e in new:
            self.seen += 1
            e["id"] = self.seen
            e["time"] = hhmm(e["i"])
            self.feed.append(e)
            if e["kind"] in ("sinal", "entrada", "saida", "alerta"):
                self._log(dict(data=str(s.date), hora_ny=e["time"], estrategia=e["strategy"], evento=e["kind"],
                               lado={1: "compra", -1: "venda", 0: ""}[e["side"]],
                               preco=("%.4f" % e["px"]) if e["px"] is not None else "", texto=e["text"],
                               reconstruido="sim" if e["hist"] else "nao"))
                if not e["hist"]:
                    print("  [%s NY] %-34s %s" % (e["time"], e["label"], e["text"]))
                    if e["kind"] in ("sinal", "saida"):
                        _beep()

    def state(self):
        with self.lock:
            s = self.session
            nu = self.sim_now or now_utc()
            ny = to_ny(nu)
            out = dict(coin=self.coin, mode=self.mode, phase=self.phase, message=self.message, error=self.error,
                       ny_time=ny.strftime("%Y-%m-%d %H:%M:%S"), sp_time=(nu - timedelta(hours=3)).strftime("%H:%M:%S"),
                       ny_offset=ny_offset_hours(nu), bars=[], strategies=[], feed=[], last=None, date=None)
            if s is None:
                return out
            t0 = rth_open_utc_ms(s.date) // 1000
            idx = sorted(self.raw)
            out["date"] = str(s.date)
            out["bars"] = [[t0 + 60 * i] + list(self.raw[i][:4]) for i in idx]
            last = self.raw[idx[-1]][3] if idx else None
            out["last"] = last
            out["hist_days"] = s.ctx.get("hist_days")
            pr = s.ctx.get("prev_rng")
            out["prev_rng"] = (pr * 100) if pr is not None else None
            out["prev_date"] = s.ctx.get("prev_date")
            out["active_thr"] = ACTIVE_RNG * 100
            out["active"] = (pr >= ACTIVE_RNG) if pr is not None else None
            if idx:
                out["today_rng"] = (max(self.raw[i][1] for i in idx) / min(self.raw[i][2] for i in idx) - 1.0) * 100
            for st in s.strats:
                p = st.pos
                out["strategies"].append(dict(
                    key=st.key, label=st.label, verdict=st.verdict, rule=st.rule, stage=st.stage,
                    status=st.describe(last), levels=[[n, v] for n, v in st.levels],
                    side=(p.side if p else 0), entry=(p.entry_px if p else None), entry_t=(t0 + 60 * p.entry_i if p else None),
                    target=(p.target if p else None), stop=(p.stop if p else None),
                    live_pnl=(p.pnl(last) * 100 if (p and last) else None),
                    trades=[dict(side=t.side, entry_t=t0 + 60 * t.entry_i, exit_t=t0 + 60 * t.exit_i, entry=t.entry_px,
                                 exit=t.exit_px, reason=t.reason, pnl=t.pnl() * 100,
                                 entry_hm=hhmm(t.entry_i), exit_hm=hhmm(t.exit_i)) for t in st.trades]))
            out["feed"] = list(reversed(self.feed[-80:]))
            return out


def _beep():
    try:
        import winsound
        winsound.Beep(880, 250)
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()


def feed_session(mon, bars, n_closed, in_progress=None):
    """Alimenta o motor com candles fechados ate o indice n_closed-1 e o candle em formacao (opcional).
    bars: dict indice -> (o,h,l,c,v). Minutos sem negocio sao preenchidos com o ultimo preco."""
    s = mon.session
    last_px = s.B.c[-1] if s.B.c else None
    for i in range(s.closed + 1, n_closed):
        b = bars.get(i)
        if b is None:
            if last_px is None:
                later = [k for k in bars if k > i]
                if not later:
                    return
                last_px = bars[min(later)][0]
            b = (last_px, last_px, last_px, last_px, 0.0)
        s.close_bar(i, *b)
        mon.raw[i] = b
        last_px = b[3]
        mon.collect_events()
    if in_progress is not None and in_progress in bars and s.closed == in_progress - 1:
        b = bars[in_progress]
        if s.opened < in_progress:
            s.open_bar(in_progress, b[0])
        s.peek(in_progress, b[0], b[1], b[2])
        mon.raw[in_progress] = b
        mon.collect_events()


# ======================================================================================
# Lacos: ao vivo e replay
# ======================================================================================
def live_step(mon, poll):
    """Uma iteracao do acompanhamento ao vivo; devolve quantos segundos esperar ate a proxima.
    E chamada por live_loop (no PC) e pela versao web (no navegador): as duas rodam este mesmo codigo."""
    nu = now_utc()
    ny = to_ny(nu)
    today = ny.date()
    m = ny.hour * 60 + ny.minute
    weekday = ny.weekday() < 5
    with mon.lock:
        if mon.session is not None and mon.session.date != today and m >= RTH_OPEN_MIN - 30:
            mon.session, mon.raw, mon.feed, mon.hist_cut = None, {}, [], -1
    if not weekday or m < RTH_OPEN_MIN:
        nxt = today
        if m >= RTH_OPEN_MIN or not weekday:
            nxt = today + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        wait_s = rth_open_utc_ms(nxt) / 1000 - nu.timestamp()
        with mon.lock:
            mon.phase = "pregao fechado"
            mon.message = "Abertura em %dh%02dmin (09:30 de Nova York, %s)." % (wait_s // 3600, (wait_s % 3600) // 60, nxt.strftime("%d/%m"))
        return min(poll * 3, 15)
    if mon.session is None:
        ctx = hl_context(mon.coin, today)
        with mon.lock:
            mon.session = Session(today, ctx)
            mon.raw, mon.feed = {}, []
            t0 = rth_open_utc_ms(today)
            mon.hist_cut = min(NB, int((nu.timestamp() * 1000 - t0) // 60000)) - 1
        print("Pregao %s: historico de %s pregoes para o contexto." % (today, ctx.get("hist_days")))
    t0 = rth_open_utc_ms(today)
    bars = hl_day_bars(mon.coin, today)
    nu = now_utc()
    cur = int((nu.timestamp() * 1000 - t0) // 60000)      # indice do candle em formacao
    with mon.lock:
        if cur >= NB:
            feed_session(mon, bars, NB)
            mon.phase, mon.message = "pregao encerrado", "Pregao regular encerrado as 16:00 de Nova York."
        else:
            feed_session(mon, bars, cur, in_progress=cur)
            mon.phase, mon.message = "pregao aberto", ""
        mon.error = ""
    return poll if cur < NB else 30


def live_loop(mon, poll):
    while True:
        try:
            time.sleep(live_step(mon, poll))
        except Exception as e:                 # mantem o monitor de pe em falha de rede
            with mon.lock:
                mon.error = str(e)
            print("  [erro] %s" % e)
            time.sleep(5)


def replay_loop(mon, ny_date, speed):
    mon.mode = "replay"
    ctx = hl_context(mon.coin, ny_date)
    bars = hl_day_bars(mon.coin, ny_date)
    if len(bars) < 60:
        with mon.lock:
            mon.phase, mon.message = "sem dados", ("A Hyperliquid so guarda ~3.5 dias de candles de 1m; "
                                                   "escolha uma data util mais recente.")
        print(mon.message)
        return
    t0 = rth_open_utc_ms(ny_date)
    with mon.lock:
        mon.session = Session(ny_date, ctx)
        mon.phase, mon.message = "replay", "Replay de %s a %dx." % (ny_date, speed)
    n = max(bars) + 1
    for i in range(n):
        with mon.lock:
            feed_session(mon, bars, i + 1)
            mon.sim_now = datetime.fromtimestamp((t0 + (i + 1) * 60000) / 1000, tz=timezone.utc)
        time.sleep(60.0 / speed)
    with mon.lock:
        mon.phase, mon.message = "replay concluido", "Replay de %s concluido." % ny_date
    print("Replay concluido.")


# ======================================================================================
# Validacao contra o backtest (CSV da Massive)
# ======================================================================================
def validate(csv_path, trades_dir):
    days = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ny = to_ny(datetime.fromtimestamp(int(r["timestamp_unix_ms"]) / 1000, tz=timezone.utc))
            m = ny.hour * 60 + ny.minute
            if RTH_OPEN_MIN <= m < RTH_OPEN_MIN + NB:
                days.setdefault(ny.date(), {})[m - RTH_OPEN_MIN] = (
                    float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                    float(r["volume"]), float(r["vwap"]) if r["vwap"] not in ("", "nan") else float("nan"))
    hist, out = [], {}
    for d in sorted(days):
        day = days[d]
        if len(day) != NB:
            continue
        sess = Session(d, context_from_days(hist))
        for i in range(NB):
            sess.close_bar(i, *day[i])
        for st in sess.strats:
            for t in st.trades:
                out.setdefault(st.key, []).append((str(d), t.side, RTH_OPEN_MIN + t.entry_i, RTH_OPEN_MIN + t.exit_i,
                                                   t.entry_px, t.exit_px, t.reason))
        hist.append(dict(date=d, o0=day[0][0], marks={i: day[i][3] for i in range(14, NB - 1, 15)}, c_last=day[NB - 1][3],
                         rng=max(b[1] for b in day.values()) / min(b[2] for b in day.values()) - 1.0))
    ref_files = {"lunch_trend": "results_v2/eod/lunch_trend_1130_1300_thr2_trades.csv",
                 "mom_2h_gap": "results_v2/momentum/mom_2h_resto_gap1pct_trades.csv",
                 "noise_125": "results_v2/momentum/noise_band_open_k125_15m_eod_trades.csv",
                 "last2min": "results_v2/eod/last2min_mom_1553_1558_atr003_trades.csv",
                 "mom_2h": "results_frozen/momentum/mom_2h_resto_do_dia_rng30_trades.csv",
                 "noise_band": "results_frozen/momentum/noise_band_open_k1_15m_eod_trades.csv",
                 "fade_or15": "results_frozen/orb/fade_or15_esp30_mid_none_trades.csv",
                 "fade_or30": "results_frozen/orb/fade_or30_esp10_opp_ext_trades.csv",
                 "vwap_band": "results_frozen/meanrev/vwapband2.5_fade_h60_trades.csv",
                 "stretch_1m": "results_frozen/meanrev/stretch1m_fade_mid_h30_trades.csv"}
    ok_all = True
    print("Motor ao vivo sobre %d pregoes do CSV:" % len(hist))
    for key, rel in ref_files.items():
        mine = out.get(key, [])
        gross = sum(t[1] * (t[5] / t[4] - 1.0) for t in mine)
        line = "  %-11s %4d trades | soma bruta %+8.2f%%" % (key, len(mine), gross * 100)
        if trades_dir:
            p = Path(trades_dir) / rel
            if p.exists():
                with open(p, newline="", encoding="utf-8") as f:
                    ref = [(r["date"], int(r["side"]), int(r["entry_m"]), int(r["exit_m"]), float(r["entry_px"]), float(r["exit_px"]))
                           for r in csv.DictReader(f)]
                same = len(ref) == len(mine) and all(
                    a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
                    and abs(a[4] - b[4]) < 1e-9 and abs(a[5] - b[5]) < 1e-9 for a, b in zip(mine, ref))
                ok_all &= same
                line += " | backtest %4d trades | identico: %s" % (len(ref), "SIM" if same else "NAO")
                if not same:
                    for a, b in zip(mine, ref):
                        if not (a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
                                and abs(a[4] - b[4]) < 1e-9 and abs(a[5] - b[5]) < 1e-9):
                            line += "\n      1a diferenca: motor %s | backtest %s" % (a[:6], b)
                            break
        print(line)
    return ok_all


# ======================================================================================
# Painel (HTTP local)
# ======================================================================================
def ensure_chart_lib():
    if LIB_FILE.exists() and LIB_FILE.stat().st_size > 50000:
        return LIB_FILE.read_text(encoding="utf-8")
    for url in LIB_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                txt = resp.read().decode("utf-8")
            if len(txt) > 50000:
                LIB_FILE.write_text(txt, encoding="utf-8")
                return txt
        except Exception:
            continue
    return None


def make_handler(mon, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                body = json.dumps(mon.state()).encode("utf-8")
                ctype = "application/json"
            else:
                body, ctype = page, "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


PAGE = r"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOXL ao vivo</title>
<style>
:root{--bg:#fcfcfb;--panel:#f3f3f0;--ink:#1a1a19;--ink2:#5d5c55;--line:#e1e0da;--up:#0b8f7a;--down:#d6453d;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#6250d6;--s7:#008300;--s8:#e34948;--hl:#fff3c4}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--panel:#242422;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a36;--up:#1fa38d;--down:#e2574d;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--s7:#008300;--s8:#e66767;--hl:#4a3f12}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:16px 20px}
.top{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}.top h1{font-size:18px;margin:0}.top .px{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.top .meta{color:var(--ink2);font-size:13px}.top .sp{flex:1}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
.banner{margin:10px 0;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink2);font-size:13px}
.banner.err{border-color:var(--down);color:var(--ink)}
.regime{margin:10px 0;padding:8px 12px;border:1px solid var(--line);border-left:4px solid var(--ink2);border-radius:8px;font-size:13px;display:flex;gap:6px 18px;flex-wrap:wrap;align-items:baseline}
.regime.on{border-left-color:var(--ink)}.regime b{font-size:14px}.regime .m{color:var(--ink2)}
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(340px,1fr);gap:16px;align-items:start}
@media (max-width:1000px){.grid{grid-template-columns:1fr}}
@media (max-width:600px){.wrap{padding:10px 12px}#chart{height:340px}td,th{padding:5px 5px}}
.chartbox{border:1px solid var(--line);border-radius:8px;overflow:hidden}#chart{height:520px}
.legend{display:flex;gap:6px 14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);padding:8px 10px;border-top:1px solid var(--line)}
.legend label{display:flex;align-items:center;gap:5px;cursor:pointer}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px}
.card{border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.card.act{background:var(--hl)}.card h3{margin:0;font-size:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{border:1px solid var(--line);border-radius:10px;padding:0 8px;font-size:11px;font-weight:400;color:var(--ink2)}
.tag.c{border-color:var(--ink);color:var(--ink);font-weight:600}.stage{margin-left:auto;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}
.card .st{margin:5px 0 3px;font-variant-numeric:tabular-nums}.card .rule{color:var(--ink2);font-size:12px}
.card .tr{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums;margin-top:3px}
h2{font-size:14px;margin:18px 0 6px}.grp{font-size:12px;color:var(--ink2);margin:14px 0 6px;font-weight:600}.grp:first-child{margin-top:0}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--ink2);font-weight:500;font-size:12px}
tr.new td{background:var(--hl)}.k{font-weight:600}
</style></head><body><div class="wrap">
<div class="top"><h1 id="title">SOXL ao vivo</h1><span class="px" id="px"></span><span class="meta" id="clock"></span><span class="sp"></span>
<span class="meta" id="mode"></span><button id="snd">Ativar som</button></div>
<div class="banner" id="banner" style="display:none"></div>
<div class="regime" id="regime" style="display:none"></div>
<div class="grid"><div>
<div class="chartbox"><div id="chart"></div><div class="legend" id="legend"></div></div>
<h2>Eventos de hoje</h2><table id="feed"></table>
</div><div id="cards"></div></div>
<div class="banner">Hipóteses estatísticas medidas em 249 pregões do ETF (nov/2025 a nov/2026); nenhuma tem vantagem demonstrada com folga. O painel só sinaliza, não envia ordens. Em feriados e pregões curtos da NYSE, ignore os sinais.</div>
</div>
__LIB__
<script>
(function(){
var css=getComputedStyle(document.documentElement);function v(n){return css.getPropertyValue(n).trim()}
var COL={lunch_trend:v('--s1'),mom_2h_gap:v('--s2'),noise_125:v('--s3'),last2min:v('--s4'),mom_2h:v('--ink2'),noise_band:v('--ink2'),fade_or15:v('--s5'),fade_or30:v('--s6'),vwap_band:v('--s7'),stretch_1m:v('--s8')};
var UP=v('--up'),DOWN=v('--down'),INK2=v('--ink2'),LINE=v('--line'),BG=v('--bg');
var VD={robusta:'Mesmo sinal nos 3 períodos',anterior:'Versão da 1ª rodada',regime:'Dependente de regime'};
var GT={robusta:'Rodada 2: mesmo sinal nos três períodos (evidência ainda fraca)',anterior:'Versões da 1ª rodada das mesmas ideias',regime:'Dependentes de regime: ganharam de mai a nov/2026, perderam de nov/2025 a mai/2026'};
var show={lunch_trend:true,mom_2h_gap:true,noise_125:true,last2min:true,mom_2h:false,noise_band:false,fade_or15:false,fade_or30:false,vwap_band:false,stretch_1m:false};
var nyOff=-5,chart=null,series=null,lines=[],lastLen=-1,lastT=0,lastId=0,first=true,audio=null,fitted=false;
function f(x,d){return x==null?'-':x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d,useGrouping:false})}
function hm(sec){var d=new Date((sec+nyOff*3600)*1000);return ('0'+d.getUTCHours()).slice(-2)+':'+('0'+d.getUTCMinutes()).slice(-2)}
function beep(){if(!audio)return;try{var o=audio.createOscillator(),g=audio.createGain();o.connect(g);g.connect(audio.destination);o.frequency.value=880;g.gain.value=0.15;o.start();o.stop(audio.currentTime+0.35)}catch(e){}}
document.getElementById('snd').onclick=function(){audio=audio||new (window.AudioContext||window.webkitAudioContext)();audio.resume();beep();this.textContent='Som ativado'};
if(window.LightweightCharts){
chart=LightweightCharts.createChart(document.getElementById('chart'),{autoSize:true,
 layout:{background:{color:BG},textColor:INK2,fontSize:11},grid:{vertLines:{color:LINE},horzLines:{color:LINE}},
 rightPriceScale:{borderColor:LINE,minimumWidth:70},crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
 localization:{timeFormatter:function(t){return hm(t)+' NY'}},
 timeScale:{borderColor:LINE,timeVisible:true,secondsVisible:false,rightOffset:8,tickMarkFormatter:function(t){return hm(t)}}});
series=chart.addCandlestickSeries({upColor:UP,downColor:DOWN,borderUpColor:UP,borderDownColor:DOWN,wickUpColor:UP,wickDownColor:DOWN,
 priceFormat:{type:'price',precision:3,minMove:0.001}});}
else{document.getElementById('chart').innerHTML='<div style="padding:20px;color:'+INK2+'">Biblioteca de gráfico não carregada (sem internet na primeira execução). Os cartões e eventos funcionam normalmente.</div>'}
function render(S){
 nyOff=S.ny_offset;
 document.getElementById('title').textContent=S.coin+' ao vivo';
 document.getElementById('px').textContent=S.last==null?'':f(S.last,3);
 document.getElementById('clock').textContent='Nova York '+S.ny_time.slice(11)+' · São Paulo '+S.sp_time+(S.date?' · pregão '+S.date:'');
 document.getElementById('mode').textContent=S.mode+' · '+S.phase;
 var b=document.getElementById('banner'),msg=S.error?('Falha de conexão: '+S.error):S.message;
 b.style.display=msg?'block':'none';b.textContent=msg;b.className='banner'+(S.error?' err':'');
 var rg=document.getElementById('regime');
 if(S.prev_rng!=null){rg.style.display='flex';rg.className='regime'+(S.active?' on':'');
  rg.innerHTML='<b>'+(S.active?'● DIA ATIVO':'○ DIA CALMO')+' para as estratégias de tendência</b>'+
   '<span>Amplitude do pregão anterior ('+S.prev_date.slice(8,10)+'/'+S.prev_date.slice(5,7)+'): <b>'+f(S.prev_rng,1)+'%</b> (corte: '+f(S.active_thr,0)+'%)</span>'+
   (S.today_rng!=null?'<span class="m">Hoje até agora: '+f(S.today_rng,1)+'%</span>':'')+
   '<span class="m">Só informativo: os sinais não mudam.</span>'}
 else rg.style.display='none';
 // grafico
 if(series){
  var data=S.bars.map(function(r){return{time:r[0],open:r[1],high:r[2],low:r[3],close:r[4]}});
  if(data.length){var t=data[data.length-1].time;
   if(data.length===lastLen||(data.length===lastLen+1&&t>lastT)){if(data.length>lastLen&&lastLen>0)series.update(data[data.length-2]);series.update(data[data.length-1])}
   else series.setData(data);
   lastLen=data.length;lastT=t;if(!fitted){chart.timeScale().fitContent();fitted=true}}
  lines.forEach(function(l){series.removePriceLine(l)});lines=[];var mk=[];
  S.strategies.forEach(function(st){if(!show[st.key])return;var c=COL[st.key];
   st.levels.forEach(function(l){lines.push(series.createPriceLine({price:l[1],color:c,lineWidth:1,lineStyle:2,axisLabelVisible:false,title:l[0]}))});
   if(st.side){if(st.target!=null)lines.push(series.createPriceLine({price:st.target,color:c,lineWidth:2,lineStyle:0,axisLabelVisible:true,title:'alvo'}));
    if(st.stop!=null)lines.push(series.createPriceLine({price:st.stop,color:c,lineWidth:2,lineStyle:1,axisLabelVisible:true,title:'stop'}));
    mk.push({time:st.entry_t,position:st.side>0?'belowBar':'aboveBar',color:c,shape:st.side>0?'arrowUp':'arrowDown',text:st.side>0?'C':'V'})}
   st.trades.forEach(function(t){mk.push({time:t.entry_t,position:t.side>0?'belowBar':'aboveBar',color:c,shape:t.side>0?'arrowUp':'arrowDown',text:t.side>0?'C':'V'});
    mk.push({time:t.exit_t,position:t.side>0?'aboveBar':'belowBar',color:c,shape:'circle',text:'sai'})})});
  mk.sort(function(a,b){return a.time-b.time});series.setMarkers(mk)}
 // legenda
 var lg=document.getElementById('legend');if(!lg.dataset.done&&S.strategies.length){lg.dataset.done=1;
  lg.innerHTML='<span>Níveis no gráfico:</span>'+S.strategies.map(function(st){return '<label><input type="checkbox" data-k="'+st.key+'"'+(show[st.key]?' checked':'')+'><i style="background:'+COL[st.key]+'"></i>'+st.label+'</label>'}).join('');
  Array.prototype.forEach.call(lg.querySelectorAll('input'),function(i){i.onchange=function(){show[i.dataset.k]=i.checked}})}
 // cartoes
 var prevV='';
 document.getElementById('cards').innerHTML=S.strategies.map(function(st){
  var head='';if(st.verdict!==prevV){prevV=st.verdict;head='<div class="grp">'+GT[st.verdict]+'</div>'}
  var act=st.stage==='sinal'||st.stage==='posicionada';
  var tr=st.trades.map(function(t){return (t.side>0?'Compra ':'Venda ')+t.entry_hm+' a '+f(t.entry,3)+' → '+t.exit_hm+' a '+f(t.exit,3)+' ('+(t.pnl>=0?'+':'')+f(t.pnl,2)+'%)'}).join('<br>');
  return head+'<div class="card'+(act?' act':'')+'" style="border-left-color:'+COL[st.key]+'"><h3>'+st.label+'<span class="tag '+(st.verdict==='robusta'?'c':'')+'">'+VD[st.verdict]+'</span><span class="stage">'+st.stage+'</span></h3><div class="st">'+(st.status||'')+'</div>'+(tr?'<div class="tr">'+tr+'</div>':'')+'<div class="rule">'+st.rule+'</div></div>'}).join('');
 // eventos
 var KN={sinal:'SINAL',entrada:'Entrada',saida:'Saída',alerta:'Alerta',info:'Info'},mx=lastId,ring=false;
 document.getElementById('feed').innerHTML='<tr><th>Hora NY</th><th>Estratégia</th><th>Evento</th><th>Detalhe</th></tr>'+S.feed.map(function(e){
  var nw=!first&&e.id>lastId;if(e.id>mx)mx=e.id;if(nw&&!e.hist&&(e.kind==='sinal'||e.kind==='saida'||e.kind==='alerta'))ring=true;
  return '<tr class="'+(nw?'new':'')+'"><td>'+e.time+'</td><td>'+e.label+'</td><td class="k">'+KN[e.kind]+'</td><td>'+e.text+(e.hist?' (reconstruído)':'')+'</td></tr>'}).join('');
 if(ring)beep();lastId=mx;first=false;
 document.title=(S.last==null?'':f(S.last,2)+' · ')+'SOXL ao vivo';
}
function tick(){fetch('/state?'+Date.now()).then(function(r){return r.json()}).then(render).catch(function(){
 var b=document.getElementById('banner');b.style.display='block';b.className='banner err';b.textContent='Sem resposta do monitor (o script Python está rodando?).'})}
tick();setInterval(tick,2000);
})();
</script></body></html>
"""


def serve(mon, port, open_browser):
    lib = ensure_chart_lib()
    if lib is None:
        lib_tag = '<script src="%s"></script>' % LIB_URLS[0]
    else:
        lib_tag = "<script>" + lib.replace("</script", "<\\/script") + "</script>"
    page = PAGE.replace("__LIB__", lib_tag).encode("utf-8")
    srv = HTTPServer(("127.0.0.1", port), make_handler(mon, page))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://localhost:%d" % port
    print("Painel: %s   (Ctrl+C para encerrar)" % url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return srv


def main(argv=None):
    p = argparse.ArgumentParser(description="Monitor em tempo real das estrategias intraday do SOXL (Hyperliquid)")
    p.add_argument("command", nargs="?", default="live", choices=["live", "replay", "validate"])
    p.add_argument("date", nargs="?", help="data do replay (AAAA-MM-DD, horario de NY)")
    p.add_argument("--coin", default="xyz:SOXL")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--poll", type=float, default=4.0, help="segundos entre consultas a API")
    p.add_argument("--speed", type=float, default=20.0, help="replay: minutos de mercado por minuto real / 60")
    p.add_argument("--csv", help="validate: CSV de 1 minuto da Massive")
    p.add_argument("--trades-dir", help="validate: pasta 'backtest' (com results_frozen e results_v2) para comparar trade a trade")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)

    if a.command == "validate":
        if not a.csv:
            p.error("validate precisa de --csv")
        ok = validate(a.csv, a.trades_dir)
        sys.exit(0 if ok else 1)

    mon = Monitor(a.coin)
    serve(mon, a.port, not a.no_browser)
    try:
        if a.command == "replay":
            if not a.date:
                p.error("replay precisa da data (AAAA-MM-DD)")
            replay_loop(mon, datetime.strptime(a.date, "%Y-%m-%d").date(), a.speed)
            while True:
                time.sleep(1)
        else:
            live_loop(mon, a.poll)
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main(sys.argv[1:])
