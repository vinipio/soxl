# Laboratorio de estrategias intraday SOXL -- regras comuns (LEIA TUDO)

Dados: a pasta backtest/data/soxl_1m_raw.csv -- SOXL (ETF 3x semis), candles de 1 minuto,
2026-05-11 a 2026-11-10, 128 pregoes, pre-market (04:00) + pregao regular (09:30-16:00, 390 candles/dia,
sem buracos) + after-hours (ate 20:00). Horario de Nova York. Precos sem ajuste (nao ha splits no periodo).

Motor OBRIGATORIO: a pasta backtest/lab.py. Leia o arquivo inteiro antes de comecar.
Use lab.load()/lab.rth()/lab.day_groups()/lab.daily_table()/lab.resample()/lab.sim_exit()/
lab.make_trade()/lab.evaluate_is()/lab.evaluate()/lab.save_result()/lab.fmt().
NAO altere lab.py nem os dados. NAO instale pacotes (pandas e numpy estao disponiveis). NAO crie subagentes.

## Regras de metodologia (inviolaveis)
1. Sem look-ahead: sinal com dados ate o FECHAMENTO do candle i, entrada no OPEN do candle i+1
   (i_entry = i+1 em lab.sim_exit). Medias/quantis "tipicos" (volume por horario, ATR etc.) so com dias ANTERIORES.
2. Posicoes abertas e fechadas no mesmo dia (day trade), salvo se sua familia disser o contrario explicitamente.
3. Escolha de parametros SOMENTE no in-sample (2026-05-11..2026-08-31, 78 dias) usando lab.evaluate_is().
   NAO olhe o OOS (2026-09-01..2026-11-10, 50 dias) antes de congelar os finalistas. Depois de ver o OOS,
   NAO reajuste nada. Reporte o OOS como saiu, bom ou ruim.
4. Grade de parametros modesta (ate ~150 configuracoes por ideia) e CONTE quantas configuracoes testou
   (vai em n_configs_tested). Prefira regioes de parametros estaveis (vizinhos tambem positivos) a picos isolados.
5. Minimo de 30 trades no IS para considerar uma configuracao.
6. Custos: base 3 bps/lado, estresse 10 bps/lado (lab.save_result ja calcula os dois). Selecione pelo custo base.
7. CUIDADO COM VIES DIRECIONAL: no IS o SOXL caiu muito DENTRO do pregao (soma open->close = -80%; ficar
   vendido do open ao close deu Sharpe ~+2,2 no IS e ~+0,4 no OOS) e subiu no overnight. Uma estrategia que so
   "funciona" vendida provavelmente esta apenas capturando essa deriva. Sempre reporte compra e venda
   separadamente e compare com esse benchmark. Valorize efeitos simetricos.
8. Finalistas: no maximo 3 por agente. Para cada um chame lab.save_result(family=<sua familia>, name=..., ...)
   com `live_trigger` = descricao objetiva do que observar em tempo real para ativar/entrar/sair
   (condicoes mensuraveis: horario, niveis, filtros). Se nada sobreviver no IS, diga isso claramente -- resultado
   negativo honesto vale mais que um resultado forcado. Mesmo assim salve o "menos pior" marcando em notes.
9. Coloque TODO o codigo em UM script re-executavel: a pasta backtest/agents/<familia>.py
   (rodar `python3 agents/<familia>.py` a partir de a pasta backtest deve refazer a busca no IS e regravar os
   resultados). Comece o script com: import sys; sys.path.insert(0, 'a pasta backtest'); import lab
10. Seja eficiente: 128 dias x 390 candles e pouco dado; loops simples em Python bastam. Limite-se a ~30 chamadas de ferramenta.

## Resposta final (em portugues, <= 350 palavras)
- Tabela dos finalistas: nome | regra em 1 linha | IS: n, net bps/trade, t, Sharpe | OOS: n, net bps/trade, t, Sharpe |
  OOS no custo de estresse | resultado compra vs venda.
- Numero de configuracoes testadas.
- O que NAO funcionou (1-3 linhas).
- Fatos estatisticos uteis que encontrou (ex.: medias por horario) mesmo que nao virem estrategia.
- Gatilho em tempo real de cada finalista.
