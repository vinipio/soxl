# Rodada 2 -- refazer a busca com 1 ano de dados (LEIA TUDO)

Contexto: a rodada 1 usou 128 pregoes (mai-nov/2026). Agora ha 249 pregoes (2025-11-11 a 2026-11-10) em
/home/claude/soxl_lab/data/soxl_1m_all.csv, ja carregados por lab.py (NAO altere lab.py nem os dados).
Leia /home/claude/soxl_lab/lab.py de novo (mudou) e /home/claude/soxl_lab/BRIEF.md (regras de execucao continuam valendo).

Tres periodos fixos (lab.period_of(date) -> 'P0'|'P1'|'P2'; lab.evaluate(trades) devolve blocos 'P0','P1','P2' alem de 'IS','OOS','ALL'):
  P0 = 2025-11-11..2026-05-08  121 pregoes  mercado em ALTA  (soma open->close +82%, amplitude media 8,6%)
  P1 = 2026-05-11..2026-08-31   78 pregoes  QUEDA forte intraday (soma open->close -75%, amplitude 12,9%)
  P2 = 2026-09-01..2026-11-10   50 pregoes  teste (soma open->close -5%, amplitude 9,7%)
IS (lab.IS_END = 2026-08-31) = P0+P1 = 199 pregoes. OOS = P2.

FATO CENTRAL: as 17 regras finalistas da rodada 1 foram escolhidas olhando so P1. Aplicadas sem mudanca a P0,
NENHUMA teve resultado significativo e a maioria perdeu (tabela em results_frozen/periodos.csv; ex.: o falso
rompimento da faixa de 15 min, melhor da rodada 1, deu -24,7 bps/trade em P0). Ou seja: o que parecia padrao
era em boa parte regime. O objetivo agora e achar o que e ROBUSTO AOS DOIS REGIMES.

## Sua tarefa
1. Copie seu script da rodada 1, /home/claude/soxl_lab/agents/<familia>.py, para agents/<familia>_v2.py e adapte.
   As PRIMEIRAS linhas do v2 devem ser:
       import os, sys
       os.environ['SOXL_RESULTS'] = 'results_v2'
       sys.path.insert(0, '/home/claude/soxl_lab'); os.chdir('/home/claude/soxl_lab'); import lab
   NAO altere o script original nem as pastas results, results_v1, results_frozen.
2. Refaca a busca da sua familia (mesmas ideias; pode acrescentar poucas ideias novas da mesma familia) com
   IS = P0+P1 e este criterio de selecao, usando SOMENTE os blocos 'P0' e 'P1' de lab.evaluate (os dois sao in-sample):
     - pelo menos 30 trades em P0 e 30 em P1;
     - liquido > 0 no custo base em P0 E em P1 separadamente;
     - ordenar por min(t_P0, t_P1) (o t do PIOR regime), desempatando por estabilidade nos parametros vizinhos;
     - preferir compra e venda ambas positivas.
   E PROIBIDO olhar os blocos 'P2', 'OOS' ou 'ALL' antes de congelar os finalistas. Depois de olhar, nao reajuste.
3. Permitido (opcional): regra CONDICIONAL a regime, desde que a variavel de regime use so informacao anterior ao
   trade (ex.: retorno dos N pregoes anteriores, atr14 de lab.daily_table(), gap do dia, largura da faixa vs. media).
   Mantenha pequeno: no maximo 3 variaveis x 3 limiares, e conte essas configuracoes.
4. Conte TODAS as configuracoes testadas. Com centenas de testes, t=2 em um periodo nao significa nada; o que
   interessa e consistencia de sinal nos dois regimes.
5. Finalistas: no maximo 2. Se nada satisfizer o criterio, diga isso claramente (resultado negativo e resultado) e
   salve apenas o "menos pior" marcando em notes. Use lab.save_result(family=<familia>, ...) (vai para results_v2/).
6. Limite: ~25 chamadas de ferramenta. NAO crie subagentes. NAO instale pacotes.

## Resposta final (portugues, <= 300 palavras)
- Tabela dos finalistas: nome | regra em 1 linha | P0: n, bps liquido/trade, t | P1: n, bps, t | P2: n, bps, t |
  P2 no custo de estresse | compra vs venda.
- Quantas configuracoes testadas e quantas passaram no criterio dos dois regimes.
- Os finalistas da rodada 1 da sua familia aparecem bem ranqueados na nova busca? (sim/nao e por que)
- O que distingue P0 de P1 para a sua familia (1-3 linhas, com numeros).
- Gatilho em tempo real de cada finalista.
