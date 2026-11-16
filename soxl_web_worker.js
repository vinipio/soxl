/* soxl_web_worker.js -- roda o motor do monitor (soxl_live.py) dentro do navegador.
   O arquivo soxl_live.py publicado ao lado e carregado como esta e executado pelo Pyodide (Python em WebAssembly).
   A unica troca e o transporte HTTP (soxl_live.http_post), que aqui usa XMLHttpRequest.
   A cada passo chama soxl_live.live_step() e envia soxl_live.Monitor.state() para a pagina. */
var PYODIDE_VERSION = '314.0.8';
var PYODIDE_BASE = 'https://cdn.jsdelivr.net/pyodide/v' + PYODIDE_VERSION + '/full/';
var ENGINE_FILE = 'soxl_live.py';

var GLUE = [
"import json, sys",
"sys.path.insert(0, '/home/pyodide')",
"import soxl_live as L",
"from js import XMLHttpRequest",
"",
"def _http_post(url, data):",
"    x = XMLHttpRequest.new()",
"    x.open('POST', url, False)",
"    x.setRequestHeader('Content-Type', 'application/json')",
"    try:",
"        x.send(data.decode('utf-8'))",
"    except Exception as e:",
"        raise OSError('sem conexao com a Hyperliquid (%s)' % e)",
"    if x.status != 200:",
"        raise OSError('HTTP %s da Hyperliquid' % x.status)",
"    return x.getResponseHeader('Date'), x.responseText",
"",
"L.http_post = _http_post",
"_api_post = L.api_post",
"L.api_post = lambda body, retries=1: _api_post(body, 1)   # nova tentativa fica por conta do proximo passo",
"",
"mon = L.Monitor(COIN)",
"",
"def web_step(poll):",
"    try:",
"        wait = L.live_step(mon, poll)",
"    except Exception as e:",
"        with mon.lock:",
"            mon.error = str(e)",
"        wait = 5",
"    return json.dumps(dict(wait=wait, state=mon.state()))",
""].join("\n");

var webStep = null, timer = null, POLL = 4.0, busy = false;

function status(text, isError) { postMessage({type: 'status', text: text, error: !!isError}); }

function step() {
  if (!webStep || busy) return;
  busy = true;
  if (timer) { clearTimeout(timer); timer = null; }
  var wait = 5;
  try {
    var out = JSON.parse(webStep(POLL));
    wait = out.wait;
    postMessage({type: 'state', state: out.state});
  } catch (e) {
    status('Erro no motor: ' + e, true);
  }
  busy = false;
  timer = setTimeout(step, Math.max(1, wait) * 1000);
}

async function boot(cfg) {
  try {
    POLL = cfg.poll || 4.0;
    status('Carregando o motor Python no navegador (na primeira vez são cerca de 12 MB; depois fica em cache)...');
    importScripts(PYODIDE_BASE + 'pyodide.js');
    var py = await loadPyodide({indexURL: PYODIDE_BASE});
    status('Carregando as estratégias (' + ENGINE_FILE + ')...');
    var resp = await fetch(ENGINE_FILE + '?v=' + encodeURIComponent(cfg.build || Date.now()), {cache: 'no-store'});
    if (!resp.ok) throw new Error(ENGINE_FILE + ': HTTP ' + resp.status);
    py.FS.mkdirTree('/home/pyodide');
    py.FS.writeFile('/home/pyodide/' + ENGINE_FILE, await resp.text());
    py.globals.set('COIN', cfg.coin || 'xyz:SOXL');
    py.runPython(GLUE);
    webStep = py.globals.get('web_step');
    status('Buscando os candles do dia na Hyperliquid...');
    step();
  } catch (e) {
    status('Não foi possível iniciar o motor: ' + e, true);
  }
}

onmessage = function (m) {
  var d = m.data || {};
  if (d.type === 'boot') boot(d);
  else if (d.type === 'now') step();      // a aba voltou a ficar visivel: atualiza na hora
};
