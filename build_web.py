# -*- coding: utf-8 -*-
"""build_web.py -- gera a versao web (GitHub Pages) do monitor a partir do proprio soxl_live.py.

  python build_web.py

Saida: index.html nesta pasta. O painel e exatamente o PAGE de soxl_live.py; a unica diferenca e de
onde vem o estado: em vez de consultar /state no servidor local, a pagina recebe o estado do
soxl_web_worker.js, que roda soxl_live.py dentro do navegador (Pyodide).

Rode de novo sempre que mudar o painel (PAGE) em soxl_live.py. Mudancas nas estrategias NAO exigem
rodar de novo: o navegador carrega o soxl_live.py publicado.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import soxl_live as L

WORKER = "soxl_web_worker.js"
ENGINE = "soxl_live.py"
LIB = "lightweight-charts-%s.standalone.js" % L.LIB_VERSION


def must_replace(text, old, new):
    if text.count(old) != 1:
        raise SystemExit("build_web.py: trecho esperado do painel nao encontrado (ou repetido): %r" % old[:70])
    return text.replace(old, new)


def main():
    for f in (WORKER, ENGINE, LIB):
        if not os.path.exists(os.path.join(HERE, f)):
            raise SystemExit("build_web.py: falta o arquivo %s nesta pasta" % f)
    h = hashlib.sha1()
    for f in (WORKER, ENGINE):
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read())
    build = h.hexdigest()[:10]

    page = L.PAGE
    page = must_replace(page, "__LIB__", '<script src="%s"></script>' % LIB)
    old_tick = page[page.index("function tick(){fetch('/state?'"):page.index("})();\n</script></body></html>")]
    if "setInterval(tick,2000);" not in old_tick:
        raise SystemExit("build_web.py: o trecho de atualizacao do painel mudou; revise este script")
    new_tick = (
        "function note(t,err){var b=document.getElementById('banner');b.style.display='block';b.className='banner'+(err?' err':'');b.textContent=t}\n"
        "var q=new URLSearchParams(location.search),BUILD='%s';\n"
        "note('Iniciando...');\n"
        "var W=new Worker('%s?v='+BUILD,{type:'module'});\n"
        "W.onmessage=function(m){var d=m.data;if(d.type==='state')render(d.state);else if(d.type==='status')note(d.text,d.error)};\n"
        "W.onerror=function(e){note('Falha ao carregar o motor: '+(e.message||e),true)};\n"
        "W.postMessage({type:'boot',coin:q.get('coin')||'xyz:SOXL',poll:parseFloat(q.get('poll'))||4,build:BUILD});\n"
        "document.addEventListener('visibilitychange',function(){if(!document.hidden)W.postMessage({type:'now'})});\n"
    ) % (build, WORKER)
    page = must_replace(page, old_tick, new_tick)
    page = must_replace(page, "<title>SOXL ao vivo</title>",
                        '<title>SOXL ao vivo</title>\n<meta name="robots" content="noindex,nofollow">\n'
                        '<meta name="apple-mobile-web-app-capable" content="yes">\n'
                        '<meta name="apple-mobile-web-app-title" content="SOXL ao vivo">')
    out = os.path.join(HERE, "index.html")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("<!-- GERADO por build_web.py a partir de soxl_live.py (PAGE). Nao edite a mao. build %s -->\n" % build)
        fh.write(page)
    nojekyll = os.path.join(HERE, ".nojekyll")
    if not os.path.exists(nojekyll):
        open(nojekyll, "w").close()
    print("index.html gerado (build %s, %d bytes)." % (build, os.path.getsize(out)))


if __name__ == "__main__":
    main()
