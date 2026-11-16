@echo off
rem publicar_github.bat -- envia o projeto para o GitHub (cria o repositorio local na primeira vez).
rem Uso: dois cliques neste arquivo, ou "publicar_github.bat" no prompt, dentro desta pasta.
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 goto nogit

rem 1) regera o index.html a partir do painel do soxl_live.py (se o Python estiver no PATH)
where python >nul 2>nul
if errorlevel 1 goto afterbuild
python build_web.py
if errorlevel 1 goto buildfail
:afterbuild

rem 2) repositorio local
if exist ".git" goto haverepo
git init
if errorlevel 1 goto fail
git symbolic-ref HEAD refs/heads/main
:haverepo

rem 3) identidade dos commits (so neste repositorio; usa o e-mail "noreply" do GitHub)
git config user.email >nul 2>nul
if not errorlevel 1 goto haveid
set "GHUSER="
set /p GHUSER=Seu usuario do GitHub: 
if "%GHUSER%"=="" goto fail
git config user.name "%GHUSER%"
git config user.email "%GHUSER%@users.noreply.github.com"
:haveid

rem 4) endereco do repositorio no GitHub
git remote get-url origin >nul 2>nul
if not errorlevel 1 goto haveremote
echo.
echo Crie antes um repositorio VAZIO em https://github.com/new (publico, sem README).
set "REPO="
set /p REPO=Cole o endereco dele (ex.: https://github.com/usuario/candle-analyzer.git): 
if "%REPO%"=="" goto fail
git remote add origin "%REPO%"
if errorlevel 1 goto fail
:haveremote

rem 5) commit (se houver mudanca) e envio
git add -A
if errorlevel 1 goto fail
git diff --cached --quiet
if not errorlevel 1 goto push
git commit -m "Atualizacao %date% %time:~0,5%"
if errorlevel 1 goto fail
:push
git push -u origin main
if errorlevel 1 goto fail

echo.
echo Enviado. Na PRIMEIRA vez, ative o site: no GitHub, Settings ^> Pages ^> Source "Deploy from a branch" ^> Branch "main" e pasta "/ (root)" ^> Save.
echo O painel fica em https://SEU_USUARIO.github.io/NOME_DO_REPOSITORIO/ (leva 1 a 2 minutos para atualizar).
pause
exit /b 0

:nogit
echo Git nao encontrado. Instale em https://git-scm.com/download/win e rode este arquivo de novo.
pause
exit /b 1

:buildfail
echo build_web.py falhou; nada foi enviado.
pause
exit /b 1

:fail
echo.
echo Nao foi possivel concluir. Veja a mensagem acima; nada alem do que aparece ali foi enviado.
pause
exit /b 1
