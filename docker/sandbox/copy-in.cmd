@echo off
rem Copy a project INTO the sandbox /workspace volume with correct (non-root)
rem ownership so Claude Code can edit it. Windows version of copy-in.sh.
rem
rem `docker compose cp` writes files as root; the sandbox runs as the non-root
rem 'claude' user and has cap_drop: ALL (no CAP_CHOWN), so we fix ownership with
rem a throwaway root container that has default caps. The running sandbox is not
rem weakened (it keeps cap_drop: ALL + user: claude).
rem
rem Usage (run from docker\sandbox):
rem   copy-in <path-to-project> [dest-name]
setlocal enabledelayedexpansion

if "%~1"=="" (
  echo Usage: copy-in ^<path-to-project^> [dest-name]
  exit /b 1
)

set "SRC=%~1"
set "NAME=%~2"
if "%NAME%"=="" set "NAME=%~nx1"

rem Compose project name = current directory's name ("sandbox").
for %%I in ("%CD%") do set "PROJECT=%%~nxI"
set "VOLUME=!PROJECT!_workspace"
set "IMAGE=!PROJECT!-claude"

echo ^>^> Copying "%SRC%" to /workspace/%NAME% ...
docker compose cp "%SRC%" "claude:/workspace/%NAME%" || exit /b 1

echo ^>^> Fixing ownership (throwaway root container; sandbox stays locked) ...
docker run --rm -u 0:0 -v "!VOLUME!:/workspace" "!IMAGE!" chown -R claude:claude "/workspace/%NAME%" || exit /b 1

echo ^>^> Done. /workspace/%NAME% is owned by the non-root claude user.
endlocal
