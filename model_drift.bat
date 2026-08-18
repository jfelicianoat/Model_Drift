@echo off
setlocal
cd /d "%~dp0"

rem Envoltorio de la linea de comandos: evita tener que recordar la ruta del
rem venv. Todo lo que se le pasa va tal cual a `model-drift`.
rem
rem   model_drift.bat modelos
rem   model_drift.bat ejecutar suites\ejemplo --modelo ollama/local/qwen3:8b
rem   model_drift.bat comparar run-A run-B --suite suites\ejemplo --informe informes\x.html
rem   model_drift.bat potencia --casos 50

if not exist ".venv\Scripts\python.exe" (
  echo No encuentro .venv\Scripts\python.exe
  echo Crea el entorno desde esta carpeta:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e .[dev]
  exit /b 1
)

".venv\Scripts\python.exe" -c "import model_drift" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo El entorno .venv esta roto, se creo en otro PC, o Model_Drift no esta
  echo instalado en el. Recrealo desde esta carpeta:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e .[dev]
  exit /b 1
)

if "%~1"=="" (
  echo ==========================================
  echo  Model_Drift - Linea de comandos
  echo ==========================================
  echo.
  ".venv\Scripts\python.exe" -m model_drift.cli --help
  echo.
  echo Ejemplos:
  echo   model_drift.bat modelos
  echo   model_drift.bat ejecutar suites\ejemplo --modelo ollama/local/qwen3:6b --relaciones relaciones
  echo   model_drift.bat tiradas
  echo   model_drift.bat comparar run-A run-B --suite suites\ejemplo --politica politica-ejemplo.yaml --informe informes\comparacion.html
  echo.
  echo Para la ventana de escritorio: arrancar_model_drift.bat
  echo.
  exit /b 0
)

rem Sin `pause` a proposito: este .bat se usa tambien desde scripts y desde CI,
rem y su codigo de salida es el del comando (2 = la politica bloquea la
rem comparacion). Un `pause` lo volveria inutilizable ahi.
".venv\Scripts\python.exe" -m model_drift.cli %*
exit /b %ERRORLEVEL%
