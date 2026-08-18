@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo  Model_Drift - Arranque
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo No encuentro .venv\Scripts\python.exe
  echo.
  echo Crea el entorno desde esta carpeta:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e .[dev]
  echo.
  echo Carpeta actual: %CD%
  echo.
  pause
  exit /b 1
)

rem El venv guarda rutas absolutas del Python con el que se creo: al alternar
rem entre los dos PCs queda roto aunque el .exe siga estando ahi.
".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo El entorno .venv esta roto o se creo en otro PC.
  echo Recrealo desde esta carpeta:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e .[dev]
  echo.
  pause
  exit /b 1
)

rem Comprobar los imports AQUI, con python.exe y consola visible. Despues se
rem lanza con pythonw.exe, que no tiene consola: un fallo de importacion alli
rem seria un fallo mudo y la aplicacion simplemente no abriria.
echo Comprobando el entorno...
".venv\Scripts\python.exe" -c "import model_drift.gui" 2>error_arranque.tmp
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Model_Drift no se puede importar. Detalle:
  echo.
  type error_arranque.tmp
  del error_arranque.tmp >nul 2>&1
  echo.
  echo Si falta el paquete, instalalo con:
  echo   .venv\Scripts\pip install -e .[dev]
  echo.
  pause
  exit /b 1
)
del error_arranque.tmp >nul 2>&1

if not defined MODEL_DRIFT_BROKER_TOKEN (
  echo.
  echo AVISO: no esta definida MODEL_DRIFT_BROKER_TOKEN.
  echo Puedes escribir el token en la propia ventana, pestana "Modelos".
  echo El Broker rota su token en cada arranque: cogelo de su ventana.
  echo.
)

echo Abriendo la ventana de Model_Drift...

rem pythonw.exe: sin ventana de consola detras de la aplicacion. Si no existiera
rem (instalaciones minimas de Python), se cae a python.exe, que funciona igual
rem pero deja una consola abierta.
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m model_drift.gui
) else (
  start "" ".venv\Scripts\python.exe" -m model_drift.gui
)

rem Sin `pause`: la ventana ya esta abierta y este .bat ha terminado su trabajo.
exit /b 0
