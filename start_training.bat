@echo off
setlocal

:: ============================================================
:: PokeRL Training Launcher for Windows
:: Starts Pokemon Showdown server, then launches training.
:: Press Ctrl+C to stop.
:: ============================================================

set SHOWDOWN_DIR=%~dp0pokemon-showdown
set SHOWDOWN_PORT=8000
set LOG_FILE=%~dp0logs\training.log

:: ----------------------------------------------------------
:: Step 1: Clone Pokemon Showdown if not present
:: ----------------------------------------------------------
if not exist "%SHOWDOWN_DIR%\pokemon-showdown" (
    if exist "%SHOWDOWN_DIR%" (
        echo [launcher] Incomplete Pokemon Showdown found. Removing and re-cloning...
        rmdir /s /q "%SHOWDOWN_DIR%"
    )
    echo [launcher] Cloning Pokemon Showdown...
    git clone --depth 1 https://github.com/smogon/pokemon-showdown.git "%SHOWDOWN_DIR%"
    if errorlevel 1 (
        echo [launcher] ERROR: Failed to clone Pokemon Showdown. Is git installed?
        pause
        exit /b 1
    )
    echo [launcher] Installing Showdown dependencies...
    cd /d "%SHOWDOWN_DIR%"
    call npm install
    node build
    cd /d "%~dp0"
)

:: ----------------------------------------------------------
:: Step 2: Create logs directory
:: ----------------------------------------------------------
if not exist "%~dp0logs" mkdir "%~dp0logs"

:: ----------------------------------------------------------
:: Step 3: Start Pokemon Showdown in the background
:: ----------------------------------------------------------
echo [launcher] Starting Pokemon Showdown on port %SHOWDOWN_PORT%...
start /b "showdown" node "%SHOWDOWN_DIR%\pokemon-showdown" start --no-security --skip-build --port=%SHOWDOWN_PORT%

:: Wait for Showdown to be ready
echo [launcher] Waiting for Showdown to be ready...
set RETRIES=0
:wait_loop
if %RETRIES% GEQ 30 (
    echo [launcher] ERROR: Showdown did not start within 30 seconds.
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul
node -e "const net=require('net');const c=net.connect(%SHOWDOWN_PORT%,'localhost',()=>{c.end();process.exit(0)});c.on('error',()=>process.exit(1))" 2>nul
if errorlevel 1 (
    set /a RETRIES+=1
    goto wait_loop
)
echo [launcher] Showdown is ready.

:: ----------------------------------------------------------
:: Step 4: Launch training
:: ----------------------------------------------------------
set DASHBOARD_PORT=5555
echo [launcher] Starting training... (logs: %LOG_FILE%)
echo [launcher] Errors only: %LOG_FILE:.log=_errors.log%
echo [launcher] Dashboard: http://localhost:%DASHBOARD_PORT%
echo [launcher] Press Ctrl+C to stop.
python -u "%~dp0train.py" ^
    --headless ^
    --log-file "%LOG_FILE%" ^
    --server-url localhost ^
    --server-port %SHOWDOWN_PORT% ^
    --checkpoint-dir "%~dp0checkpoints" ^
    --dashboard ^
    --dashboard-port %DASHBOARD_PORT% ^
    --resume ^
    %*

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [launcher] Training exited with an error!
    echo [launcher] Check logs\training_errors.log for error details
    echo [launcher] Full log: %LOG_FILE%
    echo ============================================================
)

echo.
echo [launcher] Training stopped.
pause
