@echo off
echo Starting UniBuddy Services...
echo.

REM Check if package.json exists
if not exist "package.json" (
    echo Error: Please run this script from the UniBuddy root directory
    pause
    exit /b 1
)

echo Starting services...
echo.

REM Start Auth Backend
echo Starting Authentication Backend on Port 5000...
start "Auth Backend" cmd /k "cd backend\Unibuddy-Auth\Unibuddy-MERN-Authentication\backend && npm start"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start Chatbot Backend
echo Starting Chatbot Backend on Port 9000...
start "Chatbot Backend" cmd /k "cd backend && python api.py"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start Frontend
echo Starting Frontend on Port 5173...
start "Frontend" cmd /k "npm run dev"

echo.
echo All services started!
echo.
echo Access Points:
echo    Frontend:        http://localhost:5173
echo    Auth Backend:    http://localhost:5000
echo    Chatbot Backend: http://localhost:9000
echo.
echo Tip: Close all command prompt windows to stop all services
echo.
pause
