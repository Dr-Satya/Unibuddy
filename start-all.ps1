# UniBuddy - Start All Services
# This script starts all required services for the UniBuddy application

Write-Host "Starting UniBuddy Services..." -ForegroundColor Cyan
Write-Host ""

$frontendUrl = $env:FRONTEND_URL
$authUrl = $env:AUTH_URL
$chatbotUrl = $env:BACKEND_URL
if (-not $frontendUrl -or -not $authUrl -or -not $chatbotUrl) {
    throw "Missing required environment variables: FRONTEND_URL, AUTH_URL, BACKEND_URL"
}

# Check if running in the correct directory
if (-not (Test-Path "package.json")) {
    Write-Host "Error: Please run this script from the UniBuddy root directory" -ForegroundColor Red
    exit 1
}

Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host ""

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Green
npm install

# Install auth backend dependencies
Write-Host "Installing auth backend dependencies..." -ForegroundColor Green
Set-Location "backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend"
npm install
Set-Location "../../../.."

Write-Host ""
Write-Host "Dependencies installed!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting services..." -ForegroundColor Yellow
Write-Host ""

# Start Auth Backend (Node.js)
Write-Host "Starting Authentication Backend on Port 5000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend'; npm start"

# Wait a bit for auth backend to start
Start-Sleep -Seconds 3

# Start Chatbot Backend (Python)
Write-Host "Starting Chatbot Backend on Port 9000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'backend'; python api.py"

# Wait a bit for chatbot backend to start
Start-Sleep -Seconds 3

# Start Frontend (Vite)
Write-Host "Starting Frontend on Port 5173..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev"

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Yellow
Write-Host "   Frontend:        $frontendUrl" -ForegroundColor White
Write-Host "   Auth Backend:    $authUrl" -ForegroundColor White
Write-Host "   Chatbot Backend: $chatbotUrl" -ForegroundColor White
Write-Host ""
Write-Host "Tip: Close all PowerShell windows to stop all services" -ForegroundColor Cyan
Write-Host ""
