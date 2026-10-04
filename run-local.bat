@echo off
title AI Image Agent - Local Windows Server
echo ===================================================
echo   AI Image Agent - Local Windows Server
echo ===================================================
echo.

if not exist .env (
    if exist .env.example (
        echo [INFO] Creating .env from .env.example...
        copy .env.example .env
        echo [NOTICE] Please open .env and set your GEMINI_API_KEY.
        echo.
    )
)

echo Starting local agent server on port 3000...
echo Ensure ComfyUI is running at http://127.0.0.1:8188
echo.
npm run dev
pause
