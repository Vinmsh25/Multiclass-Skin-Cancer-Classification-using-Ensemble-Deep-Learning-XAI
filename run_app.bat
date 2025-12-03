@echo off
echo Setting up Skin Cancer Classification App...
echo.

REM Check if requirements are already installed
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error installing packages. Please check your internet connection and try again.
        pause
        exit /b 1
    )
    echo Packages installed successfully!
) else (
    echo Required packages already installed.
)

echo.
echo Starting Streamlit application...
echo.

REM Use the full path to the Streamlit executable
"C:\Users\sudha\AppData\Roaming\Python\Python313\Scripts\streamlit.exe" run skin_cancer_app.py

pause