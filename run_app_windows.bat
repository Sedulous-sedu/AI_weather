@echo off
echo Starting AURAK Shuttle Arrival Predictor...
echo ==================================================

if not exist "win_venv" (
    echo Creating virtual environment...
    py -m venv win_venv
    call win_venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call win_venv\Scripts\activate
)

echo Launching application...
python main.py
pause
