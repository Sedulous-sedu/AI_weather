#!/bin/bash
# AURAK Shuttle Predictor Startup Script

echo "🚌 Starting AURAK Shuttle Arrival Predictor..."
echo "=================================================="

# Check if Tkinter is available
echo "Checking Tkinter support..."
python3 -c "import tkinter" 2>/dev/null || {
    echo "❌ Tkinter not available. Installing python-tk..."
    brew install python-tk
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "📚 Installing dependencies..."
pip install -r requirements.txt --quiet

# Test CustomTkinter
echo "🧪 Testing CustomTkinter..."
python3 -c "import customtkinter; print('✅ CustomTkinter ready!')" || {
    echo "❌ CustomTkinter test failed. Please check installation."
    exit 1
}

# Run the application
echo "🚀 Launching AURAK Shuttle Predictor..."
python3 aurak_shuttle_predictor.py
