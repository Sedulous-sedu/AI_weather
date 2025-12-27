# AURAK Shuttle Arrival Predictor

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

A professional, industrial-grade desktop application for predicting shuttle bus arrival times at AURAK University. Leveraging machine learning and a modern, dark-themed GUI, this tool provides real-time predictions, delay analysis, and interactive simulations.

## 🚀 Features

- **Advanced Machine Learning**: Predicts "On-Time" vs. "Late" arrivals using Logistic Regression, Decision Trees, and more.
- **Modern UI/UX**: Built with CustomTkinter for a sleek, responsive, and dark-themed interface.
- **Live Simulation**: Visualizes shuttle movement with smooth animations and environmental effects.
- **Real-Time Analytics**: View classification reports, confusion matrices, and feature importance charts.
- **Interactive Predictor**: "What-If" scenarios with adjustable parameters (weather, traffic, time, etc.).
- **Data Management**: Support for loading large datasets (CSV) and generating mock data for testing.

## 📂 Project Structure

The project is organized as follows:

```
AI_weather/
├── assets/                 # Data and media resources
│   ├── sounds/             # Audio files for alerts
│   └── aurak_shuttle_data_full.csv  # Default dataset
├── src/                    # Source code package
│   ├── __init__.py
│   ├── config.py           # Configuration and constants
│   ├── engine.py           # ML logic and data processing
│   ├── gui.py              # Main GUI implementation
│   ├── train_cli.py        # CLI for model training
│   └── utils.py            # Utility functions
├── tests/                  # Unit tests
├── main.py                 # Application entry point
├── build.spec              # PyInstaller build specification
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/aurak-shuttle-predictor.git
    cd aurak-shuttle-predictor
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Running the Application

To start the main application:

```bash
python main.py
```

### Using the Interface

1.  **Data & Training**:
    - Go to the "Control Panel" on the left.
    - Click **Load CSV Data** to load `assets/aurak_shuttle_data_full.csv`.
    - Select a model (e.g., Logistic Regression) and click **Train Model**.

2.  **Live Prediction**:
    - Adjust parameters like Route, Time, Traffic, and Weather.
    - Click **Predict Arrival**.
    - Watch the **Live Simulation** on the top right.

3.  **Analytics**:
    - Check the **Analytics Dashboard** on the bottom right for performance metrics and logs.

### Command Line Interface

You can also train models via the CLI:

```bash
python src/train_cli.py assets/aurak_shuttle_data_full.csv --model logreg
```

## ⚙️ Configuration

Application settings (window size, colors, paths) can be modified in `src/config.py`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
