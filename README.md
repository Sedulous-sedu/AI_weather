# Intelligent Bus Arrival Prediction System

[![Paper](https://img.shields.io/badge/Paper-ICERAI%202026-blue.svg)](./ICERAI_paper2.pdf)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/Sedulous-sedu/AI_weather)

> **Research Implementation**: Machine learning-based bus arrival prediction system for AURAK campus shuttle service and RAKTA public transport. Presented at ICERAI 2026.

---

## 📖 Overview

This repository contains the **official implementation** of the intelligent bus arrival prediction system described in our ICERAI 2026 conference paper. The system leverages supervised machine learning to predict bus arrival delays (Late vs. On-Time) using multiple environmental, temporal, and operational features.

### Key Contributions

- **Dual Dataset Analysis**: Comparative evaluation on both campus shuttle (AURAK) and public transport (RAKTA) datasets
- **Feature Engineering**: 11+ contextual features including weather, traffic, time-of-day, and route characteristics
- **Model Comparison**: Systematic evaluation of Logistic Regression and XGBoost classifiers
- **Real-Time GUI**: Production-ready desktop application with interactive prediction and visualization
- **Reproducible Research**: Complete pipeline from data generation to model deployment

### Performance Highlights

| Model | Accuracy | Macro-F1 | Balanced Acc | F1 (Late) | AUPRC |
|-------|----------|----------|--------------|-----------|-------|
| **XGBoost** | **94.2%** | **0.921** | **0.953** | **0.961** | **0.998** |
| Logistic Regression | 84.7% | 0.807 | 0.864 | 0.895 | 0.987 |
| Majority Baseline | 78.0% | 0.438 | 0.500 | 0.877 | 0.780 |

*Evaluated on RAKTA dataset (97,676 trips after deduplication, stratified group split)*

---

## 🏗️ Repository Structure

```
AI_weather/
├── src/                          # Core application source
│   ├── engine.py                 # ML pipeline & prediction engine
│   ├── gui.py                    # CustomTkinter GUI application
│   ├── config.py                 # Configuration & hyperparameters
│   ├── train_cli.py              # CLI for batch training
│   └── utils.py                  # Utility functions
├── assets/                       # Data resources
│   ├── sounds/                   # Audio alerts
│   └── aurak_shuttle_data_full.csv
├── paper_assets/                 # Research artifacts
│   ├── LR_Coefficients.png       # Feature importance (LR)
│   ├── SHAP_Summary.png          # SHAP analysis (XGBoost)
│   ├── PR_Curve_*.png            # Precision-Recall curves
│   └── model_summary.json        # Experimental results
├── figures/                      # Generated visualizations
├── data_curator.py               # Synthetic data generation
├── scraper.py                    # RAKTA web scraping utility
├── make_figures.py               # Reproduce paper figures
├── make_lr_plots.py              # Logistic Regression analysis
├── main.py                       # GUI entry point
├── requirements.txt              # Python dependencies
├── ICERAI_paper2.pdf             # Full research paper
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested on 3.8–3.14)
- 8 GB RAM (recommended for XGBoost training on large datasets)
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sedulous-sedu/AI_weather.git
   cd AI_weather
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the GUI Application

```bash
python main.py
```

**Application Features**:
- 📊 **Dashboard**: Real-time metrics, classification reports, confusion matrices
- 🔮 **Interactive Predictor**: Adjust route, time, weather, traffic conditions
- 🚍 **Live Simulation**: Animated bus movement with environmental effects
- 📈 **Performance Analytics**: Model comparison, feature importance (SHAP)
- 💾 **Data Management**: Load CSV datasets, train multiple models

---

## 🔬 Reproducing Research Results

### 1. Generate Synthetic Dataset (Optional)

To create the AURAK shuttle dataset used in the paper:

```bash
python data_curator.py
```

This generates `aurak_shuttle_data_full.csv` (100k trips) with realistic distributions.

### 2. Reproduce Paper Figures

Run the complete analysis pipeline:

```bash
python make_figures.py
```

**Outputs** (saved to `figures/`):
- `LR_Coefficients.png`: Top 25 logistic regression coefficients
- `SHAP_Summary.png`: SHAP feature importance for XGBoost
- `PR_Curve_XGB.png`: Precision-Recall curve with no-skill baseline

**Console Output**:
- Classification metrics (accuracy, F1, balanced accuracy)
- LaTeX table rows (ready for paper insertion)
- Inference latency benchmarks (ms/query)

### 3. Train Models via CLI

For batch experiments:

```bash
python src/train_cli.py assets/aurak_shuttle_data_full.csv --model logreg
python src/train_cli.py rakta_trips_large.csv --model xgboost
```

---

## 📊 Datasets

### AURAK Shuttle Dataset
- **Source**: Synthetic data generated from real timetables
- **Size**: 100,000 trips (full) / 250k (extended)
- **Features**: Route, time, weather, traffic, special events
- **Label**: Late (>5 min delay) vs. On-Time

### RAKTA Public Transport Dataset
- **Source**: Web scraping (see `scraper.py`)
- **Size**: 97,676 unique trips (after deduplication)
- **Features**: Similar to AURAK + direction, destination
- **Label**: Late (>5 min delay) vs. On-Time
- **Split**: Stratified group split (prevents timetable leakage)

---

## 🧪 Methodology

### Feature Engineering

**11 Core Features**:
- **Temporal**: `day_of_week`, `time_of_day` (Morning/Afternoon/Evening/Night)
- **Route**: `route`, `direction`, `trip_no`, `destination`
- **Operational**: `stop_distance_km`
- **Environmental**: `weather` (Clear/Rainy/Foggy), `traffic_condition` (Light/Moderate/Heavy)
- **Contextual**: `special_event` (NoEvent/Festival/Sports/Holiday), `temperature_celsius`

### Preprocessing Pipeline

```python
ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(with_mean=False), numerical_features),
])
```

### Models

1. **Logistic Regression** (`max_iter=3000`, `class_weight="balanced"`)
   - Interpretable baseline
   - L2 regularization (default)

2. **XGBoost Classifier**
   - `n_estimators=400`, `learning_rate=0.05`, `max_depth=6`
   - `subsample=0.8`, `colsample_bytree=0.8`
   - `scale_pos_weight` set to class imbalance ratio

### Evaluation Metrics

- **Balanced Accuracy**: Accounts for class imbalance (78% Late on RAKTA)
- **Macro-F1**: Equal weight to On-Time and Late classes
- **F1-Late**: Performance on minority class (On-Time arrivals)
- **AUPRC**: Area Under Precision-Recall Curve (more informative than AUC-ROC for imbalanced data)

### Data Leakage Prevention

- **Deduplication**: Removed exact duplicate (features + label) rows
- **Group Split**: Prevents identical timetable templates in train/test (uses `StratifiedGroupKFold`)
- **No Target Leakage**: `delay_minutes` excluded from features (used only for label generation)

---

## 📈 Key Findings

### Feature Importance (SHAP Analysis)

**Top Predictors of Late Arrivals** (XGBoost on RAKTA):
1. **Traffic Condition** (Heavy traffic → +40% late probability)
2. **Time of Day** (Evening rush hour → +25%)
3. **Route** (Long-distance routes → +18%)
4. **Weather** (Rainy conditions → +12%)
5. **Special Events** (Festivals/Sports → +10%)

### Inference Performance

| Model | Median Latency | P90 Latency | Memory (RSS) |
|-------|----------------|-------------|--------------|
| Logistic Regression | **3.2 ms** | 4.5 ms | 411 MB |
| XGBoost | **5.8 ms** | 8.2 ms | 340 MB |

*Measured on single-query prediction (batch_size=1) for real-time GUI use*

---

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
# Window settings
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

# Model hyperparameters
RANDOM_STATE = 42
LATE_THRESHOLD_MINUTES = 5.0

# Paths
DEFAULT_DATA_PATH = "assets/aurak_shuttle_data_full.csv"
MODEL_SAVE_PATH = "model1.joblib"
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{author2026bus,
  title={Intelligent Bus Arrival Prediction Using Machine Learning: A Comparative Study},
  author={[Your Name] and [Co-authors]},
  booktitle={Proceedings of the International Conference on Engineering Research and Applications (ICERAI)},
  year={2026},
  organization={IEEE}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

**Areas for Contribution**:
- Additional ML models (Random Forest, LightGBM, Neural Networks)
- Real-time data integration (live GPS feeds)
- Mobile app development
- Multi-city deployment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AURAK University** for providing campus shuttle timetables
- **RAKTA** for public transport schedule data
- **Scikit-Learn**, **XGBoost**, and **SHAP** communities for excellent ML tools
- **CustomTkinter** for modern Python GUI framework

---

## 📧 Contact

**Principal Investigator**: [Your Name]  
**Email**: [your.email@aurak.ac.ae]  
**GitHub**: [@Sedulous-sedu](https://github.com/Sedulous-sedu)

**Research Supervisor**: Dr. Khaled Abdulgader  
**Collaborator**: Haseeb [Last Name]

---

## 🔗 Related Resources

- [ICERAI 2026 Conference](https://icerai.org)
- [Project Repository](https://github.com/Sedulous-sedu/AI_weather)
- [Research Paper](./ICERAI_paper2.pdf)
- [Live Demo](https://youtu.be/demo-video) *(if available)*

---

<div align="center">
  <strong>Built with ❤️ for smarter public transportation</strong>
</div>
