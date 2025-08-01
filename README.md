# AutoEDA: Automated Exploratory Data Analysis & AutoML Suite

## Overview
AutoEDA is a professional, user-friendly Streamlit application for automated data exploration, preprocessing, and machine learning (AutoML). It is designed for data scientists, analysts, and ML practitioners who want to:
- Quickly explore and visualize datasets
- Effortlessly preprocess data
- Automatically train, evaluate, and compare ML models with FLAML
- Interpret results with explainability and error analysis tools

---

## Features

### 🚀 Automated EDA
- Dataset summary, missing values, data types
- Interactive visualizations (scatter, pairplot, heatmap, etc.)
- Categorical & numerical feature exploration

### 🛠️ Data Preprocessing
- Remove or fill missing values (mean, median, mode)
- Scale features (Standard, MinMax)
- Outlier detection (z-score)
- Column selection/removal

### 🤖 AutoML (FLAML)
- Supports classification & regression
- Rich model coverage: LightGBM, XGBoost, Random Forest, Extra Trees, HistGB, SVC, KNeighbors, L1/L2-regularized models, CatBoost, etc.
- Smart target/feature selection
- Cross-validation, hyperparameter tuning
- Model leaderboard
- Download predictions as CSV
- Download trained AutoML object (pickle)

### 📊 Results & Explainability
- Performance metrics (Accuracy, F1, MAE, R2, etc.) in a clear table
- Confusion matrix, ROC curve, regression plots
- SHAP explainability with summary plots
- Feature importance
- Error analysis: top misclassified samples/largest errors

### 🖥️ Professional UI/UX
- Modern Streamlit interface
- Sidebar navigation
- Tooltips, badges, and user feedback
- Robust error handling

---

## Folder Structure

```
Auto_EDA/
├── .Streamlit/                 # Streamlit config (optional)
├── __pycache__/                # Python cache files (auto-generated)
├── automl_functions.py         # All AutoML, evaluation, explainability logic
├── data_analysis_functions.py  # EDA and visualization helpers
├── data_preprocessing_function.py # Data cleaning, scaling, outlier, etc.
├── example_dataset/
│   └── titanic.csv             # Example dataset for demo/testing
├── home_page.py                # Home tab and landing page UI
├── logs.log                    # App logs
├── main.py                     # Main Streamlit app (UI, navigation, orchestration)
├── requirements.txt            # All Python dependencies
├── test_automl.py              # Test script for AutoML module
├── README.md                   # (You are here)
├── venv/                       # Python virtual environment (optional)
└── .gitignore                  # Git ignore rules
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Auto_EDA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run main.py
```

### 4. Upload your dataset or use the example Titanic dataset

---

## Usage Guide

- **Home**: Overview and quick start
- **Data Exploration**: Visualize and understand your data
- **Data Preprocessing**: Clean, scale, and prepare your data
- **AutoML**: Select problem type, target, features, run AutoML, and review results

---

## Requirements
- Python 3.8–3.11 (FLAML may not support 3.12+)
- See `requirements.txt` for all packages

---

## Notes
- SHAP explainability works best with tree-based models (LightGBM, XGBoost, RF, etc.)
- For reproducibility, set random seeds in your data and model configs
- Downloaded models are pickled FLAML AutoML objects

---

## Contributing
Pull requests and issues are welcome! Please open an issue for bugs or feature requests.

---

## License
[MIT License](LICENSE) (add your license file if needed)

---

## Author
- [Your Name/Handle]
- [LinkedIn/GitHub]

---

## Acknowledgements
- [FLAML](https://github.com/microsoft/FLAML)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://github.com/slundberg/shap)
- [Plotly](https://plotly.com/python/)
