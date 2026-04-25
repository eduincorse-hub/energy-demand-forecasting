# ⚡ Energy Demand Forecasting

Forecasting of electrical energy demand using classical and deep learning models.

## 🎯 Problem
Predict daily electrical energy consumption to help utilities optimize generation and distribution planning.

## 📊 Dataset
- **Source:** PJM Hourly Energy Consumption (Kaggle)
- **Period:** 2004 - 2018
- **Frequency:** Hourly → resampled to Daily
- **Size:** 121,273 records

## 🤖 Models Used

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| SARIMA | - MW | - MW | -% |
| LSTM | - MW | - MW | -% |

*(Fill with your actual results)*

## 🛠️ Tech Stack
- Python 3.x
- Pandas, NumPy, Matplotlib
- Statsmodels (SARIMA)
- TensorFlow/Keras (LSTM)
- Streamlit (Web App)

## 📁 Project Structure
energy-demand-forecasting/
│
├── data/
│   └── AEP_hourly.csv
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_sarima_model.ipynb
│   └── 03_lstm_model.ipynb
├── app.py
├── requirements.txt
└── README.md
## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📈 Results
The LSTM model captures non-linear patterns in energy consumption,
while SARIMA provides interpretable seasonal decomposition.
Both models achieve competitive accuracy for industrial forecasting.

## 👤 Author
**Edwin** - Data Analyst & Electrical Engineer
