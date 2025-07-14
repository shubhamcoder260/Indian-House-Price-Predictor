# 🏠 Indian House Price Predictor

A complete end-to-end machine learning project that predicts house prices based on various property features. Built with a clean modular structure, this project includes data cleaning, model training, and an interactive Streamlit web app.

---

📁 Project Structure
```plaintext
Indian-House-Price-Predictor/
│
├── data/
│   ├── House Price India.csv
│   └── house_price_india_cleaned.csv
│
├── models/
│   ├── house_price_model.pkl
│   ├── input_columns.pkl
│   ├── pca.pkl
│   └── scaler.pkl
│
├── app.py                   # Streamlit web application
├── house_pred.ipynb         # Model usage and testing
├── house_price_prediction.ipynb  # EDA and model building notebook
├── .gitignore
└── README.md
```
## 🚀 Features

- 📊 Data Cleaning & Feature Engineering
- 🔍 PCA for dimensionality reduction
- 🧠 ML model trained on Indian housing data
- 🌐 Streamlit interface for real-time prediction

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Regression, PCA)
- Streamlit
- Pickle

---

## ▶️ How to Run

```bash
streamlit run app.py
