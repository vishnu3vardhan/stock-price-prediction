# 📈 Stock Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the next day's stock closing price using historical stock data from the past 10 years (2016–2026).

It uses Machine Learning techniques to analyze patterns and trends in stock prices and generate predictions based on technical indicators.

---

## 🎯 Objective

To build a machine learning model that predicts the next day's closing price of stocks using historical data.

---

## 📊 Dataset

* Source: Yahoo Finance (via yfinance)
* Stocks: Top 30 NIFTY 50 companies
* Time Range: January 2016 – January 2026
* Data Includes:

  * Open
  * High
  * Low
  * Close
  * Volume

---

## ⚙️ Features Used

* Previous Day Close (Prev_Close)
* 5-day Moving Average (MA5)
* 10-day Moving Average (MA10)
* 50-day Moving Average (MA50)
* Daily Returns

---

## 🤖 Model Used

* Random Forest Regressor

---

## 🧪 Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## 📈 Results

The model is able to capture general stock price trends and patterns. However, due to the volatile and unpredictable nature of the stock market, it cannot predict exact prices.

---

## 📂 Project Structure

```
stock-price-prediction/
│
├── data/                # Ignored (auto-generated)
├── outputs/             # Predictions and graphs
│   ├── predictions/
│   └── *.png
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python main.py
```

---

## 📥 Data Collection

Stock data is automatically downloaded using the `yfinance` library.

---

## 📊 Output

* CSV files containing predictions
* Graphs comparing Actual vs Predicted prices

---

## ⚠️ Disclaimer

Stock price prediction is inherently uncertain and influenced by many external factors. This project is for educational purposes only and should not be used for financial decisions.

---

## 👨‍💻 Author

**Vishnu Vardhan**
