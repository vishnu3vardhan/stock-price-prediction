# NSE(National Stock Exchange) Stock Price Prediction using Machine Learning

## Project Overview

- Stock price prediction is a challenging task that aims to forecast market trends, helping investors and analysts make informed decisions.

- This project predicts the **next day's stock closing price** using historical **NSE(National stock exchange)** stock market data from the past 10 years (2016–2026).

- It combines **Machine Learning** with **data visualization** and an **interactive dashboard** to analyze stock trends and generate predictions based on technical indicators.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4eb713f-ed03-4fbb-878b-8b3fe8e461f6" width="700"/>
</p>

---

## Objective

To build an end-to-end Machine Learning pipeline that:

* Processes historical stock data
* Trains predictive models
* Evaluates performance
* Provides an interactive dashboard for visualization

---

## Key Features

* Predicts next-day stock prices
* Uses technical indicators (moving averages, returns)
* Machine Learning model (Random Forest)
* Performance evaluation using MAE & RMSE
* Interactive dashboard built with Streamlit & Plotly
* Automated data pipeline (download → preprocess → train → visualize)

---

## Dataset

* **Source:** Yahoo Finance (via `yfinance`)
* **Stocks:** Top companies from NIFTY 50.
* **Time Range:** January 2016 – January 2026

### Data includes:

* Open
* High
* Low
* Close
* Volume

---

## Features Used

* Previous Day Close (`Prev_Close`)
* 5-day Moving Average (`MA5`)
* 10-day Moving Average (`MA10`)
* 50-day Moving Average (`MA50`)
* Daily Returns

---

## Model Used

* **Random Forest Regressor**

Chosen for:

* Good performance on tabular data
* Ability to capture non-linear relationships
* Minimal tuning required

---

## Evaluation Metrics

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

---

## Results

The model successfully captures **general stock price trends** and patterns.

However:

* Stock markets are highly volatile
* Predictions are approximate, not exact

---

## Interactive Dashboard

This project includes a **Streamlit + Plotly dashboard** for visualization.

### Features:

*  Select stock from dropdown
*  View Actual vs Predicted prices
*  Interactive Plotly charts
*  Performance metrics (MAE, RMSE)
*  Trend insights (Uptrend / Downtrend)

### Run Dashboard:

```bash
streamlit run app.py
```

## Dashboard Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/c7e12b09-4560-48ae-8c76-48f226a5ff35" width="45%"/>
  <img src="https://github.com/user-attachments/assets/67376aae-cd78-432a-a625-c0bf961d9925" width="45%"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6a5cdef7-c78f-46fc-ba21-af5932965010" width="45%"/>
  <img src="https://github.com/user-attachments/assets/8f2f9b70-fc17-4bbe-a330-0877be92a135" width="45%"/>
</p>

<details>
<summary>More Screenshots</summary>

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7ff0309c-a677-477d-a3a3-9298d6c66c66" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d5bbf1a8-2171-4e13-b3c2-ff304d4405ab" width="600"/>
</p>

</details>

---

## Project Structure

```
stock-price-prediction/
│
├── data/                # Ignored (auto-generated)
├── outputs/
│   └── predictions/     # Used by dashboard
│
├── notebooks/
│     └── reliance_analysis.ipynb 
│     └── wipro_stock_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
├── app.py               # Streamlit dashboard
├── main.py              # Pipeline execution
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run ML pipeline

```bash
python main.py
```

This will:

* Download stock data
* Process data
* Train models
* Generate predictions

---

## Data Handling

* Raw data is fetched dynamically using `yfinance`
* Only prediction outputs are stored for dashboard use
* Large datasets are excluded from GitHub for efficiency

---

## Output

* Prediction CSV files
* Actual vs Predicted graphs
* Interactive dashboard

---

## Disclaimer

Stock price prediction is inherently uncertain and influenced by many external factors such as market sentiment, news, and global events.

This project is for **educational purposes only** and should not be used for financial decisions.

---

## Author

[**Vishnu Vardhan**](https://github.com/vishnu3vardhan)
---
