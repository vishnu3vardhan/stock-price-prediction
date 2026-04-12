import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from src.utils import plot_predictions


def train_and_evaluate(file_path):
    df = pd.read_csv(file_path)

    features = ['Prev_Close', 'MA5', 'MA10', 'MA50', 'Return']
    target = 'Target'

    X = df[features]
    y = df[target]

    # Time-based split
    split_index = int(len(df) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    dates = df['Date'][split_index:]

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    stock_name = os.path.basename(file_path).replace(".csv", "")

    print(f"\n📊 {stock_name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Save predictions CSV
    results = pd.DataFrame({
        "Date": dates,
        "Actual": y_test.values,
        "Predicted": predictions
    })

    os.makedirs("outputs/predictions", exist_ok=True)
    results.to_csv(f"outputs/predictions/{stock_name}.csv", index=False)

    # Plot graph
    plot_predictions(dates, y_test.values, predictions, stock_name)

    return model


def train_all_models(data_dir="data/processed"):
    models = {}

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            path = os.path.join(data_dir, file)

            try:
                model = train_and_evaluate(path)
                models[file] = model

            except Exception as e:
                print(f"❌ Error with {file}: {e}")

    return models