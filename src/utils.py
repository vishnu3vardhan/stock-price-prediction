import matplotlib.pyplot as plt
import os


def plot_predictions(dates, actual, predicted, stock_name, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual Price")
    plt.plot(dates, predicted, label="Predicted Price")

    plt.title(f"{stock_name} - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    file_path = os.path.join(save_dir, f"{stock_name}.png")
    plt.savefig(file_path)
    plt.close()

    print(f"📈 Saved plot: {file_path}")