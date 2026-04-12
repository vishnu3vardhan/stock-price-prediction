from src.data_loader import fetch_all_stocks
from src.preprocessing import process_all_data
from src.model import train_all_models

if __name__ == "__main__":
    print("Downloading stock data...")
    fetch_all_stocks()

    print("\nProcessing data...")
    process_all_data()

    print("\nTraining models & generating visuals...")
    train_all_models()

    print("\n✅ Day 2 Complete (Visualization Ready!)")