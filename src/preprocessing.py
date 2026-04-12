import pandas as pd
import os


def create_features(df):
    # Fix columns
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Keep only needed columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Convert to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    # Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Return'] = df['Close'].pct_change()

    # Target
    df['Target'] = df['Close'].shift(-1)

    df.dropna(inplace=True)

    return df


def process_all_data(input_dir="data", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            path = os.path.join(input_dir, file)

            print(f"Processing {file}...")

            try:
                df = pd.read_csv(path)
                df = create_features(df)

                output_path = os.path.join(output_dir, file)
                df.to_csv(output_path, index=False)

                print(f"✅ Processed: {file}")

            except Exception as e:
                print(f"❌ Skipped {file}: {e}")