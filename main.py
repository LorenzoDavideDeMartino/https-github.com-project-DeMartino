from pathlib import Path
from src.data_loader import load_commodity_prices

def main():
    print("Commodity Volatility Under Armed Conflict")
    print("Project setup OK.")

    # Example path (will be added later)
    example_file = Path("data/processed/example_prices.csv")
    if example_file.exists():
        df = load_commodity_prices(example_file)
        print("Loaded:", example_file)
        print(df.head())
    else:
        print("No data file found yet at:", example_file)

if __name__ == "__main__":
    main()
