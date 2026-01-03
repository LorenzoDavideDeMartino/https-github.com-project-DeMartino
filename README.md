# Commodity Volatility Forecasting Under Armed Conflict

This project studies whether armed-conflict information contains predictive signals for commodity price volatility. Using daily data for major commodities (WTI crude oil futures, natural gas futures, gold futures), the project compares standard econometric benchmarks (e.g., GARCH) with machine-learning models that incorporate conflict-event characteristics.

## Research question
Do armed-conflict characteristics (timing, location, intensity) contain predictive information that improves out-of-sample forecasts of commodity price volatility compared to standard econometric benchmarks?

## Data
### Commodity prices (daily)
The analysis relies on daily settlement prices of futures contracts for the following commodities:
- WTI Crude Oil futures
- Natural Gas futures
- Gold futures
Source: Investing.com

Raw data are downloaded as CSV files (US settings), often split into multiple parts due to data-length restrictions. These raw files are intentionally kept unmodified and are cleaned and merged using a fully reproducible Python pipeline (src/data_loader.py).

### Conflict events (daily, georeferenced)
- UCDP Georeferenced Event Dataset (GED), version 25.1

The UCDP GED provides daily, georeferenced information on organized violence worldwide. Conflict intensity is proxied using the variable best, which represents the best estimate of total fatalities for each event.

To ensure economic relevance and reduce noise, conflict data are later filtered and aggregated according to commodity-specific exposur

## Target variable
For each commodity, daily log-returns are computed as:

- rt​=log(𝑃𝑡​)−log(𝑃𝑡−1​)

where 𝑃𝑡 denotes the daily futures price. 

Realized volatility is constructed as a proxy for the conditional variance using a rolling window of 21 trading days:

- RVt(21)​=i=0∑20​rt−i2​

This realized volatility measure is observed ex post and serves as the benchmark against which all volatility forecasts—both econometric and machine-learning based—are evaluated.

## Project Structure

The repository is organized to ensure reproducibility, clarity, and modularity.

```text
commodity-volatility-conflict/
├── README.md                  # Project overview and instructions
├── PROPOSAL.md                # Project proposal
├── main.py                    # Orchestrator script (ETL Pipeline)
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Cleaning & merging raw commodity price data
│   ├── conflict_loader.py     # Cleaning & reducing UCDP GED data
│   ├── features.py            # Returns and realized volatility construction
│   ├── models.py              # (Upcoming) Econometric and ML models
│   └── evaluation.py          # (Upcoming) Forecast evaluation
│
├── data/
│   ├── raw/
│   │   ├── commodities/       # Raw CSV downloads from Investing.com
│   │   └── conflicts/         # Raw UCDP GED CSV
│   └── processed/
│       ├── commodities/       # Cleaned commodity price series
│       ├── features/          # Volatility features
│       └── conflicts/         # Cleaned and aggregated conflict data
│
└── results/                   # (Upcoming) Figures and tables

### Reproducibility

All data cleaning and feature construction steps are implemented in Python and can be reproduced by running:

        python main.py

No manual data manipulation is required.
Random seeds are fixed where applicable to ensure reproducibility.