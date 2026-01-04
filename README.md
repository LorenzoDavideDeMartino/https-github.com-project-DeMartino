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

r_t = log(P_t) − log(P_{t−1}),

where P_t denotes the daily futures price.

Following standard practice in the volatility forecasting literature, realized volatility is used as an ex post proxy for the conditional variance. At each date t, realized volatility over a monthly horizon is defined as the sum of squared daily returns over a rolling window of 21 trading days:

RV_t^{(21)} = ∑_{i=0}^{20} r_{t−i}².

This backward-looking measure captures recent market uncertainty and serves as a key explanatory variable in baseline volatility models.

The main forecast target of the study is future realized volatility over the next 21 trading days:

RV_{t+1}^{(21)} = ∑_{i=1}^{21} r_{t+i}².

This quantity is not observable at time t and is therefore suitable for out-of-sample forecast evaluation. All models use only information available up to date t to forecast future volatility, ensuring the absence of look-ahead bias.

## Conflict indices and features

Armed-conflict information is incorporated through a set of daily conflict indices constructed from the UCDP Georeferenced Event Dataset (GED). At the event level, conflict intensity is proxied by the variable best, which represents the best estimate of total fatalities associated with each event.

Conflict events are aggregated at the daily frequency to ensure consistency with the financial data. To reduce skewness and mitigate the influence of extreme observations, daily fatalities are transformed using a logarithmic transformation defined as log(1 + fatalities). These series are then smoothed using exponentially weighted moving averages (EWMA) with decay parameters λ = 0.94 and λ = 0.97, capturing persistence in geopolitical risk while assigning greater weight to recent events.

To ensure economic relevance, conflict indices are constructed at different levels of aggregation. These include region-specific indices (e.g. Middle East) as well as commodity-specific indices based on groups of key producing countries (e.g. oil focus and gas focus). All conflict variables are strictly lagged (t−1, t−5) before entering the models, ensuring that only information available prior to the forecast date is used and preventing any form of information leakage.

## Project Structure

The repository is organized to ensure reproducibility, clarity, and modularity.

```text
commodity-volatility-conflict/
├── README.md              # Project overview and instructions
├── PROPOSAL.md            # Project proposal
├── requirements.txt       # Python dependencies
├── main.py                # Orchestrator script (Runs the full ETL Pipeline)
│
├── data/
│   ├── raw/
│   │   ├── commodities/   # Raw CSV downloads from Investing.com (split in parts)
│   │   └── conflicts/     # Raw UCDP GED CSV (GEDEvent_v25_1.csv)
│   └── processed/
│       ├── commodities/   # Cleaned continuous price series
│       ├── features/      # Intermediate volatility features
│       ├── conflicts/     # Reduced and sorted conflict events
│       ├── indices/       # Daily Conflict Indices (EWMA, Regional, Focus Countries)
│       └── model_datasets/# FINAL DATASETS: Aligned Price + Volatility + Conflict Lags
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Step 1: Cleaning & merging raw commodity price data
│   ├── features.py              # Step 1b: Realized volatility (RV) calculation
│   ├── conflict_loader.py       # Step 2: Cleaning & reducing raw UCDP data
│   ├── conflict_index_builder.py # Step 3: Aggregating events into daily time-series (EWMA)
│   ├── build_model_dataset.py   # Step 4: Merging commodities with conflict indices (Lags/Target)
│   └── models/                  # (Upcoming) HAR, GARCH, and ML model definitions
│
└── results/               # (Upcoming) Figures and forecast evaluation tables

### Reproducibility

1. **Clean commodity prices**  
   `src/data_loader.py` → `data/processed/commodities/*_clean.csv`
2. **Reduce and sort UCDP GED** (event-level)  
   `src/conflict_loader.py` → `data/processed/conflicts/ucdp_ged_reduced_sorted.csv`
3. **Build daily conflict indices** (EWMA, regions, focus countries)  
   `src/conflict_index_builder.py` → `data/processed/indices/*.csv`
4. **Build final model datasets** (merge + RV target + lags)  
   `src/build_model_dataset.py` → `data/processed/model_datasets/*_dataset.csv`
5. **Model estimation and evaluation**  
   HAR / GARCH / ML with walk-forward evaluation