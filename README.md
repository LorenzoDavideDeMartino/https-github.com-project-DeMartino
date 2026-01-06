# Commodity Volatility Forecasting Under Armed Conflict

This project studies whether armed-conflict information contains predictive signals for commodity price volatility. Using daily data for major commodities (WTI crude oil futures, natural gas futures, and gold futures), the project compares a standard econometric baseline (HAR) with an augmented HAR-X specification that incorporates conflict intensity indices.

## Research question
Do armed-conflict intensity and geographic location contain predictive information that improves out-of-sample forecasts of commodity price volatility compared to a standard HAR benchmark?

## Hypotheses 

### H1 — Conflict Intensity and Volatility
- H1: The intensity of armed conflicts contains relevant information for the future dynamics of commodity price volatility.

In this project, conflict intensity is measured using the number of fatalities reported in the UCDP dataset (best variable). Because fatality counts are highly skewed and display strong persistence over time, they are transformed using a logarithmic function and smoothed through an exponentially weighted moving average (EWMA).

This hypothesis examines whether conflict intensity provides predictive power for commodity volatility independently of geographical location, relying on a global conflict index.

### H2 — Role of Geographic Exposure
- H2: The effect of armed conflicts on commodity price volatility is likely to depend on where these conflicts occur, particularly when they affect regions that are economically or strategically important for the commodity in question.

For this reason, the analysis focuses on specific geographic exposures:
- For WTI crude oil, conflicts located in the Middle East are expected to be most relevant.
- For natural gas, conflicts occurring in Europe are considered particularly important.
- For gold, which is commonly viewed as a safe-haven asset, global conflict intensity is used.

This hypothesis motivates the construction of region-specific and commodity-focused conflict indices rather than relying solely on global measures.

## Data
### Commodity prices (daily)
The analysis relies on daily settlement prices of futures contracts for the following commodities:
- WTI Crude Oil futures
- Natural Gas futures
- Gold futures

The data are obtained from Investing.com.

Raw price data are downloaded as CSV files using U.S. regional settings and are often split into multiple parts due to platform-imposed length restrictions. These raw files are intentionally left unmodified and are cleaned, parsed, and merged using a fully reproducible Python pipeline (`src/data_loader.py`). The final output consists of one clean daily price series per commodity.

### Conflict events (daily, georeferenced)
- UCDP Georeferenced Event Dataset (GED), version 25.1

The UCDP GED provides daily, georeferenced information on organized violence worldwide. Each observation corresponds to a violent event and includes information on the date, location, type of violence, and the estimated number of fatalities. Conflict intensity is proxied using the variable `best`, which represents the best available estimate of total fatalities for each event.

To ensure economic relevance and reduce noise, event-level data are filtered, aggregated to daily frequency, and transformed before entering the empirical analysis. Fatality counts are aggregated by day, transformed using a logarithmic function, and smoothed via an exponentially weighted moving average (EWMA). In addition to a global conflict index, region-specific and commodity-focused indices are constructed based on geographic exposure and key producer regions.

## Target variable

For each commodity, daily log-returns are first computed as

- r_t = log(P_t) − log(P_{t−1}),

where P_t denotes the daily futures price.

Following standard practice in the volatility forecasting literature, realized volatility is used as an ex post proxy for conditional variance. At each date t, realized volatility over a monthly horizon is defined as the sum of squared daily returns over a rolling window of 21 trading days,

- RV_t^{(21)} = ∑_{i=0}^{20} r_{t−i}².

This backward-looking measure captures recent market uncertainty and forms a key component of the baseline volatility dynamics.

The main forecast target of the study is future realized volatility over the next 21 trading days,

- RV_{t+1}^{(21)} = ∑_{i=1}^{21} r_{t+i}².

This quantity is not observable at time t and therefore provides a natural target for out-of-sample forecast evaluation. All models rely exclusively on information available up to date t when forming forecasts, ensuring that no look-ahead bias is introduced.

## Conflict indices and features

Armed-conflict information is incorporated through a set of daily conflict intensity indices constructed from the UCDP Georeferenced Event Dataset.

At the event level, conflict intensity is proxied by the number of fatalities reported for each event (variable `best`). Event-level observations are aggregated to the daily frequency in order to align conflict information with financial data. Because fatality counts are highly skewed and characterized by occasional extreme values, daily fatalities are transformed using a logarithmic transformation of the form

- log(1 + fatalities).

To capture persistence in geopolitical risk while emphasizing recent events, the transformed series are smoothed using an exponentially weighted moving average (EWMA) with a fixed decay parameter λ = 0.94.

Conflict indices are constructed at different levels of aggregation. In addition to a global conflict intensity index, region-specific and commodity-focused indices are built based on geographic exposure and the economic relevance of conflict locations for each commodity.

All conflict variables enter the empirical models in lagged form, ensuring that only information available prior to the forecast date is used. This strict lag structure prevents information leakage and guarantees a clean out-of-sample evaluation.

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
│   └── models/                  # Step 5: HAR, HAR-X
│
└── results/               # (Upcoming) Figures and forecast evaluation tables
```

## How to run the project

The entire project is executed through the main pipeline script.

After installing the required dependencies, running

python main.py

will reproduce the full workflow, including data cleaning, feature construction, conflict index building, dataset assembly, model estimation, and out-of-sample evaluation.

All results are generated deterministically from the raw input data.

## Requirements

The project is implemented in Python 3.11 and relies on the following libraries:

### Data manipulation
- pandas
- numpy

### Visualization
- matplotlib
- seaborn

### Econometrics and statistics
- statsmodels

### Utilities
- pathlib (standard library)


## Author
Lorenzo Davide De Martino  
