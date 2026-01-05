# Commodity Volatility Forecasting Under Armed Conflict

This project studies whether armed-conflict information contains predictive signals for commodity price volatility. Using daily data for major commodities (WTI crude oil futures, natural gas futures, gold futures), the project compares standard econometric benchmarks (e.g., GARCH) with machine-learning models that incorporate conflict-event characteristics.

## Research question
Do armed-conflict characteristics (timing, location, and intensity) contain predictive information that improves out-of-sample forecasts of commodity price volatility compared to standard econometric benchmarks?

## Hypotheses 

### H1 — Conflict Intensity and Volatility
- H1: The intensity of armed conflicts contains relevant information for the future dynamics of commodity price volatility.

Conflict intensity is proxied by the number of fatalities reported in the UCDP dataset (best variable). To reduce skewness and capture persistence, fatalities are transformed using a logarithmic transformation and smoothed via an exponentially weighted moving average (EWMA).

This hypothesis tests whether conflict intensity has predictive power independently of geographical location, using global conflict indices.

### H2 — Role of Geographic Exposure
- H2: The impact of conflict intensity on commodity volatility depends on the geographical location of the conflict and is stronger when conflicts affect regions that are economically relevant for the production or strategic importance of the commodity.

Specifically:
- WTI crude oil: conflicts in the Middle East
- Natural gas: conflicts in Europe
- Gold: global conflicts (safe-haven asset)

This hypothesis motivates the use of region-specific conflict indices rather than purely global measures for certain commodities.

### H3 — Structural Breaks and Model Adaptability
- H3: Major armed conflicts represent structural breaks in the volatility process, and augmented or more flexible models are better able to capture these changes than standard linear benchmarks.

This hypothesis motivates:
- the comparison between HAR and HAR-X models (econometric framework),
- and, in a second step, the use of more complex or non-linear models.

The objective is not to maximize predictive performance mechanically, but to assess whether richer models better adapt to volatility dynamics following large geopolitical shocks.

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

Armed-conflict information is incorporated through a set of daily conflict indices constructed from the UCDP GED.

At the event level, conflict intensity is proxied by the number of fatalities (best). Events are aggregated at the daily frequency to align with financial data. To mitigate skewness and extreme observations, daily fatalities are transformed as:

log(1+fatalities)

These daily series are smoothed using exponentially weighted moving averages (EWMA) with decay parameters 𝜆=0.94 and 𝜆=0.97, capturing persistence in geopolitical risk while emphasizing recent events.

Conflict indices are constructed at different aggregation levels:

- Region-specific indices (e.g. Middle East, Europe) based on key producing regions (e.g. oil focus, gas focus)

All conflict variables enter the models with strict lags (e.g. 𝑡−1, 𝑡−5) to prevent information leakage.

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