# Commodity Volatility Forecasting Under Armed Conflict

This project studies whether armed-conflict information contains predictive signals for commodity price volatility. Using daily data for major commodities (WTI crude oil futures, natural gas futures, gold futures, and an agricultural commodity index), the project compares standard econometric benchmarks (e.g., GARCH) with machine-learning models that incorporate conflict-event characteristics.

## Research question
Do conflict characteristics (timing, location, intensity) improve out-of-sample forecasts of commodity volatility, especially around major conflict-related regime breaks?

## Data
### Commodity prices (daily)
- WTI Crude Oil futures
- Natural Gas futures
- Gold futures
- S&P GSCI Agriculture index (aggregated agricultural exposure)

Source: Investing.com (downloaded as raw CSV and cleaned with a reproducible Python script).

### Conflict events (daily, georeferenced)
- UCDP Georeferenced Event Dataset (GED) v25.1

Conflict intensity is proxied using the `best` fatalities estimate.

## Target variable
For each commodity, daily log-returns are computed:
\[
r_t = \log(P_t) - \log(P_{t-1})
\]

Realized volatility (proxy for variance) is constructed over a 21-day rolling window:
\[
RV_t^{(21)} = \sum_{i=0}^{20} r_{t-i}^2
\]

This realized volatility serves as the ex post benchmark against which forecasts are evaluated.

## Repository structure (planned)
