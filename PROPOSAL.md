# Project Proposal

## Motivation and Research Question

Commodity markets are highly exposed to geopolitical risk due to their dependence on geographically concentrated production regions. Armed conflicts can disrupt supply chains, increase uncertainty, and generate abrupt changes in price volatility. Accurately forecasting volatility during such periods is therefore crucial for investors, risk managers, and policymakers.

While existing literature documents that conflicts affect commodity prices and volatility, most studies rely on descriptive analyses or standard econometric models such as GARCH. These approaches assume relatively stable volatility dynamics and may struggle to capture nonlinear effects and sudden regime shifts associated with major geopolitical events.

This project investigates the following research question:

**Do armed-conflict characteristics contain predictive information that improves out-of-sample forecasts of commodity price volatility, particularly around conflict-related regime breaks?**

The goal is to assess whether machine-learning models that incorporate detailed conflict-event information outperform traditional econometric benchmarks in forecasting commodity volatility.

## Data

The analysis combines two main data sources at a daily frequency.

Commodity price data consist of futures prices for WTI crude oil, natural gas, and gold, as well as an aggregated agricultural commodity index (S&P GSCI Agriculture). Futures prices are used instead of spot prices due to their higher liquidity, longer and more continuous historical availability, and their central role in price discovery. Daily log-returns are computed from these prices.

Conflict-related information is drawn from the UCDP Georeferenced Event Dataset (GED). This dataset provides daily, geolocated records of organized violence, including the timing, location, type of violence, and estimated number of fatalities. The daily granularity of the GED allows for precise alignment with financial market data and enables the identification of volatility responses around conflict onsets.

The primary target variable is realized volatility, constructed as the sum of squared daily returns over a rolling 21-day window. This measure serves as an ex post proxy for market risk and as the benchmark against which volatility forecasts are evaluated.

## Methodology

The project follows a comparative forecasting framework. At each point in time, models are estimated using only information available up to that date and are used to predict future realized volatility.

First, standard benchmark models are implemented, including a naive random-walk volatility model and a GARCH(1,1) model estimated on daily returns. These models serve as reference points commonly used in the volatility forecasting literature.

Second, machine-learning models such as Random Forests and Gradient Boosting are trained using both traditional market-based predictors (lagged returns and lagged volatility measures) and conflict-related variables derived from the UCDP GED. The flexibility of these models allows them to capture nonlinear relationships and interactions between geopolitical events and volatility dynamics.

Model performance is evaluated strictly out-of-sample using a walk-forward validation procedure. Forecast accuracy is assessed using loss functions appropriate for variance forecasting, including the QLIKE loss and mean squared error. Statistical significance of performance differences is assessed using Diebold–Mariano tests.

## Expected Contribution

This project aims to contribute to the literature by providing a rigorous, out-of-sample comparison between econometric and machine-learning approaches to volatility forecasting in the presence of armed conflicts. By exploiting high-frequency conflict-event data, the study seeks to clarify whether geopolitical information delivers incremental predictive power beyond standard market indicators, particularly during periods of heightened instability.

