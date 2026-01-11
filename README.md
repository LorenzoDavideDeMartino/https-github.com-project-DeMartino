# Commodity Volatility Forecasting Under Armed Conflict
This project examines whether information on armed conflicts helps predict commodity price volatility. It relies on daily data for WTI crude oil, natural gas, and gold futures. The analysis starts from a standard HAR model and extends it with a HAR-X specification that includes conflict intensity indices built from UCDP data. The project also compares these econometric models to a machine learning benchmark.

## Research question
The research question examines whether the intensity and geographic location of armed conflicts contain predictive information that improves out-of-sample forecasts of commodity price volatility relative to a standard HAR benchmark, and whether these effects remain competitive when compared to a simple machine learning reference model.

## Hypotheses 

### H1 — Conflict Intensity and Volatility
- H1: The first hypothesis is that the intensity of armed conflicts contains relevant information for the future dynamics of commodity price volatility.

In this project, conflict intensity is measured using the number of fatalities reported in the UCDP dataset, based on the best estimate variable. Fatality counts are highly skewed and display strong persistence over time. For this reason, they are transformed using a logarithmic function and smoothed with an exponentially weighted moving average.

This hypothesis tests whether conflict intensity provides predictive information for commodity volatility independently of geographic location. The analysis therefore relies on a global conflict index as a baseline measure.

### H2 — Role of Geographic Exposure
- H2: The second hypothesis states that the impact of armed conflicts on commodity price volatility depends on where these conflicts occur. The effect is expected to be stronger when conflicts take place in regions that are economically or strategically important for the commodity considered.

The empirical analysis therefore focuses on commodity-specific geographic exposures. For WTI crude oil, conflicts located in the Middle East are considered the most relevant. For natural gas, conflicts occurring in Europe receive particular attention. For gold, which is commonly viewed as a safe-haven asset, global conflict intensity is used as the primary measure.

This hypothesis motivates the construction of region-specific conflict indices rather than relying exclusively on global measures, and it guides both the econometric and machine learning comparisons conducted in the project.

### H3 — Non-linear effects and model comparison
- H3: The third hypothesis states that the relationship between conflict intensity and commodity price volatility may be non-linear and therefore not fully captured by linear econometric models such as HAR or HAR-X.

While HAR-X allows conflict information to enter the model in a linear way, it imposes strong functional form assumptions. Machine learning models, by contrast, can capture non-linear effects and interactions between volatility dynamics and conflict variables without imposing a predefined structure.

This hypothesis tests whether a machine learning benchmark, using the same information set as HAR-X, delivers superior out-of-sample volatility forecasts. The comparison focuses on predictive performance rather than interpretability, and serves as a robustness check for the linear HAR-X results.

## Data
### Commodity prices (daily)
The analysis relies on daily settlement prices of futures contracts for WTI crude oil, natural gas, and gold. The data are sourced from Investing.com.

Raw price series are downloaded as CSV files using U.S. regional settings. Due to platform length restrictions, the data are often split into several parts. These raw files are deliberately kept unchanged.

All cleaning, parsing, and merging steps are handled by a fully reproducible Python pipeline implemented in (`src/data_loader.py`). The process produces one clean daily price series per commodity, which is then used consistently across the econometric and machine learning analyses.

### Conflict events (daily, georeferenced)
- UCDP Georeferenced Event Dataset (GED), version 25.1

The conflict data are drawn from the UCDP Georeferenced Event Dataset version 25.1. This dataset provides detailed, daily, and georeferenced information on organized violence worldwide. Each observation corresponds to a single violent event and reports the date, geographic location, type of violence, and an estimate of the number of fatalities. Conflict intensity is proxied using the variable best, which represents the most reliable estimate of total fatalities associated with each event.

This dataset is well suited for the analysis because fatality counts provide an objective and economically meaningful measure of conflict severity. Unlike simple event counts, fatalities capture both the intensity and persistence of violence, which are more likely to affect production conditions, trade flows, and investor uncertainty. To align conflict information with financial data, event-level observations are aggregated to the daily frequency.

## Target variable

Daily log returns are computed as:
- r_t = log(P_t) - log(P_{t-1})

Daily realized volatility is defined as the squared log return:
- RV_d,t = r_t^2

To capture volatility persistence at different horizons, weekly and monthly realized volatility measures are constructed as rolling averages of daily realized volatility over 5 and 22 trading days.

The forecast target is next-day realized volatility:
- Target_RVt = RVd,t+1
	​
This target is not observable at time 𝑡. All models therefore generate forecasts using only information available up to date 𝑡, including lagged realized volatility components and lagged conflict indices. This timing structure strictly avoids any look-ahead bias and ensures a clean out-of-sample forecasting framework.

## Conflict indices and features

Armed-conflict information is incorporated using daily conflict intensity indices constructed from the UCDP Georeferenced Event Dataset. Conflict intensity is proxied by the number of fatalities reported for each event, measured by the variable best.

Event-level observations are aggregated to the daily frequency to align conflict data with financial markets. Because fatality counts are highly skewed and noisy, daily totals are transformed using a logarithmic transformation and smoothed with an exponentially weighted moving average using a fixed decay parameter of 0.94.

In addition to a global conflict index, region-specific and commodity-relevant indices are constructed to reflect geographic exposure. All conflict variables enter the models in lagged form, ensuring that forecasts rely only on information available at the time of prediction and avoiding any look-ahead bias.

## Project Structure

The repository is organized to ensure reproducibility, clarity, and modularity.

```text
commodity-volatility-conflict/
├── README.md              # Project overview, data description, and instructions
├── PROPOSAL.md            # Project proposal
├── requirements.txt       # Python dependencies
├── main.py                # Main script (runs the full pipeline end-to-end)
│
├── data/
│   ├── raw/
│   │   ├── commodities/   # Raw CSV downloads from Investing.com
│   │   └── conflicts/     # Raw UCDP GED data (NOT versioned on GitHub due to size)
│   │
│   └── processed/
│       ├── commodities/   # Cleaned daily price series
│       ├── features/      # Realized volatility features (RV)
│       ├── conflicts/     # Reduced and sorted UCDP events
│       ├── indices/       # Daily conflict indices (EWMA, regional, global)
│       └── model_datasets/# Final datasets aligned for modeling
│
├── src/
│   ├── data_loader.py            # Step 1: Commodity data cleaning
│   ├── features.py               # Step 1b: Realized volatility construction
│   ├── conflict_loader.py        # Step 2: Conflict data reduction
│   ├── conflict_index_builder.py # Step 3: Conflict index construction
│   ├── build_model_dataset.py    # Step 4: Final dataset assembly
│   ├── models.py                 # Step 5: HAR and HAR-X models (in-sample), RF
│   └── evaluation.py             # Step 6: Walk-forward out-of-sample evaluation
│
└── results/
    ├── in_sample/         # In-sample HAR / HAR-X comparison tables
    │   ├── WTI/
    │   ├── GAS/
    │   └── GOLD/
    │
    └── out_of_sample/     # Walk-forward forecast evaluation
        ├── WTI/
        ├── GAS/
        └── GOLD/
```
## How to run the project

The entire project is executed through the main pipeline script.

After installing the required dependencies, running

- python main.py

will reproduce the full workflow, including data cleaning, feature construction, conflict index building, dataset assembly, model estimation, and out-of-sample evaluation.

All results are generated deterministically from the raw input data.


All raw commodity price files used in the project are included in the GitHub repository. The raw UCDP Georeferenced Event Dataset is not included because of its large size. Instead, the repository contains a reduced and preprocessed version of the dataset that is sufficient to reproduce all results.

If the project is run fully from scratch, the original UCDP GED file must be downloaded manually from the UCDP website and placed in the directory data/raw/conflicts/. Once the file is available, running python main.py will rebuild the reduced conflict dataset, construct the conflict indices, and reproduce the full analysis pipeline.

(The original UCDP Georeferenced Event Dataset can be downloaded from the [UCDP download page](https://ucdp.uu.se/downloads/).)
(UCDP Georeferenced Event Dataset (GED), version 25.1)

## Requirements
The project is implemented in Python 3.11 and relies on the following libraries:

### Data manipulation
- pandas
- numpy

### Visualization
- matplotlib

### Econometrics, statistics and ML
- statsmodels
- scipy
- scikit-learn

### Utilities
- pathlib (standard library)

## Author
Lorenzo Davide De Martino
