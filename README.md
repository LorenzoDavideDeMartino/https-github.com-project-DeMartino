# https-github.com-project-DeMartino

Can We Predict and Quantify the Impact of Armed Conflicts on Commodity Price Volatility Using Machine Learning?

This project aims to investigate whether armed conflicts systematically influence the volatility of global commodity prices and how these effects can be modeled and integrated into portfolio optimization frameworks. By combining conflict event data with financial market indicators, the project seeks to quantify geopolitical risk and demonstrate how its inclusion can improve risk-adjusted returns in commodity investment strategies.

The study will integrate multiple data sources, including the Uppsala Conflict Data Program (UCDP/PRIO Armed Conflict Dataset) for conflict characteristics (type, intensity, and duration) and commodity market data (spot and futures prices for oil, gold, wheat, and natural gas) collected from yfinance or S&P Capital. Control variables such as macroeconomic indicators and climatic events (NOAA Climate Data) will be incorporated to isolate the geopolitical component of volatility.

Methodologically, the project will combine traditional econometric models (ARIMA-GARCH) with machine learning techniques, including Long Short-Term Memory (LSTM) networks for volatility forecasting and Random Forest regressors for estimating the impact of conflicts. A Geopolitical Risk Index derived from conflict characteristics will serve as a key explanatory variable for volatility dynamics.

The second component of the project will implement a conflict-aware portfolio optimizer, extending the classical Markowitz mean-variance framework by integrating predicted volatility and conflict-based risk factors. This optimizer will construct commodity portfolios that balance expected returns against both market and geopolitical risk exposures. Historical backtesting (1990–2024) will evaluate performance improvements in terms of Sharpe ratio, drawdown, and return stability compared to traditional models.

Finally, an interactive Streamlit application will visualize conflict events, volatility forecasts, and optimized portfolio performance, offering an intuitive decision-support tool for investors navigating uncertainty in commodity markets.
