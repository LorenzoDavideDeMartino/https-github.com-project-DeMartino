# AI_USAGE.md

## Project: Commodity Volatility Forecasting with Conflict Indices (HAR / HAR-X / Random Forest)

This document discloses how AI tools were used during the development of this project.

## Tools Used
- ChatGPT - for code review, debugging suggestions, refactoring ideas
- Gemini - for code review, debugging suggestions, and writing assistance (documentation)
- Deepl -  for writing assistance

## What AI Was Used For
### Code support 
AI was used to:
- suggest refactoring for readability (naming, modularization, removing redundancy);
- propose defensive programming patterns (early checks, handling missing columns, avoiding silent parsing errors);
- troubleshoot runtime/performance issues in walk-forward evaluation loops (reducing re-fit frequency, reducing repeated computations);
- suggest numerically stable evaluation practices (handling zeros/near-zeros in volatility forecasts, avoiding invalid logs);
- format and simplify LaTeX tables and report text to fit Overleaf constraints.

### Writing support
AI was used to:
- rewrite and compress parts of the methodology/results/discussion into clearer academic English;
- help structure sections of the report (Methodology, Results, Discussion, Conclusion/Future Work).

## What AI Was NOT Used For
AI was not used to:
- coding overall project
- fabricate results, metrics, or statistical significance
- invent data sources
- make decisions about the final modeling choices without verification
- run experiments on my behalf (all experiments were run locally by the author)


## Author Responsibility and Verification
All final choices, interpretations, and results are the author’s responsibility.  
Every AI suggestion was reviewed and either:
- implemented with understanding and testing, or
- rejected if it did not match the project design, reproducibility constraints, or econometric validity.

## AI-Assisted Components (Concrete Examples)
The following parts received AI assistance in the sense described above:
- Walk-forward evaluation design: reducing computation time by producing forecasts every *k* trading days and periodically refitting the Random Forest.
- Report tables: generating overleaf table skeletons and formatting coefficient/metric summaries.
