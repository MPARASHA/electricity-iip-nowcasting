# Electricity and Indian industrial production

This repository contains the code, fixed source data, saved forecasts and research draft for a study of whether national electricity supply improves early estimates of Indian industrial production growth.

The original study evaluates national manufacturing during 2022–2023. The extension evaluates manufacturing, basic metals, other non-metallic mineral products and wearing apparel during 2024–2025. It compares ordinary least squares (OLS) and ridge regression in four forecasting setups, using both full-history fitting and a strict COVID exposure exclusion.

## Finding

National electricity does not establish a dependable forecasting improvement over the released-IIP OLS benchmark in this sample. The benchmark wins for all four targets with full-history fitting. Strict COVID exclusions reduce errors for every method, including the benchmark. The remaining electricity gains are small and uncertain.

Ridge versus the IIP-only OLS benchmark compares complete methods because both inputs and regularization change. There is no IIP-only ridge ablation. The strict validation sample contains only 16 months, including just two recent months. These limitations and the upper boundary selected for the full-history ridge penalty are retained in the draft.

## Repository contents

| Path | Contents |
| --- | --- |
| `prepare_data.py`, `backtest.py`, `config.json` | Original national experiment and its source checks |
| `report.ipynb` | Original national analysis of saved results; fits no models |
| `extension/prepare_data.py` | Four targets, daily inputs, publication dates and original-report checks |
| `extension/backtest.py` | Chronological validation, experiment freeze and holdout forecasts |
| `extension/report.py` | Accuracy, paired uncertainty, influential-month checks and figures |
| `extension/verify.py` | Independent arithmetic and forecast reconstruction |
| `research/feasibility_2026-10-22`, `extension/raw` | Fixed input files with source URLs and SHA-256 hashes |
| `data/processed`, `extension/data` | Validated inputs and source audit tables |
| `results`, `extension/results` | Saved validation, holdout, coefficient and uncertainty results |
| `output/Electricity_IIP_Extended_Research_Draft.docx` | Integrated research draft |

## Run the saved experiment

Use Python 3.12. The original calculations used Python 3.12.14. Run commands from the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r extension/requirements.txt
.venv/bin/python prepare_data.py
.venv/bin/python -m unittest test_prepare_data test_backtest -v
.venv/bin/python backtest.py --stage test
.venv/bin/python -m nbconvert --to notebook --execute report.ipynb --inplace --ExecutePreprocessor.timeout=120
.venv/bin/python -m extension.prepare_data
.venv/bin/python -m unittest extension.test_extension -v
.venv/bin/python -m extension.backtest reproduce
.venv/bin/python -m extension.report
.venv/bin/python -m extension.verify
```

The normal preparation and reproduction commands use the included cache without network access. Changed source bytes fail their hash checks. Validation and freezing refuse to overwrite an existing frozen experiment. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before rebuilding validation or creating a new experiment.

The test suite checks publication timing, training-only preprocessing, shock exposure boundaries, leap-year alignment, complete weekly windows, missing values, source integrity and paired block resampling. Verification checks all 32 extension accuracy cells, reconstructs forecasts from saved coefficients and independently rebuilds first-origin forecasts.

## Data and interpretation

The electricity series is Robbie Andrew’s (CICERO) extraction of original Grid India reports. Source attribution and evidence links are in [SOURCES.md](SOURCES.md). National grid supply includes households, services and agriculture; it does not measure each selected industry’s electricity consumption or all captive generation.

Targets use the historical 2011–12 IIP base. Revised IIP and reanalysis weather make this a pseudo-real-time exercise. The forecast origin is the fifth of the following month, and each fit uses only labels released by that origin. The April 2025 IIP publication timetable change is handled explicitly in the extension.

The strict case excludes March 2020–June 2022 daily fitting observations and monthly rows exposed through outcomes, annual comparison bases, released IIP lags or electricity windows. It preserves calendar gaps. The 768 extension forecasts share only 24 test dates and are not 768 independent time observations. Paired block intervals are exploratory. Differences between fitting cases do not identify a causal effect of COVID.

Repository packaging updates are documented separately from the original experimental freeze. Data values, model choices, selected penalties and numerical forecasts are preserved.
