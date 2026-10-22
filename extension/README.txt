ELECTRICITY AND IIP INDUSTRY EXTENSION

Run this extension from the repository root. The root README gives the complete
installation and reproduction commands; REPRODUCIBILITY.md distinguishes the
experimental freezes from later repository packaging changes.

Four targets: national manufacturing, basic metals, other non-metallic mineral
products and wearing apparel. Four forecasting setups: released-IIP OLS,
raw-electricity OLS, adjusted-electricity OLS and adjusted-electricity ridge.
The full-history and strict COVID cases share 24 test months in 2024–2025.

The strict case removes March 2020–June 2022 daily fitting dates and monthly
rows exposed through outcomes, annual bases, released IIP lags or electricity
windows. Only 16 validation origins remain, including two in late 2023.

The saved experiment does not establish a dependable added forecasting gain
from national electricity. The full-history benchmark wins on every target.
Strict exclusions improve every method; remaining electricity gains are small
and uncertain. Ridge versus benchmark OLS compares complete methods, and no
IIP-only ridge ablation is present. See the integrated draft for all results.

prepare_data.py validates sources and creates the input and exposure records.
backtest.py runs chronological validation, freezing, testing and reproduction.
report.py reads saved predictions to produce metrics, uncertainty and figures.
verify.py independently checks arithmetic and reconstructs forecasts.

Source attribution is in ../SOURCES.md. Raw files and publication dates retain
their recorded URLs and hashes. Original-workbook visual checks are recorded
in data/source_review.json; automated comparisons use the stored XLS precision.
