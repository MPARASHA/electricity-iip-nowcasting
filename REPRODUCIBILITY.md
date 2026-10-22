# Reproducing the study

## Fixed experiments

The national experiment uses 2015–2018 for initial training, 2019–2021 for chronological validation and 2022–2023 for testing. The extension uses the same initial training years, validation through November 2023 and a separate 2024–2025 test. Its two fitting cases are scored on the same 24 test months.

The original model choices and penalties were fixed before the corresponding holdout scores. This is a historical exercise performed in October 2026, rather than a prospective deployment during those test years. The extension was motivated by the original findings.

## Repository packaging revision

The repository contains the scientific files needed to reproduce the calculations. The baseline manifest checks the distributed national experiment files. It is a distribution subset of the larger original workspace audit described in the draft.

Repository preparation on 22 October 2026 standardized documentation and removed administrative scope-approval metadata. Two administrative assertions were removed from the national input loader; all data-integrity, availability, training-window and freeze checks remain. Extension freeze metadata was simplified. These edits do not alter model fitting, predictor construction, tuning or scoring.

Consequently, current file fingerprints differ from the original experimental fingerprints. The publication revision in each freeze identifies its source freeze and records when repository hashes were updated. This revision occurred after results were available; it is not a new prospective experiment. The experimental freeze dates retain their historical meaning. The national forecast `experiment_id` identifies the distributed freeze, while its numerical forecasts remain unchanged.

The extension also retains the earlier preparation-metadata amendment: cached reruns preserve a timestamp when checked contents are identical. That amendment occurred after the initial evaluation and changed no prepared CSV values, predictions or selected penalties. It is disclosed in the freeze provenance.

## Check the delivered results

Follow the commands in the README in order. Run national preparation and testing before extension preparation, since the extension reads the distributed national baseline. The extension reproduction command writes `reproduce_*.csv` beside the saved `test_*.csv`. Its verification command requires all six corresponding CSV files to agree byte for byte. It also checks raw source hashes, prepared inputs, freeze fingerprints, paired intervals and independent coefficient calculations.

The repository includes the exact cached data used for these calculations. The raw manifests record source URLs, retrieval times and content hashes. Normal cached preparation rejects changed bytes. The national pipeline's `--download-missing` option checks restored files against their recorded hashes.

The extension's optional `--fetch` command checks existing cached files, but accepts downloaded bytes and updates the manifest when a file is missing. It does not guarantee restoration of that file's frozen version. To reproduce the extension, restore missing cache files from the same repository commit and use the normal cached commands above.

## Rebuild validation separately

Use a separate copy of the repository and retain the delivered results for comparison. Never overwrite the delivered freezes to disguise a new choice made after inspecting holdout errors.

For the national experiment, move that copy’s `results` directory aside and run:

```sh
.venv/bin/python prepare_data.py
.venv/bin/python backtest.py --stage validation
```

Compare the regenerated validation predictions and accuracy with the delivered values. A new prospective experiment then needs its own validation review and explicit freeze before the test command is run. The CLI requires a technical `--review-note` describing the validation decision when using `--stage freeze`.

For an independent extension rebuild, preserve `extension/data/source_review.json`, its referenced source tables and the national baseline. In the separate copy, move `extension/results` aside and create an empty replacement directory. Run:

```sh
.venv/bin/python -m extension.prepare_data
.venv/bin/python -m unittest extension.test_extension -v
.venv/bin/python -m extension.backtest validate
.venv/bin/python -m extension.backtest freeze
.venv/bin/python -m extension.backtest test
.venv/bin/python -m extension.backtest reproduce
.venv/bin/python -m extension.report
.venv/bin/python -m extension.verify
```

New timestamps and experiment identifiers will differ. Numerical validation and holdout forecasts should agree when inputs, versions and specifications are unchanged.
