"""Independent arithmetic and first-origin reconstruction, without predict()."""

import importlib.metadata
import json
import math
from statistics import fmean
import sys

import numpy as np
import pandas as pd

from extension.prepare_data import DATA, HERE, ROOT, read_config, save_json, verify_original
from prepare_data import require, sha256


def first_origin_check(config):
    daily = pd.read_csv(DATA / "daily.csv", index_col="date", parse_dates=["date"])
    source = pd.read_csv(DATA / "iip.csv", parse_dates=["release_date"])
    source["month"] = pd.PeriodIndex(source.month, freq="M")
    ref = source.loc[source.target.eq("manufacturing")].set_index("month")
    saved = pd.read_csv(HERE / "results/test_predictions.csv")
    forecast_month = pd.Period("2024-01", "M")
    forecast_date = pd.Timestamp("2024-02-05")
    issues = {}
    for path in [ROOT / "data/processed/report_audit.csv", DATA / "report_audit.csv"]:
        for row in pd.read_csv(path).itertuples():
            issues[pd.Timestamp(row.date)] = pd.Timestamp(row.printed_reporting_date)
    lower, upper = [pd.Period(value, "M") for value in config["shock"]]

    def available(date):
        mask = daily.energy_usable & daily.weather_complete & (daily.index + pd.Timedelta(days=5) <= date)
        for day, issued in issues.items():
            if day in mask.index and issued > date:
                mask.loc[day] = False
        return mask

    info = {}
    for month in pd.period_range("2015-01", "2024-01", freq="M"):
        origin = (month + 1).start_time + pd.Timedelta(days=4)
        cutoff = origin - pd.Timedelta(days=5)
        base_cutoff = cutoff - pd.DateOffset(years=1)
        windows = [pd.date_range(month.start_time, cutoff), pd.date_range((month - 12).start_time, base_cutoff),
                   pd.date_range(cutoff - pd.Timedelta(days=6), cutoff),
                   pd.date_range(cutoff - pd.Timedelta(days=13), cutoff - pd.Timedelta(days=7))]
        mask = available(origin)
        eligible = all(mask.reindex(days, fill_value=False).mean() >= .9 for days in windows)
        released = ref.loc[(ref.release_date <= origin) & (ref.index < month) & ref.yoy_pct.notna()].sort_index()
        lags = list(released.index[-2:][::-1])
        components = [month, month - 12] + [value for lag in lags for value in [lag, lag - 12]]
        components += [day.to_period("M") for window in windows for day in window]
        clean = not any(lower <= item <= upper for item in components)
        info[month] = {"origin": origin, "windows": [window[mask.reindex(window).to_numpy()] for window in windows],
                       "lags": lags, "eligible": eligible, "clean": clean}
    # Construct the nuisance design explicitly rather than call daily_design().
    elapsed = (daily.index - pd.Timestamp("2014-01-01")).days.to_numpy() / 365.25
    columns = [np.ones(len(daily)), elapsed]
    for harmonic in [1, 2]:
        columns += [np.sin(2 * np.pi * harmonic * elapsed), np.cos(2 * np.pi * harmonic * elapsed)]
    columns += [(daily.index.dayofweek == weekday).astype(float) for weekday in range(1, 7)]
    columns += [daily.fixed_holiday.to_numpy(float), daily.diwali_window.to_numpy(float)]
    temperatures = daily[[item["id"] + "_T2M_C" for item in config["weather_locations"]]].to_numpy()
    columns += [np.maximum(18 - temperatures, 0).mean(axis=1), np.maximum(temperatures - 24, 0).mean(axis=1)]
    design = np.column_stack(columns)
    differences = []
    for case in config["cases"]:
        mask = available(forecast_date) & (daily.index <= pd.Timestamp("2023-11-30"))
        if case == "strict_covid":
            mask &= ~((daily.index.to_period("M") >= lower) & (daily.index.to_period("M") <= upper))
        beta = np.linalg.lstsq(design[mask], np.log(daily.loc[mask, "energy_MU"]), rcond=None)[0]
        adjusted = pd.Series(daily.energy_MU.to_numpy() / np.exp(design[:, 2:] @ beta[2:]), index=daily.index)
        released = ref.index[(ref.release_date <= forecast_date) & (ref.index < forecast_month)]
        training = [month for month, item in info.items() if month in released and item["eligible"] and (case == "full" or item["clean"])]
        recorded = pd.read_csv(HERE / "results/test_ledger.csv").query("month == '2024-01' and case == @case").iloc[0]
        require(training == list(pd.PeriodIndex(recorded.training_month_list.split("|"), freq="M")), "Independent training set differs")
        for target in config["targets"]:
            target_id = target["id"]
            iip = source.loc[source.target.eq(target_id)].set_index("month")
            features = {}
            for month in training + [forecast_month]:
                item = info[month]
                row = [iip.loc[lag, "yoy_pct"] for lag in item["lags"]]
                row += [math.sin(2 * math.pi * month.month / 12), math.cos(2 * math.pi * month.month / 12),
                        daily.loc[month.start_time:month.end_time, "diwali_window"].mean()]
                for energy in [daily.energy_MU, adjusted]:
                    means = [energy.reindex(days).mean() for days in item["windows"]]
                    row += [100 * (means[0] / means[1] - 1), 100 * (means[2] / means[3] - 1)]
                features[month] = row
            y = iip.loc[training, "yoy_pct"].to_numpy()
            for model in ["iip_benchmark", "raw_ols", "adjusted_ols", "adjusted_ridge"]:
                selected = saved.query("month == '2024-01' and case == @case and target == @target_id and model == @model").iloc[0]
                which = list(range(5)) + ([] if model == "iip_benchmark" else ([5, 6] if model == "raw_ols" else [7, 8]))
                x = np.array([features[month] for month in training])[:, which]
                future = np.array(features[forecast_month])[which]
                mean, std = x.mean(axis=0), x.std(axis=0)
                standardized = (x - mean) / std
                # Augmented least squares gives the same ridge solution by a
                # different numerical path from the pipeline's normal equations.
                alpha = selected.alpha
                augmented = np.vstack([standardized, np.sqrt(alpha) * np.eye(len(which))])
                response = np.r_[y - y.mean(), np.zeros(len(which))]
                slopes = np.linalg.lstsq(augmented, response, rcond=None)[0]
                prediction = y.mean() + ((future - mean) / std) @ slopes
                difference = float(prediction - selected.prediction_pct)
                require(abs(difference) < 1e-8, f"Independent first-origin mismatch: {case}/{target_id}/{model}")
                differences.append(abs(difference))
    return {"independently_reconstructed_forecasts": len(differences), "maximum_absolute_difference_pp": max(differences)}


def main():
    config = read_config()
    results = HERE / "results"
    frame = pd.read_csv(results / "test_predictions.csv")
    metrics = pd.read_csv(results / "metrics.csv")
    differences = []
    for row in metrics.itertuples():
        group = frame.loc[frame.case.eq(row.case) & frame.target.eq(row.target) & frame.model.eq(row.model)]
        errors = [float(predicted - actual) for predicted, actual in zip(group.prediction_pct, group.actual_pct)]
        mae, rmse = fmean(map(abs, errors)), math.sqrt(fmean(value * value for value in errors))
        differences += [abs(mae - row.MAE_pp), abs(rmse - row.RMSE_pp)]
    require(max(differences) < 1e-8, "Independent score calculation differs")
    # Rebuild each saved forecast directly from its coefficient and feature rows.
    reconstruct = []
    for prefix in ["validation", "test"]:
        predictions = pd.read_csv(results / f"{prefix}_predictions.csv")
        coefficients = pd.read_csv(results / f"{prefix}_coefficients.csv")
        features = pd.read_csv(results / f"{prefix}_features.csv").set_index(["month", "case", "target"])
        for key, group in coefficients.groupby(["month", "case", "target", "model", "alpha"]):
            row = features.loc[key[:3]]
            value = sum(item.coefficient * (1 if item.term == "intercept" else row[item.term]) for item in group.itertuples())
            saved = predictions.loc[(predictions[["month", "case", "target", "model", "alpha"]] == key).all(axis=1), "prediction_pct"].iloc[0]
            reconstruct.append(abs(value - saved))
    require(max(reconstruct) < 1e-7, "Forecast/coefficient reconstruction differs")
    reproduced = []
    for path in sorted(results.glob("test_*.csv")):
        other = path.with_name(path.name.replace("test_", "reproduce_", 1))
        require(other.exists() and sha256(path) == sha256(other), f"Nonidentical clean rerun: {path.name}")
        reproduced.append(path.name)
    for item in json.loads((HERE / "raw/manifest.json").read_text()):
        require(sha256(HERE / "raw" / item["file"]) == item["sha256"], "Raw source changed")
    frozen = json.loads((results / "experiment_freeze.json").read_text())
    for name, checksum in frozen["fingerprints"].items():
        require(sha256(ROOT / name) == checksum, f"Frozen file changed: {name}")
    spec = config["uncertainty"]
    months = sorted(frame.month.unique())
    rng = np.random.default_rng(spec["seed"])
    independent_draws = np.array([[position for start in rng.integers(0, len(months) - 2, size=8)
                                   for position in range(start, start + 3)] for _ in range(spec["repetitions"])])
    require(np.array_equal(independent_draws, np.load(results / "bootstrap_month_indices.npy")), "Independent block draws differ")
    intervals = pd.read_csv(results / "paired_contrasts.csv")
    interval_errors = []
    for row in intervals.itertuples():
        left = frame.query("case == @row.case and target == @row.target and model == @row.left_model").sort_values("month")
        right = frame.query("case == @row.case and target == @row.target and model == @row.right_model").sort_values("month")
        delta = np.abs(left.prediction_pct.to_numpy() - left.actual_pct.to_numpy()) - np.abs(right.prediction_pct.to_numpy() - right.actual_pct.to_numpy())
        bounds = np.quantile(delta[independent_draws].mean(axis=1), [.025, .975])
        interval_errors.extend(np.abs(bounds - [row.interval_low_pp, row.interval_high_pp]))
    require(max(interval_errors) < 1e-8, "Independent interval calculation differs")
    verification = {"status": "passed", "metric_cells_checked": len(metrics), "holdout_forecasts": len(frame),
                    "max_metric_difference_pp": max(differences), "coefficient_reconstructions": len(reconstruct),
                    "max_coefficient_reconstruction_difference_pp": max(reconstruct),
                    "byte_identical_reproduction_files": reproduced, "independent_block_draws_identical": True,
                    "paired_intervals_checked": len(intervals), "max_interval_difference_pp": max(interval_errors),
                    "first_origin_independent_check": first_origin_check(config), "unchanged_original_files": verify_original(),
                    "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "xlrd", "pypdf"]}}
    save_json(HERE / "qa/numerical_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
