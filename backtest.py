"""Expanding pseudo-real-time nowcasts; validation, freeze and test are separate.

No target value from the forecast month enters prediction. Historical IIP values
are revised, so release-date checks do not reconstruct real-time value vintages.
"""

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import require, sha256


MODELS = ["iip_benchmark", "raw_ols", "adjusted_ols", "adjusted_ridge"]


def load_inputs(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text())
    processed = root / config["output_dir"]
    versions = json.loads((processed / "data_versions.json").read_text())
    require(versions["config_sha256"] == sha256(config_path), "Configuration changed; rerun preparation")
    require(versions["preparation_script_sha256"] == sha256(root / "prepare_data.py"),
            "Preparation code changed; rerun preparation")
    for name, checksum in versions["outputs"].items():
        require(sha256(processed / name) == checksum, f"Prepared file changed: {name}")
    gates = config["evaluation_gate"]
    require(gates["full_calendar_and_alignment"] == "complete", "Calendar/alignment gate not cleared")
    daily = pd.read_csv(processed / "daily.csv", index_col="date", parse_dates=["date"])
    iip = pd.read_csv(processed / "iip.csv", parse_dates=["release_date"])
    iip.index = pd.PeriodIndex(iip.pop("month"), freq="M")
    origins = pd.read_csv(processed / "origins.csv", parse_dates=["forecast_date", "daily_cutoff", "prior_year_cutoff"])
    origins["month"] = pd.PeriodIndex(origins["month"], freq="M")
    audit = pd.read_csv(processed / "report_audit.csv", parse_dates=["date", "printed_reporting_date"])
    require(not daily.index.duplicated().any() and not iip.index.duplicated().any(), "Duplicate prepared keys")
    return root, config, daily, iip, origins, audit


def issue_dates(audit):
    return {row.date: row.printed_reporting_date for row in audit.itertuples()
            if row.date_status == "consistent"}


def available_mask(daily, forecast, lag_days, issues):
    mask = daily["energy_usable"] & daily["weather_complete"]
    mask = mask & (daily.index + pd.Timedelta(days=lag_days) <= forecast)
    for day, issued in issues.items():
        if day in mask.index and issued > forecast:
            mask.loc[day] = False
    return mask


def daily_design(daily, config):
    """Two retained columns followed by the nuisance terms to subtract."""
    spec = config["adjustment"]
    elapsed = (daily.index - pd.Timestamp(config["history_start"])).days.to_numpy() / 365.25
    design = pd.DataFrame({"intercept": 1., "trend_years": elapsed}, index=daily.index)
    for harmonic in range(1, spec["annual_harmonics"] + 1):
        angle = 2 * np.pi * harmonic * elapsed
        design[f"annual_sin{harmonic}"] = np.sin(angle)
        design[f"annual_cos{harmonic}"] = np.cos(angle)
    for weekday in range(1, 7):
        design[f"weekday_{weekday}"] = (daily.index.dayofweek == weekday).astype(float)
    design["fixed_holiday"] = daily["fixed_holiday"].astype(float)
    design["diwali_window"] = daily["diwali_window"].astype(float)
    weights = np.array([location["weight"] for location in config["weather_locations"]])
    require(np.isclose(weights.sum(), 1) and (weights > 0).all(), "Invalid fixed weather weights")
    temperatures = daily[[f"{location['id']}_T2M_C" for location in config["weather_locations"]]].to_numpy()
    design["HDD18"] = np.maximum(spec["heating_base_C"] - temperatures, 0) @ weights
    design["CDD24"] = np.maximum(temperatures - spec["cooling_base_C"], 0) @ weights
    return design


def pandemic_mask(index, config):
    months = index.to_period("M") if isinstance(index, pd.DatetimeIndex) else index
    start, end = [pd.Period(value, "M") for value in config["pandemic_training_exclusion"]]
    return (months >= start) & (months <= end)


def adjust_energy(daily, fit_end, forecast, scenario, config, issues):
    """Fit only the released training period, then apply through the daily cutoff."""
    cutoff = forecast - pd.Timedelta(days=scenario["lag_days"])
    visible = daily.loc[:cutoff].copy()
    design = daily_design(visible, config)
    usable = available_mask(visible, forecast, scenario["lag_days"], issues)
    fit = usable & (visible.index <= fit_end)
    if scenario["exclude_pandemic_training"]:
        fit &= ~pandemic_mask(visible.index, config)
    require(int(fit.sum()) >= 730, "Too few daily adjustment training observations")
    x = design.loc[fit].to_numpy()
    y = np.log(visible.loc[fit, "energy_MU"].to_numpy())
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    require(rank == x.shape[1], "Daily adjustment design is rank deficient")
    adjusted = visible["energy_MU"].to_numpy() * np.exp(-design.iloc[:, 2:].to_numpy() @ beta[2:])
    output = pd.DataFrame({"raw_energy": visible["energy_MU"], "adjusted_energy": adjusted}, index=visible.index)
    output.loc[~usable, :] = np.nan
    residual = y - x @ beta
    denominator = np.sum((y - y.mean()) ** 2)
    metadata = {"adjustment_first_day": str(visible.index[fit].min().date()),
                "adjustment_last_day": str(visible.index[fit].max().date()),
                "adjustment_training_days": int(fit.sum()), "adjustment_rank": int(rank),
                "adjustment_r2": float(1 - np.sum(residual ** 2) / denominator),
                "adjustment_residual_log_sd": float(np.std(residual)),
                "retained_trend_log_per_year": float(beta[1])}
    coefficients = pd.DataFrame({"term": design.columns, "coefficient": beta,
                                 "role": ["retained", "retained"] + ["removed"] * (len(beta) - 2)})
    return output, metadata, coefficients


def monthly_features(energy, daily, iip, origins, scenario, config, issues):
    """Rebuild each historical month's predictors at its own forecast cutoff."""
    rows = []
    weekly = config["models"]["weekly_days"]
    threshold = scenario["coverage"]
    for origin in origins.itertuples():
        month = origin.month
        row = {"month": month, "eligible": False, "exclusion_reason": "monthly_coverage_or_missing_lag"}
        flag = f"eligible_{round(100 * threshold)}"
        if not getattr(origin, flag):
            rows.append(row)
            continue
        # Later observations cannot repair a historical information set.
        own_available = available_mask(daily, origin.forecast_date, scenario["lag_days"], issues)
        values = energy.where(own_available, axis=0)
        windows = []
        for start, end in [(month.start_time, origin.daily_cutoff),
                           ((month - 12).start_time, origin.prior_year_cutoff)]:
            window = values.reindex(pd.date_range(start, end))
            if window.notna().all(axis=1).mean() < threshold:
                break
            windows.append(window.mean())
        if len(windows) != 2:
            rows.append(row)
            continue
        recent = values.reindex(pd.date_range(origin.daily_cutoff - pd.Timedelta(days=weekly - 1), origin.daily_cutoff))
        previous = values.reindex(pd.date_range(origin.daily_cutoff - pd.Timedelta(days=2 * weekly - 1),
                                               origin.daily_cutoff - pd.Timedelta(days=weekly)))
        if min(recent.notna().all(axis=1).mean(), previous.notna().all(axis=1).mean()) < threshold:
            row["exclusion_reason"] = "weekly_coverage"
            rows.append(row)
            continue
        lags = [month - 2, month - 3]
        if not all(m in iip.index and iip.loc[m, "release_date"] <= origin.forecast_date
                   and pd.notna(iip.loc[m, "yoy_pct"]) for m in lags):
            row["exclusion_reason"] = "unreleased_or_missing_iip_lag"
            rows.append(row)
            continue
        row.update({"iip_lag2_yoy_pct": float(iip.loc[lags[0], "yoy_pct"]),
                    "iip_lag3_yoy_pct": float(iip.loc[lags[1], "yoy_pct"]),
                    "month_sin": np.sin(2 * np.pi * month.month / 12),
                    "month_cos": np.cos(2 * np.pi * month.month / 12),
                    "diwali_share": float(daily.loc[month.start_time:month.end_time, "diwali_window"].mean())})
        for prefix, column in [("raw", "raw_energy"), ("adjusted", "adjusted_energy")]:
            row[f"{prefix}_energy_yoy_pct"] = 100 * (windows[0][column] / windows[1][column] - 1)
            row[f"{prefix}_weekly_momentum_pct"] = 100 * (recent[column].mean() / previous[column].mean() - 1)
        row.update(eligible=True, exclusion_reason="")
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def fit_monthly(x, y, prediction_x, alpha=0.):
    """Training-only scaling; ridge penalizes slopes and leaves the mean free."""
    x, y, prediction_x = np.asarray(x, float), np.asarray(y, float), np.asarray(prediction_x, float)
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(prediction_x).all(), "Non-finite model input")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    centered = (x - mean) / scale
    target_mean = y.mean()
    if alpha == 0:
        beta, _, rank, _ = np.linalg.lstsq(centered, y - target_mean, rcond=None)
        require(rank == x.shape[1], "Monthly OLS design is rank deficient")
    else:
        require(alpha > 0, "Negative ridge penalty")
        beta = np.linalg.solve(centered.T @ centered + alpha * np.eye(x.shape[1]), centered.T @ (y - target_mean))
    prediction = target_mean + ((prediction_x - mean) / scale) @ beta
    slopes = beta / scale
    intercept = target_mean - mean @ slopes
    return float(prediction), float(intercept), slopes, mean, scale


def predict_origin(month, scenario, config, daily, iip, origins, audit, ridge_alphas):
    scenario_origins = origins.loc[(origins.assumed_lag_days == scenario["lag_days"]) & (origins.month <= month)]
    origin = scenario_origins.loc[scenario_origins.month.eq(month)].iloc[0]
    released = iip.loc[(iip.release_date <= origin.forecast_date) & iip.target_in_scope & (iip.index < month)]
    require(len(released) > 0, "No released training labels")
    fit_end = released.index.max().end_time.normalize()
    issues = issue_dates(audit)
    energy, adjustment_info, adjustment_coefficients = adjust_energy(daily, fit_end, origin.forecast_date,
                                                                     scenario, config, issues)
    features = monthly_features(energy, daily, iip, scenario_origins, scenario, config, issues)
    require(bool(features.loc[month, "eligible"]), f"Forecast month lacks complete features: {month}")
    training = features.index.isin(released.index) & features.eligible
    if scenario["exclude_pandemic_training"]:
        training &= ~pandemic_mask(features.index, config)
    train = features.loc[training]
    require(len(train) >= config["models"]["minimum_training_months"], f"Too few matched training months: {month}")
    y = iip.loc[train.index, "yoy_pct"].to_numpy()
    benchmark = config["models"]["benchmark_features"]
    predictions, coefficients = [], []
    for model in MODELS:
        columns = benchmark.copy()
        if model != "iip_benchmark":
            prefix = "raw" if model == "raw_ols" else "adjusted"
            columns += [f"{prefix}_{name}" for name in config["models"]["electricity_features"]]
        penalties = ridge_alphas if model == "adjusted_ridge" else [0.]
        for alpha in penalties:
            prediction, intercept, slopes, means, scales = fit_monthly(train[columns], y, features.loc[month, columns], alpha)
            predictions.append({"month": str(month), "forecast_date": str(origin.forecast_date.date()),
                                "scenario": scenario["name"], "model": model, "alpha": float(alpha),
                                "prediction_pct": prediction, "training_months": len(train),
                                "last_training_label": str(train.index.max())})
            for term, coefficient, mean, scale in zip(["intercept"] + columns, [intercept] + slopes.tolist(),
                                                      [0.] + means.tolist(), [1.] + scales.tolist()):
                coefficients.append({"month": str(month), "scenario": scenario["name"], "model": model,
                                     "alpha": float(alpha), "term": term, "coefficient": coefficient,
                                     "training_mean": mean, "training_sd": scale})
    ledger = {"month": str(month), "scenario": scenario["name"], "forecast_date": str(origin.forecast_date.date()),
              "daily_cutoff": str(origin.daily_cutoff.date()), "lag_days": scenario["lag_days"],
              "coverage_threshold": scenario["coverage"], "first_training_label": str(train.index.min()),
              "last_training_label": str(train.index.max()), "training_months": len(train),
              "last_training_release_date": str(iip.loc[train.index, "release_date"].max().date()),
              "latest_released_label": str(released.index.max()),
              "excluded_pandemic_training": scenario["exclude_pandemic_training"], **adjustment_info}
    require(pd.Timestamp(ledger["adjustment_last_day"]) <= fit_end < month.start_time, "Future daily training information")
    require(pd.Timestamp(ledger["last_training_release_date"]) <= origin.forecast_date, "Unreleased training label")
    adjustment_coefficients["month"] = str(month)
    adjustment_coefficients["scenario"] = scenario["name"]
    feature_audit = features[["eligible", "exclusion_reason"]].copy()
    feature_audit["forecast_origin_month"] = str(month)
    feature_audit["scenario"] = scenario["name"]
    return predictions, ledger, coefficients, adjustment_coefficients, energy, feature_audit


def scores(predictions):
    rows = []
    for (scenario, model, alpha), group in predictions.groupby(["scenario", "model", "alpha"], sort=True):
        error = group.prediction_pct - group.actual_pct
        rows.append({"scenario": scenario, "model": model, "alpha": float(alpha), "months": len(group),
                     "MAE_pp": float(np.abs(error).mean()), "RMSE_pp": float(np.sqrt((error ** 2).mean()))})
    return pd.DataFrame(rows)


def atomic_outputs(output, frames, metadata_name, metadata):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".backtest-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, frame in frames.items():
            frame.to_csv(stage / name, index=False, float_format="%.12g", date_format="%Y-%m-%d")
        metadata["outputs"] = {p.name: sha256(p) for p in sorted(stage.iterdir())}
        (stage / metadata_name).write_text(json.dumps(metadata, indent=2) + "\n")
        for path in sorted(stage.iterdir(), key=lambda p: p.name == metadata_name):
            os.replace(path, output / path.name)


def fingerprint(root, config_path):
    config = json.loads(Path(config_path).read_text())
    return {"config_sha256": sha256(config_path), "backtest_script_sha256": sha256(Path(__file__)),
            "data_versions_sha256": sha256(root / config["output_dir"] / "data_versions.json")}


def check_freeze(output, expected):
    freeze = json.loads((output / "experiment_freeze.json").read_text())
    require(freeze["fingerprint"] == expected, "Frozen code, data or configuration changed; do not retune on the test")
    validation = json.loads((output / "validation_run.json").read_text())
    require(sha256(output / "validation_run.json") == freeze["validation_run_sha256"], "Validation metadata changed after freeze")
    for name, checksum in validation["outputs"].items():
        require(sha256(output / name) == checksum, f"Validation output changed: {name}")
    return freeze


def freeze_experiment(config_path, review_note):
    root, config, *_ = load_inputs(config_path)
    output = root / config["results_dir"]
    path = output / "experiment_freeze.json"
    require(not path.exists(), "Experiment already frozen; preserve the existing record")
    require(not (output / "predictions.csv").exists(), "Test predictions already exist; cannot claim an unseen-test freeze")
    metadata = json.loads((output / "validation_run.json").read_text())
    expected = fingerprint(root, config_path)
    require(metadata["fingerprint"] == expected, "Validation used different code/data/configuration")
    for name, checksum in metadata["outputs"].items():
        require(sha256(output / name) == checksum, f"Validation output changed: {name}")
    validation = pd.read_csv(output / "validation_predictions.csv")
    require(validation.month.between(*config["splits"]["validation"]).all(), "Non-validation target in tuning file")
    comparison = scores(validation)
    ridge = comparison.loc[comparison.model.eq("adjusted_ridge")].sort_values(["MAE_pp", "alpha"])
    require(len(ridge) == len(config["models"]["ridge_alphas"]), "Incomplete ridge validation grid")
    require(comparison.months.nunique() == 1 and comparison.months.min() == 36, "Validation models use different or missing months")
    freeze = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "fingerprint": expected,
              "validation_run_sha256": sha256(output / "validation_run.json"),
              "selected_ridge_alpha": float(ridge.iloc[0].alpha), "selection_metric": "validation MAE_pp",
              "test_errors_seen_at_freeze": False, "review_note": review_note,
              "specification": {key: config[key] for key in ["splits", "forecast_day", "adjustment", "models",
                                                               "weather_time_rule", "scenarios", "evaluation", "pandemic_training_exclusion"]},
              "validation_metrics": comparison.to_dict("records")}
    path.write_text(json.dumps(freeze, indent=2) + "\n")
    return freeze


def run(config_path, stage):
    root, config, daily, iip, origins, audit = load_inputs(config_path)
    output = root / config["results_dir"]
    expected = fingerprint(root, config_path)
    if stage == "validation":
        require(not (output / "experiment_freeze.json").exists(), "Preserve validation after experiment freeze")
        scenarios, penalties = [config["scenarios"][0]], config["models"]["ridge_alphas"]
        bounds = config["splits"]["validation"]
        frozen_id = "validation_unfrozen"
    else:
        freeze = check_freeze(output, expected)
        scenarios, penalties = config["scenarios"], [freeze["selected_ridge_alpha"]]
        bounds = config["splits"]["test"]
        frozen_id = sha256(output / "experiment_freeze.json")
    predictions, ledgers, coefficients, adjustments, audits = [], [], [], [], []
    snapshots = []
    for scenario in scenarios:
        for month in pd.period_range(*bounds, freq="M"):
            result, ledger, coef, adj, energy, feature_audit = predict_origin(month, scenario, config, daily, iip, origins, audit, penalties)
            # Scoring labels are attached only after every prediction is computed.
            for row in result:
                row.update(actual_pct=float(iip.loc[month, "yoy_pct"]), experiment_id=frozen_id)
            predictions.extend(result)
            ledgers.append(ledger)
            coefficients.extend(coef)
            adjustments.append(adj)
            audits.append(feature_audit.reset_index())
            if scenario["name"] == "primary" and month == pd.Period(bounds[-1], "M"):
                snapshot = energy.reset_index()
                snapshot["fitted_at_origin_month"] = str(month)
                snapshots.append(snapshot)
    prediction_frame = pd.DataFrame(predictions)
    expected_months = set(str(m) for m in pd.period_range(*bounds, freq="M"))
    for _, group in prediction_frame.groupby(["scenario", "model", "alpha"]):
        require(set(group.month) == expected_months and not group.month.duplicated().any(), "Unequal evaluation months")
    require(stage == "validation" or len(expected_months) >= config["minimum_test_months"], "Too few test months")
    prefix = "validation_" if stage == "validation" else ""
    frames = {f"{prefix}predictions.csv": prediction_frame,
              f"{prefix}information_sets.csv": pd.DataFrame(ledgers),
              f"{prefix}coefficients.csv": pd.DataFrame(coefficients),
              f"{prefix}adjustment_coefficients.csv": pd.concat(adjustments, ignore_index=True),
              f"{prefix}feature_eligibility.csv": pd.concat(audits, ignore_index=True),
              f"{prefix}adjusted_daily_snapshot.csv": pd.concat(snapshots, ignore_index=True),
              f"{prefix}metrics.csv": scores(prediction_frame)}
    metadata = {"stage": stage, "fingerprint": expected, "experiment_id": frozen_id,
                "evaluation_months": len(expected_months), "scenarios": scenarios,
                "prediction_rows": len(prediction_frame), "value_vintage": "revised_IIP_pseudo_real_time"}
    atomic_outputs(output, frames, f"{stage}_run.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--stage", choices=["validation", "freeze", "test"], required=True)
    parser.add_argument("--review-note", default="", help="Required for freeze; record the completed validation review")
    args = parser.parse_args()
    if args.stage == "freeze":
        require(args.review_note.strip(), "A validation review note is required before opening the test")
        result = freeze_experiment(args.config.resolve(), args.review_note)
        print(json.dumps({"frozen_utc": result["frozen_utc"], "selected_ridge_alpha": result["selected_ridge_alpha"]}, indent=2))
    else:
        result = run(args.config.resolve(), args.stage)
        print(json.dumps({key: result[key] for key in ["stage", "evaluation_months", "prediction_rows"]}, indent=2))
