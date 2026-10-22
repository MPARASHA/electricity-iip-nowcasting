"""Chronological validation, prospective freeze, and separate holdout evaluation."""

import argparse
import datetime as dt
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from backtest import daily_design, fit_monthly
from prepare_data import require, sha256
from extension.prepare_data import DATA, HERE, ROOT, read_config, save_json, verify_original

MODELS = ["iip_benchmark", "raw_ols", "adjusted_ols", "adjusted_ridge"]
BENCHMARK = ["iip_latest_yoy_pct", "iip_previous_yoy_pct", "month_sin", "month_cos", "diwali_share"]
RESULTS = HERE / "results"


def load_inputs():
    config = read_config()
    versions = json.loads((DATA / "versions.json").read_text())
    require(versions["config_sha256"] == sha256(HERE / "config.json"), "Configuration changed; prepare again")
    require(versions["preparation_sha256"] == sha256(HERE / "prepare_data.py"), "Preparation code changed")
    for name, checksum in versions["outputs"].items():
        require(sha256(DATA / name) == checksum, f"Prepared input changed: {name}")
    require(versions["raw_manifest_sha256"] == sha256(HERE / "raw/manifest.json"), "Raw manifest changed")
    for name, checksum in versions["original_inputs"].items():
        require(sha256(ROOT / name) == checksum, f"Original source changed: {name}")
    verify_original()
    daily = pd.read_csv(DATA / "daily.csv", index_col="date", parse_dates=["date"])
    iip = pd.read_csv(DATA / "iip.csv", parse_dates=["release_date"])
    iip["month"] = pd.PeriodIndex(iip["month"], freq="M")
    return config, daily, iip


def touches_shock(month, config):
    lower, upper = [pd.Period(value, "M") for value in config["shock"]]
    return lower <= month <= upper


def exposure_reasons(month, lags, windows, config):
    """Audit actual components; never compress the calendar after exclusion."""
    reasons = []
    for name, value in [("outcome", month), ("outcome_annual_base", month - 12)]:
        if touches_shock(value, config):
            reasons.append(name)
    for number, lag in enumerate(lags, 1):
        for name, value in [(f"lag{number}_index", lag), (f"lag{number}_annual_base", lag - 12)]:
            if touches_shock(value, config):
                reasons.append(name)
    for name, days in windows.items():
        if any(touches_shock(day.to_period("M"), config) for day in days):
            reasons.append(name)
    return reasons


class Experiment:
    """One explicit information ledger and matched sample shared by all setups."""

    def __init__(self, config, daily, iip):
        self.config, self.daily = config, daily
        self.targets = [item["id"] for item in config["targets"]]
        self.iip = {target: iip.loc[iip.target.eq(target)].set_index("month").sort_index() for target in self.targets}
        self.reference = self.iip[self.targets[0]]
        for target in self.targets[1:]:
            require(self.iip[target].release_date.equals(self.reference.release_date), "Target release calendars differ")
        self.design = daily_design(daily, config)
        self.usable = (daily.energy_usable & daily.weather_complete).to_numpy()
        self.issues = {}
        for path in [ROOT / "data/processed/report_audit.csv", DATA / "report_audit.csv"]:
            audit = pd.read_csv(path)
            for row in audit.itertuples():
                self.issues[pd.Timestamp(row.date)] = pd.Timestamp(row.printed_reporting_date)
        self.months = pd.period_range(config["target_start"], config["test"][1], freq="M")
        self.static, self.windows = self.build_calendar()

    def own_usable(self, forecast):
        mask = self.usable & (self.daily.index + pd.Timedelta(days=self.config["daily_lag_days"]) <= forecast)
        for day, issued in self.issues.items():
            if day in self.daily.index and issued > forecast:
                mask[self.daily.index.get_loc(day)] = False
        return mask

    def build_calendar(self):
        rows, all_windows = [], {}
        for month in self.months:
            forecast = (month + 1).start_time + pd.Timedelta(days=self.config["forecast_day"] - 1)
            cutoff = min(forecast - pd.Timedelta(days=self.config["daily_lag_days"]), month.end_time.normalize())
            previous_cutoff = cutoff - pd.DateOffset(years=1)
            weeks = self.config["weekly_days"]
            days = {"electricity_current": pd.date_range(month.start_time, cutoff),
                    "electricity_annual_base": pd.date_range((month - 12).start_time, previous_cutoff),
                    "electricity_recent_week": pd.date_range(cutoff - pd.Timedelta(days=weeks - 1), cutoff),
                    "electricity_previous_week": pd.date_range(cutoff - pd.Timedelta(days=2 * weeks - 1), cutoff - pd.Timedelta(days=weeks))}
            windows = {name: self.daily.index.get_indexer(value) for name, value in days.items()}
            require(all((indices >= 0).all() for indices in windows.values()), "Electricity window outside history")
            available = self.own_usable(forecast)
            windows = {name: indices[available[indices]] for name, indices in windows.items()}
            coverage = {name: len(windows[name]) / len(value) for name, value in days.items()}
            eligible = all(value >= self.config["coverage"] for value in coverage.values())
            # At a 90% threshold a seven-day window must contain all seven days.
            released = self.reference.loc[(self.reference.release_date <= forecast) & (self.reference.index < month) & self.reference.yoy_pct.notna()]
            lags = list(released.index[-2:][::-1])
            if len(lags) != 2:
                eligible = False
            reasons = exposure_reasons(month, lags, days, self.config)
            row = {"month": month, "forecast_date": forecast, "daily_cutoff": cutoff,
                   "lag1_month": str(lags[0]) if lags else "", "lag2_month": str(lags[1]) if len(lags) > 1 else "",
                   "lag1_release_date": released.release_date.iloc[-1] if len(lags) else pd.NaT,
                   "lag2_release_date": released.release_date.iloc[-2] if len(lags) > 1 else pd.NaT,
                   "eligible": eligible, "strict_eligible": eligible and not reasons,
                   "shock_exposure": "|".join(reasons), "exclusion_reason": "" if eligible else "coverage_or_unreleased_lag",
                   "month_sin": np.sin(2 * np.pi * month.month / 12), "month_cos": np.cos(2 * np.pi * month.month / 12),
                   "diwali_share": self.daily.loc[month.start_time:month.end_time, "diwali_window"].mean(),
                   **{name + "_coverage": value for name, value in coverage.items()}}
            for name, window_days in days.items():
                row[name + "_first"] = window_days.min()
                row[name + "_last"] = window_days.max()
            rows.append(row)
            all_windows[month] = windows
        return pd.DataFrame(rows).set_index("month"), all_windows

    def eligible_months(self, case):
        return self.static.index[self.static.strict_eligible if case == "strict_covid" else self.static.eligible]

    def adjusted(self, month, case):
        origin = self.static.loc[month]
        released = self.reference.loc[(self.reference.release_date <= origin.forecast_date) &
                                      (self.reference.index < month) & self.reference.yoy_pct.notna()]
        fit_end = released.index.max().end_time.normalize()
        fit = self.own_usable(origin.forecast_date) & (self.daily.index <= fit_end)
        if case == "strict_covid":
            low, high = [pd.Period(value, "M") for value in self.config["shock"]]
            fit &= ~((self.daily.index.to_period("M") >= low) & (self.daily.index.to_period("M") <= high))
        require(int(fit.sum()) >= 730, "Too few daily adjustment rows")
        x = self.design.loc[fit].to_numpy()
        y = np.log(self.daily.loc[fit, "energy_MU"].to_numpy())
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        require(rank == x.shape[1], "Rank-deficient weather adjustment")
        values = self.daily.energy_MU.to_numpy() * np.exp(-self.design.iloc[:, 2:].to_numpy() @ beta[2:])
        # No feature lookup may use future data even though a vector is computed for convenience.
        values[self.daily.index > origin.daily_cutoff] = np.nan
        meta = {"daily_fit_first": str(self.daily.index[fit].min().date()),
                "daily_fit_last": str(self.daily.index[fit].max().date()), "daily_fit_rows": int(fit.sum()),
                "daily_fit_shock_rows": int(sum(touches_shock(day.to_period("M"), self.config) for day in self.daily.index[fit])),
                "daily_design_rank": int(rank), "latest_released_label": str(released.index.max())}
        require(pd.Timestamp(meta["daily_fit_last"]) <= fit_end < month.start_time, "Future adjustment fit")
        if case == "strict_covid":
            require(meta["daily_fit_shock_rows"] == 0, "Strict fit contains shock days")
        return values, meta, beta

    def energy_features(self, months, adjusted):
        rows = []
        raw = self.daily.energy_MU.to_numpy()
        for month in months:
            windows = self.windows[month]
            row = {"month": month}
            for prefix, values in [("raw", raw), ("adjusted", adjusted)]:
                means = {name: np.mean(values[indices]) for name, indices in windows.items()}
                row[prefix + "_energy_yoy_pct"] = 100 * (means["electricity_current"] / means["electricity_annual_base"] - 1)
                row[prefix + "_weekly_momentum_pct"] = 100 * (means["electricity_recent_week"] / means["electricity_previous_week"] - 1)
            rows.append(row)
        return pd.DataFrame(rows).set_index("month")

    def predict(self, month, case, penalties):
        require(month in self.eligible_months(case), f"Ineligible forecast {month}, {case}")
        forecast = self.static.loc[month, "forecast_date"]
        released = self.reference.index[(self.reference.release_date <= forecast) & (self.reference.index < month)]
        training = self.eligible_months(case).intersection(released)
        require(len(training) >= self.config["minimum_training_rows"], "Insufficient matched training months")
        require(self.reference.loc[training, "release_date"].max() <= forecast, "Unreleased training label")
        require(not (training >= month).any(), "Current or future target enters training")
        adjusted, daily_meta, daily_beta = self.adjusted(month, case)
        requested = training.append(pd.PeriodIndex([month], freq="M"))
        features = self.static.loc[requested, ["month_sin", "month_cos", "diwali_share"]].join(self.energy_features(requested, adjusted))
        predictions, coefficients, feature_rows = [], [], []
        for target in self.targets:
            iip = self.iip[target]
            target_features = features.copy()
            for name, lag_column in [("iip_latest_yoy_pct", "lag1_month"), ("iip_previous_yoy_pct", "lag2_month")]:
                lags = pd.PeriodIndex(self.static.loc[requested, lag_column], freq="M")
                target_features[name] = iip.reindex(lags).yoy_pct.to_numpy()
            y = iip.loc[training, "yoy_pct"]
            for model in MODELS:
                columns = BENCHMARK.copy()
                if model != "iip_benchmark":
                    prefix = "raw" if model == "raw_ols" else "adjusted"
                    columns += [prefix + "_energy_yoy_pct", prefix + "_weekly_momentum_pct"]
                for alpha in penalties[target] if model == "adjusted_ridge" else [0.]:
                    value, intercept, slopes, means, scales = fit_monthly(target_features.loc[training, columns], y,
                                                                        target_features.loc[month, columns], alpha)
                    predictions.append({"month": str(month), "case": case, "target": target, "model": model,
                                        "alpha": alpha, "prediction_pct": value, "training_months": len(training)})
                    for term, beta, mean, scale in zip(["intercept"] + columns, [intercept] + list(slopes),
                                                      [0.] + list(means), [1.] + list(scales)):
                        coefficients.append({"month": str(month), "case": case, "target": target, "model": model,
                                             "alpha": alpha, "term": term, "coefficient": beta,
                                             "training_mean": mean, "training_sd": scale})
            feature_rows.append({"month": str(month), "case": case, "target": target,
                                 **target_features.loc[month].to_dict()})
        ledger = {"month": str(month), "case": case, "forecast_date": str(forecast.date()),
                  "training_rows": len(training), "first_training_month": str(training.min()),
                  "last_training_month": str(training.max()), "last_training_release": str(self.reference.loc[training, "release_date"].max().date()),
                  "training_month_list": "|".join(training.astype(str)), **daily_meta}
        daily_coefficients = [{"month": str(month), "case": case, "term": term, "coefficient": float(value)}
                              for term, value in zip(self.design.columns, daily_beta)]
        return predictions, ledger, coefficients, feature_rows, daily_coefficients


def scores(frame):
    rows = []
    for (case, target, model, alpha), group in frame.groupby(["case", "target", "model", "alpha"], sort=True):
        error = group.prediction_pct - group.actual_pct
        rows.append({"case": case, "target": target, "model": model, "alpha": float(alpha), "months": len(group),
                     "MAE_pp": float(np.abs(error).mean()), "RMSE_pp": float(np.sqrt((error ** 2).mean()))})
    return pd.DataFrame(rows)


def with_actuals(predictions, experiment):
    frame = pd.DataFrame(predictions)
    frame["actual_pct"] = [experiment.iip[target].loc[pd.Period(month, "M"), "yoy_pct"]
                           for target, month in zip(frame.target, frame.month)]
    return frame


def save_frames(prefix, predictions, ledgers, coefficients, features, daily_coefficients, experiment):
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = with_actuals(predictions, experiment)
    for name, value in {"predictions": frame, "scores": scores(frame), "ledger": pd.DataFrame(ledgers),
                        "coefficients": pd.DataFrame(coefficients), "features": pd.DataFrame(features),
                        "daily_coefficients": pd.DataFrame(daily_coefficients)}.items():
        value.to_csv(RESULTS / f"{prefix}_{name}.csv", index=False, float_format="%.12g")
    return frame


def validation(experiment):
    config = experiment.config
    predictions, ledgers, coefficients, features, daily_coefficients = [], [], [], [], []
    sample = []
    for case in config["cases"]:
        candidates = experiment.eligible_months(case)
        months = candidates[(candidates >= pd.Period(config["validation"][0], "M")) &
                            (candidates <= pd.Period(config["validation"][1], "M"))]
        months = months[experiment.reference.loc[months, "release_date"] < pd.Timestamp(config["first_test_origin"])]
        require(len(months) >= config["minimum_validation_origins"], "Too few validation origins")
        for month in months:
            output = experiment.predict(month, case, {target: config["ridge_alphas"] for target in experiment.targets})
            predictions.extend(output[0]); ledgers.append(output[1]); coefficients.extend(output[2])
            features.extend(output[3]); daily_coefficients.extend(output[4])
        sample.append({"case": case, "origins": len(months), "months": months.astype(str).tolist(),
                       "recent_post_exposure_origins": int(sum(month >= pd.Period("2023-10", "M") for month in months))})
        print(f"Validation {case}: {len(months)} origins", flush=True)
    frame = save_frames("validation", predictions, ledgers, coefficients, features, daily_coefficients, experiment)
    chosen = []
    table = scores(frame)
    for case in config["cases"]:
        for target in experiment.targets:
            options = table.loc[table.case.eq(case) & table.target.eq(target) & table.model.eq("adjusted_ridge")]
            winner = options.sort_values(["MAE_pp", "alpha"]).iloc[0]
            chosen.append({"case": case, "target": target, "alpha": float(winner.alpha),
                           "validation_MAE_pp": float(winner.MAE_pp), "validation_origins": int(winner.months)})
    save_json(RESULTS / "selected_alphas.json", chosen)
    save_json(RESULTS / "validation_sample.json", sample)
    experiment.static.to_csv(RESULTS / "exposure_ledger.csv", float_format="%.12g")
    print(json.dumps(chosen, indent=2))


def fingerprint():
    paths = [HERE / name for name in ["config.json", "prepare_data.py", "backtest.py", "report.py", "test_extension.py"]]
    paths += list(sorted(DATA.glob("*")))
    paths += [RESULTS / name for name in ["selected_alphas.json", "validation_sample.json", "validation_predictions.csv", "exposure_ledger.csv"]]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def freeze(experiment):
    require(not (RESULTS / "test_predictions.csv").exists(), "Holdout predictions already exist; cannot make a new prospective freeze")
    require((RESULTS / "selected_alphas.json").exists(), "Validate before freezing")
    checked = subprocess.run([sys.executable, "-m", "unittest", "extension.test_extension", "-v"], cwd=ROOT,
                             capture_output=True, text=True)
    (HERE / "qa/test_results.txt").write_text(checked.stdout + checked.stderr)
    require(checked.returncode == 0, "Critical checks failed; see extension/qa/test_results.txt")
    sample = json.loads((RESULTS / "validation_sample.json").read_text())
    review = json.loads((DATA / "source_review.json").read_text())
    require(len(review["reports"]) >= experiment.config["required_new_reports"], "Visual source audit incomplete")
    for row in review["reports"]:
        require(row["visual_status"] == "checked" and sha256(HERE / "raw" / row["file"]) == row["sha256"],
                "Reviewed source bytes changed")
    common = pd.period_range(*experiment.config["test"], freq="M")
    for case in experiment.config["cases"]:
        common = common.intersection(experiment.eligible_months(case))
    require(len(common) >= experiment.config["minimum_test_months"], "Insufficient common test months")
    counts = []
    for case in experiment.config["cases"]:
        for month in common:
            forecast = experiment.static.loc[month, "forecast_date"]
            released = experiment.reference.index[(experiment.reference.release_date <= forecast) & (experiment.reference.index < month)]
            training = experiment.eligible_months(case).intersection(released)
            require(len(training) >= experiment.config["minimum_training_rows"], "Holdout training sample below minimum")
            counts.append({"month": str(month), "case": case, "training_rows": len(training)})
    evidence = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "fingerprints": fingerprint(),
                "test_months": common.astype(str).tolist(), "validation_sample": sample, "prospective_training_counts": counts,
                "accepted_limitation": "Strict validation has only two recent post-exposure origins; penalties and intervals are exploratory",
                "protocol": experiment.config["protocol"], "uncertainty": experiment.config["uncertainty"]}
    path = RESULTS / "experiment_freeze.json"
    require(not path.exists(), "Freeze already exists; it must not be silently overwritten")
    save_json(path, evidence)
    print(f"Frozen {len(common)} common holdout months before test scoring")


def test(experiment, output_prefix="test"):
    frozen = json.loads((RESULTS / "experiment_freeze.json").read_text())
    require(frozen["fingerprints"] == fingerprint(), "Frozen code/data/validation changed; stop before holdout scoring")
    selection = {(row["case"], row["target"]): row["alpha"] for row in json.loads((RESULTS / "selected_alphas.json").read_text())}
    predictions, ledgers, coefficients, features, daily_coefficients = [], [], [], [], []
    for case in experiment.config["cases"]:
        for month in pd.PeriodIndex(frozen["test_months"], freq="M"):
            output = experiment.predict(month, case, {target: [selection[case, target]] for target in experiment.targets})
            predictions.extend(output[0]); ledgers.append(output[1]); coefficients.extend(output[2])
            features.extend(output[3]); daily_coefficients.extend(output[4])
        print(f"Holdout {case}: {len(frozen['test_months'])} common months", flush=True)
    frame = save_frames(output_prefix, predictions, ledgers, coefficients, features, daily_coefficients, experiment)
    require(len(frame) == len(frozen["test_months"]) * 4 * 4 * 2, "Missing holdout cells")
    save_json(RESULTS / f"{output_prefix}_run.json", {"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "freeze_sha256": sha256(RESULTS / "experiment_freeze.json"),
              "outputs": {path.name: sha256(path) for path in sorted(RESULTS.glob(f"{output_prefix}_*.csv"))}})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["validate", "freeze", "test", "reproduce"])
    args = parser.parse_args()
    config, daily, iip = load_inputs()
    experiment = Experiment(config, daily, iip)
    if args.stage == "validate":
        require(not (RESULTS / "experiment_freeze.json").exists(), "Do not retune after freeze")
        validation(experiment)
    elif args.stage == "freeze":
        freeze(experiment)
    else:
        test(experiment, "reproduce" if args.stage == "reproduce" else "test")


if __name__ == "__main__":
    main()
