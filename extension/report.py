"""Read saved forecasts; calculate paired uncertainty and scientific figures."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extension.backtest import MODELS, RESULTS
from extension.prepare_data import HERE, read_config, save_json
from prepare_data import require, sha256


def bootstrap_indices(months, spec):
    """Noncircular moving blocks inside real contiguous calendar runs."""
    months = pd.PeriodIndex(months, freq="M")
    require(months.is_monotonic_increasing and not months.duplicated().any(), "Bootstrap calendar must be unique and sorted")
    breaks = np.flatnonzero(np.diff(months.asi8) != 1) + 1
    runs = np.split(np.arange(len(months)), breaks)
    rng = np.random.default_rng(spec["seed"])
    outputs = []
    for run in runs:
        length = min(spec["block_months"], len(run))
        starts = rng.integers(0, len(run) - length + 1, size=(spec["repetitions"], int(np.ceil(len(run) / length))))
        positions = (starts[:, :, None] + np.arange(length)).reshape(spec["repetitions"], -1)[:, :len(run)]
        outputs.append(run[positions])
    return np.concatenate(outputs, axis=1)


def summarize(frame, config):
    months = sorted(frame.month.unique())
    expected = {(case, target["id"], model) for case in config["cases"] for target in config["targets"] for model in MODELS}
    require(set(frame.groupby(["case", "target", "model"]).groups) == expected, "Missing result cells")
    require(not frame[["case", "target", "model", "month"]].duplicated().any(), "Duplicate forecast cells")
    errors = {}
    metrics = []
    for key, group in frame.groupby(["case", "target", "model"], sort=True):
        group = group.sort_values("month")
        require(group.month.tolist() == months, "Unmatched test calendar")
        error = group.prediction_pct.to_numpy() - group.actual_pct.to_numpy()
        errors[key] = np.abs(error)
        metrics.append({"case": key[0], "target": key[1], "model": key[2], "months": len(error),
                        "MAE_pp": float(np.abs(error).mean()), "RMSE_pp": float(np.sqrt((error ** 2).mean()))})
    table = pd.DataFrame(metrics)
    table["relative_MAE_gain_pct"] = [100 * (errors[case, target, "iip_benchmark"].mean() - mae) / errors[case, target, "iip_benchmark"].mean()
                                        if errors[case, target, "iip_benchmark"].mean() != 0 else np.nan
                                        for case, target, mae in zip(table.case, table.target, table.MAE_pp)]
    draws = bootstrap_indices(months, config["uncertainty"])
    contrasts, influence = [], []
    comparisons = [("adjusted_ridge", "iip_benchmark", "complete_method_primary"),
                   ("raw_ols", "iip_benchmark", "added_raw_electricity_OLS"),
                   ("adjusted_ols", "iip_benchmark", "added_adjusted_electricity_OLS"),
                   ("adjusted_ols", "raw_ols", "adjustment_OLS")]
    tail = (1 - config["uncertainty"]["coverage"]) / 2
    for case in config["cases"]:
        for target in [item["id"] for item in config["targets"]]:
            for left, right, meaning in comparisons:
                delta = errors[case, target, left] - errors[case, target, right]
                low, high = np.quantile(delta[draws].mean(axis=1), [tail, 1 - tail])
                contrasts.append({"case": case, "target": target, "left_model": left, "right_model": right,
                                  "interpretation": meaning, "MAE_difference_pp": float(delta.mean()),
                                  "interval_low_pp": float(low), "interval_high_pp": float(high)})
                for index, month in enumerate(months):
                    influence.append({"case": case, "target": target, "left_model": left, "right_model": right,
                                      "omitted_month": month, "MAE_difference_without_month_pp": float(np.delete(delta, index).mean()),
                                      "that_month_absolute_error_difference_pp": float(delta[index])})
    case_contrasts = []
    for target in [item["id"] for item in config["targets"]]:
        for model in MODELS:
            delta = errors["strict_covid", target, model] - errors["full", target, model]
            low, high = np.quantile(delta[draws].mean(axis=1), [tail, 1 - tail])
            case_contrasts.append({"target": target, "model": model, "strict_minus_full_MAE_pp": float(delta.mean()),
                                   "interval_low_pp": float(low), "interval_high_pp": float(high)})
    return table, pd.DataFrame(contrasts), pd.DataFrame(case_contrasts), pd.DataFrame(influence), draws


def figures(frame, table, config):
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    labels = {"iip_benchmark": "IIP OLS", "raw_ols": "Raw + OLS", "adjusted_ols": "Adjusted + OLS", "adjusted_ridge": "Adjusted + ridge"}
    colors = ["#50545c", "#75a1c7", "#376da0", "#af6639"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, target in zip(axes.flat, config["targets"]):
        for offset, case in [(-.18, "full"), (.18, "strict_covid")]:
            group = table.loc[table.target.eq(target["id"]) & table.case.eq(case)].set_index("model").reindex(MODELS)
            ax.bar(np.arange(4) + offset, group.MAE_pp, .34, color=colors if case == "full" else colors,
                   alpha=1 if case == "full" else .45, hatch=None if case == "full" else "//", label="Full history" if case == "full" else "Strict COVID exclusion")
        ax.set_title(target["name"])
        ax.set_xticks(np.arange(4), [labels[name] for name in MODELS], rotation=18, ha="right")
        ax.set_ylabel("MAE (percentage points)")
    axes.flat[0].legend(fontsize=8)
    fig.savefig(folder / "test_mae.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
    for ax, target in zip(axes, config["targets"]):
        group = frame.loc[frame.target.eq(target["id"]) & frame.case.eq("strict_covid")]
        dates = pd.to_datetime(sorted(group.month.unique()))
        for model, color, line in [("iip_benchmark", "#666666", "--"), ("adjusted_ridge", "#a75a2d", "-")]:
            series = group.loc[group.model.eq(model)].sort_values("month")
            ax.plot(dates, series.prediction_pct, line, color=color, label=labels[model])
        actual = group.loc[group.model.eq("iip_benchmark")].sort_values("month")
        ax.plot(dates, actual.actual_pct, color="#193f58", marker="o", markersize=3, label="Revised actual IIP")
        ax.axhline(0, color="#bbbbbb", linewidth=.6)
        ax.set_title(target["name"], loc="left")
        ax.set_ylabel("Growth (%)")
    axes[0].legend(ncol=3, fontsize=8)
    fig.savefig(folder / "strict_test_paths.png", dpi=180)
    plt.close(fig)


def main():
    config = read_config()
    run = json.loads((RESULTS / "test_run.json").read_text())
    for name, checksum in run["outputs"].items():
        require(sha256(RESULTS / name) == checksum, f"Saved test result changed: {name}")
    frame = pd.read_csv(RESULTS / "test_predictions.csv")
    table, contrasts, cases, influence, draws = summarize(frame, config)
    for name, value in [("metrics", table), ("paired_contrasts", contrasts), ("case_contrasts", cases), ("influence", influence)]:
        value.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.12g")
    np.save(RESULTS / "bootstrap_month_indices.npy", draws)
    save_json(RESULTS / "uncertainty_metadata.json", {**config["uncertainty"], "test_months": sorted(frame.month.unique()),
              "intervals": "exploratory paired differences, no multiplicity correction or causal inference"})
    figures(frame, table, config)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
