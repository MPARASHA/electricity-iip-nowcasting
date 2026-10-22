"""Information-set, transformation and experiment-freeze regression checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from prepare_data import sha256


ROOT = Path(__file__).parent


class EstimationTests(unittest.TestCase):
    def test_ridge_does_not_shrink_the_intercept_and_scales_only_training_rows(self):
        x = np.array([[1., 5.], [2., 4.], [3., 8.], [5., 0.]])
        prediction, intercept, slopes, means, scales = bt.fit_monthly(x, np.full(4, 17.), [10000., -5000.], 1000.)
        self.assertAlmostEqual(prediction, 17.)
        self.assertAlmostEqual(intercept, 17.)
        np.testing.assert_allclose(slopes, 0.)
        np.testing.assert_allclose(means, x.mean(axis=0))
        np.testing.assert_allclose(scales, x.std(axis=0))

    def test_ols_recovers_linear_prediction_and_rejects_nonfinite_inputs(self):
        x = np.array([[1., 5.], [2., 4.], [3., 8.], [5., 0.]])
        y = 3 + 2 * x[:, 0] - x[:, 1]
        prediction, intercept, slopes, *_ = bt.fit_monthly(x, y, [7., 2.])
        self.assertAlmostEqual(prediction, 15.)
        self.assertAlmostEqual(intercept, 3.)
        np.testing.assert_allclose(slopes, [2., -1.])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            bt.fit_monthly(x, y, [np.nan, 2.])

    def test_freeze_detects_changed_inputs_or_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "validation_predictions.csv").write_text("original")
            validation = {"outputs": {"validation_predictions.csv": sha256(output / "validation_predictions.csv")}}
            (output / "validation_run.json").write_text(json.dumps(validation))
            frozen = {"fingerprint": {"config": "locked"}, "validation_run_sha256": sha256(output / "validation_run.json")}
            (output / "experiment_freeze.json").write_text(json.dumps(frozen))
            bt.check_freeze(output, {"config": "locked"})
            with self.assertRaisesRegex(ValueError, "Frozen code"):
                bt.check_freeze(output, {"config": "changed"})
            (output / "validation_predictions.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Validation output changed"):
                bt.check_freeze(output, {"config": "locked"})


class HistoricalInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.config, cls.daily, cls.iip, cls.origins, cls.audit = bt.load_inputs(ROOT / "config.json")
        cls.scenario = cls.config["scenarios"][0]

    def predict(self, month, daily=None, iip=None, scenario=None):
        return bt.predict_origin(pd.Period(month, "M"), scenario or self.scenario, self.config,
                                 self.daily if daily is None else daily,
                                 self.iip if iip is None else iip, self.origins, self.audit, [1.])

    def test_future_daily_data_and_unreleased_iip_cannot_change_prediction(self):
        original = self.predict("2019-01")
        daily, iip = self.daily.copy(), self.iip.copy()
        daily.loc[daily.index > "2019-01-31", "energy_MU"] *= 100
        columns = [column for column in daily if column.endswith("_C")]
        daily.loc[daily.index > "2019-01-31", columns] = 65.
        # Includes December 2018, the current target and every later label.
        iip.loc[iip.release_date > "2019-02-05", "yoy_pct"] += 10000.
        changed = self.predict("2019-01", daily, iip)
        np.testing.assert_allclose([row["prediction_pct"] for row in original[0]],
                                   [row["prediction_pct"] for row in changed[0]], rtol=0, atol=0)
        pd.testing.assert_frame_equal(original[3], changed[3])
        self.assertEqual(original[1]["last_training_label"], "2018-11")
        self.assertLessEqual(original[1]["adjustment_last_day"], "2018-11-30")

    def test_unlabelled_current_period_does_not_enter_adjustment_fit(self):
        first = self.predict("2019-01")
        changed = self.daily.copy()
        changed.loc["2018-12-01":"2019-01-31", "energy_MU"] *= 1.2
        second = self.predict("2019-01", changed)
        pd.testing.assert_frame_equal(first[3], second[3])
        self.assertNotAlmostEqual(first[0][1]["prediction_pct"], second[0][1]["prediction_pct"])

    def test_weekly_gap_is_not_hidden_by_good_monthly_coverage(self):
        changed = self.daily.copy()
        changed.loc["2019-01-31", "energy_usable"] = False
        with self.assertRaisesRegex(ValueError, "lacks complete features"):
            self.predict("2019-01", changed)

    def test_adjustment_keeps_trend_and_residual_level(self):
        daily = self.daily.loc[:"2018-12-31"].copy()
        design = bt.daily_design(daily, self.config)
        beta = np.zeros(design.shape[1])
        beta[0], beta[1] = 8., 0.02
        beta[2:] = np.linspace(-0.01, 0.01, len(beta) - 2)
        daily["energy_MU"] = np.exp(design.to_numpy() @ beta)
        daily["energy_usable"] = True
        adjusted, info, coefficients = bt.adjust_energy(daily, pd.Timestamp("2018-12-31"), pd.Timestamp("2019-01-05"),
                                                        self.scenario, self.config, {})
        expected_log = 8 + 0.02 * design.trend_years
        np.testing.assert_allclose(np.log(adjusted.adjusted_energy), expected_log, atol=1e-11)
        self.assertAlmostEqual(info["retained_trend_log_per_year"], 0.02)
        self.assertEqual(coefficients.role.tolist()[:2], ["retained", "retained"])

    def test_primary_first_test_origin_is_released_and_models_share_training_months(self):
        predictions, ledger, *_ = self.predict("2022-01")
        self.assertEqual(ledger["latest_released_label"], "2021-11")
        self.assertEqual(ledger["last_training_label"], "2021-11")
        self.assertEqual(len({row["training_months"] for row in predictions}), 1)
        self.assertLessEqual(ledger["last_training_release_date"], ledger["forecast_date"])
        self.assertLessEqual(ledger["adjustment_last_day"], "2021-11-30")

    def test_pandemic_sensitivity_removes_training_days_and_labels_only(self):
        primary = self.predict("2022-01")
        scenario = copy.deepcopy(self.scenario)
        scenario.update(name="pandemic_excluded", exclude_pandemic_training=True)
        alternative = self.predict("2022-01", scenario=scenario)
        self.assertLess(alternative[1]["adjustment_training_days"], primary[1]["adjustment_training_days"])
        self.assertEqual(primary[1]["training_months"] - alternative[1]["training_months"], 16)
        self.assertEqual(primary[1]["daily_cutoff"], alternative[1]["daily_cutoff"])


if __name__ == "__main__":
    unittest.main()
