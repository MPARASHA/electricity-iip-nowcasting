"""Critical information-timing, exclusion, and estimator checks."""

import unittest

import numpy as np
import pandas as pd

from backtest import fit_monthly
from extension.backtest import Experiment, exposure_reasons, load_inputs
from extension.report import bootstrap_indices


class ExtensionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.daily, cls.iip = load_inputs()
        cls.experiment = Experiment(cls.config, cls.daily, cls.iip)

    def test_actual_release_change_and_delayed_october(self):
        ledger = self.experiment.static
        self.assertEqual(ledger.loc[pd.Period("2025-03", "M"), "lag1_month"], "2025-01")
        self.assertEqual(ledger.loc[pd.Period("2025-04", "M"), "lag1_month"], "2025-03")
        # October was actually released 1 December, before the 5 December origin.
        self.assertEqual(ledger.loc[pd.Period("2025-11", "M"), "lag1_month"], "2025-10")
        self.assertTrue((ledger.lag1_release_date <= ledger.forecast_date).all())
        self.assertTrue((ledger.lag2_release_date <= ledger.forecast_date).all())

    def test_strict_boundary_and_exact_validation_calendar(self):
        ledger = self.experiment.static
        september = ledger.loc[pd.Period("2023-09", "M")]
        self.assertIn("lag2_annual_base", september.shock_exposure)
        self.assertFalse(september.strict_eligible)
        self.assertTrue(ledger.loc[pd.Period("2023-10", "M"), "strict_eligible"])
        available = self.experiment.eligible_months("strict_covid")
        validation = available[(available >= pd.Period("2019-01", "M")) & (available <= pd.Period("2023-11", "M"))]
        nominal = pd.period_range("2019-01", "2020-02", freq="M").append(pd.period_range("2023-10", "2023-11", freq="M"))
        self.assertTrue(validation.isin(nominal).all())
        self.assertGreaterEqual(len(validation), self.config["minimum_validation_origins"])
        self.assertEqual(sum(validation >= pd.Period("2023-10", "M")), 2)

    def test_electricity_window_and_annual_base_exposure(self):
        month = pd.Period("2023-01", "M")
        reasons = exposure_reasons(month, [month - 2, month - 3],
                                   {"annual_electricity": pd.date_range("2022-01-01", "2022-01-31")}, self.config)
        self.assertIn("outcome_annual_base", reasons)
        self.assertIn("annual_electricity", reasons)
        self.assertFalse(exposure_reasons(pd.Period("2024-01", "M"), [pd.Period("2023-11", "M"), pd.Period("2023-10", "M")],
                                         {"current": pd.date_range("2024-01-01", "2024-01-31")}, self.config))

    def test_leap_year_and_complete_week(self):
        month = pd.Period("2024-02", "M")
        row = self.experiment.static.loc[month]
        self.assertEqual(row.daily_cutoff, pd.Timestamp("2024-02-29"))
        self.assertEqual(row.electricity_annual_base_last, pd.Timestamp("2023-02-28"))
        self.assertEqual(len(self.experiment.windows[month]["electricity_recent_week"]), 7)
        changed = self.daily.copy()
        changed.loc["2024-02-29", "energy_usable"] = False
        modified = Experiment(self.config, changed, self.iip)
        self.assertFalse(modified.static.loc[month, "eligible"])

    def test_late_report_cannot_repair_historical_feature(self):
        day = pd.Timestamp("2024-01-31")
        origin = pd.Timestamp("2024-02-05")
        previous = self.experiment.issues.get(day)
        try:
            self.experiment.issues[day] = pd.Timestamp("2024-02-06")
            self.assertFalse(self.experiment.own_usable(origin)[self.daily.index.get_loc(day)])
        finally:
            if previous is None:
                del self.experiment.issues[day]
            else:
                self.experiment.issues[day] = previous

    def test_future_labels_and_daily_values_do_not_change_prediction(self):
        month = pd.Period("2024-01", "M")
        penalties = {target: [1.] for target in self.experiment.targets}
        original = self.experiment.predict(month, "strict_covid", penalties)
        changed_iip = self.iip.copy()
        changed_iip.loc[changed_iip.month >= month, "yoy_pct"] = 1e8
        changed_daily = self.daily.copy()
        changed_daily.loc["2024-02-01":, "energy_MU"] *= 1000
        modified = Experiment(self.config, changed_daily, changed_iip).predict(month, "strict_covid", penalties)
        np.testing.assert_allclose([row["prediction_pct"] for row in original[0]], [row["prediction_pct"] for row in modified[0]], rtol=0, atol=1e-10)
        self.assertEqual(original[1]["daily_fit_shock_rows"], 0)
        months = original[1]["training_month_list"].split("|")
        self.assertNotIn(str(month), months)

    def test_training_scaling_and_ridge_translation(self):
        x = np.array([[1.], [2.], [3.], [4.]])
        y = 2 + 3 * x[:, 0]
        prediction, _, _, means, scales = fit_monthly(x, y, np.array([1000.]), 0.)
        self.assertAlmostEqual(prediction, 3002.)
        self.assertEqual(means[0], 2.5)
        self.assertAlmostEqual(scales[0], np.std(x))
        a = fit_monthly(x, y, [5.], 10.)[0]
        b = fit_monthly(x, y + 20, [5.], 10.)[0]
        self.assertAlmostEqual(b - a, 20.)

    def test_bootstrap_preserves_gaps_and_short_runs(self):
        months = ["2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2023-10", "2023-11"]
        spec = {"seed": 241022, "block_months": 3, "repetitions": 40}
        draws = bootstrap_indices(months, spec)
        self.assertEqual(draws.shape, (40, 7))
        self.assertTrue((draws[:, :5] < 5).all())
        np.testing.assert_array_equal(draws[:, 5:], np.tile([5, 6], (40, 1)))
        np.testing.assert_array_equal(np.diff(draws[:, :3], axis=1), np.ones((40, 2)))
        np.testing.assert_array_equal(draws, bootstrap_indices(months, spec))


if __name__ == "__main__":
    unittest.main(verbosity=2)
