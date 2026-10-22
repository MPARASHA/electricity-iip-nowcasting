"""Regression checks for source integrity, date ambiguity and information timing."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfWriter

import prepare_data as prep


ROOT = Path(__file__).parent
RAW = ROOT / "research/feasibility_2026-10-22"


class InputRulesTests(unittest.TestCase):
    def test_changed_frozen_file_fails_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("original")
            record = {"file": path.name, "sha256": prep.sha256(path)}
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                prep.checked_file(Path(directory), record, download_missing=True)
            self.assertEqual(path.read_text(), "changed")

    def test_missing_energy_remains_missing_and_conflicting_values_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "metadata.json").write_text(json.dumps({"fields": [{"name": "India: EnergyMet", "units": "GWh"}]}))
            (raw / "energy.csv").write_text("yyyymmdd,India: EnergyMet\n20140101,2630\n20140102,\n")
            config = {"energy": {"metadata_file": "metadata.json", "file": "energy.csv", "column": "India: EnergyMet"},
                      "history_start": "2014-01-01", "history_end": "2014-01-02",
                      "energy_exclusions": [{"date": "2014-01-01", "reason": "unresolved_printed_year"}]}
            daily = prep.load_energy(raw, config)
            self.assertEqual(daily.iloc[0].energy_MU, 2630)
            self.assertFalse(daily.iloc[0].energy_usable)
            self.assertTrue(pd.isna(daily.iloc[1].energy_MU))
            self.assertFalse(daily.iloc[1].energy_usable)
            (raw / "energy.csv").write_text("yyyymmdd,India: EnergyMet\n20140101,2630\n20140101,2640\n")
            with self.assertRaisesRegex(ValueError, "Duplicate energy dates"):
                prep.load_energy(raw, config)
            (raw / "energy.csv").write_text("yyyymmdd,India: EnergyMet\n20140101,0\n")
            with self.assertRaisesRegex(ValueError, "Non-positive"):
                prep.load_energy(raw, config)

    def test_weather_fill_value_does_not_become_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            payload = {"geometry": {"coordinates": [77.21, 28.61]},
                       "header": {"time_standard": "UTC", "fill_value": -999},
                       "parameters": {name: {"units": "C"} for name in ["T2M", "T2M_MAX", "T2M_MIN"]},
                       "properties": {"parameter": {"T2M": {"20140101": 20, "20140102": -999},
                                                     "T2M_MIN": {"20140101": 15, "20140102": -999},
                                                     "T2M_MAX": {"20140101": 25, "20140102": 26}}}}
            (raw / "weather.json").write_text(json.dumps(payload))
            daily = pd.DataFrame(index=pd.date_range("2014-01-01", "2014-01-02"))
            config = {"weather_locations": [{"id": "delhi", "file": "weather.json", "latitude": 28.61, "longitude": 77.21}]}
            result = prep.add_weather(daily, raw, config)
            self.assertTrue(result.iloc[0].weather_complete)
            self.assertFalse(result.iloc[1].weather_complete)
            self.assertTrue(pd.isna(result.iloc[1].delhi_T2M_C))


class OriginalReportTests(unittest.TestCase):
    def test_evening_demand_is_never_used_as_daily_peak(self):
        early = prep.parse_report(RAW / "grid_original_140102.pdf")
        self.assertEqual(early["energy_MU"], 2630)
        self.assertTrue(np.isnan(early["national_peak_MW"]))
        self.assertEqual(early["evening_demand_MW"], 120228)
        self.assertEqual(early["printed_reporting_date"], pd.Timestamp("2013-01-02"))
        self.assertNotEqual(early["covered_date"], pd.Timestamp("2014-01-01"))
        later = prep.parse_report(RAW / "original_190101.pdf")
        self.assertEqual(later["national_peak_MW"], 158835)
        self.assertEqual(later["evening_demand_MW"], 149734)
        self.assertEqual(later["covered_date"], pd.Timestamp("2019-01-01"))

    def test_zero_scada_peak_is_explicitly_unusable(self):
        parsed = prep.parse_report(RAW / "grid_sample_20200325.pdf")
        self.assertEqual(parsed["national_peak_MW"], 0)
        self.assertEqual(parsed["national_peak_status"], "reported_zero_not_usable")

    def test_unrecognized_report_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=300, height=300)
            writer.write(path)
            with self.assertRaisesRegex(ValueError, "No supported"):
                prep.parse_report(path)


class FrozenHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text())
        raw, inventory = prep.verify_inputs(cls.config, ROOT)
        cls.daily = prep.add_weather(prep.load_energy(raw, cls.config), raw, cls.config)
        cls.iip = prep.load_iip(raw, cls.config, prep.load_release_calendar(raw, cls.config, inventory))
        cls.audit = prep.audit_reports(raw, cls.config, cls.daily)
        cls.origins = prep.build_origins(cls.daily, cls.iip, cls.config, cls.audit)

    def test_a_configured_source_must_be_in_the_frozen_manifests(self):
        config = copy.deepcopy(self.config)
        config["energy"]["file"] = "unmanifested_candidate.csv"
        with self.assertRaisesRegex(ValueError, "absent from the frozen manifests"):
            prep.verify_inputs(config, ROOT)

    def test_release_rule_at_first_validation_and_test_origins(self):
        core = self.origins.loc[self.origins.assumed_lag_days.eq(5)].set_index("month")
        first = core.loc["2019-01"]
        self.assertEqual(first.forecast_date, pd.Timestamp("2019-02-05"))
        self.assertEqual(first.latest_released_iip_month, "2018-11")
        self.assertEqual(first.latest_label_release_date, pd.Timestamp("2019-01-11"))
        self.assertEqual(first.eligible_released_training_labels, 47)
        self.assertGreater(self.iip.loc[pd.Period("2018-12"), "release_date"], first.forecast_date)
        self.assertEqual(core.loc["2022-01", "latest_released_iip_month"], "2021-11")
        self.assertEqual(core.loc["2022-01", "eligible_released_training_labels"], 83)
        evaluated = self.origins.loc[~self.origins.split.eq("training")]
        self.assertTrue((evaluated.latest_label_release_date <= evaluated.forecast_date).all())

    def test_pandemic_growth_uses_index_ratio_even_when_published_rate_is_blank(self):
        row = self.iip.loc[pd.Period("2021-04")]
        self.assertTrue(pd.isna(row.reported_yoy_pct))
        self.assertAlmostEqual(row.yoy_pct, 195.96199524940613)
        self.assertEqual(row.base_year, "2011-12")

    def test_coverage_checks_both_years_and_keeps_test_months(self):
        core = self.origins.loc[self.origins.assumed_lag_days.eq(5)].set_index("month")
        jan = core.loc["2015-01"]
        self.assertEqual(jan.prior_energy_days, 28)  # two date flags and one missing day
        self.assertEqual(jan.prior_expected_days, 31)
        self.assertTrue(jan.eligible_90)
        self.assertFalse(jan.eligible_95)
        early = self.origins.loc[self.origins.month.eq("2015-01") & self.origins.assumed_lag_days.eq(7)].iloc[0]
        self.assertFalse(early.eligible_90)
        test = self.origins.loc[self.origins.split.eq("test")]
        self.assertEqual(len(test), 72)
        self.assertTrue(test.eligible_90.all())
        self.assertTrue(test.eligible_95.all())

    def test_a_later_documented_issue_date_overrides_assumed_lag(self):
        audit = pd.concat([self.audit, pd.DataFrame([{"date": "2019-01-31", "printed_reporting_date": pd.Timestamp("2019-02-06"), "date_status": "consistent"}])], ignore_index=True)
        origins = prep.build_origins(self.daily, self.iip, self.config, audit)
        row = origins.loc[origins.month.eq("2019-01") & origins.assumed_lag_days.eq(5)].iloc[0]
        self.assertEqual(row.current_energy_days, 30)
        self.assertEqual(row.current_weather_coverage, 1)

    def test_leap_year_cutoff_and_source_conflict_are_preserved(self):
        leap = self.origins.loc[self.origins.month.eq("2020-02") & self.origins.assumed_lag_days.eq(5)].iloc[0]
        self.assertEqual(leap.daily_cutoff, pd.Timestamp("2020-02-29"))
        self.assertEqual(leap.prior_year_cutoff, pd.Timestamp("2019-02-28"))
        row = self.audit.set_index("date").loc["2017-02-15"]
        self.assertEqual(row.energy_MU, 3050)
        self.assertEqual(row.secondary_original_energy_MU, 3200)
        self.assertEqual(row.candidate_energy_status, "conflicting_originals_excluded")
        self.assertFalse(self.daily.loc["2017-02-15", "energy_usable"])
        self.assertEqual(self.daily.loc["2017-02-15", "energy_MU"], 3200)

    def test_offline_run_repeats_exactly_and_does_not_change_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["raw_dir"] = str(RAW.resolve())
            config["output_dir"] = "processed"
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            energy_hash = prep.sha256(RAW / self.config["energy"]["file"])
            summary = prep.run(path)
            self.assertEqual(summary["reports_audited"], 30)
            self.assertEqual(summary["target_months"], 108)
            output = Path(directory) / "processed"
            first = {file.name: prep.sha256(file) for file in output.iterdir()}
            prep.run(path)
            self.assertEqual(first, {file.name: prep.sha256(file) for file in output.iterdir()})
            self.assertEqual(energy_hash, prep.sha256(RAW / self.config["energy"]["file"]))
            config["output_dir"] = str(RAW.resolve())
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "inside the raw cache"):
                prep.run(path)
            self.assertEqual(energy_hash, prep.sha256(RAW / self.config["energy"]["file"]))


if __name__ == "__main__":
    unittest.main()
