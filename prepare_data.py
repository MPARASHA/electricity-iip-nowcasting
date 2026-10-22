"""Validate the frozen project inputs and build the daily/target/origin tables.

The default run is offline. Raw files are never edited. Use --download-missing
only to restore a missing, hash-locked input from its recorded public URL.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(root, record, download_missing=False):
    """Check frozen bytes; a changed existing file is always an error."""
    root = Path(root).resolve()
    path = (root / record["file"]).resolve()
    require(path.is_relative_to(root), f"Input path escapes raw directory: {path}")
    if not path.exists() and download_missing:
        url = record.get("source_url", "")
        require(url.startswith("https://"), f"No public HTTPS source for {path.name}")
        require("range_start" not in record, "An archive range cannot be restored as a whole file")
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Electricity-IIP-study/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read()
        require(hashlib.sha256(content).hexdigest() == record["sha256"],
                f"Downloaded content differs from frozen input: {path.name}")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    require(path.is_file(), f"Missing input: {path}")
    require(sha256(path) == record["sha256"], f"Input hash changed: {path.name}")
    return path


def verify_inputs(config, root, download_missing=False):
    raw = root / config["raw_dir"]
    inventory = {}
    for reference in config["download_manifests"]:
        path = checked_file(raw, reference)
        for record in json.loads(path.read_text()):
            checked_file(raw, record, download_missing)
            if record["file"] in inventory:
                require(inventory[record["file"]]["sha256"] == record["sha256"],
                        f"Conflicting input manifests: {record['file']}")
            inventory[record["file"]] = record
        inventory[reference["file"]] = reference
    for key in ["release_calendar", "report_sample", "calendar"]:
        reference = config[key]
        checked_file(raw, reference)
        inventory[reference["file"]] = reference
    consumed = [config["energy"]["file"], config["energy"]["metadata_file"], config["iip"]["manifest_file"]]
    consumed.extend(location["file"] for location in config["weather_locations"])
    consumed.extend(pd.read_csv(raw / config["calendar"]["file"])["source_file"])
    sample = pd.read_csv(raw / config["report_sample"]["file"], keep_default_na=False)
    for field in ["file", "precision_xls_file", "secondary_pdf_file"]:
        consumed.extend(value for value in sample[field] if value)
    require(all(name in inventory for name in consumed), "A configured input is absent from the frozen manifests")
    return raw, inventory


def positive_numeric(series, name):
    values = pd.to_numeric(series, errors="raise")
    require(not np.isinf(values).any(), f"Non-finite {name}")
    require(not (values.dropna() <= 0).any(), f"Non-positive {name}; missing is not zero")
    return values


def load_energy(raw, config):
    source = config["energy"]
    fields = json.loads((raw / source["metadata_file"]).read_text())["fields"]
    metadata = {field["name"]: field for field in fields}
    require(metadata[source["column"]]["units"] == "GWh", "Unexpected national energy units")
    frame = pd.read_csv(raw / source["file"], usecols=["yyyymmdd", source["column"]],
                        dtype={"yyyymmdd": str})
    frame["date"] = pd.to_datetime(frame.pop("yyyymmdd"), format="%Y%m%d", errors="raise")
    require(not frame["date"].duplicated().any(), "Duplicate energy dates; reconcile with originals")
    frame["energy_MU"] = positive_numeric(frame.pop(source["column"]), "national energy")
    frame = frame.set_index("date").sort_index()
    dates = pd.date_range(config["history_start"], config["history_end"], name="date")
    daily = frame.reindex(dates)
    daily["energy_quality_flag"] = ""
    for exception in config["energy_exclusions"]:
        day = pd.Timestamp(exception["date"])
        require(day in daily.index, f"Date exclusion outside history: {day}")
        daily.loc[day, "energy_quality_flag"] = exception["reason"]
    daily["energy_usable"] = daily["energy_MU"].notna() & daily["energy_quality_flag"].eq("")
    daily["energy_source_sha256"] = sha256(raw / source["file"])
    return daily


def add_weather(daily, raw, config):
    """Retain each fixed location; no weights or adjustment are estimated here."""
    daily = daily.copy()
    for location in config["weather_locations"]:
        payload = json.loads((raw / location["file"]).read_text())
        require(payload["header"]["time_standard"] == "UTC", "Unexpected weather time convention")
        longitude, latitude = payload["geometry"]["coordinates"][:2]
        require(np.isclose(latitude, location["latitude"]) and
                np.isclose(longitude, location["longitude"]), "Weather coordinates changed")
        for parameter in ["T2M", "T2M_MAX", "T2M_MIN"]:
            require(payload["parameters"][parameter]["units"] == "C", "Unexpected temperature units")
            series = pd.Series(payload["properties"]["parameter"][parameter], dtype=float)
            series.index = pd.to_datetime(series.index, format="%Y%m%d", errors="raise")
            require(not series.index.duplicated().any(), "Duplicate weather dates")
            series = series.replace(payload["header"]["fill_value"], np.nan)
            require(not np.isinf(series).any(), "Non-finite weather values")
            require(series.dropna().between(-70, 70).all(), "Temperature outside documented sanity bounds")
            daily[f"{location['id']}_{parameter}_C"] = series.reindex(daily.index)
        prefix = location["id"]
        pair = daily[[f"{prefix}_T2M_MIN_C", f"{prefix}_T2M_MAX_C"]].dropna()
        require((pair.iloc[:, 0] <= pair.iloc[:, 1]).all(),
                f"Weather minimum exceeds maximum: {prefix}")
    columns = [column for column in daily if column.endswith("_C")]
    daily["weather_complete"] = daily[columns].notna().all(axis=1)
    daily["weather_time_standard"] = "UTC"
    return daily


def load_release_calendar(raw, config, inventory, download_missing=False):
    frame = pd.read_csv(raw / config["release_calendar"]["file"], dtype=str, keep_default_na=False)
    require(not frame["target_month"].duplicated().any(), "Duplicate IIP release months")
    frame["month"] = pd.PeriodIndex(frame["target_month"], freq="M")
    frame["release_date"] = pd.to_datetime(frame["first_release_date"], errors="raise")
    for row in frame.itertuples():
        record = {"file": row.local_file, "sha256": row.sha256, "source_url": row.source_url}
        checked_file(raw, record, download_missing)
        inventory[row.local_file] = record
        require(row.release_date > row.month.end_time.normalize(), "IIP release precedes target month end")
    return frame.set_index("month")


def add_calendar(daily, raw, config):
    """Small, source-checked calendar; no exhaustive state shutdown claim."""
    events = pd.read_csv(raw / config["calendar"]["file"], parse_dates=["date"])
    require(not events[["date", "event"]].duplicated().any(), "Duplicate calendar event")
    years = set(range(pd.Timestamp(config["history_start"]).year, pd.Timestamp(config["history_end"]).year + 1))
    require(set(events.date.dt.year) == years, "Calendar does not cover every project year")
    require(events.review_status.eq("checked_primary_calendar").all(), "Calendar review incomplete")
    daily = daily.copy()
    daily["fixed_holiday"] = False
    daily["diwali_window"] = False
    for year, group in events.groupby(events.date.dt.year):
        require(len(group) == 4 and set(group.event) == {"Republic Day", "Independence Day", "Gandhi Jayanti", "Diwali"},
                f"Incomplete small calendar: {year}")
        fixed = group.loc[group.event.ne("Diwali"), "date"]
        require(set(fixed.dt.strftime("%m-%d")) == set(config["adjustment"]["fixed_holidays"]), "Fixed holiday changed")
        daily.loc[daily.index.isin(fixed), "fixed_holiday"] = True
        diwali = group.loc[group.event.eq("Diwali"), "date"].iloc[0]
        require(diwali.month in [10, 11], "Unexpected Diwali month")
        width = pd.Timedelta(days=config["adjustment"]["diwali_window_days_each_side"])
        daily.loc[(daily.index >= diwali - width) & (daily.index <= diwali + width), "diwali_window"] = True
    return daily


def load_iip(raw, config, releases):
    source = config["iip"]
    records = []
    manifest = json.loads((raw / source["manifest_file"]).read_text())
    for item in manifest:
        payload = json.loads((raw / item["file"]).read_text())
        require(payload["statusCode"] is True, f"Unsuccessful IIP source: {item['file']}")
        for row in payload["data"]:
            if row["type"] == "Sectoral" and row["category"] == "Manufacturing" and not row["sub_category"]:
                require(row["base_year"] == source["base_year"], "IIP base change; do not splice bases")
                month = pd.Timestamp(f"{row['month']} 1 {row['year']}").to_period("M")
                records.append({"month": month, "manufacturing_index": row["index"],
                                "reported_yoy_pct": row["growth_rate"] or np.nan,
                                "source_sha256": item["sha256"]})
    frame = pd.DataFrame(records).set_index("month").sort_index()
    require(not frame.index.duplicated().any(), "Duplicate aggregate manufacturing IIP months")
    frame["manufacturing_index"] = positive_numeric(frame["manufacturing_index"], "manufacturing IIP")
    frame["reported_yoy_pct"] = pd.to_numeric(frame["reported_yoy_pct"], errors="raise")
    expected = pd.period_range(config["history_start"], config["history_end"], freq="M")
    require(expected.isin(frame.index).all(), "Missing project IIP months")
    # Explicit month joins avoid confusing a missing month with a twelve-row lag.
    prior = frame["manufacturing_index"].reindex(frame.index - 12).to_numpy()
    frame["yoy_pct"] = 100 * (frame["manufacturing_index"] / prior - 1)
    frame = frame.reindex(expected).rename_axis("month")
    published = frame["reported_yoy_pct"].notna() & frame["yoy_pct"].notna()
    require((frame.loc[published, "reported_yoy_pct"] - frame.loc[published, "yoy_pct"]).abs().max() <= 0.051,
            "IIP published/computed growth differs by more than rounding")
    frame["base_year"] = source["base_year"]
    frame["release_date"] = [(month + 2).to_timestamp() + pd.Timedelta(days=11) for month in frame.index]
    frame["release_date_status"] = "nominal_12th_unverified_legacy_or_outside_evaluation"
    frame["release_source_url"] = ""
    common = frame.index.intersection(releases.index)
    frame.loc[common, "release_date"] = releases.loc[common, "release_date"]
    frame.loc[common, "release_date_status"] = "verified_first_quick_estimate_date"
    frame.loc[common, "release_source_url"] = releases.loc[common, "source_url"]
    frame["value_vintage_status"] = "latest_revised_indices_not_release_vintages"
    frame["target_in_scope"] = frame.index >= pd.Period(config["target_start"], freq="M")
    require(frame.loc[frame["target_in_scope"], "yoy_pct"].notna().all(), "Missing in-scope IIP growth")
    return frame


def _table_values(line):
    number = r"-?\d[\d,]*(?:\.\d+)?"
    match = re.search(rf"((?:{number}\s+){{5}}{number})\s*$", line)
    require(match is not None, f"Expected six regional/national values: {line[:150]}")
    return [float(token.replace(",", "")) for token in match[1].split()]


def parse_report(path):
    """Parse Table A only, with/without the explicit national SCADA maximum.

    Evening demand is kept as a crosscheck. It is never substituted for peak.
    Page positions and filenames do not determine the day covered.
    """
    reader = PdfReader(path)
    cover = reader.pages[0].extract_text()
    reference = re.search(r"Sub:\s*Daily\s+PSP\s+Report\s+for\s+the\s+date\s+"
                          r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", cover, re.I)
    covered_date = pd.Timestamp(int(reference[3]), int(reference[2]), int(reference[1])) if reference else None
    for page_number, page in enumerate(reader.pages[:3], 1):
        if "/Contents" not in page:
            continue
        text = page.extract_text(extraction_mode="layout")
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        header = next((i for i, line in enumerate(lines) if re.fullmatch(r"NR WR SR ER NER (?:Total|TOTAL)", line)), None)
        if header is None:
            continue
        end = next((i for i in range(header + 1, len(lines)) if lines[i].startswith("B.")), None)
        require(end is not None, "Unknown Table A boundary")
        table = lines[header + 1:end]
        keys = [re.sub(r"[^a-z]", "", line.lower()) for line in table]
        energy_rows = [line for line, key in zip(table, keys) if key.startswith("energymetmu")]
        require(len(energy_rows) == 1, "Unknown/ambiguous national energy row")
        energy = _table_values(energy_rows[0])
        require(all(value > 0 for value in energy), "Non-positive report energy")
        # Six independently rounded whole-MU entries can differ by up to 3 MU.
        require(abs(sum(energy[:5]) - energy[5]) <= 3.01, "Inconsistent national/regional energy total")
        evening_rows = [line for line, key in zip(table, keys) if key.startswith("demandmetduringeveningpeak")]
        require(len(evening_rows) == 1, "Unknown evening demand row")
        evening = _table_values(evening_rows[0])[-1]
        peak_rows = [line for line, key in zip(table, keys) if key.startswith("maximumdemandmetduringtheday")]
        peak = np.nan
        layout = "table_a_without_national_scada_peak"
        if peak_rows:
            require(len(peak_rows) == 1 and "scada" in re.sub(r"[^a-z]", "", " ".join(table).lower()),
                    "Maximum demand is not identified as NLDC SCADA")
            values = _table_values(peak_rows[0])
            peak = values[-1]
            require(max(values[:5]) <= peak <= sum(values[:5]), "Implausible national coincident peak")
            layout = "table_a_with_national_scada_peak"
        date_match = re.search(r"Date\s*of\s*Reporting\s*:?\s*(\d{1,2}-[A-Za-z]{3}-\d{2,4})", " ".join(lines), re.I)
        require(date_match is not None, "Missing printed reporting date")
        printed = pd.to_datetime(date_match[1], format="%d-%b-%Y" if len(date_match[1].split("-")[-1]) == 4 else "%d-%b-%y")
        return {"table_page": page_number, "layout": layout, "energy_MU": energy[-1],
                "national_peak_MW": peak, "evening_demand_MW": evening,
                "national_peak_status": "not_reported" if pd.isna(peak) else ("reported_zero_not_usable" if peak == 0 else "reported_positive"),
                "printed_reporting_date": printed, "covered_date": covered_date if covered_date is not None else printed - pd.Timedelta(days=1),
                "date_basis": "explicit_cover_reference" if covered_date is not None else "printed_reporting_date_minus_one_day"}
    raise ValueError(f"No supported national Table A in {Path(path).name}")


def audit_reports(raw, config, daily):
    sample = pd.read_csv(raw / config["report_sample"]["file"], keep_default_na=False)
    require(not sample["date"].duplicated().any(), "Duplicate dates in report audit")
    rows = []
    exclusions = {item["date"] for item in config["energy_exclusions"]}
    for item in sample.to_dict("records"):
        parsed = parse_report(raw / item["file"])
        day = pd.Timestamp(item["date"])
        require(item["visual_review_status"] == "checked", f"Visual audit incomplete: {item['date']}")
        require(parsed["table_page"] == int(item["manual_table_page"]), "Manual/parser page disagreement")
        for field in ["energy_MU", "national_peak_MW", "evening_demand_MW"]:
            manual = float(item[f"manual_{field}"]) if item[f"manual_{field}"] != "" else np.nan
            require((pd.isna(manual) and pd.isna(parsed[field])) or np.isclose(manual, parsed[field], rtol=0, atol=1e-8),
                    f"Manual/parser {field} disagreement: {item['date']}")
        require(parsed["printed_reporting_date"].strftime("%Y-%m-%d") == item["manual_reporting_date"],
                f"Manual/parser reporting-date disagreement: {item['date']}")
        date_ok = parsed["covered_date"] == day
        require(date_ok or (item["date"] in exclusions and item["date_status"] == "unresolved_printed_year"),
                f"Unreviewed report/date disagreement: {item['date']}")
        candidate = daily.loc[day, "energy_MU"]
        tolerance = float(item["pdf_energy_rounding_tolerance_MU"])
        require(0 <= tolerance <= 0.5001, "Overly broad PDF energy tolerance")
        energy_ok = pd.notna(candidate) and abs(candidate - parsed["energy_MU"]) <= tolerance
        conflict_excluded = item["date"] in exclusions and item["energy_status"] == "conflicting_originals_excluded"
        require(energy_ok or conflict_excluded, f"Unreviewed candidate/PDF energy disagreement: {item['date']}")
        secondary_energy = np.nan
        if item.get("secondary_pdf_file"):
            secondary = parse_report(raw / item["secondary_pdf_file"])
            secondary_energy = secondary["energy_MU"]
            require(np.isclose(secondary_energy, float(item["manual_secondary_energy_MU"]), rtol=0, atol=1e-8),
                    "Secondary report manual/parser energy disagreement")
            require(secondary["covered_date"] == day, "Secondary report covers another day")
            require(conflict_excluded and not np.isclose(secondary_energy, parsed["energy_MU"]),
                    "Secondary conflict evidence does not match the recorded exclusion")
        xls_energy = np.nan
        if item.get("precision_xls_file"):
            import xlrd
            sheet = xlrd.open_workbook(raw / item["precision_xls_file"]).sheet_by_name("MOP_E")
            xls_energy = float(sheet.cell_value(7, 7))  # Official MOP_E!H8, not a state sum.
            require(np.isclose(candidate, xls_energy, rtol=0, atol=1e-8), "Candidate/official XLS energy mismatch")
            require(np.isclose(parsed["national_peak_MW"], float(sheet.cell_value(12, 7)), rtol=0, atol=1e-8),
                    "PDF/XLS national peak disagreement")
        rows.append({"date": item["date"], "file": item["file"], **parsed,
                     "candidate_energy_MU": candidate, "pdf_difference_MU": candidate - parsed["energy_MU"],
                     "candidate_energy_status": "within_source_precision" if energy_ok else "conflicting_originals_excluded",
                     "secondary_original_energy_MU": secondary_energy,
                     "official_xls_energy_MU": xls_energy, "date_status": item["date_status"],
                     "numeric_audit_status": "pass", "source_url": item["source_url"],
                     "source_sha256": sha256(raw / item["file"])})
    audit = pd.DataFrame(rows)
    require(len(audit) >= config["required_audit_reports"], "Fewer than 30 original reports audited")
    require(audit["layout"].nunique() <= 2, "More than two report layouts")
    return audit


def build_origins(daily, iip, config, audit):
    """Availability counts only; feature completeness is checked in backtesting."""
    rows = []
    known_issue_dates = {pd.Timestamp(row.date): row.printed_reporting_date
                         for row in audit.itertuples() if row.date_status == "consistent"}
    for month in pd.period_range(config["target_start"], config["history_end"], freq="M"):
        forecast = (month + 1).to_timestamp() + pd.Timedelta(days=config["forecast_day"] - 1)
        released = iip.loc[(iip["release_date"] <= forecast) & (iip.index < month)]
        latest = released.index.max()
        split = next(name for name, bounds in config["splits"].items()
                     if pd.Period(bounds[0], "M") <= month <= pd.Period(bounds[1], "M"))
        if split != "training":
            require(latest == month - 2, f"Unexpected released label at {forecast.date()}: {latest}")
            require(iip.loc[latest, "release_date_status"] == "verified_first_quick_estimate_date", "Unverified evaluated latest label")
            require(iip.loc[month - 1, "release_date_status"] == "verified_first_quick_estimate_date", "Unverified next-label release date")
            require(iip.loc[month - 1, "release_date"] > forecast, "Unreleased t-1 target leaked")
        for lag in config["daily_lags"]:
            cutoff = min(forecast - pd.Timedelta(days=lag), month.end_time.normalize())
            prior_cutoff = cutoff - pd.DateOffset(years=1)
            info = {"month": str(month), "split": split, "forecast_date": forecast,
                    "assumed_lag_days": lag, "daily_cutoff": cutoff, "prior_year_cutoff": prior_cutoff,
                    "latest_released_iip_month": str(latest), "latest_label_release_date": iip.loc[latest, "release_date"],
                    "release_date_status": iip.loc[latest, "release_date_status"],
                    "eligible_released_training_labels": int((released["target_in_scope"] & released["yoy_pct"].notna()).sum())}
            for prefix, start, end in [("current", month.start_time, cutoff),
                                       ("prior", (month - 12).start_time, prior_cutoff)]:
                dates = pd.date_range(start, end)
                window = daily.reindex(dates)
                available = pd.Series(dates + pd.Timedelta(days=lag) <= forecast, index=dates)
                weather_available = available.copy()
                for day, issued in known_issue_dates.items():
                    if day in available.index and issued > forecast:
                        available.loc[day] = False
                energy = window["energy_usable"].fillna(False) & available
                weather = window["weather_complete"].fillna(False) & weather_available
                info[f"{prefix}_expected_days"] = len(dates)
                info[f"{prefix}_energy_days"] = int(energy.sum())
                info[f"{prefix}_energy_coverage"] = float(energy.mean())
                info[f"{prefix}_weather_coverage"] = float(weather.mean())
                info[f"{prefix}_joint_coverage"] = float((energy & weather).mean())
            required_coverages = [info[f"{prefix}_joint_coverage"] for prefix in ["current", "prior"]]
            target_ok = pd.notna(iip.loc[month, "yoy_pct"])
            lag_months = [month - 2, month - 3]
            lag_ok = all(m in released.index and pd.notna(released.loc[m, "yoy_pct"]) for m in lag_months)
            for threshold in config["coverage_thresholds"]:
                info[f"eligible_{round(100 * threshold)}"] = target_ok and lag_ok and min(required_coverages) >= threshold
            rows.append(info)
    return pd.DataFrame(rows)


def data_dictionary(daily, iip, origins):
    rows = []
    special = {
        "date": ("date", "Grid energy day; each weather series retains its UTC date label"),
        "month": ("YYYY-MM", "Reference month of the manufacturing IIP target"),
        "energy_MU": ("MU = GWh = million kWh", "National Table A energy supplied; missing stays missing"),
        "energy_usable": ("boolean", "Positive nonmissing energy with no unresolved date or conflicting-source flag"),
        "energy_quality_flag": ("text", "Unresolved reporting-year or source-value conflict; nonempty flags exclude usable energy"),
        "energy_source_sha256": ("SHA-256", "Hash of the frozen national electricity CSV"),
        "weather_complete": ("boolean", "All 15 temperature fields present; fixed UTC calendar-label proxy for grid energy days"),
        "weather_time_standard": ("UTC", "NASA POWER daily time convention; no local-day conversion at preparation"),
        "fixed_holiday": ("boolean", "Republic Day, Independence Day or Gandhi Jayanti from checked primary annual calendars"),
        "diwali_window": ("boolean", "Checked Diwali date plus/minus one day; small aggregate festival proxy"),
        "manufacturing_index": ("index, 2011-12 = 100", "Official aggregate manufacturing index; excludes manufacturing subcategories and other IIP sectors"),
        "yoy_pct": ("percentage points", "100 * (index for month / index for same month last year - 1)"),
        "reported_yoy_pct": ("percent", "Published annual growth, retained for rounding comparison; April 2021 is blank"),
        "base_year": ("text", "Consistent 2011-12 IIP base; other bases are not spliced"),
        "release_date": ("date", "First-release date where verified; explicitly nominal elsewhere"),
        "release_date_status": ("text", "Distinguishes verified primary release dates from nominal legacy dates"),
        "release_source_url": ("HTTPS URL", "Primary first quick-estimate release; blank for unverified legacy dates"),
        "source_sha256": ("SHA-256", "Hash of the official API page containing the aggregate monthly index"),
        "value_vintage_status": ("text", "Latest revised indices and backcasts; original release values are not reconstructed"),
        "target_in_scope": ("boolean", "January 2015 onward; 2014 indices provide prior-year comparisons"),
        "split": ("text", "Training 2015-18, validation 2019-21, held-out test 2022-23"),
        "forecast_date": ("date", "Fifth calendar day of the month after the target month"),
        "assumed_lag_days": ("calendar days", "Prespecified daily-data availability assumption; longer documented issue lags override energy availability"),
        "daily_cutoff": ("date", "Earlier of target month end and forecast date minus assumed daily lag"),
        "prior_year_cutoff": ("date", "Same calendar date one year earlier; Feb 29 maps to Feb 28"),
        "latest_released_iip_month": ("YYYY-MM", "Latest past IIP label with a first-release date no later than the forecast date"),
        "latest_label_release_date": ("date", "First-release date of the latest past label"),
        "eligible_released_training_labels": ("months", "In-scope labels released by this origin, before feature/coverage exclusions; not the final fitted sample size"),
    }
    for table, frame, key in [("daily.csv", daily, "date"), ("iip.csv", iip, "month"), ("origins.csv", origins, None)]:
        for column in ([key] if key else []) + list(frame.columns):
            unit, rule = special.get(column, ("", ""))
            if column.endswith("_C"):
                unit, rule = "degrees Celsius", "Fixed NASA POWER point, UTC; same labelled date retained with a documented 5.5-hour grid-day mismatch"
            elif "coverage" in column:
                unit, rule = "fraction 0–1", "Usable available observations / expected calendar days; current and prior year checked separately"
                if "joint" in column:
                    rule += "; energy and all weather fields must be available on the same day"
            elif column.endswith("_expected_days"):
                unit, rule = "days", "Expected calendar dates from the window's month start through its cutoff"
            elif column.endswith("_energy_days"):
                unit, rule = "days", "Nonmissing, quality-approved energy dates available by the forecast origin"
            elif column.startswith("eligible_"):
                unit, rule = "boolean", "Input eligibility at the stated joint coverage threshold; feature readiness is not implied"
            rows.append({"table": table, "field": column, "unit": unit, "definition": rule or column.replace("_", " ")})
    return pd.DataFrame(rows)


def run(config_path, download_missing=False):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text())
    raw, inventory = verify_inputs(config, root, download_missing)
    daily = add_calendar(add_weather(load_energy(raw, config), raw, config), raw, config)
    releases = load_release_calendar(raw, config, inventory, download_missing)
    iip = load_iip(raw, config, releases)
    audit = audit_reports(raw, config, daily)
    origins = build_origins(daily, iip, config, audit)
    counts = []
    for (lag, split), group in origins.groupby(["assumed_lag_days", "split"], sort=True):
        for threshold in config["coverage_thresholds"]:
            field = f"eligible_{round(100 * threshold)}"
            counts.append({"lag_days": int(lag), "split": split, "coverage_threshold": threshold,
                           "eligible_months": int(group[field].sum()), "total_months": len(group)})
    require(all(row["eligible_months"] >= config["minimum_test_months"] for row in counts if row["split"] == "test"),
            "Fewer than 20 matched test months; review scope before modelling")
    summary = {"config_sha256": sha256(config_path), "daily_dates": len(daily),
               "nonmissing_energy_days": int(daily["energy_MU"].notna().sum()),
               "usable_energy_days": int(daily["energy_usable"].sum()),
               "missing_energy_dates": daily.index[daily["energy_MU"].isna()].strftime("%Y-%m-%d").tolist(),
               "excluded_quality_flags": daily.index[daily["energy_quality_flag"].ne("")].strftime("%Y-%m-%d").tolist(),
               "weather_complete_days": int(daily["weather_complete"].sum()),
               "iip_index_months": len(iip), "target_months": int(iip["target_in_scope"].sum()),
               "reports_audited": len(audit), "report_layouts": sorted(audit["layout"].unique()),
               "sample_counts": counts, "sample_status": "input eligibility checked; model timing and evaluation are checked by the backtest freeze",
               "model_evaluation_gate": config["evaluation_gate"],
               "vintage_limit": "Revised IIP and reanalysis weather; pseudo-real-time, no original-vintage reconstruction"}
    output = root / config["output_dir"]
    require(not output.resolve().is_relative_to(raw.resolve()), "Outputs must not be written inside the raw cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, frame, index in [("daily.csv", daily, True), ("iip.csv", iip, True),
                                   ("origins.csv", origins, False), ("report_audit.csv", audit, False),
                                   ("data_dictionary.csv", data_dictionary(daily, iip, origins), False),
                                   ("sample_counts.csv", pd.DataFrame(counts), False)]:
            frame.to_csv(stage / name, index=index, date_format="%Y-%m-%d", float_format="%.12g")
        (stage / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        versions = {"config_sha256": sha256(config_path),
                    "preparation_script_sha256": sha256(Path(__file__)),
                    "inputs": [inventory[name] for name in sorted(inventory)],
                    "outputs": {path.name: sha256(path) for path in sorted(stage.iterdir())}}
        (stage / "data_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage.iterdir(), key=lambda p: p.name == "data_versions.json"):
            os.replace(path, output / path.name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--download-missing", action="store_true", help="Restore missing locked inputs; changed files still fail")
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.download_missing)
    print(json.dumps({key: result[key] for key in ["daily_dates", "usable_energy_days", "target_months", "reports_audited", "model_evaluation_gate"]}, indent=2))
