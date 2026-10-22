"""Fetch immutable public evidence and prepare the extension's audited inputs.

Run from the repository root with ``python -m extension.prepare_data``.
The original experiment and its caches are read, never written.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import html
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
import pandas as pd

from prepare_data import add_calendar, add_weather, load_energy, require, sha256

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "extension"
RAW = HERE / "raw"
DATA = HERE / "data"


def read_config():
    return json.loads((HERE / "config.json").read_text())


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch_all(requests):
    """Never silently replace bytes after a successful download."""
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    known = {row["file"]: row for row in previous}

    def fetch(item):
        path = RAW / item["file"]
        if path.exists():
            require(item["file"] in known, f"Unmanifested existing raw file: {path}")
            record = known[item["file"]]
            require(sha256(path) == record["sha256"], f"Cached bytes changed: {path}")
            require(record["url"] == item["url"] and record.get("post") == item.get("post"),
                    f"Source request changed: {path}")
            return record
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(item["post"]).encode() if "post" in item else None
        headers = {"User-Agent": "Mozilla/5.0 Electricity-IIP academic audit", "Accept": "*/*"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Origin": "https://grid-india.in"})
        error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(item["url"], data=body, headers=headers)
                with urllib.request.urlopen(request, timeout=45) as response:
                    content = response.read()
                    content_type = response.headers.get("Content-Type", "")
                    final_url = response.url
                require(len(content) > 100, f"Empty/short response: {item['url']}")
                path.write_bytes(content)
                return {**item, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "final_url": final_url, "content_type": content_type,
                        "bytes": len(content), "sha256": sha256(path)}
            except Exception as exc:
                error = exc
                time.sleep(attempt + 1)
        return {"file": item["file"], "error": str(error)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, requests))
    failures = [row for row in results if "error" in row]
    for row in results:
        if "error" not in row:
            known[row["file"]] = row
    save_json(manifest_path, list(sorted(known.values(), key=lambda row: row["file"])))
    save_json(RAW / "fetch_errors.json", failures)
    require(not failures, f"Source downloads failed: {failures}")
    return known


def source_requests(config):
    requests = []
    for location in config["weather_locations"]:
        url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M%2CT2M_MAX%2CT2M_MIN"
               f"&community=RE&latitude={location['latitude']}&longitude={location['longitude']}"
               "&start=20240101&end=20251231&format=JSON&time-standard=UTC")
        requests.append({"file": f"weather/{location['id']}.json", "url": url})
    for row in config["new_releases"]:
        requests.append({"file": f"releases/pib_{row['id']}.html", "url": row["url"]})
    requests.extend(config["evidence_sources"])
    return requests


def plain_text(path):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", path.read_text(errors="replace"))).split())


def releases(config):
    old = ROOT / config["original_raw_dir"]
    rows = pd.read_csv(old / "iip_first_release_calendar_verified.csv").to_dict("records")
    for row in rows:
        require(sha256(old / row["local_file"]) == row["sha256"], "Original release evidence changed")
        row["local_file"] = str((old / row["local_file"]).relative_to(ROOT))
    for item in config["new_releases"]:
        path = RAW / f"releases/pib_{item['id']}.html"
        text = plain_text(path)
        date = pd.Timestamp(item["date"])
        # Check the actual release's publication line, not a planned calendar date.
        match = re.search(r"(?:Posted On|प्रविष्टि तिथि)\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})", text, re.I)
        require(match is not None, f"Cannot parse publication date: {path.name}")
        require(pd.Timestamp(match[1]).normalize() == date, f"Publication date mismatch: {path.name}")
        month = pd.Period(item["month"], "M")
        month_word = month.start_time.strftime("%B")
        require(re.search(rf"{month_word}\s*[,\-]?\s*{month.year}", text, re.I) is not None,
                f"Target month absent from release: {path.name}")
        require("2011-12" in text or "2011–12" in text, f"Unexpected release base: {path.name}")
        rows.append({"target_month": item["month"], "first_release_date": item["date"],
                     "source_url": item["url"], "local_file": str(path.relative_to(ROOT)),
                     "title": f"IIP quick estimates for {month_word} {month.year}",
                     "date_evidence": match[0], "sha256": sha256(path),
                     "verified_date_status": "actual_primary_release_publication_line",
                     "value_vintage_status": "dates_checked_but_values_latest_revised"})
    frame = pd.DataFrame(rows).sort_values("target_month")
    require(not frame.target_month.duplicated().any(), "Duplicate first-release month")
    expected = pd.period_range("2018-11", "2025-12", freq="M").astype(str)
    require(set(frame.target_month) == set(expected), "Incomplete evaluation release ledger")
    frame.to_csv(DATA / "release_calendar.csv", index=False)
    return frame.set_index("target_month")


def iip_data(config, release):
    old = ROOT / config["original_raw_dir"]
    manifest = json.loads((old / "iip_verified_download_manifest.json").read_text())
    labels = {item["sub_category"]: item["id"] for item in config["targets"]}
    rows = []
    for page in manifest:
        path = old / page["file"]
        require(sha256(path) == page["sha256"], f"IIP cache changed: {path}")
        payload = json.loads(path.read_text())
        require(payload["statusCode"] is True, "Unsuccessful IIP API response")
        for row in payload["data"]:
            if row["type"] != "Sectoral" or row["category"] != "Manufacturing" or row["sub_category"] not in labels:
                continue
            require(row["base_year"] == "2011-12", "IIP base changed")
            month = pd.Timestamp(f"{row['month']} 1 {row['year']}").to_period("M")
            if month > pd.Period(config["test"][1], "M"):
                continue
            rows.append({"month": month, "target": labels[row["sub_category"]], "index": float(row["index"]),
                         "reported_yoy_pct": float(row["growth_rate"]) if row["growth_rate"] else np.nan,
                         "source_sha256": page["sha256"]})
    frame = pd.DataFrame(rows)
    require(not frame[["month", "target"]].duplicated().any(), "Duplicate IIP target/month")
    result = []
    for target, group in frame.groupby("target"):
        group = group.set_index("month").sort_index()
        require((group["index"] > 0).all() and np.isfinite(group["index"]).all(), "Invalid IIP levels")
        expected = pd.period_range(config["history_start"], config["history_end"], freq="M")
        require(expected.isin(group.index).all(), f"Incomplete continuous history: {target}")
        group["yoy_pct"] = 100 * (group["index"] / group["index"].reindex(group.index - 12).to_numpy() - 1)
        group = group.reindex(expected)
        published = group[["yoy_pct", "reported_yoy_pct"]].dropna()
        require((published.yoy_pct - published.reported_yoy_pct).abs().max() <= .051, "IIP growth/level mismatch")
        group["release_date"] = [(month + 2).to_timestamp() + pd.Timedelta(days=11) for month in group.index]
        group["release_status"] = "nominal_12th_second_following_month_training_only"
        for month in group.index:
            if str(month) in release.index:
                group.loc[month, "release_date"] = pd.Timestamp(release.loc[str(month), "first_release_date"])
                group.loc[month, "release_status"] = "verified_primary_release"
        group["base_year"] = "2011-12"
        group["target"] = target
        result.append(group.reset_index(names="month"))
    out = pd.concat(result, ignore_index=True)
    require(out.loc[out.month >= pd.Period(config["target_start"], "M"), "yoy_pct"].notna().all(), "Missing target growth")
    out.to_csv(DATA / "iip.csv", index=False, float_format="%.12g")
    return out


def daily_data(config):
    old = ROOT / config["original_raw_dir"]
    daily = load_energy(old, config)
    old_config = {**config, "history_end": "2023-12-31"}
    history = add_weather(daily.loc[:"2023-12-31"], old, old_config)
    new_locations = [{**item, "file": f"weather/{item['id']}.json"} for item in config["weather_locations"]]
    extra = add_weather(daily.loc["2024-01-01":], RAW, {**config, "weather_locations": new_locations})
    daily = pd.concat([history, extra])
    events = pd.read_csv(old / "calendar_events.csv")
    added = []
    for year, day in [(2024, "2024-10-31"), (2025, "2025-10-20")]:
        source = next(row for row in config["evidence_sources"] if row["file"] == f"calendar/{year}.html")
        text = plain_text(RAW / source["file"])
        require("Diwali" in text and str(year) in text, f"Holiday evidence missing: {year}")
        for name, date in [("Republic Day", f"{year}-01-26"), ("Independence Day", f"{year}-08-15"),
                           ("Gandhi Jayanti", f"{year}-10-02"), ("Diwali", day)]:
            added.append({"date": date, "event": name, "review_status": "checked_primary_calendar",
                          "source_url": source["url"], "source_file": source["file"]})
    pd.concat([events, pd.DataFrame(added)], ignore_index=True).to_csv(DATA / "calendar_events.csv", index=False)
    daily = add_calendar(daily, DATA, {**config, "calendar": {"file": "calendar_events.csv"}})
    daily.to_csv(DATA / "daily.csv", float_format="%.12g")
    return daily


def grid_samples(config, daily):
    import xlrd
    # Official API/CDN connections failed. These are original NLDC XLS documents
    # preserved in a third-party archive, not that project's derived CSV dataset.
    commit = json.loads((RAW / "grid/mirror_commit.json").read_text())["sha"]
    fetch_all([{"file": "grid/mirror_tree.json", "url": f"https://api.github.com/repos/HalcyonVector/Grid-Sentinel/git/trees/{commit}?recursive=1"}])
    tree = json.loads((RAW / "grid/mirror_tree.json").read_text())
    paths = {row["path"]: row for row in tree["tree"]}
    # Six regularly spaced months per year, plus extremes and largest daily jumps.
    selected = {pd.Timestamp(year, month, 15): "calendar_spread" for year in [2024, 2025] for month in [1, 3, 5, 7, 9, 11]}
    new = daily.loc["2024-01-01":, "energy_MU"]
    for day in [new.idxmax(), new.idxmin(), *new.pct_change(fill_method=None).abs().nlargest(2).index]:
        selected[day] = "extreme_or_large_change"
    requests, sample = [], []
    for day, reason in sorted(selected.items()):
        filename = day.strftime("%d.%m.%y_NLDC_PSP.xls")
        options = sorted(path for path in paths if path.endswith("/" + filename) and "/File3_Raw/" not in path)
        require(options and len({paths[path]["sha"] for path in options}) == 1,
                f"Missing or conflicting archived original workbooks: {day.date()}")
        archive_path = options[0]  # Identical copies across archive studies are one document.
        file = f"grid/reports/{day.date()}.xls"
        url = f"https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/{commit}/{archive_path}"
        requests.append({"file": file, "url": url})
        sample.append({"date": str(day.date()), "reason": reason, "file": file, "source_url": url,
                       "archive_git_blob": paths[archive_path]["sha"], "archive_commit": commit,
                       "source_status": "original_NLDC_document_from_third_party_mirror_official_endpoint_unavailable"})
    fetch_all(requests)
    rows = []
    for row in sample:
        book = xlrd.open_workbook(RAW / row["file"])
        sheet = book.sheet_by_name("MOP_E")
        # Locate the original labelled table; never rely on a filename's day.
        text = " ".join(str(cell.value) for cells in sheet.get_rows() for cell in cells)
        reporting = re.search(r"Date\s*of\s*Reporting\s*:?\s*(\d{1,2}-[A-Za-z]+-\d{2,4})", text, re.I)
        require(reporting is not None, f"Printed issue date absent: {row['file']}")
        issued = pd.Timestamp(reporting[1])
        energy_row = next((index for index in range(sheet.nrows) if "energymetmu" in re.sub(r"[^a-z]", "", str(sheet.cell_value(index, 0)).lower())), None)
        require(energy_row is not None, "Labelled energy row missing")
        require(str(sheet.cell_value(4, 7)).upper() == "TOTAL", "National TOTAL column changed")
        values = [float(sheet.cell_value(energy_row, col)) for col in range(2, 8)]
        # NR is recorded in whole MU, WR in tenths; the other displayed
        # components and national total use hundredths. Keep the discrepancy.
        regional_difference = sum(values[:5]) - values[-1]
        require(abs(regional_difference) <= .601, "Regional/national energy sum differs beyond mixed-precision rounding")
        parsed = {"table_page": 1, "layout": f"MOP_E_energy_row_{energy_row+1}", "energy_MU": values[-1],
                  "printed_reporting_date": issued, "covered_date": issued - pd.Timedelta(days=1),
                  "date_basis": "printed_previous_day_report_and_issue_date", "table_cell": f"MOP_E!H{energy_row+1}",
                  "regional_total_difference_MU": regional_difference}
        day = pd.Timestamp(row["date"])
        require(parsed["covered_date"] == day, f"Printed report day differs: {row['file']}")
        require(day <= parsed["printed_reporting_date"] <= day + pd.Timedelta(days=7), "Unusual report issue lag")
        candidate = float(daily.loc[day, "energy_MU"])
        difference = candidate - parsed["energy_MU"]
        require(abs(difference) <= .00001, f"National energy mismatch at XLS precision: {day.date()}: {difference}")
        rows.append({**row, **parsed, "candidate_energy_MU": candidate, "difference_MU": difference,
                     "sha256": sha256(RAW / row["file"]), "status": "matches_original_XLS_precision"})
    audit = pd.DataFrame(rows)
    require(len(audit) >= config["required_new_reports"], "Insufficient new independent report checks")
    audit.to_csv(DATA / "report_audit.csv", index=False)
    return audit


def verify_original():
    baseline = json.loads((HERE / "baseline/original_experiment_hashes.json").read_text())
    files = baseline["files"] if "files" in baseline else baseline
    if isinstance(files, list):
        files = {row["path"]: row["sha256"] for row in files}
    for name, checksum in files.items():
        require(sha256(ROOT / name) == checksum, f"Original experiment changed: {name}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    config = read_config()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_all(source_requests(config))
    require((RAW / "manifest.json").exists(), "Fetch public evidence first with --fetch")
    for row in json.loads((RAW / "manifest.json").read_text()):
        require(sha256(RAW / row["file"]) == row["sha256"], f"Raw evidence changed: {row['file']}")
    unchanged = verify_original()
    release = releases(config)
    iip = iip_data(config, release)
    daily = daily_data(config)
    audit = grid_samples(config, daily)
    summary = {"iip_rows": len(iip), "iip_targets": sorted(iip.target.unique()), "iip_months_each": int(len(iip) / 4),
               "daily_days": len(daily), "usable_energy_days": int(daily.energy_usable.sum()),
               "complete_weather_days": int(daily.weather_complete.sum()), "new_report_checks": len(audit),
               "new_report_layouts": audit.layout.unique().tolist(), "verified_release_months": len(release),
               "original_unchanged_files": unchanged,
               "new_daily_missing_energy": daily.loc["2024-01-01":].index[daily.loc["2024-01-01":, "energy_MU"].isna()].strftime("%Y-%m-%d").tolist(),
               "value_vintage": "revised IIP and reanalysis weather; pseudo-real-time only"}
    save_json(DATA / "source_audit.json", summary)
    consumed = [ROOT / config["original_raw_dir"] / name for name in ["andrew_POSOCO_data.csv", "andrew_POSOCO_metadata.json", "iip_verified_download_manifest.json"]]
    consumed += [ROOT / config["original_raw_dir"] / loc["file"] for loc in config["weather_locations"]]
    versions = {"config_sha256": sha256(HERE / "config.json"),
                "preparation_sha256": sha256(Path(__file__)),
                "original_inputs": {str(path.relative_to(ROOT)): sha256(path) for path in consumed},
                "raw_manifest_sha256": sha256(RAW / "manifest.json"),
                "outputs": {path.name: sha256(path) for path in sorted(DATA.glob("*.csv"))}}
    previous_path = DATA / "versions.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_time = previous.pop("created_utc", None)
    versions["created_utc"] = previous_time if previous == versions and previous_time else dt.datetime.now(dt.timezone.utc).isoformat()
    save_json(previous_path, versions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
