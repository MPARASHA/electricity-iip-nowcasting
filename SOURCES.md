# Sources

The source files retain their providers’ attribution and notices. The repository does not claim authorship of external datasets or documents. Their respective terms govern reuse; the national CSV redistribution terms have not been independently established.

| Input or evidence | Source |
| --- | --- |
| National energy extraction | [Robbie Andrew’s Indian energy data](https://robbieandrew.github.io/india/); [CICERO researcher profile](https://cicero.oslo.no/en/employees/robbie-andrew) |
| Original daily reports | [Grid India daily PSP reports](https://grid-india.in/en/reports/daily-psp-report) and the [historical archive](https://zenodo.org/records/7963862) |
| Added original NLDC workbooks | [Public archive of original tables](https://github.com/HalcyonVector/Grid-Sentinel/tree/main/Dataset/Raw), retrieved at the commit recorded in the extension manifest; the archive’s derived modeling dataset is unused |
| Historical manufacturing and industry IIP | [MoSPI IIP API](https://api.mospi.gov.in/api/iip/getIIPData?frequency=Monthly&base_year=2011-12&category_code=2&limit=200&page=1), using the historical 2011–12 base |
| Industry definitions and December 2025 release | [MoSPI release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2219602&lang=1&reg=3) |
| April 2025 publication timetable change | [MoSPI announcement](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2122427&lang=2&reg=3) |
| Actual October 2025 publication date | [MoSPI release of 1 December 2025](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2196938&lang=1&reg=3) |
| Daily temperature | [NASA POWER API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), fixed coordinates, UTC dates |
| Industry energy context | [Bureau of Energy Efficiency sector information](https://www.beeindia.gov.in/view_content.php?lang=1&lid=613) |
| Electricity as an economic indicator | [Blonz and Williams, Federal Reserve](https://www.federalreserve.gov/econres/notes/feds-notes/electricity-demand-as-a-high-frequency-economic-indicator-20201021.html) |

Calendar and publication-date source documents, exact retrieval URLs and SHA-256 hashes are retained in the raw manifests and release ledgers. The added report audit used preserved original workbooks because the official host was unavailable. Byte identity with current official-host copies could not be confirmed. The 0.12 MU regional-sum discrepancy on 15 September 2024 is retained rather than silently corrected.
