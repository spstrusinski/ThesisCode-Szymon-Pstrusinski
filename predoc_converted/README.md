# Coding sample (thesis code)

This repository contains code used in my thesis project on Polish exam outcomes and educational inequality.

## Data sources (official)
- Raw merged data up to 2018: OKE Kraków website (https://www.oke.krakow.pl/inf/staticpages/index.php?page=20160429131847153), section **"Wyniki egzaminu gimnazjalnego w szkołach"**.
- All other years: `mapa.wyniki.edu.pl`.

**HSEE** = High School Entrance Exam.

## Languages
- Python is used for data preparation, summary statistics, and figure generation.
- Stata is used only for the FE specification in `Regressions/High School Entry Exams/FE-hsee-Others.do`.

## Notes on methods
- Fixed effects (FE) is the main empirical approach.
- Difference-in-differences (DiD) results were generated but are not part of the main analysis.

## How to use
Scripts are organized by function in the existing folders (DataCleaning, Summary Statistics, Regressions, Graphs).
Some scripts contain local file paths and may require updating paths to match your local setup.
