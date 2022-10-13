
This project is a progression from my capstone project for my Master's Degree in Data Science and Analytics.  File descriptions are below:

ratings-final1.xlsx - full dataset for 2013-2020 games

ratings-2021.xlsx - full dataset for 2021 games

ratings-2022.xlsx - full dataset (to date) of 2022 games

model-run.py - python code that creates models and generates predictions for 2021 and 2022 seasons.

lookupdf.csv - file created by model-run.py that characterizes quality of teams from 1-5 based on a three-year span of the team being televised

tuned_etr.sav - tuned extra trees regressor sklearn model generated from model-run.py in pickle format

tuned_gbr.sav - tuned gradient boosting regressor model generated from model-run.py in pickle format

model-saved.py - python code that generates predictions for 2022 season using previously saved models

2021withpredictions.csv - output with predictions for viewership and comparisons with actual viewership for the 2021 season

2022withpredictions.csv - output with predictions for viewership and comparisons with actual viewership (where available) for 2022 season