PARQUET_PATH = "statcast_2015_2023.parquet"
SHEET_URL = "https://docs.google.com/spreadsheets/u/1/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/edit"

FASTBALLS = {"FF", "SI", "FT", "FC"} # four-seam, sinker, two-seam, cutter
BREAKING = {"SL", "CU", "KC", "ST", "SV", "CS"} # slider, curve, knuckle-curve, sweeper, etc.

SEASON_FEATURES = [
    "outings", "total_pitches", "max_pitches", "avg_pitches",
    "fb_velo_mean", "velo_decline", "acwr_peak", "drift_max",
    "spin_mean", "breaking_pct", "fastball_pct", "arsenal_size",
    "rel_x_std", "rel_z_std", "arm_angle_mean", "days_rest_mean",
    "age", "prior_tj",
]

TEST_YEARS = [2019, 2020, 2021, 2022, 2023] # for cross-validation folds