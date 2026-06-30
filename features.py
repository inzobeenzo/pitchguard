import pandas as pd
import numpy as np
from config import FASTBALLS, BREAKING, SHEET_URL, PARQUET_PATH

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch-level rows -> one row per OUTING (pitcher + game), including the
    rolling, time-aware features (ACWR, velocity trend, release drift)."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    is_fb = df["pitch_type"].isin(FASTBALLS)
    df["fb_speed"] = df["release_speed"].where(is_fb)
    df["fb_eff_speed"] = df["effective_speed"].where(is_fb)
    df["is_breaking"] = df["pitch_type"].isin(BREAKING).astype(float)
    df["is_fastball"] = df["pitch_type"].isin(FASTBALLS).astype(float)

    outings = (
        df.groupby(["pitcher", "player_name", "game_date"])
          .agg(
              pitch_count = ("release_speed", "size"),
              fb_velo_mean = ("fb_speed", "mean"),
              fb_velo_max = ("fb_speed", "max"),
              fb_eff_velo_mean = ("fb_eff_speed", "mean"),
              spin_mean = ("release_spin_rate", "mean"),
              pfx_x_mean = ("pfx_x", "mean"),
              pfx_z_mean = ("pfx_z", "mean"),
              rel_x_mean = ("release_pos_x", "mean"),
              rel_z_mean = ("release_pos_z", "mean"),
              rel_x_std = ("release_pos_x", "std"),
              rel_z_std = ("release_pos_z", "std"),
              arm_angle_mean = ("arm_angle", "mean"),
              breaking_pct = ("is_breaking", "mean"),
              fastball_pct = ("is_fastball", "mean"),
              arsenal_size = ("pitch_type", "nunique"),
              days_rest = ("pitcher_days_since_prev_game", "first"),
              age = ("age_pit", "first"),
          ).reset_index().sort_values(["pitcher", "game_date"])
    )

    def per_pitcher(g):
        g = g.set_index("game_date").sort_index()
        acute = g["pitch_count"].rolling("7D").sum()
        chronic = g["pitch_count"].rolling("28D").sum() / 4
        g["acwr"] = acute / chronic
        recent_velo = g["fb_velo_mean"].rolling(3, min_periods=1).mean()
        baseline_velo = g["fb_velo_mean"].shift(1).expanding(min_periods=1).mean()
        g["velo_trend"] = recent_velo - baseline_velo
        base_x = g["rel_x_mean"].shift(1).expanding(min_periods=1).mean()
        base_z = g["rel_z_mean"].shift(1).expanding(min_periods=1).mean()
        g["release_drift"] = np.sqrt((g["rel_x_mean"] - base_x) ** 2 + (g["rel_z_mean"] - base_z) ** 2)
        return g.reset_index()

    features = outings.groupby("pitcher", group_keys=True).apply(per_pitcher)
    return features.reset_index(level="pitcher", drop=False)

def load_surgeries(sheet_url: str = SHEET_URL) -> pd.DataFrame:
    """Load the Tommy John surgery list -> columns [pitcher, surgery_date]."""
    raw = pd.read_csv(sheet_url.replace("/edit", "/export?format=csv"), skiprows=1)
    surg = raw[["mlbamid", "TJ Surgery Date"]].dropna().copy()
    surg.columns = ["pitcher", "surgery_date"]
    surg["surgery_date"] = pd.to_datetime(surg["surgery_date"], errors="coerce")
    return surg.dropna(subset=["surgery_date"])

def build_season_table(outings: pd.DataFrame, surgeries: pd.DataFrame,horizon_months: int = 18) -> pd.DataFrame:
    """Outing rows -> one row per PITCHER-SEASON, with the TJ label and a
    prior-surgery count."""
    surg_dates = surgeries.groupby("pitcher")["surgery_date"].apply(list).to_dict()

    outings = outings.copy()
    outings["game_date"] = pd.to_datetime(outings["game_date"])
    outings["season"] = outings["game_date"].dt.year

    season = (
        outings.groupby(["pitcher", "season"])
        .agg(
            outings = ("game_date",     "size"),
            total_pitches = ("pitch_count",   "sum"),
            max_pitches = ("pitch_count",   "max"),
            avg_pitches = ("pitch_count",   "mean"),
            fb_velo_mean = ("fb_velo_mean",  "mean"),
            velo_decline = ("velo_trend",    "min"),
            acwr_peak = ("acwr",          "max"),
            drift_max = ("release_drift", "max"),
            spin_mean = ("spin_mean",     "mean"),
            breaking_pct = ("breaking_pct",  "mean"),
            fastball_pct = ("fastball_pct",  "mean"),
            arsenal_size = ("arsenal_size",  "max"),
            rel_x_std = ("rel_x_std",     "mean"),
            rel_z_std = ("rel_z_std",     "mean"),
            arm_angle_mean = ("arm_angle_mean","mean"),
            days_rest_mean = ("days_rest",     "mean"),
            age = ("age",           "max"),
        ).reset_index()
    )

    def season_start(year):
        return pd.Timestamp(year=int(year), month=3, day=1)

    def make_label(row):
        start = season_start(row["season"])
        end = start + pd.DateOffset(months=horizon_months)
        return int(any(start <= d <= end for d in surg_dates.get(row["pitcher"], [])))

    def make_prior_tj(row):
        start = season_start(row["season"])
        return sum(d < start for d in surg_dates.get(row["pitcher"], []))

    season["TJ"] = season.apply(make_label, axis=1)
    season["prior_tj"] = season.apply(make_prior_tj, axis=1)
    return season

if __name__ == "__main__":
    df = pd.read_parquet(PARQUET_PATH)
    season = build_season_table(build_features(df), load_surgeries())
    print(f"{len(season)} pitcher-seasons, {int(season['TJ'].sum())} injuries")