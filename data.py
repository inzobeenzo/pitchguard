import time
import pandas as pd
from pybaseball import statcast, cache

cache.enable()

COLS = [
    "pitch_type", "game_date", "release_speed", "release_pos_x", "release_pos_z",
    "player_name", "pitcher", "spin_dir", "zone", "game_type", "home_team",
    "away_team", "type", "game_year", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "inning", "inning_topbot", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "sz_top", "sz_bot", "effective_speed", "release_spin_rate", "release_extension",
    "game_pk", "release_pos_y", "pitch_number", "pitch_name", "spin_axis",
    "age_pit_legacy", "age_pit", "pitcher_days_since_prev_game",
    "pitcher_days_until_next_game", "api_break_z_with_gravity",
    "api_break_x_arm", "api_break_x_batter_in", "arm_angle",
]

START_YEAR, END_YEAR = 2015, 2023
MIN_ROWS = 100_000


def pull_year(year, retries=2):
    """Pull March-November (regular season + playoffs) for one year,
    retrying if the result comes back empty or suspiciously short."""
    last = None
    for attempt in range(retries + 1):
        try:
            df = statcast(start_dt=f"{year}-03-01", end_dt=f"{year}-11-30")
            last = df
            if df is not None and len(df) >= MIN_ROWS:
                return df
            got = 0 if df is None else len(df)
            print(f"  {year}: only {got:,} rows, retrying ({attempt+1})...")
        except Exception as e:
            print(f"  {year}: error ({e}), retrying ({attempt+1})...")
        time.sleep(5)
    return last


def main():
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"pulling {year} ...", flush=True)
        season = pull_year(year)
        n = 0 if season is None else len(season)
        print(f"  {year}: {n:,} rows")
        if n:
            frames.append(season)

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    df = df.reindex(columns=COLS)

    print("\nrows per season:")
    print(df["game_year"].value_counts().sort_index())
    print("date range:", df["game_date"].min(), "->", df["game_date"].max())
    print(f"total rows: {len(df):,}")

    df.to_parquet("statcast_2015_2023.parquet", index=False)
    print("\nsaved -> statcast_2015_2023.parquet")
    print("reload later with: df = pd.read_parquet('statcast_2015_2023.parquet')")


if __name__ == "__main__":
    main()