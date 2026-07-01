import pandas as pd
from config import PARQUET_PATH, SHEET_URL
from features import build_features, build_season_table, load_surgeries
from model import train_model, explain_prediction
from crossval import cross_validate
from visualize import plot_results
from rag.engine import build_index, explain

def main():
    # 1. load the parquet your data pull saved, plus the surgery list
    df = pd.read_parquet(PARQUET_PATH)
    surgeries = load_surgeries(SHEET_URL)

    # 2. raw pitches -> outings -> labeled pitcher-seasons
    outings = build_features(df)
    season = build_season_table(outings, surgeries)
    print(f"{len(season)} pitcher-seasons, {int(season['TJ'].sum())} injuries\n")

    # 3. train + explain one prediction
    result = train_model(season)
    plot_results(result)
    print("\ntop risk factors for one pitcher:", explain_prediction(result["model"], result["X_test"]))
    top = explain_prediction(result["model"], result["X_test"])
    coll = build_index()
    print(explain(coll, top))

    # 4. trustworthy averaged score + SHAP stability
    print("\n--- cross-validation ---")
    cross_validate(season)


if __name__ == "__main__":
    main()