"""
train a model to predict delivery time (in minutes) from the delivery details.

usage:
    python train.py                       # uses data/sample_deliveries.csv
    python train.py --data mydata.csv     # use the real kaggle csv

expects columns: Agent_Age, Agent_Rating, Store/Drop lat+long,
Weather, Traffic, Vehicle, Area, Category, Delivery_Time (target).
"""
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_deliveries.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"[*] loaded {len(df)} rows from {args.data}")

    # basic cleaning - drop rows missing the target
    df = df.dropna(subset=["Delivery_Time"])

    # feature engineering: turn the two lat/long pairs into one distance feature
    df["distance_km"] = haversine(df["Store_Latitude"], df["Store_Longitude"],
                                   df["Drop_Latitude"], df["Drop_Longitude"])

    # pick the features i actually want to use
    numeric = ["Agent_Age", "Agent_Rating", "distance_km"]
    categorical = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
    X = pd.get_dummies(df[numeric + categorical], columns=categorical)
    y = df["Delivery_Time"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== results on test set ===")
    print(f"MAE  : {mae:.2f} minutes")
    print(f"RMSE : {rmse:.2f} minutes")
    print(f"R2   : {r2:.3f}")

    # which features mattered most
    importances = sorted(zip(X.columns, model.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print("\n=== top features ===")
    for name, imp in importances[:6]:
        print(f"  {name:22} {imp:.3f}")


if __name__ == "__main__":
    main()
