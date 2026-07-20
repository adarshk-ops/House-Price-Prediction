import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("dataset/bengaluru_house_prices.csv")

print("Original Shape:", df.shape)

# ==========================
# BASIC CLEANING
# ==========================

df = df.drop("society", axis=1)
df = df.dropna()

# ==========================
# CREATE BHK
# ==========================

df["bhk"] = df["size"].apply(lambda x: int(x.split()[0]))

# ==========================
# CLEAN TOTAL SQFT
# ==========================

def convert_sqft(x):
    tokens = str(x).split("-")

    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2

    try:
        return float(x)
    except:
        return None


df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

# ==========================
# FEATURE ENGINEERING
# ==========================

df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# ==========================
# REMOVE IMPOSSIBLE HOUSES
# ==========================

df = df[df.total_sqft / df.bhk >= 300]

# ==========================
# CLEAN LOCATION
# ==========================

df.location = df.location.apply(lambda x: x.strip())

location_stats = df.groupby("location").size()

rare_locations = location_stats[location_stats <= 10]

df.location = df.location.apply(
    lambda x: "other" if x in rare_locations else x
)

print("Unique Locations:", df.location.nunique())

# ==========================
# REMOVE PRICE/SQFT OUTLIERS
# ==========================

def remove_pps_outliers(df):

    final = pd.DataFrame()

    for location, subdf in df.groupby("location"):

        mean = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)

        reduced = subdf[
            (subdf.price_per_sqft > mean - std) &
            (subdf.price_per_sqft <= mean + std)
        ]

        final = pd.concat([final, reduced], ignore_index=True)

    return final


df = remove_pps_outliers(df)

print("Shape after outlier removal:", df.shape)

# ==========================
# SAVE CLEANED DATASET
# ==========================

df.to_csv("dataset/cleaned_house_data.csv", index=False)

print("Cleaned dataset saved.")

# ==========================
# PREPARE DATA
# ==========================

model_df = df.drop(
    ["size", "price_per_sqft", "availability"],
    axis=1
)

model_df = pd.get_dummies(
    model_df,
    columns=["location", "area_type"],
    drop_first=True
)

print("Encoded Shape:", model_df.shape)

X = model_df.drop("price", axis=1)
y = model_df.price

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n========== RESULTS ==========")
print("R2 :", r2_score(y_test, pred))
print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, pred)))

joblib.dump(model, "model/house_price_model.pkl")

print("Model Saved.")

import json

columns = {
    "data_columns": list(X.columns)
}

with open("model/columns.json", "w") as f:
    json.dump(columns, f, indent=4)

print("Columns saved successfully!")