import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# LOAD CLEANED DATA
# ==========================

df = pd.read_csv("dataset/cleaned_house_data.csv")

print("Cleaned Dataset Shape:", df.shape)

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
y = model_df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODELS
# ==========================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
}

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    cv = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1
    ).mean()

    print("-" * 40)
    print(name)
    print("R2 Score :", round(r2, 4))
    print("Cross Validation :", round(cv, 4))
    print("MAE :", round(mae, 2))
    print("RMSE :", round(rmse, 2))