import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("dataset/cleaned_house_data.csv")

# Remove unwanted columns
df = df.drop(
    ["size", "price_per_sqft", "availability"],
    axis=1
)

# One Hot Encoding
df = pd.get_dummies(
    df,
    columns=["location", "area_type"],
    drop_first=True
)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# RANDOM FOREST GRID SEARCH
# =====================================================

rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\n==============================")
print("BEST PARAMETERS")
print("==============================")

print(grid.best_params_)

print("\nBest Cross Validation Score:")
print(grid.best_score_)

best_model = grid.best_estimator_

test_score = best_model.score(X_test, y_test)

print("\nTest R2 Score:")
print(test_score)