import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("dataset/cleaned_house_data.csv")

# Remove unnecessary columns
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

# ===============================
# TRAIN RANDOM FOREST
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# FEATURE IMPORTANCE
# ===============================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features\n")
print(importance.head(20))

# ===============================
# PLOT
# ===============================

top20 = importance.head(20)

plt.figure(figsize=(12,8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig("model/feature_importance.png")

plt.show()

print("\nFeature importance graph saved!")