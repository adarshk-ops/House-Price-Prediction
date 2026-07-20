import json
import joblib
import pandas as pd
import os

# ==========================================
# Load Model
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
COLUMN_PATH = os.path.join(BASE_DIR, "model", "columns.json")

model = joblib.load(MODEL_PATH)

with open(COLUMN_PATH, "r") as f:
    data_columns = json.load(f)["data_columns"]


# ==========================================
# Prediction Function
# ==========================================

def predict_price(total_sqft, bath, balcony, bhk, location, area_type):

    # Clean user input
    location = location.strip().lower()
    area_type = area_type.strip().lower()

    # Create dataframe with all zeros
    input_df = pd.DataFrame(
        [[0] * len(data_columns)],
        columns=data_columns
    )

    # Numerical features
    input_df.at[0, "total_sqft"] = float(total_sqft)
    input_df.at[0, "bath"] = int(bath)
    input_df.at[0, "balcony"] = int(balcony)
    input_df.at[0, "bhk"] = int(bhk)

    # --------------------------------------
# Match Location
# --------------------------------------

    location = " ".join(location.split()).lower()

    for column in data_columns:

        if column.startswith("location_"):

            actual_location = " ".join(
            column.replace("location_", "").split()
        ).lower()

            if actual_location == location:

                input_df.at[0, column] = 1
                break

   # --------------------------------------
# Match Area Type (Ignore Extra Spaces)
# --------------------------------------

    area_type = " ".join(area_type.split()).lower()

    for column in data_columns:

        if column.startswith("area_type_"):

            actual_area = " ".join(
            column.replace("area_type_", "").split()
        ).lower()

            if actual_area == area_type:

                input_df.at[0, column] = 1
                break
    print("\n==============================")
    print(input_df.loc[:, (input_df != 0).any(axis=0)])
    print("==============================\n")

    # Predict
    prediction = model.predict(input_df)[0]

    return round(prediction, 2)


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    print()

    print(
        predict_price(
            1200,
            2,
            2,
            2,
            "Whitefield",
            "Super built-up Area"
        )
    )

    print(
        predict_price(
            2500,
            3,
            2,
            3,
            "Electronic City",
            "Built-up Area"
        )
    )

    print(
        predict_price(
            500,
            1,
            1,
            1,
            "Yelahanka",
            "Plot Area"
        )
    )