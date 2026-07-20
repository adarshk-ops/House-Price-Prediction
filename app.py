from flask import Flask, render_template, request
from predict import predict_price

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        form={}
    )


@app.route("/predict", methods=["POST"])
def predict():

    form = request.form

    try:
        area_type = form.get("area_type", "").strip()
        location = form.get("location", "").strip()

        total_sqft = form.get("total_sqft", "").strip()
        bath = form.get("bath", "").strip()
        balcony = form.get("balcony", "").strip()
        bhk = form.get("bhk", "").strip()

        # Validation
        if not total_sqft:
            raise ValueError("Please enter Total Square Feet.")

        if not bath:
            raise ValueError("Please enter Bathrooms.")

        if not balcony:
            raise ValueError("Please enter Balconies.")

        if not bhk:
            raise ValueError("Please enter BHK.")

        total_sqft = float(total_sqft)
        bath = int(bath)
        balcony = int(balcony)
        bhk = int(bhk)

        if total_sqft <= 0:
            raise ValueError("Total Square Feet must be greater than zero.")

        if bath < 0 or balcony < 0 or bhk < 0:
            raise ValueError("Values cannot be negative.")

        price = predict_price(
            total_sqft,
            bath,
            balcony,
            bhk,
            location,
            area_type
        )

        return render_template(
            "index.html",
            prediction=f"₹ {price} Lakhs",
            error=None,
            form=form
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=None,
            error=str(e),
            form=form
        )


if __name__ == "__main__":
    app.run(debug=True)