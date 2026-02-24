import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import json

def main():
    X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")
    model = joblib.load("models/trained_model.pkl")

    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({
        "y_true": y_test.iloc[:, 0],
        "y_pred": y_pred
    })
    results.to_csv("data/processed_data/predictions.csv")

    scores = {
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()