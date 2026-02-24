import joblib
import pandas as pd

def main():
    spec = joblib.load("models/model_spec.pkl")
    model = spec["model_class"](**spec["best_params"])

    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    model.fit(X_train_scaled, y_train)
    joblib.dump(model, "models/trained_model.pkl")

if __name__ == "__main__":
    main()