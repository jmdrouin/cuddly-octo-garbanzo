import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib

def main():
    # Load data:
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    # Find best params:
    model = Ridge()
    param_grid = {
        "alpha": np.logspace(-4, 4, 17)
    }
    gs = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train_scaled, y_train)

    # Export model specs:
    model_spec = {
        "model_class": Ridge,
        "best_params": gs.best_params_,
    }
    joblib.dump(model_spec, "models/model_spec.pkl")

if __name__ == "__main__":
    main()