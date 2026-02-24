import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    scaler = StandardScaler()

    source = "data/processed_data/"
    X_train = pd.read_csv(f"{source}X_train.csv")
    X_test = pd.read_csv(f"{source}X_test.csv")

    scaler.fit(X_train)

    def scale_and_export(df, name):
        scaled = scaler.transform(df)
        scaled_df = pd.DataFrame(
            scaled,
            columns=df.columns,
            index=df.index
        )
        scaled_df.to_csv(f"data/processed_data/{name}", index=False)

    scale_and_export(X_train, "X_train_scaled.csv")
    scale_and_export(X_test, "X_test_scaled.csv")

if __name__ == "__main__":
    main()