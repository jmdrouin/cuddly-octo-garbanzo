import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("data/raw_data/data.csv")
    target = "silica_concentrate"
    ignore = ["date"]
    X = df.drop(columns=ignore+[target])
    y = df[target]
    X_test, X_train, y_test, y_train = train_test_split(
        X, y, test_size=0.2, random_state=42)

    dest = "data/processed_data/"
    X_train.to_csv(f"{dest}X_train.csv", index=False)
    X_test.to_csv(f"{dest}X_test.csv", index=False)
    y_train.to_csv(f"{dest}y_train.csv", index=False)
    y_test.to_csv(f"{dest}y_test.csv", index=False)

if __name__ == "__main__":
    main()