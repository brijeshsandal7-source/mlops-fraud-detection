import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    return parser.parse_args()
def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data_path)

    # Separate features and target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Encode categorical features
    if "transaction_type" in X.columns:
        encoder = LabelEncoder()
        X["transaction_type"] = encoder.fit_transform(X["transaction_type"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"Model saved at {model_path}")
if __name__ == "__main__":
    main()

