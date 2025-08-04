import os
import joblib
import argparse
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def train_model(save_path="outputs/model.pkl"):
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Show dataset info
    print("Feature names:", iris.feature_names)
    print("Target names:", iris.target_names)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and print first few results
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save Iris classifier")
    parser.add_argument("--save_path", type=str, default="outputs/model.pkl",
                        help="Path to save the trained model")
    args = parser.parse_args()

    train_model(save_path=args.save_path)
