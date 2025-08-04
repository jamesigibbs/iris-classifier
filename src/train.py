import os
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def train_model(save_model_path="outputs/model.pkl", save_fig_path="outputs/confusion_matrix.png"):
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Feature names:", iris.feature_names)
    print("Target names:", iris.target_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    # Save model
    joblib.dump(model, save_model_path)
    print(f"Model saved to: {save_model_path}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save confusion matrix image
    plt.savefig(save_fig_path)
    print(f"Confusion matrix saved to: {save_fig_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export an Iris classifier.")
    parser.add_argument("--save_model_path", type=str, default="outputs/model.pkl",
                        help="Path to save the trained model (default: outputs/model.pkl)")
    parser.add_argument("--save_fig_path", type=str, default="outputs/confusion_matrix.png",
                        help="Path to save the confusion matrix image (default: outputs/confusion_matrix.png)")
    args = parser.parse_args()

    train_model(save_model_path=args.save_model_path, save_fig_path=args.save_fig_path)


