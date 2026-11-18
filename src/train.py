"""Train and evaluate the Iris classifier, then save it to models/."""
from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # Scaling is included in the pipeline so training and inference use the same steps.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "target_names": iris.target_names.tolist(),
         "feature_names": iris.feature_names},
        MODEL_PATH
    )
    metrics = {
        "accuracy": accuracy,
        "test_samples": int(len(y_test)),
        "classification_report": classification_report(
            y_test, predictions, target_names=iris.target_names, output_dict=True
        )
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Metrics saved to: {METRICS_PATH}")
    return accuracy

if __name__ == "__main__":
    train_model()
