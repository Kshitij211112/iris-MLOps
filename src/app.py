"""FastAPI application serving the trained Iris classifier."""
from pathlib import Path
import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_model.joblib"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Flower Classification API",
    description="A beginner-friendly ML model served with FastAPI.",
    version="1.0.0"
)

class FlowerFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm")

def load_model_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. Run 'python -m src.train' before starting the API."
        )
    return joblib.load(MODEL_PATH)

bundle = None

@app.on_event("startup")
def startup_event():
    global bundle
    bundle = load_model_bundle()
    logger.info("API started and model loaded")

@app.get("/")
def home():
    return {"message": "Iris ML API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bundle is not None}

@app.post("/predict")
def predict(features: FlowerFeatures):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    values = [[
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width
    ]]
    prediction = int(bundle["model"].predict(values)[0])
    species = bundle["target_names"][prediction]
    logger.info("prediction=%s input=%s", species, values[0])
    return {"prediction": species, "class_id": prediction}
