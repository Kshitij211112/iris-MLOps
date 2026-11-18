# Iris Flower Classification with MLOps

A beginner-friendly local MLOps project demonstrating model training, evaluation,
serialization, REST API deployment, Docker packaging, Git version control,
GitHub Actions CI, and prediction logging.

## 1. Requirements
- Python 3.11 recommended
- Git
- VS Code (optional)
- Docker Desktop (only for the Docker section)

## 2. Setup on macOS

Open Terminal and move into this project folder:

```bash
cd iris-mlops-project
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.11` is not installed, install Python 3.11 first (for example with
Homebrew: `brew install python@3.11`), then repeat the commands.

## 3. Train the model

```bash
python -m src.train
```

This downloads/loads the built-in Iris dataset from scikit-learn, splits it into
training and testing sets, trains a preprocessing + Random Forest pipeline,
evaluates it, and creates:
- `models/iris_model.joblib`
- `models/metrics.json`

The dataset is built into scikit-learn, so no manual CSV download is needed.
The training/test split uses a fixed random seed for repeatability.

## 4. Start the API

```bash
uvicorn src.app:app --reload
```

Open:
- API home: http://127.0.0.1:8000/
- Interactive Swagger documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

In `/docs`, open `POST /predict`, select **Try it out**, and submit:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Expected species: `setosa`. Stop the server with `Ctrl+C`.

## 5. Run automated tests

Make sure you have trained the model first, then run:

```bash
pytest -v
```

Tests cover the home endpoint, health endpoint, a known prediction, and invalid
input validation.

## 6. Docker

Install and start Docker Desktop. From the project root:

```bash
docker build -t iris-mlops:1.0 .
docker run --name iris-api -p 8000:8000 iris-mlops:1.0
```

Visit http://127.0.0.1:8000/docs. Stop the container with `Ctrl+C` in its terminal,
or from another terminal run:

```bash
docker stop iris-api
docker rm iris-api
```

The Dockerfile installs dependencies, trains the model inside the image, and
starts the API. This makes the image self-contained for this demonstration.

## 7. Git and GitHub

```bash
git init
git add .
git commit -m "Initial Iris MLOps project"
git branch -M main
```

Create an empty GitHub repository, then use the remote URL shown by GitHub:

```bash
git remote add origin https://github.com/YOUR-USERNAME/iris-mlops-project.git
git push -u origin main
```

Do not type the placeholder URL literally. Replace it with your own repository URL.

Every push to `main` and every pull request targeting `main` triggers the GitHub
Actions workflow in `.github/workflows/ci.yml`. It installs dependencies, trains
the model, and runs pytest. Check the **Actions** tab in GitHub for the result.

## 8. Monitoring

Every successful prediction is recorded in `logs/predictions.log` with a timestamp,
predicted class, and input values. This is basic application/request monitoring.
It does not automatically detect data drift or calculate live production accuracy.
For a real production system, monitoring would also compare predictions with
later-confirmed ground-truth labels.

## 9. Important project files

- `src/train.py`: training, evaluation, model saving
- `src/app.py`: API endpoints, input validation, logging
- `tests/test_api.py`: automated API tests
- `Dockerfile`: container build instructions
- `.github/workflows/ci.yml`: CI automation
- `models/metrics.json`: measured evaluation results

## 10. Troubleshooting

- `python3.11: command not found`: install Python 3.11 and retry.
- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt`.
- Model file missing: run `python -m src.train` from the project root.
- Port 8000 already in use: stop the other server or run `uvicorn src.app:app --port 8001`.
- Docker command fails: start Docker Desktop and wait until it says Docker is running.
- CI fails: open the failed GitHub Actions run and inspect the first error.

## 11. Scope and limitations

This is an educational, local demonstration. It uses a small built-in dataset and
a simple classifier. It does not include production-grade authentication, cloud
hosting, Kubernetes, automatic retraining, or advanced drift detection. These
are possible future extensions.
# iris-MLOps
