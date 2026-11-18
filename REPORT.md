# END-TO-END IRIS FLOWER CLASSIFICATION USING MLOPS

**Student Name:** __________________________  
**Roll Number:** __________________________  
**Class / Division:** ______________________  
**Subject:** Machine Learning Operations (MLOps)  
**Institute:** _____________________________  
**Academic Year:** _________________________  

## Abstract
Machine Learning Operations (MLOps) combines machine learning, software engineering,
and operational practices to manage the lifecycle of machine learning systems.
This project develops an Iris Flower Classification model and demonstrates
foundational MLOps practices: data handling, model training and evaluation,
version control, API deployment, containerization, automated testing, and logging.
A Random Forest classifier is trained using the Iris dataset available in
scikit-learn. FastAPI exposes the model through a REST endpoint, Docker packages
the application, and GitHub Actions runs training and tests after code changes.
The project is intended as a small educational demonstration that can run locally.

## 1. Introduction
A model that performs well in a notebook is not automatically ready for real-world
use. It must be packaged with its dependencies, made accessible to applications,
tested after code changes, and monitored after deployment. MLOps applies automation
and software engineering discipline to these tasks. This project demonstrates a
small end-to-end workflow rather than a production-scale platform.

## 2. Problem Statement
Manual model development and deployment can lead to inconsistent environments,
difficult-to-track code changes, and errors that are discovered late. The project
addresses these issues by building a repeatable training process, exposing the
model as an API, packaging it with Docker, and automating tests with CI.

## 3. Objectives
1. Load and prepare a standard classification dataset.
2. Train and evaluate a machine learning classifier.
3. Save the trained model for reuse.
4. Track source code using Git and GitHub.
5. Serve predictions through FastAPI.
6. Package the application using Docker.
7. Run automated tests using GitHub Actions.
8. Record prediction events and basic application logs.

## 4. Dataset Description
The Iris dataset is included with scikit-learn. It contains 150 flower observations,
four numerical input features (sepal length, sepal width, petal length, petal width),
and three target classes: setosa, versicolor, and virginica. The dataset is used
for educational classification tasks. The project uses an 80:20 train/test split
with stratification and a fixed random seed.

## 5. Tools and Technologies
- Python 3.11: primary programming language.
- scikit-learn: dataset access, preprocessing, classifier, and evaluation.
- Joblib: saving and loading the fitted model bundle.
- FastAPI: REST API for prediction.
- Pydantic: request validation.
- Pytest: automated tests.
- Git and GitHub: source-code version control and collaboration.
- Docker: packaging the API and dependencies into a container.
- GitHub Actions: continuous integration workflow.

## 6. System Architecture
The workflow is:

Iris dataset → train/test split → preprocessing and Random Forest training →
evaluation → saved model → FastAPI prediction service → Docker container.

GitHub stores source code. A GitHub Actions workflow runs when code is pushed
to the main branch or a pull request targets main. It installs dependencies,
trains the model, and executes automated tests. The API records successful
prediction requests in a local log file.

## 7. Methodology and Implementation
### 7.1 Data loading and preprocessing
The Iris dataset is loaded using `sklearn.datasets.load_iris`. The features and
labels are separated and split into training and test sets. A scikit-learn
Pipeline combines StandardScaler and RandomForestClassifier so the preprocessing
step is fitted during training and consistently applied during prediction.

### 7.2 Model training
A Random Forest classifier with 100 trees is trained. A fixed random state is
used to make the experiment repeatable.

### 7.3 Model evaluation
The model is evaluated on the held-out test set. Accuracy and a classification
report containing precision, recall, and F1-score are saved in
`models/metrics.json`. The actual measured values should be copied from that
file after running the project; they should not be guessed.

### 7.4 Model serialization
Joblib saves the fitted pipeline together with class names and feature names in
`models/iris_model.joblib`. This allows the API to load the trained model without
retraining it for every request.

### 7.5 API deployment
FastAPI provides:
- `GET /`: basic service information.
- `GET /health`: application health and model-loaded status.
- `POST /predict`: accepts four flower measurements and returns a predicted class.

Pydantic validates that measurements are positive and within a reasonable
educational input range. Interactive API documentation is available at `/docs`.

### 7.6 Containerization
The Dockerfile starts from a Python 3.11 slim image, installs dependencies,
copies the project, trains the model inside the image, and starts Uvicorn.
This provides a consistent runtime environment for local demonstration.

### 7.7 Continuous Integration
The GitHub Actions workflow runs on pushes to `main` and pull requests targeting
`main`. It checks out the repository, configures Python, installs dependencies,
trains the model, and runs Pytest. This catches basic integration and API errors
early. This is CI; it does not automatically deploy to a remote production server.

### 7.8 Monitoring and logging
The API writes a timestamp, input measurements, and predicted class to
`logs/predictions.log`. This provides basic request and error investigation
capability. It is not advanced model monitoring or drift detection.

## 8. Testing
The test suite checks:
1. The home endpoint returns a successful response.
2. The health endpoint reports a healthy service.
3. A known Setosa example is classified as setosa.
4. Invalid negative measurements are rejected with HTTP 422.

Record the actual test outcome after running `pytest -v`.

## 9. Results
Complete this table after executing the project:

| Measure | Actual result |
|---|---|
| Test-set accuracy | __________________ |
| Number of test samples | 30 |
| API prediction test | Pass / Fail: __________ |
| Invalid-input test | Pass / Fail: __________ |
| Docker run | Pass / Fail: __________ |
| GitHub Actions CI | Pass / Fail: __________ |

Attach screenshots of model training output, Swagger `/docs` prediction,
health endpoint, passing Pytest output, Docker container, GitHub Actions run,
and prediction log.

## 10. Advantages
- Repeatable model training using a fixed split and random seed.
- Reusable saved model.
- Clear API interface for predictions.
- Consistent dependencies through containerization.
- Automated tests after code changes.
- Basic logging for troubleshooting.

## 11. Limitations
This is a small educational project using a standard dataset. It does not provide
production-grade authentication, cloud hosting, Kubernetes orchestration, automated
retraining, centralized experiment tracking, or advanced data/model drift detection.
The local log file is not a scalable centralized monitoring solution.

## 12. Future Scope
The project can be extended with MLflow for experiment tracking, cloud deployment
on AWS/Azure/GCP, authentication, centralized monitoring, model drift detection,
automated retraining, and Kubernetes orchestration.

## 13. Conclusion
The project demonstrates how MLOps practices can be applied to a simple machine
learning application. It combines model development with version control, API
deployment, Docker containerization, CI testing, and basic logging. The exercise
shows that MLOps covers the complete operational lifecycle of a model, not only
the algorithm used to make predictions.

## References
1. Scikit-learn documentation: https://scikit-learn.org/
2. FastAPI documentation: https://fastapi.tiangolo.com/
3. Docker documentation: https://docs.docker.com/
4. Git documentation: https://git-scm.com/doc
5. GitHub Actions documentation: https://docs.github.com/actions
6. Pytest documentation: https://docs.pytest.org/
